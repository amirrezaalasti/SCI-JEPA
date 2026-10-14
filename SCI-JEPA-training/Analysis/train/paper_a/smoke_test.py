"""
CPU smoke test on the synthetic graph: the invariants checked directly, then a few
representative arms (one per experiment family, all three KU modes, the reasoning-task
matrix) train for a few steps and evaluate. About a minute.

    python -m train.paper_a.smoke_test           # small (default)
    python -m train.paper_a.smoke_test --full    # every arm of every experiment
"""
import math
import sys

import numpy as np
import torch

from train.paper_a.config import EXPERIMENTS, SIZES, PaperAConfig
from train.paper_a.graph_index import HUB, get_index
from train.paper_a.instances import Sampler, collate
from train.paper_a.model import PaperAModel, to_device
from train.paper_a.train import train_one

FAST = dict(synthetic=True, embedder="hash", device="cpu", time_T=2016, proxy_tau=0.6, steps=12, batch_size=16, log_every=1000, warmup=5,
            eval_instances=48, probe_papers=15, seeds=[0], eval_tasks=False, task_eval_instances=32, eval_worldview=False, save_model=False,
            epochs=0.0)                                   # fixed 12 steps instead of the paper's 10 epochs
# the small test: one arm per experiment family (tag, with the reasoning tasks on or off)
SMALL = [("BENCH", "text_baseline", True), ("E10", "N=8", False), ("E11", "inv=0.3", False), ("E12", "reg=within", False),
         ("E4", "frozen+census", False), ("E5", "contrast=on", False), ("E15", "only=result", False),
         ("E17", "section", True), ("E17", "paper_masked", True), ("E18", "pm_only=claim", False),
         ("E19", "world_model", True), ("E20", "graph_jepa+world", True), ("APPX_EVAL", "reference_old_protocol", False), ("E1", "shacl=tight", False)]


def check_invariants(gi, cfg):
    cfg = cfg.override(task_mask_p=0.0)                  # the invariants test the masking policy alone
    s = Sampler(gi, cfg.override(target_types=("claim",)), "train", 0)
    leaky = Sampler(gi, cfg.override(target_types=("claim",), mask_policy="token"), "train", 0)
    safe_i, leak_i = s.sample(64), leaky.sample(64)
    kids_vis = lambda insts: np.mean([np.isin(i["ctx"]["comp"], gi.children.get(i["target_comp"], [])).any()
                                      for i in insts])
    assert kids_vis(safe_i) == 0.0 and kids_vis(leak_i) > 0.9, (kids_vis(safe_i), kids_vis(leak_i))
    sim = lambda insts: np.mean([i["audit"]["max_ctx_sim"] for i in insts])
    print(f"  [ok] leak-safe masking: claim's evidence/implications visible in "
          f"{kids_vis(leak_i):.0%} (token only) -> {kids_vis(safe_i):.0%}; "
          f"max ctx sim {sim(leak_i):.3f} -> {sim(safe_i):.3f}")
    sib = Sampler(gi, cfg.override(target_types=("method",), mask_policy="token+edges+children+siblings"), "train", 0)
    proc = gi.comp_type == __import__("train.paper_a.ku_schema", fromlist=["x"]).COMP_TYPES.index("procedure")
    for i in sib.sample(16):
        own = i["ctx"]["slot"] == 0
        assert not np.any(proc[i["ctx"]["comp"][own & (i["ctx"]["comp"] >= 0)]]), "procedure visible"
    print("  [ok] siblings policy hides the masked method's procedure")
    assert gi.n_papers == 300 and len(gi.papers("test")) > 0
    names = set(gi.field_names)
    print(f"  [ok] KU graph from normalized-schema records: {gi.summary()['kus_per_paper']} KUs/paper, "
          f"{len(names)} fields, splits {gi.summary()['splits']}")
    out = gi.summary()["build"]["ku_outcomes"]
    assert out.get("contradicting_evidence:null_statement", 0) > 0 and out.get("figures:placeholder", 0) == 0
    print(f"  [ok] null statements filtered: {out.get('contradicting_evidence:null_statement')}, "
          f"placeholders: {out.get('contradicting_evidence:placeholder', 0)}")

    for inst in s.sample(32):
        c, m = inst["ctx"], inst["mask_pos"]
        assert np.all(c["x"][m] == 0), "masked token still carries features"
        assert not np.any(c["comp"][~inst["is_mask"]] == inst["target_comp"]), "target visible twice"
        assert set(np.unique(c["rel"][m])) <= {0, 1, 2}, "masked token kept a role edge"
        hubs = c["type"] == HUB
        assert np.all(c["x"][hubs].shape[0] == len(inst["papers"]))
    print("  [ok] masked token: no features, no role edges, not duplicated; one hub per paper")

    n1 = Sampler(gi, cfg.override(n_subgraphs=1), "train", 0).sample(16)
    n8 = Sampler(gi, cfg.override(n_subgraphs=8), "train", 0).sample(16)
    assert all(len(i["papers"]) == 1 for i in n1) and all(len(i["papers"]) == 8 for i in n8)
    tr = set(gi.papers("train").tolist())
    assert all(set(i["papers"]) <= tr for i in n8), "neighbourhood leaked across splits"
    print("  [ok] neighbourhood size N honoured and confined to the split")

    model = PaperAModel(cfg)
    batch = to_device(collate(n8), "cpu")
    with torch.no_grad():
        out = model(batch)
    assert torch.allclose(out["target"], model.encode_target(batch["tgt"], batch["target_pos"]))
    print("  [ok] target encoder is deterministic and gradient-free")


def main():
    cfg = PaperAConfig().override(**SIZES["tiny"], **FAST)
    gi = get_index(cfg)
    check_invariants(gi, cfg)
    failures = []
    if "--full" in sys.argv:
        todo = [(e, t, o, False) for e, arms in EXPERIMENTS.items() for t, o in arms]
    else:
        todo = [(e, t, dict(EXPERIMENTS[e])[t], tasks) for e, t, tasks in SMALL]
    for exp, tag, over, tasks in todo:
        if True:
            try:
                acfg = cfg.override(**over, eval_tasks=tasks, eval_worldview=tasks)
                agi = get_index(acfg) if acfg.ku_mode != cfg.ku_mode else gi   # E16: its own KU graph
                if acfg.baseline:
                    from train.paper_a.baselines import text_baseline
                    m = text_baseline(acfg, agi, "test")
                    print(f"  [ok] {exp:<5s} {tag:<22s} balanced MRR {m['retrieval_balanced']['MRR']:.3f} "
                          f"(chance {m['retrieval_balanced']['chance_MRR']:.3f}) forward {m['tasks']['forward']['ratio']:.2f}x")
                    continue
                m = train_one(acfg, agi, 0, log=lambda *_: None)
                bm = m["retrieval_balanced"]["MRR"]
                assert math.isfinite(bm) and math.isfinite(m["history"][-1]["jepa"])
                print(f"  [ok] {exp:<4s} {tag:<22s} balanced MRR {bm:.3f} chi {m['target_space']['chi']:.3f} "
                      f"probe {m['partition_probe']['probe_acc']:.3f} inv {m['inverse_acc']:.3f} "
                      f"det {m.get('detector', {}).get('discrimination', float('nan')):+.3f}")
                if tasks:
                    s_, a_ = m["structure"], m["actions"]
                    print(f"       world: x chance with edges {s_['with']:.2f} / without {s_['without']:.2f}; "
                          f"action direction swap {a_['swap']['direction_cos']} (n={a_['swap']['n']}) "
                          f"flip {a_['flip']['direction_cos']} (n={a_['flip']['n']})")
                    T = m["tasks"]
                    assert all(k in T for k in ("forward", "inverse", "abduction", "consistency", "intervention")), T
                    print(f"       tasks: forward {T['forward']['ratio']:.2f}x  inverse {T['inverse']['ratio']:.2f}x  "
                          f"abduction {T['abduction']['ratio']:.2f}x  consistency {T['consistency']['bal_acc']}  "
                          f"intervention n={T['intervention']['condition']['n']}")
            except Exception as e:                       # noqa: BLE001
                failures.append((exp, tag, repr(e)))
                print(f"  [FAIL] {exp} {tag}: {e!r}")
    print(f"\n{'ALL ARMS RAN' if not failures else f'{len(failures)} FAILURE(S)'}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
