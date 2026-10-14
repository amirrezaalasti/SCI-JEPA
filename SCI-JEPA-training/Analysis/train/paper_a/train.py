"""
Paper A runner.

    # one configuration
    python -m train.paper_a.train --size S --set n_subgraphs=8 lambda_inv=0.3

    # an ablation from the plan (all arms x all seeds, paired tests vs the first arm)
    python -m train.paper_a.train --exp E10 --size S

    # Theorem A2 tightness audit of the shape vocabulary (no training)
    python -m train.paper_a.train --audit-shapes [--set shape_spec=path.json]

    # smoke test on a synthetic graph, CPU, ~1 min
    python -m train.paper_a.train --synthetic --size tiny --exp E10 --set steps=60 seeds=[0,1]

Outputs (cfg.out_dir/<exp>/): one JSON per arm x seed, and summary.json with
mean / std per metric and paired t-tests against the reference arm.
"""
import argparse
import ast
import json
import math
import os
import time

import numpy as np
import torch

from train.paper_a import losses as L
from train.paper_a.config import EXPERIMENTS, SIZES, PaperAConfig, resolve_device
from train.paper_a.diagnostics import evaluate
from train.paper_a.graph_index import get_index
from train.paper_a.instances import Sampler, collate, minimal_pair
from train.paper_a.model import PaperAModel, to_device
from train.paper_a.neurons import neuron_report
from train.paper_a.shapes import ShapeBank, SlotLabels, audit_tightness

# Metrics compared across arms in summary.json; (path, higher_is_better).
KEY_METRICS = [
    (("retrieval_balanced", "MRR"), True),
    (("retrieval_standard", "MRR"), True),
    (("retrieval_standard", "Hits@1"), True),
    (("target_space", "chi"), True),
    (("target_space", "chi_raw"), True),
    (("context_space", "chi"), True),
    (("partition_probe", "between_subgraph_var_share"), False),
    (("partition_probe", "probe_acc"), False),
    (("relation_sensitivity",), True),
    (("inverse_acc",), True),
    (("detector", "discrimination"), True),
]


def _get(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _grad_norms(model):
    groups = {"encoder": model.context, "predictor": model.predictor, "inverse": model.inverse}
    if model.shapes is not None:
        groups["shapes"] = model.shapes
    out = {}
    for k, mod in groups.items():
        sq = [p.grad.detach().float().pow(2).sum() for p in mod.parameters() if p.grad is not None]
        out[k] = float(torch.stack(sq).sum().sqrt()) if sq else 0.0
    return out


@torch.no_grad()
def _step_stats(out, batch, cfg):
    """Per-step learning signals: loss per target KU type, prediction/target
    agreement, and whether the predictions are collapsing (std across the batch)."""
    pred, tgt, tt = out["pred"].float(), out["target"].float(), batch["target_type"]
    per = (pred - tgt).pow(2).mean(-1)
    cos = torch.nn.functional.cosine_similarity(pred, tgt, dim=-1)
    rec = {"cos_pred_target": float(cos.mean()),
           "pred_std": float(pred.std(0).mean()), "target_std": float(tgt.std(0).mean()),
           "pred_pairwise_cos": float(torch.nn.functional.normalize(pred - pred.mean(0), dim=-1)
                                      .matmul(torch.nn.functional.normalize(pred - pred.mean(0), dim=-1).T).mean()),
           "tokens_per_instance": float((~batch["ctx"]["pad"]).sum(1).float().mean())}
    from train.paper_a.ku_schema import COMP_TYPES
    rec["loss_by_type"] = {COMP_TYPES[int(t)]: float(per[tt == t].mean()) for t in tt.unique()}
    return rec


def mask_plan(cfg, sampler):
    """-> (n_steps, plan_fn, info). plan_fn(step) -> ({type: count} or None, phase label).

      mixed        every target type, in proportion to its KU count (None = one shared pool)
      balanced     equal numbers of each type in every batch
      round_robin  batch k is all one type, cycling through the types
      sequential   one phase per type, in target_types order: a full pass over every KU
                   of that type before moving on; replay_frac of each batch re-draws the
                   types of earlier phases (0 = pure sequential, prone to forgetting)
    With epochs > 0 the run is `epochs` full passes over all target KUs; sequential phases
    are sized by each type's KU count, so each type is masked `epochs` times over."""
    types = [t for t in cfg.target_types if sampler.n_anchors(t) > 0]
    missing = [t for t in cfg.target_types if t not in types]
    if missing:
        print(f"[paper_a] no KUs of type {missing} in this split: skipped")
    counts = {t: sampler.n_anchors(t) for t in types}
    B = cfg.batch_size
    steps = int(math.ceil(sum(counts.values()) * cfg.epochs / B)) if cfg.epochs > 0 else cfg.steps
    sched = cfg.mask_schedule
    bounds, acc = [], 0.0
    for t in types:                                    # sequential phase ends
        acc += counts[t] / max(sum(counts.values()), 1) * steps
        bounds.append((t, int(round(acc))))

    def split(n, keys):
        base, rem = divmod(n, len(keys))
        return {k: base + (i < rem) for i, k in enumerate(keys)}

    def plan(step):
        if sched == "mixed":
            return None, "mixed"
        if sched == "balanced":
            return split(B, types), "balanced"
        if sched == "round_robin":
            t = types[step % len(types)]
            return {t: B}, t
        if sched == "sequential":
            k = next(i for i, (_, end) in enumerate(bounds) if step < end or i == len(bounds) - 1)
            t, prev = types[k], types[:k]
            r = int(round(cfg.replay_frac * B)) if prev else 0
            out = {t: B - r}
            for pt, n in split(r, prev).items() if r else []:
                out[pt] = out.get(pt, 0) + n
            return out, t
        raise ValueError(f"unknown mask_schedule {sched!r}")

    info = {"schedule": sched, "steps": steps, "types": types, "anchors_per_type": counts,
            "phases": [{"type": t, "end_step": e} for t, e in bounds] if sched == "sequential" else None,
            "replay_frac": cfg.replay_frac if sched == "sequential" else None}
    return steps, plan, info


def epoch_checkpoint(model, gi, cfg, dev, epoch, step, train_jepa, sec, live=None):
    """Validation + neuron summary at the end of one epoch (one row of the
    dashboard's Epochs tab)."""
    m = evaluate(model, gi, cfg.override(eval_instances=cfg.epoch_eval_instances,
                                         probe_papers=min(cfg.probe_papers, 80)), dev, None, split="val")
    rb, pt = m["retrieval_balanced"], m.get("per_type", {})
    rec = {"epoch": epoch, "step": step, "sec": round(sec, 1), "train_jepa": train_jepa,
           "bal_MRR": rb["MRR"], "bal_chance": rb.get("chance_MRR"), "bal_H1": rb["Hits@1"], "bal_H10": rb["Hits@10"],
           "std_MRR": m["retrieval_standard"]["MRR"], "std_chance": m["chance_MRR_standard"],
           "chi": m["target_space"]["chi"], "eff_rank": m["target_space"]["eff_rank"],
           "probe_acc": m["partition_probe"].get("probe_acc"), "probe_chance": m["partition_probe"].get("probe_chance"),
           "between_var": m["partition_probe"].get("between_subgraph_var_share"),
           "inverse_acc": m["inverse_acc"], "rel_sens": m["relation_sensitivity"],
           "flagged": m["audit"]["flagged_frac"],
           "per_type": {k: {"MRR": v["MRR_bal"], "chance": v["chance_bal"], "n": v["n"],
                            "same_paper": v["top1_same_paper"]} for k, v in pt.items()},
           "types_learned": sum(v["MRR_bal"] >= 1.5 * v["chance_bal"] for v in pt.values())}
    if cfg.eval_tasks:
        from train.paper_a.tasks import task_eval
        rec["tasks"] = task_eval(model, gi, cfg, dev, "val")
    if cfg.eval_worldview:
        from train.paper_a.worldview import action_eval, structure_reliance
        rec["structure"] = structure_reliance(model, gi, cfg.override(eval_instances=cfg.epoch_eval_instances,
                                                                      probe_papers=min(cfg.probe_papers, 80)),
                                              dev, "val", base=m)
        rec["actions"] = action_eval(model, gi, cfg, dev, "val", n=min(200, cfg.epoch_eval_instances))
        if live is not None:                          # the World model tab updates every epoch
            from train.paper_a.worldview import examples
            live.worldview(step, epoch, examples(model, gi, cfg, dev, "val"))
    if cfg.neuron_report:
        rep = neuron_report(model, gi, cfg, dev, "val", cfg.neuron_instances)
        out = rep["layers"][-1]
        n = max(sum(out["counts"].values()), 1)
        rec["neurons"] = {"identity": out["counts"]["identity"] / n, "content": out["counts"]["content"] / n,
                          "type": out["counts"]["type"] / n, "dead": out["counts"]["dead"] / n,
                          "leak": out["counts"]["leak"] / n,
                          "pred_learned": rep["predictor"]["learned"], "pred_n": rep["predictor"]["n"]}
        if live is not None:
            live.neurons(step, epoch, rep)
    if live is not None:
        live.epoch_end(rec)
    return rec


def save_model(model, cfg, save_dir, metrics=None, best_state=None, gi=None):
    """Write a reusable model folder:
       model.safetensors  weights at the end of training (all modules, incl. the EMA target encoder)
       best.safetensors   weights at the best end-of-epoch validation score (if tracked)
       config.json        the full PaperAConfig, so inference rebuilds the identical model
       summary.json       test metrics, reasoning tasks, data fingerprint, text encoder
    Load with train.paper_a.inference.SciJEPA.from_pretrained(save_dir)."""
    from safetensors.torch import save_file
    os.makedirs(save_dir, exist_ok=True)
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(save_dir, "model.safetensors"))
    if best_state is not None:
        save_file({k: v.contiguous() for k, v in best_state["state"].items()}, os.path.join(save_dir, "best.safetensors"))
    json.dump(cfg.as_dict(), open(os.path.join(save_dir, "config.json"), "w"), indent=2, default=str)
    summ = {"best_epoch": best_state["epoch"] if best_state else None,
            "best_val_bal_MRR": best_state["score"] if best_state else None,
            "text_encoder": cfg.embed_model, "feat_dim": cfg.feat_dim, "ku_mode": cfg.ku_mode,
            "params_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad))}
    if gi is not None:
        from train.paper_a.ku_graph import data_fingerprint
        summ["data_fingerprint"] = data_fingerprint(cfg)
    if metrics:
        rb = metrics.get("retrieval_balanced", {})
        summ["test"] = {"balanced_MRR": rb.get("MRR"), "chance": rb.get("chance_MRR"),
                        "tasks": {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (dict, list))}
                                  for k, v in (metrics.get("tasks") or {}).items() if isinstance(v, dict)}}
    json.dump(summ, open(os.path.join(save_dir, "summary.json"), "w"), indent=2, default=str)
    return save_dir


def train_one(cfg, gi, seed, log=print, live=None, save_dir=None):
    """live: optional telemetry.LiveRun -- streams every log step and every
    cfg.eval_every steps a quick validation pass to the dashboard.
    save_dir: write the trained model there (see save_model); None = do not save."""
    set_seed(seed)
    dev = torch.device(resolve_device(cfg.device))
    if gi.comp_x.shape[1] != cfg.feat_dim:            # node features come from the index's text encoder
        log(f"  [model] feat_dim {cfg.feat_dim} -> {gi.comp_x.shape[1]} (dimension of the KU index's text encoder)")
        cfg = cfg.override(feat_dim=int(gi.comp_x.shape[1]))
    model = PaperAModel(cfg)
    # optional modules draw from their own seed AFTER the shared model, so every
    # arm of an ablation starts from identical encoder / predictor weights
    shapes = None
    if cfg.lambda_shacl > 0 or cfg.shape_spec:
        torch.manual_seed(seed + 10_007)
        shapes = ShapeBank(cfg.shape_spec, SlotLabels(gi), cfg.d_model, cfg)
        model.shapes = shapes
    model = model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(cfg.warmup, 1)))
    sampler = Sampler(gi, cfg, "train", seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    history, t0 = [], time.time()
    n_steps, plan_fn, sched_info = mask_plan(cfg, sampler)
    log(f"  [mask] schedule={sched_info['schedule']} steps={n_steps} anchors={sched_info['anchors_per_type']}"
        + (f" phases={sched_info['phases']}" if sched_info["phases"] else ""))
    total_anchors = max(sum(sched_info["anchors_per_type"].values()), 1)
    epoch_of = lambda st: round((st + 1) * cfg.batch_size / total_anchors, 4)   # passes over all target KUs
    if live is not None:
        live.state["steps_total"] = n_steps
        live.state["schedule"] = sched_info
        live.state["epochs_total"] = round(n_steps * cfg.batch_size / total_anchors, 3)
        live.state["steps_per_epoch"] = total_anchors / cfg.batch_size
    # end-of-epoch checkpoints: the last step of every full pass over the target KUs
    epoch_ends = {}
    if cfg.eval_at_epochs and cfg.epochs > 0:
        for e in range(1, int(math.floor(cfg.epochs + 1e-9)) + 1):
            epoch_ends[min(int(math.ceil(e * total_anchors / cfg.batch_size)) - 1, n_steps - 1)] = e
    epoch_results, ep_sum, ep_n = [], 0.0, 0
    best_state = None                                  # weights at the best end-of-epoch validation
    phase = None

    for step in range(n_steps):
        plan, phase_now = plan_fn(step)
        if phase_now != phase and cfg.mask_schedule == "sequential":
            log(f"  [mask] step {step}: phase -> masking '{phase_now}'"
                + (f" (+{cfg.replay_frac:.0%} replay)" if cfg.replay_frac and step else ""))
        phase = phase_now
        insts = sampler.sample(cfg.batch_size, plan)
        if not insts:
            continue
        batch = to_device(collate(insts), dev)
        out = model(batch)
        if "pred_extra" in out:                         # E20: every hidden motif node counts equally
            parts = {"jepa": L.jepa_loss(torch.cat([out["pred"], out["pred_extra"]]),
                                         torch.cat([out["target"], out["target_extra"]]), cfg.jepa_loss)}
        else:
            parts = {"jepa": L.jepa_loss(out["pred"], out["target"], cfg.jepa_loss)}
        if "inv_logits" in out:
            if cfg.lambda_inv > 0:                      # E11: the IDM-analogue barrier
                parts["inverse"] = cfg.lambda_inv * L.inverse_loss(out["inv_logits"], out["inv_labels"])
            else:                                       # detached probe: head learns, encoder does not
                b, i, j, _ = batch["probe"]
                h = out["h"].detach()
                parts["inverse_probe"] = L.inverse_loss(model.inverse(h[b, i], h[b, j]), out["inv_labels"])
        if cfg.lambda_reg > 0:
            c = batch["ctx"]
            parts["reg"] = cfg.lambda_reg * L.grouping_regulariser(
                out["h"], c["slot"], c["type"], c["pad"], cfg.reg_grouping, cfg.reg_projections, gen)
        if cfg.lambda_align > 0:
            parts["align"] = cfg.lambda_align * L.align_loss(out["pred"], out["target"],
                                                             batch["target_type"], cfg.align_tau)
        if cfg.lambda_shacl > 0 and shapes is not None:
            parts["shacl"] = cfg.lambda_shacl * shapes.loss(out["pred"], batch["target_comp"],
                                                            batch["target_type"])
        if cfg.lambda_contrast > 0:
            pairs = [minimal_pair(gi, i, sampler.rng, sampler.papers) for i in insts]
            keep = [k for k, p in enumerate(pairs) if p is not None]
            if keep:
                pb = to_device(collate([pairs[k] for k in keep]), dev)
                p2, _ = model.predict(pb)
                parts["contrast"] = cfg.lambda_contrast * L.contrast_margin(
                    out["pred"][keep], p2, cfg.contrast_margin)
        loss = sum(parts.values())
        ep_sum, ep_n = ep_sum + parts["jepa"].detach(), ep_n + 1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        logging_now = step % cfg.log_every == 0 or step == n_steps - 1
        gnorm = _grad_norms(model) if (logging_now and live is not None) else None
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step(); sched.step()
        m = cfg.ema_start + (cfg.ema_end - cfg.ema_start) * step / max(n_steps - 1, 1)
        model.ema(m)

        if logging_now:
            rec = {"step": step, "epoch": epoch_of(step), "phase": phase,
                   **{k: float(v.detach()) for k, v in parts.items()},
                   "flagged": float(np.mean([a["flagged"] for a in batch["audit"]])),
                   "sec": round(time.time() - t0, 1)}
            history.append(rec)
            log("  " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in rec.items()))
            if live is not None:
                live.relbias(model)
                live.step({**rec, "total": float(loss.detach()), "lr": float(sched.get_last_lr()[0]), "ema": m,
                           "grad_norm": gnorm, **_step_stats(out, batch, cfg)})
        if step in epoch_ends:
            rec = epoch_checkpoint(model, gi, cfg, dev, epoch_ends[step], step, float(ep_sum) / max(ep_n, 1),
                                   time.time() - t0, live)
            epoch_results.append(rec)
            ep_sum, ep_n = 0.0, 0
            if save_dir and (best_state is None or rec["bal_MRR"] > best_state["score"]):
                best_state = {"epoch": rec["epoch"], "score": rec["bal_MRR"],
                              "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            log(f"  [epoch {rec['epoch']}] val balanced MRR {rec['bal_MRR']:.4f} (chance {rec['bal_chance']:.4f}) "
                f"train jepa {rec['train_jepa']:.4g} types learned {rec['types_learned']}/{len(rec['per_type'])}")
        if live is not None and cfg.eval_every and step > 0 and step % cfg.eval_every == 0:
            quick = cfg.override(eval_instances=cfg.live_eval_instances, probe_papers=min(cfg.probe_papers, 40))
            live.eval(step, evaluate(model, gi, quick, dev, None, split="val"), epoch=epoch_of(step))
            if cfg.neuron_report and step not in epoch_ends:
                live.neurons(step, epoch_of(step), neuron_report(model, gi, cfg, dev, "val", cfg.neuron_instances))

    metrics = evaluate(model, gi, cfg, dev, shapes)
    if cfg.eval_worldview:
        from train.paper_a import worldview as W
        metrics["structure"] = W.structure_reliance(model, gi, cfg, dev, "test", base=metrics)
        metrics["actions"] = W.action_eval(model, gi, cfg, dev, "test")
        ex = W.examples(model, gi, cfg, dev, "test")
        if live is not None:
            live.worldview(n_steps, epoch_of(n_steps - 1), ex, final=True)
        s_, a_ = metrics["structure"], metrics["actions"]
        log(f"  [world] x chance with typed edges {s_['with']:.2f} / without {s_['without']:.2f} | action direction "
            f"swap {a_['swap']['direction_cos']} flip {a_['flip']['direction_cos']}")
    if cfg.eval_tasks:
        from train.paper_a.tasks import task_eval
        metrics["tasks"] = task_eval(model, gi, cfg, dev, "test")
        log("  [tasks] " + " | ".join(f"{k} {v['ratio']:.2f}x chance" for k, v in metrics["tasks"].items()
                                      if isinstance(v, dict) and v.get("ratio") is not None))
    if live is not None:
        live.relbias(model)
        live.eval(n_steps, evaluate(model, gi, cfg.override(eval_instances=cfg.live_eval_instances,
                                                              probe_papers=min(cfg.probe_papers, 40)),
                                      dev, None, split="val"), epoch=epoch_of(n_steps - 1))
        if cfg.neuron_report:
            live.neurons(n_steps, epoch_of(n_steps - 1),
                         neuron_report(model, gi, cfg, dev, "test", cfg.neuron_instances * 2), final=True)
    metrics["history"] = history
    metrics["epoch_results"] = epoch_results
    metrics["train_sec"] = round(time.time() - t0, 1)
    metrics["params_trainable"] = int(sum(p.numel() for p in params))
    metrics["samples_seen"] = int(n_steps * cfg.batch_size)
    metrics["schedule"] = {**sched_info, "passes_completed": dict(getattr(sampler, "passes", {}))}
    if save_dir:
        metrics["model_dir"] = save_model(model, cfg, save_dir, metrics, best_state, gi)
        log(f"  [model] saved to {save_dir}" + (f" (best epoch {best_state['epoch']})" if best_state else ""))
    if live is not None:
        live.finish(metrics)
    return metrics


def summarise(results, ref_tag):
    from scipy import stats
    tags = list(results)
    summary = {}
    for tag in tags:
        row = {}
        for path, _ in KEY_METRICS:
            vals = [_get(r, path) for r in results[tag]]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
            if not vals:
                continue
            row[".".join(path)] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
            ref = [_get(r, path) for r in results[ref_tag]]
            if tag != ref_tag and len(vals) == len(ref) >= 2 and None not in ref:
                t, p = stats.ttest_rel(vals, ref)
                row[".".join(path)]["paired_vs_ref"] = {"t": float(t), "p": float(p),
                                                       "delta": float(np.mean(vals) - np.mean(ref))}
        summary[tag] = row
    return summary


def parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser("paper_a")
    ap.add_argument("--exp", default=None, help=f"one of {list(EXPERIMENTS)}")
    ap.add_argument("--size", default="S", choices=list(SIZES))
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--audit-shapes", action="store_true")
    a = ap.parse_args()

    cfg = PaperAConfig().override(**SIZES[a.size], **parse_overrides(a.set))
    if a.synthetic:
        cfg = cfg.override(synthetic=True, out_dir=os.environ.get("PAPER_A_OUT", "./paper_a_out"))
    gi = get_index(cfg)
    S = gi.summary()
    print(f"[paper_a] index: {S['papers']} papers, {S['kus']} KUs ({S['kus_per_paper']}/paper), "
          f"splits {S['splits']}, citations {S['paper_edges_cites']}, proxy edges {S['paper_edges_proxy']}")

    if a.audit_shapes:
        rows = audit_tightness(cfg.shape_spec, SlotLabels(gi), gi)
        for r in rows:
            print("  " + json.dumps(r))
        os.makedirs(cfg.out_dir, exist_ok=True)
        json.dump(rows, open(os.path.join(cfg.out_dir, "shape_tightness.json"), "w"), indent=2)
        return

    run_experiment(cfg, gi, a.exp)


def _resumable(path, acfg):
    """A finished result for the same arm, seed and run settings (papers, epochs, size)."""
    if not os.path.exists(path):
        return None
    try:
        m = json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return None
    c = m.get("config", {})
    same = all(c.get(k) == getattr(acfg, k) for k in ("max_papers", "epochs", "d_model", "enc_layers", "ku_mode",
                                                      "time_T", "n_subgraphs", "batch_size"))
    return m if same else None


def run_experiment(cfg, gi, exp=None, arms=None, log=print, live_root=None, resume=True):
    """Train + evaluate every arm x seed of an experiment, write per-run JSON and
    summary.json, return (results, summary). exp: a key of EXPERIMENTS, or None
    for a single run of `cfg` (or pass `arms` = [(tag, overrides), ...]).
    resume: an arm x seed whose result file already exists with the same settings is
    loaded instead of retrained (so an interrupted benchmark continues where it stopped)."""
    arms = arms or (EXPERIMENTS[exp] if exp else [("single", {})])
    name = exp or "single"
    out_dir = os.path.join(cfg.out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    results, base_gi = {}, gi
    for tag, over in arms:
        acfg = cfg.override(**over)
        if acfg.ku_mode == "paper" and over.keys() & {"target_types", "mask_schedule", "replay_frac"}:
            log(f"\n[paper_a] {name} | {tag}: skipped -- with ku_mode='paper' there is one KU type, "
                f"so arms that vary the masked type / schedule are identical")
            continue
        results[tag] = []
        done = {s: _resumable(os.path.join(out_dir, f"{tag}_seed{s}.json"), acfg) for s in acfg.seeds} if resume else {}
        for s, m in done.items():
            if m is not None:
                log(f"\n[paper_a] {name} | {tag} | seed {s}: already finished, loaded from {out_dir}")
                results[tag].append(m)
        todo = [s for s in acfg.seeds if done.get(s) is None]
        if not todo:
            continue
        # an arm that changes the KU graph itself (E16: ku_mode) gets its own index
        gi = get_index(acfg) if "ku_mode" in over and acfg.ku_mode != cfg.ku_mode else base_gi
        for seed in todo:
            log(f"\n[paper_a] {name} | {tag} | seed {seed}")
            live = None
            if live_root:
                from train.paper_a.telemetry import LiveRun
                live = LiveRun(live_root, f"{name}__{tag}__seed{seed}".replace("=", "-"), acfg, gi,
                               extra={"experiment": name, "arm": tag, "seed": seed})
            if acfg.baseline:                            # BENCH: a no-training reference row
                from train.paper_a.baselines import text_baseline
                m = text_baseline(acfg, gi, "test", seed=1234 + seed)
                if live is not None:
                    live.state["steps_total"] = 0
                    live.finish(m)
            else:
                m = train_one(acfg, gi, seed, log=log, live=live,
                              save_dir=os.path.join(out_dir, "models", f"{tag}_seed{seed}") if acfg.save_model else None)
            m["config"] = acfg.as_dict(); m["tag"] = tag; m["seed"] = seed
            json.dump(m, open(os.path.join(out_dir, f"{tag}_seed{seed}.json"), "w"), indent=2, default=str)
            results[tag].append(m)
            log(f"  -> balanced MRR {m['retrieval_balanced']['MRR']:.4f} "
                f"(chance {m['retrieval_balanced'].get('chance_MRR', float('nan')):.4f}) | "
                f"standard MRR {m['retrieval_standard']['MRR']:.4f} (chance {m['chance_MRR_standard']:.2e}) | "
                f"chi {m['target_space']['chi']:.3f} | probe {(m['partition_probe'].get('probe_acc') or float('nan')):.3f} | "
                f"rel-sens {m['relation_sensitivity']:.3f} | inv {m['inverse_acc']:.3f} | "
                f"audit flagged {m['audit']['flagged_frac']:.2f}")
    if not results:
        log(f"[paper_a] {name}: no arm ran"); return results, {}
    ref = next(iter(results))
    summary = summarise(results, ref)
    json.dump({"experiment": name, "reference_arm": ref, "summary": summary,
               "index": base_gi.summary(), "note": "grounded edges are PROXIES"},
              open(os.path.join(out_dir, "summary.json"), "w"), indent=2, default=str)
    log(f"\n[paper_a] summary -> {os.path.join(out_dir, 'summary.json')}")
    for tag, row in summary.items():
        bm = row.get("retrieval_balanced.MRR", {})
        ch = row.get("target_space.chi", {})
        log(f"  {tag:<22s} balanced MRR {bm.get('mean', float('nan')):.4f}±{bm.get('std', 0):.4f}  "
            f"chi {ch.get('mean', float('nan')):.3f}  "
            f"{'p=%.3g' % bm['paired_vs_ref']['p'] if 'paired_vs_ref' in bm else ''}")
    return results, summary


if __name__ == "__main__":
    main()
