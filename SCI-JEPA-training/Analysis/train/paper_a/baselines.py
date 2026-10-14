"""
No-training baselines for the benchmark (experiment BENCH).

text  The frozen text encoder alone (the model's own input features): the query is the
      mean embedding of the hidden KU's VISIBLE paper-mates (the same instances, with
      the same leak-safe masking, as the trained models see), and candidates are ranked
      by cosine to their own text embedding. Same splits, same 100-candidate pools,
      same chance levels and the same reasoning-task templates, so its numbers sit in
      the same table as the trained models. Embeddings are centred first (gte-modernbert
      is anisotropic: two random KUs have cosine ~0.53).

A trained model is only worth publishing if it beats this row.
"""
import math
import time

import numpy as np

from train.paper_a.diagnostics import _chance, hard_pool
from train.paper_a.graph_index import COMP_TYPES, HUB
from train.paper_a.instances import Sampler
from train.paper_a.ku_schema import REL_ID, TEMPLATES

NAN = float("nan")


def _bank(gi, split):
    papers = np.asarray(gi.papers(split))
    ids = np.concatenate([gi.paper_comps[int(p)] for p in papers]).astype(np.int64)
    X = gi.comp_x[ids].astype(np.float32)
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    Xc /= np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-8
    return ids, Xc, mu, {int(c): k for k, c in enumerate(ids)}


def _query(inst, mu):
    c = inst["ctx"]
    vis = (c["slot"] == 0) & (c["type"] != HUB) & (~inst["is_mask"]) & (c["comp"] >= 0)
    if not vis.any():
        return None
    q = c["x"][vis].astype(np.float32).mean(0, keepdims=True) - mu
    return q / (np.linalg.norm(q) + 1e-8)


def _rank_instances(gi, cfg, sampler, bank, n, rng, variants=False):
    ids, Xc, mu, pos_of = bank
    ctype, cpaper = gi.comp_type[ids], gi.comp_paper[ids]
    rec, left = [], n
    while left > 0:
        insts = sampler.sample(min(cfg.batch_size, left))
        left -= cfg.batch_size
        for inst in insts:
            g = pos_of.get(int(inst["target_comp"]))
            q = _query(inst, mu)
            if g is None or q is None:
                continue
            t = int(inst["target_type"]); same = np.nonzero(ctype == t)[0]
            s_all = Xc[same] @ q[0]
            std = 1 + int((s_all > s_all[np.nonzero(same == g)[0][0]]).sum())
            pool = [g] + hard_pool(gi, int(inst["papers"][0]), same, cpaper, cfg.balanced_k, rng, mode=cfg.pool_mode)
            s = Xc[pool] @ q[0]
            var = gi.mp_of.get(int(inst["target_comp"]), []) if (variants and cfg.minimal_pairs) else []
            sv = np.zeros(0)
            if var:                                      # counterfactual variants of the true KU
                V = gi.mp_x[var].astype(np.float32) - mu
                V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
                sv = V @ q[0]
            rec.append({"type": COMP_TYPES[t], "rank_bal": 1 + int((s[1:] > s[0]).sum()) + int((sv > s[0]).sum()),
                        "pool": len(pool) + len(var), "rank_std": std, "bank": len(same),
                        "mp_pair": list((s[0] > sv).astype(float)) if var else None,
                        "mp_all": float((s[0] > sv).all()) if var else None})
    return rec


def _agg(rec):
    if not rec:
        return {"MRR": NAN, "Hits@1": NAN, "Hits@10": NAN, "chance_MRR": NAN, "n": 0}
    r = np.asarray([x["rank_bal"] for x in rec], float)
    return {"MRR": float((1 / r).mean()), "Hits@1": float((r <= 1).mean()), "Hits@10": float((r <= 10).mean()),
            "chance_MRR": float(np.mean([_chance(x["pool"]) for x in rec])), "mean_pool": float(np.mean([x["pool"] for x in rec])),
            "n": len(rec)}


def _consistency(gi, split):
    """Support vs contradict without training: cosine(claim, evidence) below the median
    over all claim-evidence pairs of the split -> 'contradicts'. Labels are only used to
    score the guess."""
    allowed = set(int(p) for p in gi.papers(split))
    s_rel, c_rel = REL_ID["supported_by"], REL_ID["challenged_by"]
    pairs = [(a, b, int(r == c_rel)) for a, b, r in zip(gi.role_src.tolist(), gi.role_dst.tolist(), gi.role_rel.tolist())
             if r in (s_rel, c_rel) and int(gi.comp_paper[a]) in allowed]
    if not pairs:
        return {"bal_acc": None, "n": 0}
    X = gi.comp_x.astype(np.float32)
    mu = X[[a for a, _, _ in pairs] + [b for _, b, _ in pairs]].mean(0)
    def cos(a, b):
        u, v = X[a] - mu, X[b] - mu
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-8))
    c = np.asarray([cos(a, b) for a, b, _ in pairs]); y = np.asarray([l for _, _, l in pairs])
    yhat = (c < np.median(c)).astype(int)
    rec = [float((yhat[y == k] == k).mean()) for k in (0, 1) if (y == k).any()]
    return {"bal_acc": float(np.mean(rec)), "acc": float((yhat == y).mean()), "chance": 0.5, "n": int(len(y)),
            "n_supports": int((y == 0).sum()), "n_contradicts": int((y == 1).sum()),
            "recall_supports": rec[0] if (y == 0).any() else None,
            "recall_contradicts": rec[-1] if (y == 1).any() else None}


def _pairs(gi, papers):
    allowed = set(int(p) for p in papers)
    s_rel, c_rel = REL_ID["supported_by"], REL_ID["challenged_by"]
    return [(a, b, int(r == c_rel)) for a, b, r in zip(gi.role_src.tolist(), gi.role_dst.tolist(), gi.role_rel.tolist())
            if r in (s_rel, c_rel) and int(gi.comp_paper[a]) in allowed]


def _consistency_probe(gi, split):
    """Support vs contradict from frozen text with a supervised linear probe trained on the
    TRAIN split's claim-evidence pairs (class-balanced), scored on `split`."""
    from sklearn.linear_model import LogisticRegression
    tr, te = _pairs(gi, gi.papers("train")), _pairs(gi, gi.papers(split))
    if not tr or not te or len({l for *_, l in tr}) < 2:
        return {"bal_acc": None, "n": 0}
    X = gi.comp_x.astype(np.float32); mu = X.mean(0)
    def feats(P):
        c = X[[a for a, _, _ in P]] - mu; e = X[[b for _, b, _ in P]] - mu
        c /= np.linalg.norm(c, axis=1, keepdims=True) + 1e-8; e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-8
        return np.concatenate([c, e, c * e, np.abs(c - e)], 1)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(feats(tr), np.asarray([l for *_, l in tr]))
    y = np.asarray([l for *_, l in te]); yhat = clf.predict(feats(te))
    rec = [float((yhat[y == k] == k).mean()) for k in (0, 1) if (y == k).any()]
    return {"bal_acc": float(np.mean(rec)), "acc": float((yhat == y).mean()), "chance": 0.5, "n": int(len(y)),
            "n_supports": int((y == 0).sum()), "n_contradicts": int((y == 1).sum()),
            "recall_supports": rec[0] if (y == 0).any() else None,
            "recall_contradicts": rec[-1] if (y == 1).any() else None, "n_train": len(tr)}


def _intervention(gi, cfg, split, bank, rng, seed):
    """The models' intervention test on frozen text: forward query (problem, method,
    procedure visible), swap the procedure (condition) or method for a donor paper's,
    and check whether the query moves toward the donor's RESULT (centred embeddings)."""
    ids, Xc, mu, pos_of = bank
    t_res = COMP_TYPES.index("result")
    res_of = {int(gi.comp_paper[c]): k for k, c in enumerate(ids) if gi.comp_type[c] == t_res}
    allowed = set(int(p) for p in gi.papers(split))
    centre = Xc[list(res_of.values())].mean(0) if res_of else 0
    out = {}
    for name, swap in (("condition", "procedure"), ("method", "method")):
        t_sw = COMP_TYPES.index(swap)
        s = Sampler(gi, cfg.override(target_types=("result",)), split, seed + 1)
        s.given = set(TEMPLATES["forward"][0])
        q1s, q2s, toward = [], [], []
        for inst in s.sample(max(cfg.task_eval_instances // 2, 16)):
            c = inst["ctx"]; p0 = int(inst["papers"][0])
            own = (c["slot"] == 0) & (c["type"] != HUB) & (~inst["is_mask"]) & (c["comp"] >= 0)
            k = np.nonzero(own & (c["type"] == t_sw))[0]
            pool = [q for q in gi.subfield_papers[int(gi.paper_subfield[p0])] if q != p0 and q in allowed and q in res_of
                    and q not in inst["papers"]] or [q for q in res_of if q != p0 and q in allowed and q not in inst["papers"]]
            if not len(k) or not pool:
                continue
            d = int(rng.choice(pool))
            dc = [int(x) for x in gi.paper_comps[d] if gi.comp_type[x] == t_sw]
            if not dc:
                continue
            V = c["x"][own].astype(np.float32); V2 = V.copy()
            V2[np.nonzero(np.nonzero(own)[0] == k[0])[0][0]] = gi.comp_x[dc[0]].astype(np.float32)
            q1, q2 = V.mean(0) - mu[0], V2.mean(0) - mu[0]
            q1 /= np.linalg.norm(q1) + 1e-8; q2 /= np.linalg.norm(q2) + 1e-8
            q1s.append(q1); q2s.append(q2)
            toward.append(float((q2 - q1) @ (Xc[res_of[d]] - centre) > 0))
        if len(q1s) > 1:
            Q1, Q2 = np.stack(q1s), np.stack(q2s)
            shift = 1 - (Q1 * Q2).sum(1)
            base = 1 - (Q1 @ Q1.T)[~np.eye(len(Q1), dtype=bool)].mean()
            out[name] = {"swap": swap, "shift": float(shift.mean()), "rel_shift": float(shift.mean() / max(base, 1e-12)),
                         "toward_donor": float(np.mean(toward)), "away_from_own": None, "chance": 0.5, "n": len(q1s)}
        else:
            out[name] = {"swap": swap, "rel_shift": None, "toward_donor": None, "n": len(q1s)}
    out["note"] = "frozen text embeddings: the swapped section changes the mean query directly"
    return out


def text_baseline(cfg, gi, split="test", seed=1234):
    t0 = time.time()
    bank = _bank(gi, split)
    rng = np.random.default_rng(seed)
    rec = _rank_instances(gi, cfg, Sampler(gi, cfg, split, seed), bank, cfg.eval_instances, rng)
    rb = _agg(rec)
    rs = np.asarray([x["rank_std"] for x in rec], float) if rec else np.asarray([NAN])
    per_type = {}
    for t in sorted({x["type"] for x in rec}):
        g = [x for x in rec if x["type"] == t]
        a = _agg(g)
        per_type[t] = {"n": a["n"], "MRR_bal": a["MRR"], "chance_bal": a["chance_MRR"], "Hits1_bal": a["Hits@1"],
                       "MRR_std": float(np.mean([1 / x["rank_std"] for x in g])), "top1_same_paper": NAN}
    # reasoning-task templates (retrieval tasks only: consistency and intervention need a trained head)
    tasks = {}
    for key, label, query in (("forward", "Forward prediction", "mask Result | given Problem, Method, Material/Condition"),
                              ("inverse", "Experiment design (inverse)", "mask Method / Material | given Problem, Result"),
                              ("abduction", "Abduction", "mask Claim / Hypothesis | given Results, Conditions")):
        given, targets = TEMPLATES[key]
        by = {}
        for tg in targets:
            s = Sampler(gi, cfg.override(target_types=(tg,)), split, seed)
            s.given = set(given)
            rr = _rank_instances(gi, cfg, s, bank, max(cfg.task_eval_instances // len(targets), 16), rng, variants=True)
            a = _agg(rr)
            mpp = [v for x in rr if x["mp_pair"] for v in x["mp_pair"]]; mpa = [x["mp_all"] for x in rr if x["mp_all"] is not None]
            if a["n"]:
                by[tg] = {**a, "chance": a["chance_MRR"], "ratio": a["MRR"] / max(a["chance_MRR"], 1e-9),
                          "mp_pair": float(np.mean(mpp)) if mpp else None, "mp_all": float(np.mean(mpa)) if mpa else None,
                          "mp_n": len(mpa)}
        n = sum(v["n"] for v in by.values())
        avg = (lambda k: float(sum(v[k] * v["n"] for v in by.values()) / n)) if n else (lambda k: NAN)
        mrr, ch = avg("MRR"), avg("chance")
        mp = [(v["mp_pair"], v["mp_all"], v["mp_n"]) for v in by.values() if v.get("mp_pair") is not None]
        mpn = sum(x[2] for x in mp)
        tasks[key] = {"MRR": mrr, "chance": ch, "ratio": mrr / ch if n and ch else NAN, "Hits@1": avg("Hits@1"),
                      "Hits@10": avg("Hits@10"), "n": n, "by_target": by, "label": label, "query": query,
                      "mp_pair": float(sum(a * w for a, _, w in mp) / mpn) if mpn else None,
                      "mp_all": float(sum(b * w for _, b, w in mp) / mpn) if mpn else None, "mp_n": mpn}
    unsup = _consistency(gi, split)
    tasks["consistency"] = {**_consistency_probe(gi, split), "label": "Consistency checking",
                            "query": "supervised probe on frozen text embeddings (logistic regression on "
                                     "[c, e, c*e, |c-e|], trained on the train split): the fair counterpart of the "
                                     "models' trained relation decoder",
                            "unsupervised": unsup}
    tasks["intervention"] = {**_intervention(gi, cfg, split, bank, rng, seed), "label": "Intervention conditioning",
                             "query": "same procedure / method swap as for the models, on the frozen text embeddings"}
    return {"baseline": "text (frozen " + str(cfg.embed_model) + ", cosine, no training)",
            "retrieval_balanced": rb,
            "retrieval_standard": {"MRR": float((1 / rs).mean()), "Hits@1": float((rs <= 1).mean()),
                                   "Hits@10": float((rs <= 10).mean())},
            "chance_MRR_standard": float(np.mean([(math.log(x["bank"]) + 0.5772) / x["bank"] for x in rec])) if rec else NAN,
            "per_type": per_type, "tasks": tasks,
            "target_space": {"chi": NAN, "chi_raw": NAN, "eff_rank": NAN}, "context_space": {"chi": NAN, "eff_rank": NAN},
            "partition_probe": {}, "relation_sensitivity": NAN, "inverse_acc": NAN,
            "audit": {"flagged_frac": NAN}, "shortcut_margin_MRR": {"mean": NAN},
            "history": [], "epoch_results": [], "n_eval": len(rec),
            "train_sec": round(time.time() - t0, 1), "params_trainable": 0, "samples_seen": 0}
