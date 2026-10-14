"""
All hyperparameters and ablation switches for Paper A, in one dataclass.

Every experiment in the plan (E1-E5, E10-E12) is a preset that overrides a
few fields of `PaperAConfig`; see EXPERIMENTS at the bottom.
"""
import dataclasses
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_NFS = "/nfs/home/rabbyg/JEPA/Graph-JEPA"


@dataclass
class PaperAConfig:
    # ---- data: normalized Scientific-Summaries parquet (see ku_schema.py) ------
    parquet_dirs: tuple = tuple(os.environ.get(
        "PAPER_A_PARQUET", f"{_NFS}/Scientific-Summaries-Data/normalized/arxiv").split(","))
    subsets: tuple = ()                # () = all; e.g. ("arxiv-new", "arxiv-old", "titled")
    max_papers: int = 60000            # tier: D0 60k, D1 500k, D2 2M (see TIERS)
    data_seed: int = 0                 # which shards / papers a tier draws
    rows_per_shard: int = 500          # random rows per shuffled shard (0 = whole shards)
    year_min: int = 1900
    year_max: int = 2100
    index_dir: str = os.environ.get("PAPER_A_INDEX_DIR", f"{_NFS}/dataset/paper_a")
    out_dir: str = os.environ.get("PAPER_A_OUT", f"{_NFS}/checkpoints/paper_a")
    synthetic: bool = False            # generated records in the parquet schema, for smoke tests
    ku_mode: str = "section"           # section: ~22 KUs per paper (7 sections + claim / evidence /
                                       #   implication KUs) | paper: the WHOLE paper is ONE KU (all
                                       #   sections + claims, ~2,100 tokens); same papers in both modes
                                       # | paper_masked: the target's paper is ONE text of the visible
                                       #   ("given") KUs, encoded together; targets = section KUs
    pm_templates: str = "full"         # paper_masked: "full" (3 reasoning queries + leave-one-out per
                                       #   type, ~13 texts / paper) | "tasks" (the 3 queries only)

    # ---- splits ---------------------------------------------------------------
    split_mode: str = "time"           # time: train <= T, val T+1, test T+2 | random
    time_T: int = 2018                 # sweep >= 2 values for the paper
    encoder_cutoff_year: int = 2024    # text encoder's training data (preset); warn when T precedes it
    split_seed: int = 42
    split: tuple = (0.8, 0.1, 0.1)     # random mode only

    # ---- knowledge units ------------------------------------------------------
    ku_min_chars: int = 20             # shorter, placeholder or null-statement text is not a KU
    optional_kus: tuple = ()           # e.g. ("figures",); robustness / data_code / ethics are sparse
    min_kus_per_paper: int = 8
    required_kus: tuple = ("result", "claim")
    embedder: str = "gte-modernbert"   # a key of EMBEDDERS (sets the fields below) | hash (smoke tests)
    embed_model: str = "Alibaba-NLP/gte-modernbert-base"   # 8192-token context, 768-d
    embed_max_tokens: int = 2048       # longest KU ~1150 tokens (results); MiniLM's 256 cut 93% of methods
    embed_batch: int = 32              # texts per forward pass (long sections: keep small on a laptop)
    embed_prefix: str = ""             # some encoders want e.g. "search_document: "
    embed_chunk: int = 8192
    snippet_chars: int = 240           # KU text kept for the dashboard's example tables

    # ---- proxy grounded edges (stand-in until canonicalisation exists) ------
    proxy_types: tuple = ("method",)   # component types matched across papers
    proxy_tau: float = 0.86            # per encoder (EMBEDDERS): the same percentile of cross-paper top-1
                                       # method cosine as MiniLM's 0.80 (p99.3); the build prints the edge count
    proxy_topk: int = 10

    # ---- instance construction (the relation layer) -------------------------
    target_types: tuple = ("context", "question", "method", "procedure", "result", "interpretation",
                           "limitation", "claim", "evidence", "implication")   # all 10 (ALL_TARGET_TYPES)
    mask_schedule: str = "mixed"       # mixed | balanced | round_robin | sequential (see train.mask_plan)
    replay_frac: float = 0.0           # sequential only: share of each batch replaying earlier types
    epochs: float = 10.0               # >0: run length = epochs full passes over every target KU
                                       # (overrides steps); sequential gives each type `epochs` passes
    n_subgraphs: int = 8               # N: E10 sweeps {1, 4, 8, 16}
    hops: int = 2                      # k-hop expansion over the paper graph
    fill_policy: str = "same_subfield" # keep N fixed when citations are sparse (~22% have refs)
    max_comp_per_paper: int = 24       # a paper has ~23 KUs (7 sections + ~4.3 x 3.5 claim KUs)
    mask_policy: str = "token+edges+children+siblings"  # siblings: method<->procedure restate each other
                                       # (same-paper cosine 0.78 vs 0.55 across papers)
    audit_tau: float = 0.87            # paraphrase-leak threshold, per encoder (EMBEDDERS): MiniLM 0.75,
                                       # gte-modernbert 0.87 (same percentile of same-paper similarity)
    audit_drop: bool = False           # drop instances the audit flags
    resample_each_epoch: bool = True   # E4: False = neighbourhoods frozen at epoch 0
    use_degree_features: bool = False  # E4: True = expose the node census to the encoder
    n_probe_edges: int = 4             # role edges hidden from the encoder for the inverse head
    task_mask_p: float = 0.5           # task-aware masking: share of training instances whose own paper shows
                                       #   only a reasoning template's given parts (forward / inverse / abduction)
                                       #   so training matches the five reasoning queries; 0 = leave-one-out only
    motif_p: float = 0.0               # E20 (Graph-JEPA-style): share of training instances whose hidden KU's
                                       #   whole motif (claim+evidence+implications, method+procedure,
                                       #   result+interpretation+limitation, context+question) is predicted
    rwse_steps: int = 0                # E20: random-walk structural encoding size (0 = off; Graph-JEPA uses 15-40)
    baseline: str = ""                 # "text": no training, frozen text-encoder cosine (benchmark row)
    wm_p_action: float = 0.0           # JEPA world model (E19): share of training instances that get an
                                       #   action (swap a KU / flip a supports-contradicts edge); the target
                                       #   is the hidden KU's latent AFTER the action. 0 = off (reference)

    # ---- model --------------------------------------------------------------
    d_model: int = 512
    n_heads: int = 8
    enc_layers: int = 8
    pred_layers: int = 6
    dropout: float = 0.0
    feat_dim: int = 768                # node features = text-encoder dimension (preset)

    # ---- objective ----------------------------------------------------------
    jepa_loss: str = "l2"              # l2 | cos
    target_scope: str = "paper"        # paper: target encoder sees only the target's own subgraph
    lambda_inv: float = 0.0            # E11: inverse-relation (IDM analogue) weight
    lambda_reg: float = 0.0            # E12: grouping regulariser weight
    reg_grouping: str = "across"       # E12: within | across
    reg_projections: int = 64
    lambda_contrast: float = 0.0       # E5: relational contrast masking margin weight
    contrast_margin: float = 0.5
    lambda_align: float = 0.0          # contrastive alignment term (own it in the intro)
    align_tau: float = 0.1

    # ---- symbolic constraints (E1-E3) -----------------------------------------
    shape_spec: Optional[str] = None   # JSON spec; None = constraint loss off
    lambda_shacl: float = 0.0
    tnorm: str = "product"             # product | lukasiewicz | godel
    shape_depth: Optional[int] = None  # E2: replicate conjuncts to this depth
    shape_factored: bool = True        # E3: per-slot sub-vectors vs one opaque target
    shape_subset: str = "all"          # E1: all | tight | loose

    # ---- optimisation ---------------------------------------------------------
    batch_size: int = 128
    steps: int = 20000
    steps_per_epoch: int = 1000
    lr: float = 3e-4
    warmup: int = 1000
    weight_decay: float = 0.05
    ema_start: float = 0.996
    ema_end: float = 1.0
    grad_clip: float = 1.0
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    device: str = "auto"               # auto: cuda > mps (Apple GPU) > cpu

    # ---- evaluation -----------------------------------------------------------
    eval_instances: int = 4000
    eval_bank_papers: int = 0          # 0 = every test paper in the candidate bank
    balanced_k: int = 100              # candidates per query (besides the true KU)
    pool_mode: str = "topic"           # topic: distractors from the papers with the most similar problem
                                       # (shortcut-controlled) | subfield: the earlier protocol
    minimal_pairs: bool = True         # forward / abduction: add counterfactual variants of the true
                                       # result / claim (ku_schema.counterfactuals) to the candidates
    probe_papers: int = 200            # partition-identity probe
    log_every: int = 200
    eval_every: int = 0                # >0: quick validation pass every N steps (dashboard curves)
    live_eval_instances: int = 300
    neuron_report: bool = True         # per-neuron interpretation at every live eval (dashboard: Neurons)
    neuron_instances: int = 128
    eval_at_epochs: bool = True        # with epochs > 0: validation + neuron report at the end of EVERY epoch
    epoch_eval_instances: int = 1000   # queries per end-of-epoch validation (more = steadier per-type numbers)
    eval_tasks: bool = True            # reasoning-task matrix (tasks.py): forward / inverse / abduction /
                                       # consistency / intervention, at every epoch end (val) and at the end (test)
    task_eval_instances: int = 300     # queries per task
    save_model: bool = True            # write each trained model to <out_dir>/<exp>/models/<arm>_seed<s>/
    eval_worldview: bool = True        # structure reliance + action prediction (every epoch end, val; final, test)
                                       # and explained example subgraphs for the dashboard's World model tab

    def override(self, **kw) -> "PaperAConfig":
        bad = [k for k in kw if not hasattr(self, k)]
        if bad:
            raise KeyError(f"unknown config keys: {bad}")
        explicit = set(kw)
        if kw.get("embedder") not in (None, self.embedder):   # switching encoder: its preset brings its
            name = kw["embedder"]                    # own dim / thresholds, unless set explicitly in this call
            if name not in EMBEDDERS:
                raise KeyError(f"unknown embedder {name!r}; choose from {sorted(EMBEDDERS)}")
            kw = {**EMBEDDERS[name], **kw}
        mode = kw.get("ku_mode", self.ku_mode)
        if mode not in ("section", "paper", "paper_masked"):
            raise KeyError(f"unknown ku_mode {mode!r}; choose 'section', 'paper' or 'paper_masked'")
        emb = kw.get("embedder", self.embedder)
        if mode == "paper":                          # one KU per paper: its only type is "paper"
            kw["target_types"], kw["proxy_types"] = ("paper",), ("paper",)
            if self.ku_mode != "paper" or "embedder" in kw:   # entering paper mode: whole-paper settings
                kw = {**kw, **{k: v for k, v in PAPER_MODE.get(emb, PAPER_MODE["*"]).items() if k not in explicit}}
        elif mode == "paper_masked" and self.ku_mode != "paper_masked":
            kw.setdefault("embed_max_tokens", 8192)  # context texts reach ~4k tokens
            if self.ku_mode == "paper":
                kw.setdefault("target_types", ("context", "question", "method", "procedure", "result",
                                               "interpretation", "limitation", "claim", "evidence", "implication"))
                kw.setdefault("proxy_types", ("method",))
                kw = {**{k: v for k, v in EMBEDDERS.get(emb, {}).items() if k in ("proxy_tau", "audit_tau")}, **kw}
            if kw.get("pm_templates", self.pm_templates) == "tasks":
                tt = kw.get("target_types", self.target_types)
                kw["target_types"] = tuple(t for t in tt if t in ("result", "method", "procedure", "claim", "question"))
        elif self.ku_mode in ("paper", "paper_masked") and mode == "section":   # back to section mode
            kw.setdefault("target_types", self.target_types if self.target_types != ("paper",) else
                          ("context", "question", "method", "procedure", "result", "interpretation",
                           "limitation", "claim", "evidence", "implication"))
            kw.setdefault("proxy_types", ("method",))
            kw = {**{k: v for k, v in EMBEDDERS.get(emb, {}).items()
                     if k in ("embed_max_tokens", "proxy_tau", "audit_tau")}, **kw}
        if mode == "paper_masked" and kw.get("pm_templates") == "tasks" and "target_types" not in explicit:
            tt = kw.get("target_types", self.target_types)
            kw["target_types"] = tuple(t for t in tt if t in ("result", "method", "procedure", "claim", "question"))
        for k, v in kw.items():                      # "1000" -> 1000, "a,b" -> ("a", "b")
            cur = getattr(self, k)
            if isinstance(v, str) and not isinstance(cur, str) and cur is not None:
                if isinstance(cur, bool):
                    kw[k] = v.lower() in ("1", "true", "yes")
                elif isinstance(cur, (int, float)):
                    kw[k] = type(cur)(v)             # raises on garbage, which is the point
                elif isinstance(cur, (tuple, list)):
                    kw[k] = tuple(x for x in v.split(",") if x)
            elif isinstance(cur, tuple) and isinstance(v, list):
                kw[k] = tuple(v)
        return dataclasses.replace(self, **kw)

    def as_dict(self) -> Dict:
        return dataclasses.asdict(self)


# Text encoders for KU node features. Thresholds are cosine values, and every encoder has its own
# cosine scale (gte-modernbert: two random KUs ~0.53, MiniLM ~0.08), so each preset carries
# proxy_tau / audit_tau calibrated to the SAME percentiles on 600 real papers (13k KUs).
EMBEDDERS = {
    "gte-modernbert": dict(embed_model="Alibaba-NLP/gte-modernbert-base", embed_max_tokens=2048, feat_dim=768,
                           embed_batch=32, embed_prefix="", encoder_cutoff_year=2024,
                           proxy_tau=0.86, audit_tau=0.87),
    "nomic-modernbert": dict(embed_model="nomic-ai/modernbert-embed-base", embed_max_tokens=2048, feat_dim=768,
                             embed_batch=32, embed_prefix="search_document: ", encoder_cutoff_year=2024,
                             proxy_tau=0.86, audit_tau=0.87),   # not calibrated separately
    "bge-m3": dict(embed_model="BAAI/bge-m3", embed_max_tokens=2048, feat_dim=1024, embed_batch=16,
                   embed_prefix="", encoder_cutoff_year=2024, proxy_tau=0.86, audit_tau=0.87),  # not calibrated
    "minilm": dict(embed_model="sentence-transformers/all-MiniLM-L6-v2", embed_max_tokens=256, feat_dim=384,
                   embed_batch=256, embed_prefix="", encoder_cutoff_year=2021, proxy_tau=0.80, audit_tau=0.75),
    "hash": dict(proxy_tau=0.80, audit_tau=0.75),    # synthetic smoke tests; keeps feat_dim
}
# ku_mode="paper": whole-paper texts (median ~2,100 tokens, max ~4,200 on 600 real papers) and
# paper-to-paper thresholds. gte-modernbert: cross-paper top-1 cosine p99.3 = 0.84 (the percentile
# MiniLM's 0.80 had at section level); the leak flag uses the same value (a near-duplicate paper).
PAPER_MODE = {
    "gte-modernbert": dict(embed_max_tokens=8192, proxy_tau=0.84, audit_tau=0.84),
    "hash": dict(proxy_tau=0.80, audit_tau=0.75),
    "*": dict(embed_max_tokens=8192),                # other encoders: thresholds not calibrated
}


# ---------------------------------------------------------------- the ICML paper
# Every run uses the PaperAConfig defaults above (= the paper's configuration); an experiment
# changes ONE thing (its arms). Shown by the notebook as "the fixed configuration".
ICML_FIXED = {
    "KU graph": "each paper -> ~22 KUs: 7 sections + every claim / evidence / implication (ku_mode='section')",
    "text encoder": "gte-modernbert-base, 8192-token context, 768-d, no KU truncated",
    "split": "time split: train <= 2018, val 2019, test 2020",
    "hidden KU types": "all 10 (context, question, method, procedure, result, interpretation, limitation, "
                       "claim, evidence, implication)",
    "masking": "mixed (every type in every batch); leak-safe: the hidden KU's paraphrases are hidden too",
    "context": "N = 8 papers per training instance (E10 varies it)",
    "objective": "plain JEPA (L2 to an EMA target encoder); every extra loss off (E11, E12, E5, E1-E3 turn one on)",
    "training": "10 epochs (full passes over every hidden KU), batch 128 (64 on the MacBook), AdamW 3e-4",
    "evaluation": "every epoch on val, final on test: balanced MRR vs chance, per KU type, reasoning-task "
                  "matrix (forward, inverse, abduction, consistency, intervention), neuron report",
}
PAPER_EXPERIMENTS = {                  # key -> the question it answers (in paper order)
    "BENCH": "benchmark: the text baseline and every main model variant side by side (6 rows per seed)",
    "APPX_STRUCT": "appendix: random-walk structural encodings, alone and with motif + world model (4 runs)",
    "APPX_EVAL": "appendix: old vs redesigned evaluation (candidates, minimal pairs, task-aware masking) (5 runs)",
    "E10": "A1 degeneracy: does widening the context (N = 1, 4, 8, 16 papers) remove the identity shortcut?",
    "E11": "repair: inverse-relation head (lambda_inv = 0, 0.1, 0.3, 1)",
    "E12": "repair: grouping regulariser (off / within a paper / across papers)",
    "E4":  "repair: graph construction (frozen neighbourhoods + node census)",
    "E5":  "repair: relational contrast masking (off / on)",
    "E15": "which KU types are learnable: one model masking every type vs one model per type (11 runs)",
    "E17": "how to read a paper: separate KUs vs ONE text of its visible parts (2 runs)",
    "E18": "E15 with the paper read as one text: one model for every part vs one per part (11 runs)",
    "E19": "JEPA world model: learning how actions on the graph (swap a KU, flip an edge) change the hidden KU, "
           "without losing clean accuracy (3 runs)",
    "E20": "Graph-JEPA ideas: sub-graph (motif) targets + random-walk structural encodings, alone and with "
           "the world model, at unchanged clean accuracy (6 runs)",
    "E1":  "shape constraints: SHACL off / loose / tight (Theorem A2) [placeholder shapes until ORKG]",
    "E2":  "t-norm depth decay: product / Lukasiewicz / Goedel x depth 3, 8, 15 (Prop. A3) [placeholder shapes]",
    "E3":  "shape-factored heads vs one head x depth [placeholder shapes]",
}


def resolve_device(name):
    """'auto' -> cuda, else Apple 'mps', else cpu. An unavailable request falls back."""
    import torch
    if name in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda"
    if name in ("auto", "cuda", "mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Data tiers from the plan (papers drawn at random across shards).
TIERS = {"D0": dict(max_papers=60000), "D1": dict(max_papers=500000), "D2": dict(max_papers=2000000)}

# Model sizes from the plan. S matches D0; M/L need the D1/D2 corpora.
SIZES = {
    "tiny": dict(d_model=64, n_heads=4, enc_layers=2, pred_layers=1),
    "local": dict(d_model=256, n_heads=8, enc_layers=4, pred_layers=2),   # MacBook-sized
    "S": dict(d_model=512, n_heads=8, enc_layers=8, pred_layers=6),
    "M": dict(d_model=768, n_heads=12, enc_layers=12, pred_layers=8),
    "L": dict(d_model=1024, n_heads=16, enc_layers=16, pred_layers=12),
}

# Each experiment is a list of (tag, overrides). The first entry is the
# reference arm that paired tests compare against.
EXPERIMENTS = {
    # BENCH: the benchmark. Every main model variant + the no-training text baseline, one row each,
    # same papers, splits, candidate pools and evaluation; compared in the dashboard's Benchmark tab.
    "BENCH": [("text_baseline", dict(baseline="text")),
              ("reference", dict()),
              ("paper_as_text", dict(ku_mode="paper_masked")),
              ("world_model", dict(wm_p_action=0.25)),
              ("motif", dict(motif_p=0.5)),
              ("motif+world", dict(motif_p=0.5, wm_p_action=0.25))],
    # Appendix A: random-walk structural encodings (Graph-JEPA), kept out of BENCH: on 500 papers
    # they changed nothing and cost ~17x training time
    "APPX_STRUCT": [("reference", dict()),
                    ("rwse", dict(rwse_steps=16)),
                    ("motif+world", dict(motif_p=0.5, wm_p_action=0.25)),
                    ("motif+world+rwse", dict(rwse_steps=16, motif_p=0.5, wm_p_action=0.25))],
    # Appendix B: what the evaluation redesign changes. old_protocol = subfield candidates, no
    # minimal pairs, no task-aware masking (the first BENCH run's setting)
    "APPX_EVAL": [("text_baseline", dict(baseline="text")),
                  ("text_baseline_old", dict(baseline="text", pool_mode="subfield", minimal_pairs=False)),
                  ("reference", dict()),
                  ("reference_no_task_mask", dict(task_mask_p=0.0)),
                  ("reference_old_protocol", dict(task_mask_p=0.0, pool_mode="subfield", minimal_pairs=False))],
    "E10": [(f"N={n}", dict(n_subgraphs=n)) for n in (1, 4, 8, 16)],
    "E11": [(f"inv={w}", dict(lambda_inv=w)) for w in (0.0, 0.1, 0.3, 1.0)],
    "E12": [("reg=off", dict(lambda_reg=0.0)),
            ("reg=within", dict(lambda_reg=1.0, reg_grouping="within")),
            ("reg=across", dict(lambda_reg=1.0, reg_grouping="across"))],
    "E4": [("frozen+census", dict(resample_each_epoch=False, use_degree_features=True)),
           ("resample+census", dict(resample_each_epoch=True, use_degree_features=True)),
           ("frozen+nocensus", dict(resample_each_epoch=False, use_degree_features=False)),
           ("resample+nocensus", dict(resample_each_epoch=True, use_degree_features=False))],
    "E5": [("contrast=off", dict(lambda_contrast=0.0)),
           ("contrast=on", dict(lambda_contrast=1.0))],
    # E15 = mixed (all types together) next to one model per type: the same budget of
    # epochs each, so the dashboard's All runs tab compares mixed vs each type alone, per type
    "E15": [("mixed", dict(mask_schedule="mixed"))]
           + [(f"only={t}", dict(target_types=(t,), mask_schedule="mixed"))
              for t in ("context", "question", "method", "procedure", "result", "interpretation",
                        "limitation", "claim", "evidence", "implication")],
    # E17: the target's own paper as separate KUs (section) vs as ONE joint text of the visible
    # parts (paper_masked); same papers, same targets, same neighbours, same reasoning-task matrix
    "E17": [("section", dict(ku_mode="section")), ("paper_masked", dict(ku_mode="paper_masked"))],
    # E18 = E15 with the paper read as ONE text (ku_mode="paper_masked"): one model masking every
    # part (mixed) + one model per masked part; compare with E15 (the same in section mode)
    "E18": [("pm_mixed", dict(ku_mode="paper_masked", mask_schedule="mixed"))]
           + [(f"pm_only={t}", dict(ku_mode="paper_masked", target_types=(t,), mask_schedule="mixed"))
              for t in ("context", "question", "method", "procedure", "result", "interpretation",
                        "limitation", "claim", "evidence", "implication")],
    # E19: JEPA world model. A quarter of training instances get an action (swap a KU for another
    # paper's / flip a supports-contradicts edge) and the target is the hidden KU's latent AFTER the
    # action. Accuracy is always measured on clean instances, so it is directly comparable.
    "E19": [("reference", dict(wm_p_action=0.0)),
            ("world_model", dict(wm_p_action=0.25)),
            ("world_model+inverse", dict(wm_p_action=0.25, lambda_inv=0.3))],
    # E20: Graph-JEPA ideas for the KU graph (Skenderi et al.): sub-graph (motif) targets and random-
    # walk structural encodings (encoder input + the hidden nodes' true position for the predictor).
    # Motifs are used in training only; evaluation is the reference protocol, so accuracy compares 1:1.
    "E20": [("reference", dict()),
            ("rwse", dict(rwse_steps=16)),
            ("motif", dict(motif_p=0.5)),
            ("graph_jepa", dict(rwse_steps=16, motif_p=0.5)),
            ("graph_jepa+world", dict(rwse_steps=16, motif_p=0.5, wm_p_action=0.25)),
            ("motif+world", dict(motif_p=0.5, wm_p_action=0.25))],
    "E1": [("shacl=off", dict(lambda_shacl=0.0)),
           ("shacl=loose", dict(lambda_shacl=1.0, shape_subset="loose")),
           ("shacl=tight", dict(lambda_shacl=1.0, shape_subset="tight"))],
    "E2": [(f"{t}@{d}", dict(lambda_shacl=1.0, tnorm=t, shape_depth=d))
           for t in ("product", "lukasiewicz", "godel") for d in (3, 8, 15)],
    "E3": [(f"factored={f}@{d}", dict(lambda_shacl=1.0, shape_factored=f, shape_depth=d))
           for d in (3, 15) for f in (False, True)],
}
