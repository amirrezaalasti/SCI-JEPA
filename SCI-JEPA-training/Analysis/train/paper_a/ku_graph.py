"""
Build the Paper A component graph straight from the normalized parquet shards
(Scientific-Summaries-Data/normalized/<subset>/*.parquet). Replaces the old
raw-JSON builder (core/data_utils/paper_graph.py), which kept only five KU
types and dropped procedures whenever methodological details existed.

Streaming: shards are read a few columns at a time (never text_raw /
text_sanitized), KUs are embedded in chunks, and only float16 embeddings are
kept. Output is the blob GraphIndex consumes.

    python -m train.paper_a.ku_graph --set max_papers=60000     # build + cache D0
"""
import glob
import hashlib
import os
import time
import zlib
from collections import Counter, defaultdict

import numpy as np
import torch

from train.paper_a import ku_schema as S

INDEX_VERSION = 4          # bump when the blob format changes (2: KU text snippets)

READ_COLS = (list(S.SECTION_KUS.values()) + list(S.OPTIONAL_KUS.values())
             + ["claims", "paper_id", "subset", "source_year", "oa_year", "field_subfield",
                "type_of_paper", "openalex_id", "oa_referenced_works"])


# ---------------------------------------------------------------- embedders
class HashEmbedder:
    """Deterministic bag-of-words random projection. FOR SMOKE TESTS ONLY: it
    has no semantics beyond shared words."""

    def __init__(self, dim=384):
        self.dim = dim

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                h = zlib.crc32(w.encode())
                out[i, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-8)


class STEmbedder:
    def __init__(self, name, device, max_tokens, batch=32, prefix=""):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name, device=device)
        if max_tokens:
            self.m.max_seq_length = min(max_tokens, self.m.max_seq_length or max_tokens)
        self.dim = self.m.get_sentence_embedding_dimension()
        self.batch, self.prefix = batch, prefix
        self.n_seen = self.n_truncated = 0
        print(f"[ku_graph] text encoder {name}: {self.dim}-d, context {self.m.max_seq_length} tokens, {device}")

    def encode(self, texts, token_budget=8192, bucket=512):
        """Batches by a token budget (many short KUs, few long papers per pass) and pads
        to a multiple of `bucket`: on Apple GPUs every distinct sequence length keeps its
        own cached buffers, and thousands of lengths ran an 88 GB GPU out of memory."""
        import torch
        if self.prefix:
            texts = [self.prefix + t for t in texts]
        tok, L = self.m.tokenizer, self.m.max_seq_length
        n_tok = np.array([len(x) for x in tok(texts, add_special_tokens=True)["input_ids"]])
        self.n_seen += len(texts)
        self.n_truncated += int((n_tok > L).sum())
        order = np.argsort(-n_tok)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        dev = self.m.device
        i, t0, shown = 0, time.time(), 0
        while i < len(order):
            if len(texts) >= 500 and i >= shown:     # visible progress for long encodes
                el = time.time() - t0
                eta = f", about {el / i * (len(texts) - i) / 60:.0f} min left" if i else ""
                print(f"[ku_graph]   encoded {i} / {len(texts)} texts ({el / 60:.1f} min{eta})", flush=True)
                shown = i + max(len(texts) // 10, 1)
            padded = min(int(np.ceil(min(n_tok[order[i]], L) / bucket) * bucket), L)
            idx = order[i:i + max(1, min(self.batch, token_budget // padded))]
            f = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=L,
                    pad_to_multiple_of=bucket, return_tensors="pt")
            with torch.no_grad():
                z = self.m.forward({k: v.to(dev) for k, v in f.items()})["sentence_embedding"]
            out[idx] = z.float().cpu().numpy()
            i += len(idx)
            if dev.type == "mps":
                torch.mps.empty_cache()
        return out


def make_embedder(cfg, dev):
    if cfg.embedder == "hash":
        return HashEmbedder(cfg.feat_dim)
    return STEmbedder(cfg.embed_model, dev, cfg.embed_max_tokens, cfg.embed_batch, cfg.embed_prefix)


# ---------------------------------------------------------------- reading
def shard_files(cfg):
    files = []
    for d in cfg.parquet_dirs:
        files += sorted(glob.glob(os.path.join(d, "*.parquet")))
    rng = np.random.default_rng(cfg.data_seed)
    rng.shuffle(files)                       # tiers are random papers, not the first shards
    return files


def iter_records(cfg):
    """Each shard is a contiguous slice of one subset and a narrow year range, so
    a tier takes `rows_per_shard` random rows from many shuffled shards instead of
    whole shards (whole shards would make the time split degenerate)."""
    import pyarrow.parquet as pq
    rng = np.random.default_rng(cfg.data_seed + 1)
    for f in shard_files(cfg):
        pf = pq.ParquetFile(f)
        cols = [c for c in READ_COLS if c in pf.schema_arrow.names]
        n = pf.metadata.num_rows
        if cfg.rows_per_shard and cfg.rows_per_shard < n:
            take = np.sort(rng.choice(n, cfg.rows_per_shard, replace=False))
            t = pf.read(columns=cols).take(take)
            yield from t.to_pylist()
            del t
        else:
            for batch in pf.iter_batches(batch_size=512, columns=cols):
                yield from batch.to_pylist()


def keep_record(rec, cfg):
    """-> (keep, year or None)."""
    if cfg.subsets and rec.get("subset") not in cfg.subsets:
        return False, None
    y = S.paper_year(rec)
    if cfg.split_mode == "time" and y is None:
        return False, None
    if y is not None and not (cfg.year_min <= y <= cfg.year_max):
        return False, y
    return True, y


# ---------------------------------------------------------------- build
def build_blob(cfg, records=None):
    """records: optional iterable of dicts in the normalized schema (synthetic
    tests); default streams cfg.parquet_dirs."""
    from train.paper_a.config import resolve_device
    dev = resolve_device(cfg.device)
    emb = make_embedder(cfg, dev)
    t0 = time.time()
    records = iter_records(cfg) if records is None else records

    comp_x, comp_type, comp_paper, comp_flags, comp_snip = [], [], [], defaultdict(list), []
    role_src, role_dst, role_rel = [], [], []
    p_year, p_field, p_sub, p_type, p_key, p_id, p_refs = [], [], [], [], [], [], []
    stats, skipped = Counter(), Counter()
    buf_txt, n_comp = [], 0
    pm_parts = []                             # paper_masked: each paper's cleaned KUs + edges
    mp_src = []                               # (comp id, text) of results / claims: minimal-pair sources

    def flush():
        if buf_txt:
            comp_x.append(np.asarray(emb.encode(buf_txt), dtype=np.float16))
            buf_txt.clear()

    for rec in records:
        keep, y = keep_record(rec, cfg)
        if not keep:
            skipped["filtered"] += 1
            continue
        kus, edges, st = S.record_to_kus(rec, cfg.ku_min_chars, cfg.optional_kus)
        stats.update(st)
        types = [t for t, _, _ in kus]
        if len(kus) < cfg.min_kus_per_paper or not all(t in types for t in cfg.required_kus):
            skipped["too_few_kus"] += 1
            continue
        if cfg.ku_mode == "paper":           # same papers as section mode, each collapsed into one KU
            kus, edges = S.paper_ku(kus, edges)
        elif cfg.ku_mode == "paper_masked":
            pm_parts.append((kus, edges))
        p = len(p_year)
        p_year.append(-1 if y is None else y)
        coarse, fine = S.split_field(rec.get("field_subfield"))
        p_field.append(coarse); p_sub.append(fine)
        p_type.append(S.paper_type(rec.get("type_of_paper")))
        p_key.append(S.openalex_key(rec.get("openalex_id")))
        p_id.append(str(rec.get("paper_id")))
        p_refs.append(S.referenced(rec))
        if cfg.ku_mode != "paper":
            mp_src += [(n_comp + k, text) for k, (t, text, _) in enumerate(kus) if t in S.MP_TYPES]
        for t, text, fl in kus:
            buf_txt.append(text)
            comp_snip.append(text[:cfg.snippet_chars])
            comp_type.append(S.COMP_TYPES.index(t)); comp_paper.append(p)
            for k in ("quantitative", "cites_figure", "challenged"):
                comp_flags[k].append(bool(fl.get(k, False)))
        for a, r, b in edges:
            role_src.append(n_comp + a); role_dst.append(n_comp + b); role_rel.append(S.REL_ID[r])
        n_comp += len(kus)
        if len(buf_txt) >= cfg.embed_chunk:
            flush()
        if len(p_year) % 10000 == 0:
            print(f"[ku_graph] {len(p_year)} papers, {n_comp} KUs ({time.time() - t0:.0f}s)", flush=True)
        if cfg.max_papers and len(p_year) >= cfg.max_papers:
            break
    flush()
    P = len(p_year)
    if P == 0:
        raise RuntimeError("no papers selected -- check parquet_dirs / subsets / split_mode")

    comp_type = np.asarray(comp_type, dtype=np.int64)
    comp_paper = np.asarray(comp_paper, dtype=np.int64)
    comp_x = np.concatenate(comp_x, 0)

    # ---- fields --------------------------------------------------------------
    fields = sorted(set(p_field)); subs = sorted(set(p_sub))
    fid = {f: i for i, f in enumerate(fields)}; sid = {s: i for i, s in enumerate(subs)}
    field_x = np.asarray(emb.encode(fields), dtype=np.float16)

    # ---- citations inside the selected corpus ------------------------------------
    key_to_p = {k: i for i, k in enumerate(p_key) if k}
    pp_s, pp_d, pp_w = [], [], []
    for i, refs in enumerate(p_refs):
        for k in refs:
            j = key_to_p.get(k)
            if j is not None and j != i:
                pp_s.append(i); pp_d.append(j); pp_w.append(1)
    n_cites = len(pp_s)

    # ---- proxy grounded edges -------------------------------------------------------
    from train.paper_a.graph_index import build_proxy_edges
    for t in cfg.proxy_types:
        ids = np.nonzero(comp_type == S.COMP_TYPES.index(t))[0]
        pr = build_proxy_edges(comp_x[ids].astype(np.float32), comp_paper[ids],
                               cfg.proxy_tau, cfg.proxy_topk, dev)
        if len(pr):
            a, b = ids[pr[:, 0]], ids[pr[:, 1]]
            role_src += a.tolist(); role_dst += b.tolist(); role_rel += [S.REL_ID["proxy"]] * len(a)
            pp_s += comp_paper[a].tolist(); pp_d += comp_paper[b].tolist(); pp_w += [2] * len(a)
        print(f"[ku_graph] proxy '{t}' edges: {len(pr)} (tau={cfg.proxy_tau})")

    # ---- membership CSR + splits --------------------------------------------------
    order = np.argsort(comp_paper, kind="stable")
    pc_ptr = np.concatenate([[0], np.cumsum(np.bincount(comp_paper, minlength=P))])
    years = np.asarray(p_year)
    train, val, test = make_splits(cfg, years)

    pm = build_context_texts(cfg, emb, pm_parts, p_field, p_sub, (train, val, test)) \
        if cfg.ku_mode == "paper_masked" else {}
    pm.update(build_minimal_pairs(emb, mp_src, comp_paper, np.concatenate([val, test])))

    ku_stats = {f"{t}:{w}": int(c) for (t, w), c in sorted(stats.items())}
    print(f"[ku_graph] {P} papers, {len(comp_type)} KUs, {n_cites} in-corpus citations, "
          f"{time.time() - t0:.0f}s; skipped {dict(skipped)}")
    if getattr(emb, "n_seen", 0):
        print(f"[ku_graph] text encoder truncated {emb.n_truncated} of {emb.n_seen} texts "
              f"(context {cfg.embed_max_tokens} tokens)")
    return {"comp_x": comp_x, "comp_type": comp_type, "comp_paper": comp_paper,
            "comp_quantitative": np.asarray(comp_flags["quantitative"]),
            "comp_cites_figure": np.asarray(comp_flags["cites_figure"]),
            "comp_challenged": np.asarray(comp_flags["challenged"]),
            "comp_snippet": np.asarray(comp_snip, dtype=object),
            "pc_ptr": pc_ptr, "pc_idx": order,
            "role_src": np.asarray(role_src, dtype=np.int64), "role_dst": np.asarray(role_dst, dtype=np.int64),
            "role_rel": np.asarray(role_rel, dtype=np.int64),
            "pp_src": np.asarray(pp_s, dtype=np.int64), "pp_dst": np.asarray(pp_d, dtype=np.int64),
            "pp_w": np.asarray(pp_w, dtype=np.int64),
            "paper_field": np.asarray([fid[f] for f in p_field]), "paper_subfield": np.asarray([sid[s] for s in p_sub]),
            "paper_year": years, "paper_type": np.asarray(p_type), "paper_ids": p_id,
            "field_names": fields, "field_x": field_x,
            "train_papers": train, "val_papers": val, "test_papers": test, **pm,
            "build_stats": {"ku_outcomes": ku_stats, "skipped": dict(skipped), "in_corpus_citations": n_cites,
                            "embedder": cfg.embedder if cfg.embedder == "hash" else cfg.embed_model,
                            "embed_max_tokens": cfg.embed_max_tokens, "feat_dim": int(comp_x.shape[1]),
                            "kus_truncated": int(getattr(emb, "n_truncated", 0)),
                            "kus_embedded": int(getattr(emb, "n_seen", 0)),
                            "split_mode": cfg.split_mode, "time_T": cfg.time_T}}


def build_minimal_pairs(emb, src, comp_paper, eval_papers):
    """Minimal-pair counterfactuals (ku_schema.counterfactuals) of every validation / test
    result and claim, embedded with the same text encoder (mp_*)."""
    keep = set(int(p) for p in eval_papers)
    rows, texts = [], []
    for c, text in src:
        if int(comp_paper[c]) in keep:
            for kind, t in S.counterfactuals(text):
                rows.append((c, ("direction", "negation", "number").index(kind))); texts.append(t)
    X = np.asarray(emb.encode(texts), dtype=np.float16) if texts else np.zeros((0, 1), np.float16)
    print(f"[ku_graph] minimal pairs: {len(texts)} counterfactual texts for "
          f"{len({c for c, _ in rows})} validation / test results and claims")
    return {"mp_comp": np.asarray([c for c, _ in rows], dtype=np.int64),
            "mp_kind": np.asarray([k for _, k in rows], dtype=np.int64), "mp_x": X,
            "mp_snippet": np.asarray([t[:240] for t in texts], dtype=object)}


def build_context_texts(cfg, emb, parts, p_field, p_sub, splits):
    """ku_mode="paper_masked": for every paper and template, the given KUs joined into ONE
    text and encoded together (ctx_*). For val / test papers also counterfactual forward texts
    with the procedure (condition) or the method replaced by a donor paper's, for up to 3
    donors each (cf_*)."""
    t0 = time.time()
    names = S.active_templates(cfg.pm_templates)
    texts, c_paper, c_tpl, seen = [], [], [], {}
    for p, (kus, edges) in enumerate(parts):
        for name in names:
            given, targets = S.TEMPLATES[name]
            if not any(t == k for k, _, _ in kus for t in targets):
                continue                                   # nothing of the target type to predict
            txt = S.compose_text(kus, edges, given)
            if not txt:
                continue
            key = (p, txt)
            if key not in seen:                            # identical texts share one row
                seen[key] = len(texts); texts.append(txt)
            c_paper.append(p); c_tpl.append(S.TEMPLATE_NAMES.index(name))
    rows = [seen[(p, S.compose_text(parts[p][0], parts[p][1], S.TEMPLATES[S.TEMPLATE_NAMES[t]][0]))]
            for p, t in zip(c_paper, c_tpl)]
    # counterfactual forward contexts for evaluation papers: change(Condition := c'), change(Method := m')
    rng = np.random.default_rng(cfg.split_seed + 99)
    first = lambda kus, t: next((txt for k, txt, _ in kus if k == t), None)
    cf_txt, cf_paper, cf_kind, cf_donor = [], [], [], []
    for split in splits[1:]:
        pool = [int(p) for p in split]
        for p in pool:
            for kind, t in enumerate(("procedure", "method")):
                if first(parts[p][0], t) is None:
                    continue
                # 3 donors from different pools (same subfield, same field, anywhere in the split):
                # evaluation uses the first one that is not a visible neighbour of the query
                donors = []
                for same in (lambda q: p_sub[q] == p_sub[p], lambda q: p_field[q] == p_field[p], lambda q: True):
                    cand = [q for q in pool if q != p and q not in donors and same(q) and first(parts[q][0], t)]
                    if cand:
                        donors.append(int(rng.choice(cand)))
                for d in donors:
                    cf_txt.append(S.compose_text(parts[p][0], parts[p][1], S.TEMPLATES["forward"][0],
                                                 replace={t: first(parts[d][0], t)}))
                    cf_paper.append(p); cf_kind.append(kind); cf_donor.append(int(d))
    print(f"[ku_graph] paper_masked: {len(texts)} context texts ({len(names)} templates) + "
          f"{len(cf_txt)} counterfactual texts to encode", flush=True)
    X = np.asarray(emb.encode(texts), dtype=np.float16) if texts else np.zeros((0, 1), np.float16)
    CF = np.asarray(emb.encode(cf_txt), dtype=np.float16) if cf_txt else np.zeros((0, X.shape[1]), np.float16)
    print(f"[ku_graph] paper_masked texts encoded in {time.time() - t0:.0f}s")
    return {"ctx_x": X, "ctx_row": np.asarray(rows, dtype=np.int64), "ctx_paper": np.asarray(c_paper, dtype=np.int64),
            "ctx_tpl": np.asarray(c_tpl, dtype=np.int64), "template_names": S.TEMPLATE_NAMES,
            "cf_x": CF, "cf_paper": np.asarray(cf_paper, dtype=np.int64), "cf_kind": np.asarray(cf_kind, dtype=np.int64),
            "cf_donor": np.asarray(cf_donor, dtype=np.int64)}


def make_splits(cfg, years):
    P = len(years)
    if cfg.split_mode == "time":
        T = cfg.time_T
        if cfg.encoder_cutoff_year and T < cfg.encoder_cutoff_year:
            print(f"[ku_graph] WARNING: time_T={T} precedes the text encoder's pretraining cutoff "
                  f"({cfg.encoder_cutoff_year}); the encoder may have read the test years. "
                  f"Foresight claims are invalid for this slice.")
        train = np.nonzero(years <= T)[0]
        val = np.nonzero(years == T + 1)[0]
        test = np.nonzero(years == T + 2)[0]
        print(f"[ku_graph] time split T={T}: train<= {len(train)}  val {len(val)}  test {len(test)}")
        return train, val, test
    rng = np.random.default_rng(cfg.split_seed)
    perm = rng.permutation(P)
    a = int(cfg.split[0] * P); b = a + int(cfg.split[1] * P)
    return np.sort(perm[:a]), np.sort(perm[a:b]), np.sort(perm[b:])


def data_fingerprint(cfg):
    keys = ["parquet_dirs", "subsets", "max_papers", "data_seed", "rows_per_shard", "year_min", "year_max", "split_mode",
            "time_T", "split", "split_seed", "ku_min_chars", "optional_kus", "min_kus_per_paper",
            "required_kus", "ku_mode", "pm_templates", "embedder", "embed_model", "embed_max_tokens", "proxy_types", "proxy_tau",
            "proxy_topk", "snippet_chars"]
    return hashlib.sha1(repr([INDEX_VERSION] + [getattr(cfg, k) for k in keys]).encode()).hexdigest()[:12]


if __name__ == "__main__":
    import argparse
    from train.paper_a.config import PaperAConfig
    from train.paper_a.train import parse_overrides
    ap = argparse.ArgumentParser("ku_graph")
    ap.add_argument("--set", nargs="*", default=[])
    cfg = PaperAConfig().override(**parse_overrides(ap.parse_args().set))
    from train.paper_a.graph_index import get_index
    gi = get_index(cfg)
    import json
    print(json.dumps(gi.summary(), indent=1, default=str))
