"""
Use a trained SCI-JEPA model outside training (prototype / production).

    from train.paper_a.inference import SciJEPA
    m = SciJEPA.from_pretrained("paper_a_results/BENCH/models/motif+world_seed0")
    paper = {"research_context": "...", "research_question_hypothesis": "...",
             "methodological_details": "...", "procedures_architectures": "...",
             "key_results": "...", "claims": [{"description": "...", "supporting_evidence": "...",
             "contradicting_evidence": "", "implications": "..."}], "field_subfield": "Computer Science — ML"}
    m.embed(paper)                                   # one vector per knowledge unit (facet embeddings) + paper
    m.predict(paper, hide="result")                  # predicted representation of the result
    m.rank(paper, hide="result", candidates=[...])   # which candidate text is the most likely result
    m.consistency(claim_text, evidence_text)         # P(supports), P(contradicts)

A paper is given as a structured summary in the training data's format (the fields of the
normalized parquet; `claims` may be a list or a JSON string). The model is run on that paper
alone (no neighbouring papers), which is how its candidate keys are built in evaluation.
Candidates in `rank` are encoded alone (one KU, no paper context): an approximation of the
evaluation protocol, where each candidate is encoded inside its own paper.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from train.paper_a import ku_schema as S
from train.paper_a.config import PaperAConfig, resolve_device
from train.paper_a.graph_index import COMP_TYPES, HUB
from train.paper_a.instances import collate_views
from train.paper_a.ku_schema import LEAK_FAMILY, REL_ID, REL_MEMBER, REL_NONE, REL_SAME_PAPER, TEMPLATES
from train.paper_a.model import PaperAModel, to_device

_SUP, _CON = REL_ID["supported_by"] - 3, REL_ID["challenged_by"] - 3


class SciJEPA:
    def __init__(self, model, cfg, device, embedder=None):
        self.model, self.cfg, self.dev = model.eval(), cfg, device
        self._emb = embedder

    # ------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, path, weights="model", device="auto"):
        """path: a folder written by train.save_model. weights: "model" (end of training) or
        "best" (best validation epoch)."""
        from safetensors.torch import load_file
        raw = json.load(open(os.path.join(path, "config.json")))
        base = PaperAConfig()
        cfg = base.override(**{k: v for k, v in raw.items() if hasattr(base, k) and k not in ("embedder",)})
        dev = torch.device(resolve_device(device))
        model = PaperAModel(cfg)
        f = os.path.join(path, f"{weights}.safetensors")
        if not os.path.exists(f):
            f = os.path.join(path, "model.safetensors")
        missing, unexpected = model.load_state_dict(load_file(f), strict=False)
        if [k for k in missing if not k.startswith("shapes.")]:
            raise RuntimeError(f"weights do not match the model: missing {missing[:5]}")
        return cls(model.to(dev), cfg, dev)

    @property
    def text_encoder(self):
        if self._emb is None:                            # the same frozen encoder the model was trained on
            from train.paper_a.ku_graph import STEmbedder
            self._emb = STEmbedder(self.cfg.embed_model, str(self.dev), self.cfg.embed_max_tokens,
                                   self.cfg.embed_batch, self.cfg.embed_prefix)
        return self._emb

    # ------------------------------------------------------------ building a paper view
    def _kus(self, paper):
        rec = dict(paper)
        if isinstance(rec.get("claims"), list):
            rec["claims"] = json.dumps(rec["claims"])
        kus, edges, _ = S.record_to_kus(rec, self.cfg.ku_min_chars, self.cfg.optional_kus)
        if not kus:
            raise ValueError("no knowledge units found: give at least a few summary fields")
        return kus, edges, str(rec.get("field_subfield") or "")

    def _view(self, kus, edges, field, hide=None, given=None):
        """Tokens of one paper (hub + KUs). hide: a KU type to mask (added as a mask token if the
        paper does not have one). given: only these KU types stay visible."""
        types = [t for t, _, _ in kus]
        keep = list(range(len(kus)))
        if given is not None:
            keep = [i for i in keep if types[i] in given]
        if hide is not None:
            fam = set(LEAK_FAMILY.get(hide, (hide,)))
            first = next((i for i in range(len(kus)) if types[i] == hide), None)
            keep = [i for i in keep if types[i] not in fam]      # the hidden KU and what restates it
            keep = keep + ([first] if first is not None else [])
        texts = [kus[i][1] for i in keep]
        X = self.text_encoder.encode(texts) if texts else np.zeros((0, self.cfg.feat_dim), np.float32)
        hub = self.text_encoder.encode([field.split("—")[0].strip() or "Unknown"])[0]
        x = np.concatenate([hub[None], X], 0).astype(np.float32)
        typ = np.asarray([HUB] + [COMP_TYPES.index(types[i]) for i in keep], np.int64)
        T = len(typ)
        rel = np.full((T, T), REL_SAME_PAPER, np.int64)
        rel[0, 1:] = rel[1:, 0] = REL_MEMBER
        pos = {i: k + 1 for k, i in enumerate(keep)}
        for a, r, b in edges:
            if a in pos and b in pos:
                rel[pos[a], pos[b]] = rel[pos[b], pos[a]] = REL_ID[r]
        np.fill_diagonal(rel, REL_NONE)
        m = None
        if hide is not None:
            if first is None or first not in pos:                # no such KU: add an empty mask token
                x = np.concatenate([x, np.zeros((1, x.shape[1]), np.float32)]); typ = np.append(typ, COMP_TYPES.index(hide))
                rel = np.pad(rel, ((0, 1), (0, 1)), constant_values=REL_SAME_PAPER); rel[-1, 0] = rel[0, -1] = REL_MEMBER
                rel[-1, -1] = REL_NONE; m = len(typ) - 1
            else:
                m = pos[first]
            x[m] = 0.0
            role = ~np.isin(rel[m], [REL_NONE, REL_SAME_PAPER, REL_MEMBER])
            rel[m, role] = REL_SAME_PAPER; rel[role, m] = REL_SAME_PAPER
        view = {"x": x, "type": typ, "comp": np.arange(len(typ)) - 1, "slot": np.zeros(len(typ), np.int64),
                "deg": np.full(len(typ), len(typ) - 1, np.int64), "rel": rel}
        mask = np.zeros(len(typ), bool)
        if m is not None:
            mask[m] = True
        return view, mask, m, [types[i] for i in keep]

    # ------------------------------------------------------------ uses
    @torch.no_grad()
    def embed(self, paper):
        """Facet embeddings: the context encoder's state of every knowledge unit, plus their
        mean as a paper embedding. -> {"kus": [(type, vector)], "paper": vector}"""
        kus, edges, field = self._kus(paper)
        view, mask, _, types = self._view(kus, edges, field)
        v = to_device(collate_views([view], [mask]), self.dev)
        h = self.model.context(v)[0, 1:].float().cpu().numpy()
        return {"kus": list(zip(types, h)), "paper": h.mean(0)}

    @torch.no_grad()
    def predict(self, paper, hide="result", task=None):
        """Predicted representation of the hidden KU type. task: "forward" / "inverse" /
        "abduction" shows only that reasoning template's given parts."""
        kus, edges, field = self._kus(paper)
        given = set(TEMPLATES[task][0]) if task else None
        view, mask, m, _ = self._view(kus, edges, field, hide=hide, given=given)
        v = to_device(collate_views([view], [mask]), self.dev)
        h = self.model.context(v)
        return self.model.predictor(h, v, torch.tensor([m], device=self.dev),
                                    torch.tensor([COMP_TYPES.index(hide)], device=self.dev))[0].float().cpu()

    @torch.no_grad()
    def rank(self, paper, hide, candidates, task=None):
        """Rank candidate texts for the hidden KU type. -> [(score, text)] best first."""
        pred = self.predict(paper, hide, task)
        X = self.text_encoder.encode(list(candidates))
        views = []
        for x in X:
            views.append({"x": np.stack([np.zeros_like(x), x]).astype(np.float32),
                          "type": np.asarray([HUB, COMP_TYPES.index(hide)]), "comp": np.asarray([-1, 0]),
                          "slot": np.zeros(2, np.int64), "deg": np.ones(2, np.int64),
                          "rel": np.asarray([[REL_NONE, REL_MEMBER], [REL_MEMBER, REL_NONE]])})
        keys = self.model.target(to_device(collate_views(views), self.dev))[:, 1].float().cpu()
        s = F.cosine_similarity(pred[None], keys).numpy()
        return sorted(zip(s.tolist(), candidates), reverse=True)

    @torch.no_grad()
    def consistency(self, claim, evidence):
        """P(supports), P(contradicts) for a claim and a piece of evidence (link hidden), from
        the relation decoder."""
        X = self.text_encoder.encode([claim, evidence])
        view = {"x": np.concatenate([np.zeros((1, X.shape[1])), X]).astype(np.float32),
                "type": np.asarray([HUB, COMP_TYPES.index("claim"), COMP_TYPES.index("evidence")]),
                "comp": np.asarray([-1, 0, 1]), "slot": np.zeros(3, np.int64), "deg": np.full(3, 2),
                "rel": np.asarray([[REL_NONE, REL_MEMBER, REL_MEMBER], [REL_MEMBER, REL_NONE, REL_SAME_PAPER],
                                   [REL_MEMBER, REL_SAME_PAPER, REL_NONE]])}
        v = to_device(collate_views([view]), self.dev)
        h = self.model.context(v)
        logit = self.model.inverse(h[:, 1], h[:, 2])[0]
        p = torch.softmax(logit[[_SUP, _CON]], -1).cpu().numpy()
        return {"supports": float(p[0]), "contradicts": float(p[1])}
