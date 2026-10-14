"""
Context encoder E_theta, EMA target encoder E_bar, conditioned predictor
Pred_phi, inverse-relation head I and (optionally) the shape bank.

The encoder is a graph transformer over instance tokens: full self-attention
plus a learned per-head bias for the typed relation between every token pair
(TokenGT-style in spirit, without node identifiers). Conditioning available
on this graph: the target type z_tau. z_do / z_pub from the plan need
action and publication metadata the graph does not carry yet.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from train.paper_a.graph_index import N_RELS, N_TOKEN_TYPES, REL_NONE, REL_SAME_PAPER
from train.paper_a.instances import N_PROBE_CLASSES


class RelBiasAttention(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.rel_bias = nn.Embedding(N_RELS, heads)
        nn.init.zeros_(self.rel_bias.weight)
        self.drop = nn.Dropout(dropout)
        self.capture, self.last_att = False, None      # worldview.py reads attention when capture is on

    def forward(self, x, rel, pad):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
        att = att + self.rel_bias(rel).permute(0, 3, 1, 2)
        att = att.masked_fill(pad[:, None, None, :], float("-inf"))
        att = self.drop(att.softmax(-1))
        if self.capture:
            self.last_att = att.detach()
        return self.out((att @ v).transpose(1, 2).reshape(B, T, -1))


class Block(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = RelBiasAttention(d, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(dropout))

    def forward(self, x, rel, pad):
        x = x + self.att(self.n1(x), rel, pad)
        return x + self.ff(self.n2(x))


def rwse(rel, pad, k):
    """Random-walk structural encoding (Graph-JEPA, GraphGPS): for every token the
    probability that a k-step random walk returns to it, k = 1..K, on the graph of
    typed links (role, proxy, cites, field, membership). The 'same paper' clique and
    'none' are not edges here, so the encoding describes a token's structural role
    (e.g. a claim with three pieces of evidence). -> (B, T, K)"""
    A = ((rel != REL_NONE) & (rel != REL_SAME_PAPER)).float()
    valid = (~pad).float()
    A = A * valid[:, :, None] * valid[:, None, :]
    M = A / A.sum(-1, keepdim=True).clamp_min(1.0)
    out, P = [], M
    for _ in range(k):
        out.append(torch.diagonal(P, dim1=-2, dim2=-1))
        P = P @ M
    return torch.stack(out, -1)


class GraphEncoder(nn.Module):
    def __init__(self, cfg, n_layers):
        super().__init__()
        d = cfg.d_model
        self.inp = nn.Linear(cfg.feat_dim, d)
        self.type_emb = nn.Embedding(N_TOKEN_TYPES, d)
        self.mask_emb = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList([Block(d, cfg.n_heads, cfg.dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.deg_emb = None                              # E4 census, own seed: init of the rest unchanged
        if cfg.use_degree_features:
            with torch.random.fork_rng():
                torch.manual_seed(7_919)
                self.deg_emb = nn.Embedding(65, d)
        self.k_rwse, self.rwse_proj = cfg.rwse_steps, None   # E20 structural encoding, own seed
        if self.k_rwse:
            with torch.random.fork_rng():
                torch.manual_seed(7_927)
                self.rwse_proj = nn.Linear(self.k_rwse, d)

    def forward(self, v, return_layers=False):
        h = self.inp(v["x"]) + self.type_emb(v["type"])
        h = h + v["is_mask"].unsqueeze(-1) * self.mask_emb
        if self.deg_emb is not None:                     # E4: the node census, exposed
            h = h + self.deg_emb(v["deg"].clamp(max=64))
        if self.rwse_proj is not None:                   # E20: structural role of every token
            h = h + self.rwse_proj(rwse(v["rel"], v["pad"], self.k_rwse))
        layers = []
        for blk in self.blocks:
            h = blk(h, v["rel"], v["pad"])
            if return_layers:
                layers.append(h)
        out = self.norm(h)
        return (out, layers) if return_layers else out


class Predictor(nn.Module):
    """Pred_phi(E(x), z_tau): refines the context at the masked token, conditioned
    on the target type, and reads out a d_model prediction."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.type_cond = nn.Embedding(N_TOKEN_TYPES, d)
        self.blocks = nn.ModuleList([Block(d, cfg.n_heads, cfg.dropout) for _ in range(cfg.pred_layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.pos_proj = None                             # E20: WHERE the hidden node sits (Graph-JEPA Eq. 5)
        if cfg.rwse_steps:
            with torch.random.fork_rng():
                torch.manual_seed(7_933)
                self.pos_proj = nn.Linear(cfg.rwse_steps, d)

    def forward(self, h, v, mask_pos, target_type, extra=None, target_pos_enc=None):
        """Every hidden token is conditioned on its own KU type (masked tokens keep their
        type; for the anchor this is target_type). extra = (batch idx, ctx pos) of further
        hidden motif members -> returns (anchor prediction, extra predictions)."""
        h = h + v["is_mask"].unsqueeze(-1) * self.type_cond(v["type"])
        if self.pos_proj is not None and target_pos_enc is not None:
            h = h + v["is_mask"].unsqueeze(-1) * self.pos_proj(target_pos_enc)
        for blk in self.blocks:
            h = blk(h, v["rel"], v["pad"])
        anchor = self.head(h[torch.arange(h.size(0), device=h.device), mask_pos])
        if extra is None:
            return anchor
        return anchor, self.head(h[extra[0], extra[1]])


class InverseRelationHead(nn.Module):
    """The IDM analogue: from two component representations, recover the typed
    role edge between them (or 'no edge'). The edge was hidden from the encoder,
    so the label is free supervision the encoder must preserve."""

    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, N_PROBE_CLASSES))

    def forward(self, hi, hj):
        return self.mlp(torch.cat([hi + hj, (hi - hj).abs(), hi * hj], -1))


class PaperAModel(nn.Module):
    def __init__(self, cfg, shape_bank=None):
        super().__init__()
        self.cfg = cfg
        self.context = GraphEncoder(cfg, cfg.enc_layers)
        self.target = copy.deepcopy(self.context).requires_grad_(False)
        self.predictor = Predictor(cfg)
        self.inverse = InverseRelationHead(cfg.d_model)
        self.shapes = shape_bank

    @torch.no_grad()
    def ema(self, m):
        for pt, pc in zip(self.target.parameters(), self.context.parameters()):
            pt.mul_(m).add_(pc.detach(), alpha=1 - m)

    @torch.no_grad()
    def encode_target(self, v, pos, extra=None):
        z = self.target(v)
        za = F.layer_norm(z[torch.arange(z.size(0), device=z.device), pos], z.shape[-1:])
        if extra is None:
            return za
        return za, F.layer_norm(z[extra[0], extra[2]], z.shape[-1:])

    def _target_pos_enc(self, batch):
        """E20: the hidden nodes' TRUE structural positions (RWSE in the unmasked target
        view), placed at their context positions -> (B, T_ctx, K)."""
        k = self.cfg.rwse_steps
        if not k:
            return None
        c, t = batch["ctx"], batch["tgt"]
        r = rwse(t["rel"], t["pad"], k)
        enc = torch.zeros(c["x"].shape[0], c["x"].shape[1], k, device=c["x"].device)
        b = torch.arange(enc.size(0), device=enc.device)
        enc[b, batch["mask_pos"]] = r[b, batch["target_pos"]]
        ex = batch.get("extra")
        if ex is not None and len(ex[0]):
            enc[ex[0], ex[1]] = r[ex[0], ex[2]]
        return enc

    def predict(self, batch, with_extra=False):
        h = self.context(batch["ctx"])
        pe = self._target_pos_enc(batch)
        ex = batch.get("extra")
        if with_extra and ex is not None and len(ex[0]):
            a, e = self.predictor(h, batch["ctx"], batch["mask_pos"], batch["target_type"], (ex[0], ex[1]), pe)
            return (a, e), h
        return self.predictor(h, batch["ctx"], batch["mask_pos"], batch["target_type"], target_pos_enc=pe), h

    def forward(self, batch):
        ex = batch.get("extra")
        has_extra = ex is not None and len(ex[0]) > 0
        pred, h = self.predict(batch, with_extra=has_extra)
        if has_extra:
            (pred, pred_x), (tgt, tgt_x) = pred, self.encode_target(batch["tgt"], batch["target_pos"], ex)
            out = {"pred": pred, "h": h, "target": tgt, "pred_extra": pred_x, "target_extra": tgt_x}
        else:
            out = {"pred": pred, "h": h, "target": self.encode_target(batch["tgt"], batch["target_pos"])}
        b, i, j, lab = batch["probe"]
        if len(b):
            out["inv_logits"] = self.inverse(h[b, i], h[b, j])
            out["inv_labels"] = lab
        return out


def to_device(batch, dev):
    if isinstance(batch, dict):
        return {k: to_device(v, dev) for k, v in batch.items()}
    if isinstance(batch, (tuple, list)) and batch and torch.is_tensor(batch[0]):
        return type(batch)(t.to(dev) for t in batch)
    return batch.to(dev) if torch.is_tensor(batch) else batch
