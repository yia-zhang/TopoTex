# -*- coding: utf-8 -*-
"""Topology-aware face transformer: sparse graph attention along the shared-
edge face adjacency (self + topology neighbors only -- NO xyz/euclidean KNN).

QK attention with a learned relation bias from (shared edge length, dihedral
angle, boundary flag).
"""
import torch
import torch.nn as nn


def scatter_softmax(scores, index, num_nodes):
    """Softmax of `scores` grouped by `index` (destination node)."""
    m = torch.full((num_nodes,), -1e30, device=scores.device,
                   dtype=scores.dtype)
    m = m.index_reduce(0, index, scores, "amax", include_self=True)
    ex = torch.exp(scores - m[index])
    denom = torch.zeros(num_nodes, device=scores.device, dtype=scores.dtype)
    denom = denom.index_add(0, index, ex)
    return ex / denom[index].clamp(min=1e-20)


class GraphAttnBlock(nn.Module):
    def __init__(self, dim, heads, rel_dim=3):
        super().__init__()
        self.h = heads
        self.dk = dim // heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        # final bias would add a per-edge-constant that cancels in the grouped
        # softmax (structurally gradient-free) -> bias=False
        self.rel_bias = nn.Sequential(nn.Linear(rel_dim + 1, 32), nn.GELU(),
                                      nn.Linear(32, heads, bias=False))
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                 nn.Linear(dim * 4, dim))

    def forward(self, x, edges_full, rel_full):
        """x [F,D]; edges_full [E,2] (src,dst) INCLUDING self loops;
        rel_full [E,4] relation features (self loops flagged)."""
        F = x.shape[0]
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(F, self.h, self.dk)
        k = k.view(F, self.h, self.dk)
        v = v.view(F, self.h, self.dk)
        src, dst = edges_full[:, 0], edges_full[:, 1]
        # attention of dst (query) over src (key/value)
        logits = (q[dst] * k[src]).sum(-1) / self.dk ** 0.5    # [E,h]
        logits = logits + self.rel_bias(rel_full)              # [E,h]
        out = torch.zeros(F, self.h, self.dk, device=x.device, dtype=x.dtype)
        for hh in range(self.h):
            w = scatter_softmax(logits[:, hh], dst, F)         # [E]
            out[:, hh] = torch.zeros(F, self.dk, device=x.device,
                                     dtype=x.dtype).index_add(
                0, dst, w.unsqueeze(1) * v[src, hh])
        x = x + self.proj(out.reshape(F, -1))
        return x + self.mlp(self.norm2(x))


class TopologyTransformer(nn.Module):
    def __init__(self, dim=256, heads=8, depth=4):
        super().__init__()
        self.blocks = nn.ModuleList([GraphAttnBlock(dim, heads)
                                     for _ in range(depth)])

    def forward(self, x, graph):
        """x [F,D]; graph from build_face_graph."""
        F = x.shape[0]
        device = x.device
        loops = torch.arange(F, device=device)
        edges_full = torch.cat([graph["edges"],
                                torch.stack([loops, loops], dim=1)], dim=0)
        # rel: (edge_len, dihedral, boundary-of-src, is_self)
        rel_e = torch.cat([graph["rel"][:, :2],
                           graph["boundary"][graph["edges"][:, 0]].unsqueeze(1)
                           if len(graph["edges"]) else
                           torch.zeros(0, 1, device=device),
                           torch.zeros(len(graph["edges"]), 1, device=device)],
                          dim=1) if len(graph["edges"]) else \
            torch.zeros(0, 4, device=device)
        rel_s = torch.cat([torch.zeros(F, 2, device=device),
                           graph["boundary"].unsqueeze(1).float(),
                           torch.ones(F, 1, device=device)], dim=1)
        rel_full = torch.cat([rel_e, rel_s], dim=0).float()
        for blk in self.blocks:
            x = blk(x, edges_full, rel_full)
        return x
