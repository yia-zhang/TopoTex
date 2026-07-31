# -*- coding: utf-8 -*-
"""Pluggable topology positional encodings over the face-adjacency graph.

Variants: 'none' | 'random_walk' (default). Both are functions of graph
structure only -> permutation equivariant and rigid/scale invariant by
construction. (A spectral/laplacian variant lived here during ablations; it
is not permutation-equivariant under degenerate spectra and was retired.)
"""

import torch
import torch.nn as nn

from topotex.data.schema import FaceGraph


# --------------------------------------------------------------- graph utils
def build_face_graph(vertices, faces):
    """Shared-edge face adjacency + relation features.

    Returns dict:
      edges       int64 [E,2]  directed pairs (i,j), both directions, i!=j
      rel         float32 [E,3] (shared_edge_len/global_scale, dihedral_angle,
                                 0.0 placeholder-for-boundary on edges)
      boundary    float32 [F]   fraction of a face's edges with no neighbor
      global_scale float scalar sqrt(total mesh area)
    Non-manifold edges (>2 faces) connect ALL incident face pairs.
    """
    V = vertices
    F = faces
    device = V.device
    f = F.cpu().numpy()
    import numpy as np

    e2f = {}
    for fi in range(len(f)):
        a, b, c = int(f[fi, 0]), int(f[fi, 1]), int(f[fi, 2])
        for u, v in ((a, b), (b, c), (c, a)):
            e2f.setdefault((min(u, v), max(u, v)), []).append(fi)

    # face normals + areas (float64 for stable dihedral)
    tri = V[F].double()
    n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    area2 = n.norm(dim=1)
    normals = n / area2.clamp(min=1e-20).unsqueeze(1)
    total_area = (area2 / 2).sum()
    global_scale = torch.sqrt(total_area.clamp(min=1e-20))

    # index lists in python; ALL vertex-dependent math vectorized afterwards
    # (per-edge GPU item() syncs made this seconds-per-call on real meshes)
    src, dst, eu, ev = [], [], [], []
    boundary_cnt = np.zeros(len(f), np.float32)
    for (u, v), flist in e2f.items():
        if len(flist) == 1:
            boundary_cnt[flist[0]] += 1
            continue
        for i in flist:
            for j in flist:
                if i != j:
                    src.append(i)
                    dst.append(j)
                    eu.append(u)
                    ev.append(v)
    if src:
        edges = torch.tensor(
            np.stack([src, dst], 1), dtype=torch.long, device=device
        )
        eu_t = torch.tensor(eu, dtype=torch.long, device=device)
        ev_t = torch.tensor(ev, dtype=torch.long, device=device)
        elen = (V[eu_t] - V[ev_t]).norm(dim=1) / global_scale.float()
        # winding-invariant dihedral: angle between UNORIENTED planes
        # (|cos| kills the normal-sign dependence on face winding; convex vs
        # concave is not recoverable for inconsistently wound wild meshes)
        cosd = (normals[edges[:, 0]] * normals[edges[:, 1]]).sum(1)
        dihedral = torch.acos(cosd.abs().clamp(max=1.0)).float()
        rel = torch.stack([elen, dihedral, torch.zeros_like(elen)], dim=1)
    else:
        edges = torch.zeros(0, 2, dtype=torch.long, device=device)
        rel = torch.zeros(0, 3, device=device)
    boundary = torch.tensor(boundary_cnt / 3.0, device=device)
    return FaceGraph(
        edges=edges,
        rel=rel,
        boundary=boundary,
        global_scale=global_scale.float(),
    )


def random_walk_pe(edges, num_faces, k=16, device="cpu", chunk=4096):
    """RWSE: diagonal of P^t for t=1..k, P = D^-1 A. [F,k].

    Computed in column blocks of P^t via sparse matmuls so memory stays
    O(F * chunk) — the dense [F,F] power chain OOMs on packed groups
    with >~16k faces (a handful of source GLBs triangulate far above
    the nominal face budget). Same quantity as the dense chain up to
    float association order.
    """
    if len(edges):
        idx = edges.t().long()
        A = torch.sparse_coo_tensor(
            idx,
            torch.ones(idx.shape[1], device=device),
            (num_faces, num_faces),
        ).coalesce()
        # duplicate index pairs must count once (dense used assignment)
        A = torch.sparse_coo_tensor(
            A.indices(), A.values().clamp(max=1.0), A.shape
        ).coalesce()
        deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1.0)[:, None]
    else:
        A = None
        deg = torch.ones(num_faces, 1, device=device)
    out = torch.zeros(num_faces, k, device=device)
    for s in range(0, num_faces, chunk):
        e = min(s + chunk, num_faces)
        C = torch.zeros(num_faces, e - s, device=device)
        C[
            torch.arange(s, e, device=device),
            torch.arange(e - s, device=device),
        ] = 1.0
        rows = torch.arange(s, e, device=device)
        cols = torch.arange(e - s, device=device)
        for t in range(k):
            C = (torch.sparse.mm(A, C) if A is not None else C * 0) / deg
            out[rows, t] = C[rows, cols]
    return out


class TopologyPE(nn.Module):
    """Computes (and caches nothing -- caller may cache) a [F,k] structural
    encoding from face adjacency."""

    def __init__(self, kind="random_walk", k=16):
        super().__init__()
        assert kind in ("none", "random_walk")
        self.kind = kind
        self.k = k

    @property
    def dim(self):
        return self.k

    @torch.no_grad()
    def forward(self, graph, num_faces):
        device = graph["boundary"].device
        if self.kind == "none":
            return torch.zeros(num_faces, self.k, device=device)
        return random_walk_pe(graph["edges"], num_faces, self.k, device)


# ==================================================================
# sparse topology transformer
# ==================================================================
"""Topology-aware face transformer: sparse graph attention along the shared-
edge face adjacency (self + topology neighbors only -- NO xyz/euclidean KNN).

QK attention with a learned relation bias from (shared edge length, dihedral
angle, boundary flag).
"""


def scatter_softmax(scores, index, num_nodes):
    """Softmax of `scores` grouped by `index` (destination node)."""
    m = torch.full(
        (num_nodes,), -1e30, device=scores.device, dtype=scores.dtype
    )
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
        self.rel_bias = nn.Sequential(
            nn.Linear(rel_dim + 1, 32),
            nn.GELU(),
            nn.Linear(32, heads, bias=False),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

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
        logits = (q[dst] * k[src]).sum(-1) / self.dk**0.5  # [E,h]
        logits = logits + self.rel_bias(rel_full)  # [E,h]
        out = torch.zeros(F, self.h, self.dk, device=x.device, dtype=x.dtype)
        for hh in range(self.h):
            w = scatter_softmax(logits[:, hh], dst, F)  # [E]
            out[:, hh] = torch.zeros(
                F, self.dk, device=x.device, dtype=x.dtype
            ).index_add(0, dst, w.unsqueeze(1) * v[src, hh])
        x = x + self.proj(out.reshape(F, -1))
        return x + self.mlp(self.norm2(x))


class TopologyTransformer(nn.Module):
    def __init__(self, dim=256, heads=8, depth=4):
        super().__init__()
        self.blocks = nn.ModuleList(
            [GraphAttnBlock(dim, heads) for _ in range(depth)]
        )

    def forward(self, x, graph):
        """x [F,D]; graph from build_face_graph."""
        F = x.shape[0]
        device = x.device
        loops = torch.arange(F, device=device)
        edges_full = torch.cat(
            [graph["edges"], torch.stack([loops, loops], dim=1)], dim=0
        )
        # rel: (edge_len, dihedral, boundary-of-src, is_self)
        rel_e = (
            torch.cat(
                [
                    graph["rel"][:, :2],
                    graph["boundary"][graph["edges"][:, 0]].unsqueeze(1)
                    if len(graph["edges"])
                    else torch.zeros(0, 1, device=device),
                    torch.zeros(len(graph["edges"]), 1, device=device),
                ],
                dim=1,
            )
            if len(graph["edges"])
            else torch.zeros(0, 4, device=device)
        )
        rel_s = torch.cat(
            [
                torch.zeros(F, 2, device=device),
                graph["boundary"].unsqueeze(1).float(),
                torch.ones(F, 1, device=device),
            ],
            dim=1,
        )
        rel_full = torch.cat([rel_e, rel_s], dim=0).float()
        for blk in self.blocks:
            x = blk(x, edges_full, rel_full)
        return x
