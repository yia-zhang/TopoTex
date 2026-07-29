# -*- coding: utf-8 -*-
"""Face tokenizer: mesh -> permutation-equivariant face tokens.

Per-face INTRINSIC geometry only (no world xyz, no absolute normals, no face
index, no UV): edge lengths (mesh-scale-normalized -> rigid+scale invariant),
per-face-normalized edge lengths, corner angles, normalized face area.
Topology positional encoding is added by TopologyPE (pluggable).

face_token = MLP(intrinsic ++ topology_pe)
"""
import torch
import torch.nn as nn


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
        edges = torch.tensor(np.stack([src, dst], 1), dtype=torch.long,
                             device=device)
        eu_t = torch.tensor(eu, dtype=torch.long, device=device)
        ev_t = torch.tensor(ev, dtype=torch.long, device=device)
        elen = (V[eu_t] - V[ev_t]).norm(dim=1) / global_scale.float()
        # winding-invariant dihedral: angle between UNORIENTED planes
        # (|cos| kills the normal-sign dependence on face winding; convex vs
        # concave is not recoverable for inconsistently wound wild meshes)
        cosd = (normals[edges[:, 0]] * normals[edges[:, 1]]).sum(1)
        dihedral = torch.acos(cosd.abs().clamp(max=1.0)).float()
        rel = torch.stack([elen, dihedral,
                           torch.zeros_like(elen)], dim=1)
    else:
        edges = torch.zeros(0, 2, dtype=torch.long, device=device)
        rel = torch.zeros(0, 3, device=device)
    boundary = torch.tensor(boundary_cnt / 3.0, device=device)
    return {"edges": edges, "rel": rel, "boundary": boundary,
            "global_scale": global_scale.float()}


def face_intrinsic_features(vertices, faces, global_scale=None):
    """[F, 10] intrinsic, rigid+scale invariant by construction:
    3 edge lengths / mesh_scale, 3 edge lengths / perimeter,
    3 corner angles / pi, log-normalized face area."""
    V, F = vertices, faces
    tri = V[F]                                    # [F,3,3]
    e = torch.stack([tri[:, 1] - tri[:, 0],
                     tri[:, 2] - tri[:, 1],
                     tri[:, 0] - tri[:, 2]], dim=1)   # [F,3,3]
    L = e.norm(dim=2).clamp(min=1e-12)            # [F,3] (l01, l12, l20)
    n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    area = n.norm(dim=1) / 2
    if global_scale is None:
        global_scale = torch.sqrt(area.sum().clamp(min=1e-20))
    perim = L.sum(dim=1, keepdim=True)

    def angle(a, b):
        cos = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp(min=1e-12)
        return torch.acos(cos.clamp(-1, 1))
    ang = torch.stack([angle(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                       angle(tri[:, 2] - tri[:, 1], tri[:, 0] - tri[:, 1]),
                       angle(tri[:, 0] - tri[:, 2], tri[:, 1] - tri[:, 2])],
                      dim=1)                      # [F,3], sums to pi
    # sort within-face so features are invariant to the corner/index order a
    # given exporter happened to use (same shape => same features)
    L_sorted, _ = torch.sort(L, dim=1, descending=True)
    ang_sorted, _ = torch.sort(ang, dim=1, descending=True)
    feat = torch.cat([
        L_sorted / global_scale,
        L_sorted / perim,
        ang_sorted / torch.pi,
        torch.log(area.clamp(min=1e-20) / global_scale ** 2).unsqueeze(1) / 10.0,
    ], dim=1)
    return feat.float()


class FaceTokenizer(nn.Module):
    """intrinsic geometry (+ boundary flag) ++ topology PE -> MLP -> [F,D]."""

    IN_INTRINSIC = 10 + 1  # +1 boundary fraction

    def __init__(self, dim=256, pe_dim=16, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(self.IN_INTRINSIC + pe_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dim))

    def forward(self, vertices, faces, graph, topo_pe):
        """vertices [V,3], faces [F,3], graph from build_face_graph,
        topo_pe [F,pe_dim]. Returns [F,D]."""
        intrinsic = face_intrinsic_features(vertices, faces,
                                            graph["global_scale"])
        x = torch.cat([intrinsic, graph["boundary"].unsqueeze(1).float(),
                       topo_pe], dim=1)
        return self.mlp(x)
