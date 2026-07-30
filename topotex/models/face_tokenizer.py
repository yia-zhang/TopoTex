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


def face_intrinsic_features(vertices, faces, global_scale=None):
    """[F, 10] intrinsic, rigid+scale invariant by construction:
    3 edge lengths / mesh_scale, 3 edge lengths / perimeter,
    3 corner angles / pi, log-normalized face area."""
    V, F = vertices, faces
    tri = V[F]  # [F,3,3]
    e = torch.stack(
        [tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 1], tri[:, 0] - tri[:, 2]],
        dim=1,
    )  # [F,3,3]
    L = e.norm(dim=2).clamp(min=1e-12)  # [F,3] (l01, l12, l20)
    n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    area = n.norm(dim=1) / 2
    if global_scale is None:
        global_scale = torch.sqrt(area.sum().clamp(min=1e-20))
    perim = L.sum(dim=1, keepdim=True)

    def angle(a, b):
        cos = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp(min=1e-12)
        return torch.acos(cos.clamp(-1, 1))

    ang = torch.stack(
        [
            angle(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
            angle(tri[:, 2] - tri[:, 1], tri[:, 0] - tri[:, 1]),
            angle(tri[:, 0] - tri[:, 2], tri[:, 1] - tri[:, 2]),
        ],
        dim=1,
    )  # [F,3], sums to pi
    # sort within-face so features are invariant to the corner/index order a
    # given exporter happened to use (same shape => same features)
    L_sorted, _ = torch.sort(L, dim=1, descending=True)
    ang_sorted, _ = torch.sort(ang, dim=1, descending=True)
    feat = torch.cat(
        [
            L_sorted / global_scale,
            L_sorted / perim,
            ang_sorted / torch.pi,
            torch.log(area.clamp(min=1e-20) / global_scale**2).unsqueeze(1)
            / 10.0,
        ],
        dim=1,
    )
    return feat.float()


class FaceTokenizer(nn.Module):
    """intrinsic geometry (+ boundary flag) ++ topology PE -> MLP -> [F,D]."""

    IN_INTRINSIC = 10 + 1  # +1 boundary fraction

    def __init__(self, dim=256, pe_dim=16, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(self.IN_INTRINSIC + pe_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, vertices, faces, graph, topo_pe):
        """vertices [V,3], faces [F,3], graph from build_face_graph,
        topo_pe [F,pe_dim]. Returns [F,D]."""
        intrinsic = face_intrinsic_features(
            vertices, faces, graph["global_scale"]
        )
        x = torch.cat(
            [intrinsic, graph["boundary"].unsqueeze(1).float(), topo_pe], dim=1
        )
        return self.mlp(x)
