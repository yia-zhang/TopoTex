# -*- coding: utf-8 -*-
"""Dataset schema constants + typed batch contracts.

The on-disk layout (``topotex_dataset@1``) is FROZEN — nothing here changes
serialized files. The dataclasses type the tensors that used to travel as
plain dicts; they keep ``obj["key"]`` / ``obj.get`` compatibility so every
call site reads identically while public APIs gain real types.

Shapes (F faces, V vertices, Vuv UV vertices, E directed adjacency edges,
H=W=256 texel grid, Nv=6 views):
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np
import torch

# ------------------------------------------------------ schema constants
DATASET_SCHEMA = "topotex_dataset@1"
SOURCE_SCHEMA = "topotex_source"
RES = 256
QUERIES = ("uv_000", "uv_001", "uv_002", "uv_test")
SOURCE_FILES = (
    "reference.png",
    "mv.safetensors",
    "mesh.safetensors",
    "uv_address.safetensors",
    "gt_texture.png",
    "meta.json",
)


class _Record:
    """dict-compat mixin: subscription, mutation and .get by field name."""

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value) -> None:
        setattr(self, key, value)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) and getattr(self, key) is not None

    def keys(self):
        return [f.name for f in fields(self)]

    def items(self):
        return [(f.name, getattr(self, f.name)) for f in fields(self)]


@dataclass
class MeshBatch(_Record):
    """One triangle mesh.

    vertices: float32 [V, 3] (world units — Z_F is rigid/scale invariant)
    faces:    int64   [F, 3]
    """

    vertices: torch.Tensor
    faces: torch.Tensor


@dataclass
class FaceGraph(_Record):
    """Shared-edge face adjacency built by
    :func:`topotex.layers.topology.build_face_graph`.

    edges:        int64   [E, 2] directed face pairs (both directions)
    rel:          float32 [E, 3] (edge_len/global_scale, dihedral, 0)
    boundary:     float32 [F] fraction of boundary edges per face
    global_scale: float32 scalar — sqrt(total mesh area); a cached graph is
                  only valid for the vertices it was built from
    """

    edges: torch.Tensor
    rel: torch.Tensor
    boundary: torch.Tensor
    global_scale: torch.Tensor


@dataclass
class UVQueryBatch(_Record):
    """One UV parameterization as a (face_id, barycentric) address map.

    face_id:     int64   [H, W]  (-1 = background)
    barycentric: float32 [3, H, W]
    valid_mask:  bool    [H, W]
    gt_texture:  float32 [3, H, W] in [0,1] (None at pure inference)
    uv_vertices: float32 np [Vuv, 2] top-down v (full layout — rebake-safe)
    uv_faces:    int64   np [F, 3]
    graph:       optional FaceGraph override for encode_faces
    """

    name: str
    face_id: torch.Tensor
    barycentric: torch.Tensor
    valid_mask: torch.Tensor
    gt_texture: Optional[torch.Tensor] = None
    uv_vertices: Optional[np.ndarray] = None
    uv_faces: Optional[np.ndarray] = None
    graph: Optional[FaceGraph] = None


@dataclass
class TopoTexBatch(_Record):
    """One dataset item: a mesh with its conditioning views and UV queries.

    reference_image: uint8 [3, 512, 512]
    mv_images:       uint8 [Nv, 3, image_size, image_size]
    uv_queries:      TRAIN queries (canonical / alternative / partial)
    test_uv_queries: held-out family — EVALUATION ONLY, never trained
    """

    sample_id: str
    mesh: MeshBatch
    reference_image: Optional[torch.Tensor] = None
    mv_images: Optional[torch.Tensor] = None
    graph: Optional[FaceGraph] = None
    uv_queries: list[UVQueryBatch] = field(default_factory=list)
    test_uv_queries: list[UVQueryBatch] = field(default_factory=list)


@dataclass
class ConditionerOutput(_Record):
    """Surface-conditioner forward output.

    face_tokens:  float32 [F, D] — the Face Set Latent Z_F
    uv_condition: float32 [1, C, H, W] dense per-texel condition (0 on bg)
    uv_rgb:       float32 [1, 3, H, W] aux RGB head (training only)
    """

    face_tokens: torch.Tensor
    uv_condition: torch.Tensor
    uv_rgb: Optional[torch.Tensor] = None


@dataclass
class TopoTexOutput(_Record):
    """Inference-pipeline output.

    texture:      uint8 np [H, W, 3] — generated texture, 0 outside the
                  query's valid region
    uv_condition: float32 [1, C, H, W]
    face_tokens:  float32 [F, D] Z_F used for this generation
    """

    texture: np.ndarray
    uv_condition: torch.Tensor
    face_tokens: torch.Tensor


__all__ = [
    "DATASET_SCHEMA",
    "SOURCE_SCHEMA",
    "RES",
    "QUERIES",
    "SOURCE_FILES",
    "MeshBatch",
    "FaceGraph",
    "UVQueryBatch",
    "TopoTexBatch",
    "ConditionerOutput",
    "TopoTexOutput",
]
