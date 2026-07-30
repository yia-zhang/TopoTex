# -*- coding: utf-8 -*-
"""Face tokenizer: permutation equivariance + rigid/scale invariance."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topotex.layers.topology import (
    TopologyPE,
    TopologyTransformer,
    build_face_graph,
)
from topotex.models.face_tokenizer import (
    FaceTokenizer,
    face_intrinsic_features,
)


def _encoder(seed=1):
    """Geometry pathway of the Surface Conditioner (tokenizer + topology
    PE + topology transformer) — the modules that define Z_F invariances."""
    torch.manual_seed(seed)
    return (
        TopologyPE("random_walk", 16),
        FaceTokenizer(dim=256, pe_dim=16),
        TopologyTransformer(256, 8, 4),
    )


def _tokens_of(enc, V, F):
    pe_mod, tok, topo = enc
    g = build_face_graph(V, F)
    with torch.no_grad():
        pe = pe_mod(g, len(F))
        x = tok(V, F, g, pe)
        return topo(x, g)


def _strip_mesh():
    """A 6-face triangle strip (no symmetric automorphism thanks to varied
    geometry)."""
    torch.manual_seed(0)
    V = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0.4, 1.1, 0],
            [1.7, 0.9, 0.2],
            [2.5, 0.1, 0],
            [3.1, 1.3, 0.4],
            [3.9, 0.4, 0.1],
            [4.6, 1.2, 0],
        ],
        dtype=torch.float32,
    )
    F = torch.tensor(
        [[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 3], [4, 6, 5], [6, 7, 5]]
    )
    return V, F


def test_face_permutation_equivariance():
    enc = _encoder()
    V, F = _strip_mesh()
    t0 = _tokens_of(enc, V, F)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    t1 = _tokens_of(enc, V, F[perm])
    assert torch.allclose(t1, t0[perm], atol=1e-4), (t1 - t0[perm]).abs().max()


def test_rigid_and_scale_invariance():
    enc = _encoder()
    V, F = _strip_mesh()
    t0 = _tokens_of(enc, V, F)
    # rotation + translation + uniform scale
    ang = torch.tensor(0.7)
    R = torch.tensor(
        [
            [torch.cos(ang), -torch.sin(ang), 0],
            [torch.sin(ang), torch.cos(ang), 0],
            [0, 0, 1.0],
        ]
    )
    Vt = 2.37 * (V @ R.T) + torch.tensor([5.0, -3.0, 11.0])
    t1 = _tokens_of(enc, Vt, F)
    assert torch.allclose(t1, t0, atol=1e-3), (t1 - t0).abs().max()


def test_intrinsic_features_no_absolute_position():
    """Intrinsic features must be identical for two coincident-in-XYZ copies
    and for translated copies (they never see world coordinates)."""
    V, F = _strip_mesh()
    g = build_face_graph(V, F)
    f0 = face_intrinsic_features(V, F, g["global_scale"])
    f1 = face_intrinsic_features(V + 100.0, F, g["global_scale"])
    assert torch.allclose(f0, f1, atol=1e-4)


def test_corner_order_invariance():
    """Cyclic rotation of a face's index triple (same shape, different export
    convention) must not change its token."""
    enc = _encoder()
    V, F = _strip_mesh()
    t0 = _tokens_of(enc, V, F)
    F2 = F.clone()
    F2[2] = F[2][torch.tensor([1, 2, 0])]  # cyclic rotation
    t1 = _tokens_of(enc, V, F2)
    assert torch.allclose(t0, t1, atol=1e-4), (t0 - t1).abs().max()


def test_winding_flip_invariance():
    """Flipping one face's winding (inconsistently oriented wild mesh) must
    not change tokens: dihedral is unoriented, intrinsics are sorted."""
    enc = _encoder()
    V, F = _strip_mesh()
    t0 = _tokens_of(enc, V, F)
    F2 = F.clone()
    F2[3] = F[3][torch.tensor([0, 2, 1])]  # reversal = winding flip
    t1 = _tokens_of(enc, V, F2)
    assert torch.allclose(t0, t1, atol=1e-4), (t0 - t1).abs().max()


def test_topology_pe_variants():
    from topotex.layers.topology import TopologyPE

    V, F = _strip_mesh()
    g = build_face_graph(V, F)
    for kind in ("none", "random_walk"):
        pe = TopologyPE(kind, 16)(g, len(F))
        assert pe.shape == (len(F), 16)
        assert torch.isfinite(pe).all()
    assert TopologyPE("none", 16)(g, len(F)).abs().max() == 0
