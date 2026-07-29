# -*- coding: utf-8 -*-
"""Toy experiments as tests.

Test 1 (coincident faces): two triangles with IDENTICAL XYZ but different
topology context (face A belongs to a 3-face strip, face B is isolated).
An XYZ-query MLP provably cannot separate their colors; face tokens can.

Test 2 (rigid transform): face tokens identical under translation/rotation/
scale of the same mesh.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.surface_conditioner import UVQueryAttention
from models.surface_conditioner import (FaceTokenizer, TopologyPE,
                                        TopologyTransformer, build_face_graph,
                                        face_intrinsic_features)


def _encoder(seed=1):
    """Geometry pathway of the Surface Conditioner (tokenizer + topology
    PE + topology transformer) — the modules that define Z_F invariances."""
    torch.manual_seed(seed)
    return (TopologyPE("random_walk", 16), FaceTokenizer(dim=256, pe_dim=16),
            TopologyTransformer(256, 8, 4))


def _tokens_of(enc, V, F):
    pe_mod, tok, topo = enc
    g = build_face_graph(V, F)
    with torch.no_grad():
        pe = pe_mod(g, len(F))
        x = tok(V, F, g, pe)
        return topo(x, g)



def coincident_fixture():
    """Face 0 (in strip with faces 1,2) and face 3 (isolated) share EXACTLY
    the same vertex positions; separate vertices, separate components."""
    tri = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 1.0, 0.0]]
    V = torch.tensor(tri + [[1.3, 1.1, 0.0], [2.1, 0.2, 0.0]]   # strip extras
                     + tri, dtype=torch.float32)                # coincident copy
    F = torch.tensor([[0, 1, 2],      # face A (part of strip)
                      [1, 3, 2],      # strip
                      [1, 4, 3],      # strip
                      [5, 6, 7]])     # face B == face A in XYZ, isolated
    assert torch.allclose(V[F[0]], V[F[3]])
    return V, F


def _uv_for_two_faces():
    """Two separated UV islands querying face 0 and face 3, with IDENTICAL
    barycentric patterns: every texel of island B shares its exact xyz with a
    texel of island A, making xyz->color separation provably ill-posed."""
    fid = torch.full((64, 64), -1, dtype=torch.long)
    fid[8:28, 8:28] = 0
    fid[36:56, 8:28] = 3
    torch.manual_seed(11)
    bary = torch.rand(64, 64, 3)
    bary = bary / bary.sum(-1, keepdim=True)
    bary[36:56, 8:28] = bary[8:28, 8:28]     # mirror island A's pattern
    return {"face_id": fid, "barycentric": bary}


def test_coincident_face_separation_tokens():
    """Face tokens for the coincident pair must differ (topology context)."""
    enc = _encoder(seed=0)
    V, F = coincident_fixture()
    t = _tokens_of(enc, V, F)
    d_coincident = (t[0] - t[3]).norm()
    assert d_coincident > 1e-3, "coincident faces collapsed to same token"


def test_coincident_face_rgb_fit_vs_xyz_baseline():
    """Optimize the RGB head: face A -> red, face B -> blue. Face-token path
    must fit both; an XYZ-input MLP cannot (same xyz => same output)."""
    torch.manual_seed(0)
    V, F = coincident_fixture()
    uv = _uv_for_two_faces()
    target = torch.zeros(3, 64, 64)
    target[:, uv["face_id"] == 0] = torch.tensor([1.0, 0, 0]).view(3, 1)
    target[:, uv["face_id"] == 3] = torch.tensor([0, 0, 1.0]).view(3, 1)
    valid = uv["face_id"] >= 0

    pe_mod, tok, topo = _encoder(seed=0)
    dec = UVQueryAttention(dim=256, out_channels=64, patch=8, res=64)
    g = build_face_graph(V, F)
    params = (list(tok.parameters()) + list(topo.parameters())
              + list(dec.parameters()))
    opt = torch.optim.Adam(params, lr=1e-3)
    for _ in range(150):
        pe = pe_mod(g, len(F))
        tokens = topo(tok(V, F, g, pe), g)
        _, rgb = dec(tokens, uv["face_id"], uv["barycentric"], with_rgb=True)
        loss = (rgb - target).abs()[:, valid].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss < 0.05, f"face-token path failed to fit red/blue: {float(loss)}"

    # xyz baseline: same-capacity MLP on texel xyz. With the mirrored bary
    # pattern every xyz appears TWICE with red and blue targets, so ANY
    # function of xyz has L1 >= |red-blue|_1 / (2*3) = 1/3 -- provably stuck.
    xyz = (V[F[uv["face_id"][valid]].long()]
           * uv["barycentric"][valid].unsqueeze(-1)).sum(1)
    mlp = torch.nn.Sequential(torch.nn.Linear(3, 256), torch.nn.GELU(),
                              torch.nn.Linear(256, 256), torch.nn.GELU(),
                              torch.nn.Linear(256, 3))
    opt2 = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    tv = target.permute(1, 2, 0)[valid]
    for _ in range(300):
        l2 = (mlp(xyz) - tv).abs().mean()
        opt2.zero_grad()
        l2.backward()
        opt2.step()
    assert l2 > 0.25, f"xyz baseline beat its provable 1/3 floor?!: {float(l2)}"


def test_rigid_transform_token_consistency():
    enc = _encoder(seed=0)
    V, F = coincident_fixture()
    t0 = _tokens_of(enc, V, F)
    ang = torch.tensor(1.1)
    R = torch.tensor([[1, 0, 0],
                      [0, torch.cos(ang), -torch.sin(ang)],
                      [0, torch.sin(ang), torch.cos(ang)]])
    Vt = 0.31 * (V @ R.T) + torch.tensor([-2.0, 7.0, 0.5])
    t1 = _tokens_of(enc, Vt, F)
    assert torch.allclose(t0, t1, atol=1e-3), (t0 - t1).abs().max()
