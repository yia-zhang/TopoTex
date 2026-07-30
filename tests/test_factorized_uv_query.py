# -*- coding: utf-8 -*-
"""Factorized dense UV query encoder contracts.

Face ids are POINTERS into Z_F, not semantic embeddings — permuting the
face set while remapping ids must not change the condition. Barycentric
coordinates are the only within-face signal, so they must move the texel
feature and receive gradient. Background texels must never index Z_F.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topotex.config import SurfaceConditionerConfig, TopoTexConfig
from topotex.models.uv_query import UVQueryAttention


def _query(res, faces, seed=0, bg_border=8):
    """Random valid-interior query map with a background border."""
    g = torch.Generator().manual_seed(seed)
    fid = torch.randint(0, faces, (res, res), generator=g)
    fid[:bg_border] = -1
    fid[-bg_border:] = -1
    fid[:, :bg_border] = -1
    fid[:, -bg_border:] = -1
    bary = torch.rand(res, res, 3, generator=g)
    bary = bary / bary.sum(-1, keepdim=True)
    return fid, bary


def test_shapes_dim256_and_dim384():
    """Dq derives from config (D//4): dim256 -> 64, dim384 -> 96; query
    tokens [1024, D]; condition [64, 256, 256]."""
    for dim, dq in ((256, 64), (384, 96)):
        cfg = SurfaceConditionerConfig(cond_dim=dim)
        assert cfg.uv_texel_dim == dq
        cfg.validate()
        dec = UVQueryAttention(dim=dim, texel_dim=cfg.uv_texel_dim).eval()
        assert dec.face_proj.out_features == dq
        fid, bary = _query(256, 7)
        tokens = torch.randn(7, dim)
        seen = {}
        h = dec.norm.register_forward_hook(
            lambda m, i, o: seen.__setitem__("q", o.detach())
        )
        with torch.no_grad():
            cond, _ = dec(tokens, fid, bary)
        h.remove()
        assert seen["q"].shape == (1024, dim)
        assert cond.shape == (64, 256, 256)


def test_config_rejects_narrow_texel_dim():
    with pytest.raises(ValueError):
        SurfaceConditionerConfig(cond_dim=64).validate()  # Dq=16 < 32
    with pytest.raises(ValueError):
        TopoTexConfig.from_dict({"uv_query_encoder": "concat_bottleneck"})


def test_face_permutation_equivariance():
    """Z_F' = P Z_F with face_id' = P(face_id) leaves the condition
    unchanged: the face id is a pointer, not a semantic embedding."""
    torch.manual_seed(0)
    dec = UVQueryAttention(dim=256, res=64, patch=8).eval()
    fid, bary = _query(64, 12, bg_border=4)
    tokens = torch.randn(12, 256)
    perm = torch.randperm(12)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(12)
    fid_p = fid.clone()
    valid = fid >= 0
    fid_p[valid] = inv[fid[valid]]
    with torch.no_grad():
        c0, _ = dec(tokens, fid, bary)
        c1, _ = dec(tokens[perm], fid_p, bary)
    assert torch.allclose(c0, c1, atol=1e-5), (
        f"permutation broke equivariance: {(c0 - c1).abs().max():.3e}"
    )


def test_barycentric_sensitivity_and_gradient_flow():
    """Same face, different barycentric -> different texel feature; the
    loss must reach face projection, bary MLP, patch embedding and the
    global cross attention."""
    torch.manual_seed(0)
    dec = UVQueryAttention(dim=256, res=64, patch=8)
    fid, bary = _query(64, 5, bg_border=4)
    tokens = torch.randn(5, 256)
    seen = {}
    h = dec.texel_norm.register_forward_hook(
        lambda m, i, o: seen.__setitem__("t", o.detach())
    )
    with torch.no_grad():
        dec(tokens, fid, bary)
        t0 = seen["t"]
        dec(tokens, fid, bary[:, :, [1, 2, 0]])
        t1 = seen["t"]
    h.remove()
    assert (t0 - t1).abs().max() > 1e-4, "bary change did not move texels"

    cond, rgb = dec(tokens, fid, bary, with_rgb=True)
    (cond.square().mean() + rgb.abs().mean()).backward()
    for name in ("face_proj", "bary_mlp", "patch_embed", "blocks"):
        mod = getattr(dec, name)
        bad = [
            n
            for n, p in mod.named_parameters()
            if p.grad is None
            or not torch.isfinite(p.grad).all()
            or p.grad.abs().sum() == 0
        ]
        assert not bad, f"{name}: no finite nonzero grad for {bad}"


def test_background_never_indexes_face_tokens():
    """Background texel features equal the learned bg embedding and do not
    move when Z_F changes; masked regions are exactly zero; no NaN/Inf."""
    torch.manual_seed(0)
    dec = UVQueryAttention(dim=256, res=64, patch=8).eval()
    fid, bary = _query(64, 5, bg_border=4)
    bg_flat = (fid < 0).reshape(-1)
    dense = {}
    h = dec.patch_embed.register_forward_pre_hook(
        lambda m, i: dense.__setitem__(
            "d", i[0].detach()[0].permute(1, 2, 0).reshape(-1, dec.texel_dim)
        )
    )
    with torch.no_grad():
        c0, _ = dec(torch.randn(5, 256), fid, bary)
        d0 = dense["d"][bg_flat]
        c1, _ = dec(torch.randn(5, 256) * 7, fid, bary)
        d1 = dense["d"][bg_flat]
    h.remove()
    assert torch.equal(d0, d1), "background texels depend on Z_F"
    assert torch.equal(d0, dec.bg_texel.expand_as(d0)), (
        "background texels != learned bg embedding"
    )
    assert c0[:, fid < 0].abs().max() == 0.0
    assert torch.isfinite(c0).all() and torch.isfinite(c1).all()

    # all-background query (empty gather path)
    fid_bg = torch.full((64, 64), -1, dtype=torch.long)
    with torch.no_grad():
        cb, _ = dec(torch.randn(5, 256), fid_bg, bary)
    assert cb.abs().max() == 0.0 and torch.isfinite(cb).all()


def test_partial_query_mask_exact_zero():
    """A partial-region query (valid subset) zeroes everything outside."""
    torch.manual_seed(0)
    dec = UVQueryAttention(dim=256, res=64, patch=8).eval()
    fid, bary = _query(64, 5, bg_border=4)
    fid[:, 32:] = -1  # keep only the left half valid
    with torch.no_grad():
        cond, _ = dec(torch.randn(5, 256), fid, bary)
    assert cond[:, fid < 0].abs().max() == 0.0


def test_same_query_determinism():
    """Fixed weights + inputs -> identical output on repeat forward (CPU
    exact; CUDA within the documented nondeterminism envelope)."""
    torch.manual_seed(0)
    dec = UVQueryAttention(dim=256, res=64, patch=8).eval()
    fid, bary = _query(64, 5, bg_border=4)
    tokens = torch.randn(5, 256)
    with torch.no_grad():
        a, _ = dec(tokens, fid, bary)
        b, _ = dec(tokens, fid, bary)
    assert torch.equal(a, b)
    if torch.cuda.is_available():
        dec = dec.to("cuda:0")
        fid, bary, tokens = (
            fid.to("cuda:0"),
            bary.to("cuda:0"),
            tokens.to("cuda:0"),
        )
        with torch.no_grad():
            a, _ = dec(tokens, fid, bary)
            b, _ = dec(tokens, fid, bary)
        assert (a - b).abs().max() <= 3e-5
