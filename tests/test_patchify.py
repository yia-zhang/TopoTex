# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topotex.layers.embeddings import (  # noqa: E402
    patchify,
    sincos_2d_pos_embed,
    unpatchify,
)
from topotex.models.flow_matching import MiniDiT  # noqa: E402


def test_patchify_unpatchify_roundtrip():
    x = torch.randn(2, 3, 256, 256)
    t = patchify(x, 8)
    assert t.shape == (2, 1024, 8 * 8 * 3)
    y = unpatchify(t, 8, 3)
    assert torch.equal(x, y)


def test_patchify_spatial_layout():
    """Token k must correspond to patch (row k//g, col k%g)."""
    x = torch.zeros(1, 3, 256, 256)
    x[0, 0, 8:16, 16:24] = 1.0  # patch row 1, col 2
    t = patchify(x, 8)
    g = 32
    nz = (t.abs().sum(-1) > 0).nonzero()[:, 1]
    assert nz.tolist() == [1 * g + 2]


def test_pos_embed_shape_and_uniqueness():
    pe = sincos_2d_pos_embed(384, 32)
    assert pe.shape == (1024, 384)
    # all positions distinct
    d = torch.cdist(pe[:64], pe[:64])
    assert (d + torch.eye(64) * 1e9).min() > 1e-3


def test_model_output_shape_and_zero_init():
    """AdaLN-Zero: at init the model output must be exactly zero."""
    torch.manual_seed(0)
    m = MiniDiT(
        resolution=64, patch=8, hidden=96, depth=2, heads=4, cond_channels=64
    )
    x = torch.randn(2, 3, 64, 64)
    c = torch.randn(2, 64, 64, 64)
    mask = torch.ones(2, 1, 64, 64)
    t = torch.tensor([10, 500])
    out = m(x, c, mask, t)
    assert out.shape == (2, 3, 64, 64)
    assert out.abs().max() == 0.0
