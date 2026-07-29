# -*- coding: utf-8 -*-
"""Global UV Query Attention decoder: reconstruction from face tokens +
barycentric position, within-face variation, background masking, and
end-to-end gradient flow through the full conditioner."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.surface_conditioner import SurfaceConditioner, UVQueryAttention


def _uv_48():
    """Two rectangular UV islands querying faces 0 and 1 at 48x48."""
    fid = torch.full((48, 48), -1, dtype=torch.long)
    fid[4:44, 4:22] = 0
    fid[4:44, 26:44] = 1
    torch.manual_seed(3)
    bary = torch.rand(48, 48, 3)
    bary = bary / bary.sum(-1, keepdim=True)
    return fid, bary


def test_uv_query_reconstruction():
    """Fit a bary-dependent target: color varies WITHIN each face, so the
    decoder must use both the gathered face token and the barycentric
    position of every texel."""
    torch.manual_seed(0)
    fid, bary = _uv_48()
    valid = fid >= 0
    target = torch.zeros(48, 48, 3)
    target[fid == 0] = torch.stack(
        [bary[fid == 0][:, 0], bary[fid == 0][:, 1],
         torch.zeros_like(bary[fid == 0][:, 0])], dim=1)
    target[fid == 1] = torch.stack(
        [torch.zeros_like(bary[fid == 1][:, 0]),
         bary[fid == 1][:, 1], bary[fid == 1][:, 2]], dim=1)
    target = target.permute(2, 0, 1)

    dec = UVQueryAttention(dim=256, out_channels=64, patch=8, res=48)
    tokens = torch.nn.Parameter(0.02 * torch.randn(2, 256))
    opt = torch.optim.Adam(list(dec.parameters()) + [tokens], lr=2e-3)
    for _ in range(400):
        _, rgb = dec(tokens, fid, bary, with_rgb=True)
        loss = (rgb - target).abs()[:, valid].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss < 0.05, f"UV query reconstruction failed: {float(loss)}"


def test_within_face_variation():
    """Same face ids, different barycentric maps -> different condition."""
    torch.manual_seed(0)
    fid, bary = _uv_48()
    dec = UVQueryAttention(dim=256, out_channels=64, patch=8, res=48).eval()
    tokens = torch.randn(2, 256)
    bary2 = bary[:, :, [2, 0, 1]]                # permuted bary pattern
    with torch.no_grad():
        c0, _ = dec(tokens, fid, bary)
        c1, _ = dec(tokens, fid, bary2)
    assert (c0 - c1).abs().max() > 1e-4


def test_background_zeroed_and_shape():
    torch.manual_seed(0)
    fid, bary = _uv_48()
    dec = UVQueryAttention(dim=256, out_channels=64, patch=8, res=48).eval()
    tokens = torch.randn(2, 256)
    with torch.no_grad():
        cond, _ = dec(tokens, fid, bary)
    assert cond.shape == (64, 48, 48)
    bg = fid < 0
    assert cond[:, bg].abs().max() == 0.0
    assert torch.isfinite(cond).all()


def test_gradient_flow_all_modules():
    """Every parameter of the full conditioner receives a finite gradient."""
    torch.manual_seed(0)
    model = SurfaceConditioner(image_size=64, resolution=48)
    V = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2.1, 1.7, 0.6]],
                     dtype=torch.float32)
    F = torch.tensor([[0, 1, 2], [1, 3, 2]])
    fid, bary = _uv_48()
    imgs = torch.rand(1, 6, 3, 64, 64)
    out = model({"vertices": V, "faces": F}, imgs,
                {"face_id": fid, "barycentric": bary}, with_rgb=True)
    loss = out["uv_rgb"].abs().mean() + out["uv_condition"].square().mean()
    loss.backward()
    missing = [n for n, p in model.named_parameters()
               if p.grad is None or not torch.isfinite(p.grad).all()
               or p.grad.abs().sum() == 0]
    assert not missing, f"params without finite nonzero grad: {missing}"
