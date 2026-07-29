# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.texture_generator.diffusion import MaskedDiffusion, cosine_alphas_cumprod


def _mask():
    m = torch.zeros(2, 1, 32, 32)
    m[:, :, 8:24, 8:24] = 1.0
    return m


def test_schedule_monotone():
    ab = cosine_alphas_cumprod(1000)
    assert ab[0] > 0.99 and ab[-1] < 1e-3
    assert (ab[1:] <= ab[:-1] + 1e-12).all()


def test_q_sample_keeps_invalid_fixed():
    diff = MaskedDiffusion(T=1000, device="cpu")
    m = _mask()
    x0 = torch.randn(2, 3, 32, 32) * m            # invalid region = 0
    for t in (1, 500, 1000):
        xt, _ = diff.q_sample(x0, torch.tensor([t, t]), m)
        assert torch.equal(xt * (1 - m), x0 * (1 - m))
        assert (xt * (1 - m)).abs().max() == 0.0


def test_loss_ignores_invalid_region():
    """Perturbing the model's prediction OUTSIDE the mask must not change
    the loss."""
    diff = MaskedDiffusion(T=1000, device="cpu")
    m = _mask()
    x0 = torch.randn(2, 3, 32, 32) * m

    class Base(torch.nn.Module):
        def __init__(self, delta):
            super().__init__()
            self.delta = delta
        def forward(self, xt, cond, mask, t):
            return xt * 0.1 + self.delta * (1 - mask)

    g = torch.Generator().manual_seed(0)
    l0 = diff.loss(Base(0.0), x0, None, m, t=torch.tensor([300, 700]),
                   generator=g)
    g = torch.Generator().manual_seed(0)
    l1 = diff.loss(Base(123.0), x0, None, m, t=torch.tensor([300, 700]),
                   generator=g)
    assert torch.allclose(l0, l1)


def test_ddim_keeps_invalid_fixed_and_range():
    diff = MaskedDiffusion(T=1000, device="cpu")
    m = _mask()

    class Zero(torch.nn.Module):
        def forward(self, xt, cond, mask, t):
            return torch.zeros_like(xt)

    x = diff.ddim_sample(Zero(), None, m, steps=10,
                         generator=torch.Generator().manual_seed(0))
    assert (x * (1 - m)).abs().max() == 0.0
    assert x.min() >= -1.0 - 1e-5 and x.max() <= 1.0 + 1e-5


def test_ddim_recovers_memorized_x0():
    """A perfect eps-oracle must reconstruct x0 through DDIM."""
    diff = MaskedDiffusion(T=1000, device="cpu")
    m = _mask()
    x0 = (torch.rand(1, 3, 32, 32) * 2 - 1) * m
    ab = diff.alphas_cumprod.float()

    class Oracle(torch.nn.Module):
        def forward(self, xt, cond, mask, t):
            a = ab[t.long() - 1].view(-1, 1, 1, 1)
            return (xt - a.sqrt() * x0) / (1 - a).sqrt().clamp(min=1e-8)

    x = diff.ddim_sample(Oracle(), None, m, steps=50,
                         generator=torch.Generator().manual_seed(1))
    assert (x - x0).abs()[m.bool().expand_as(x)].max() < 0.05
