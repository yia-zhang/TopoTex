# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.texture_generator.flow_matching import MaskedFlowMatching


def _mask():
    m = torch.zeros(2, 1, 32, 32)
    m[:, :, 8:24, 8:24] = 1.0
    return m


def test_interpolant_keeps_invalid_fixed():
    """The invalid region must stay 0 at every flow time — the generator is
    defined on the valid-mask interior only."""
    fm = MaskedFlowMatching(T=1000, device="cpu")
    m = _mask()
    x0 = torch.randn(2, 3, 32, 32) * m            # invalid region = 0
    seen = []

    class Probe(torch.nn.Module):
        def forward(self, xt, cond, mask, t):
            seen.append(xt)
            return torch.zeros_like(xt)

    for t in (1, 500, 1000):
        fm.loss(Probe(), x0, None, m, t=torch.tensor([t, t]),
                generator=torch.Generator().manual_seed(0))
    for xt in seen:
        assert (xt * (1 - m)).abs().max() == 0.0


def test_loss_ignores_invalid_region():
    """Perturbing the model's prediction OUTSIDE the mask must not change
    the loss."""
    fm = MaskedFlowMatching(T=1000, device="cpu")
    m = _mask()
    x0 = torch.randn(2, 3, 32, 32) * m

    class Base(torch.nn.Module):
        def __init__(self, delta):
            super().__init__()
            self.delta = delta
        def forward(self, xt, cond, mask, t):
            return xt * 0.1 + self.delta * (1 - mask)

    g = torch.Generator().manual_seed(0)
    l0 = fm.loss(Base(0.0), x0, None, m, t=torch.tensor([300, 700]),
                 generator=g)
    g = torch.Generator().manual_seed(0)
    l1 = fm.loss(Base(123.0), x0, None, m, t=torch.tensor([300, 700]),
                 generator=g)
    assert torch.allclose(l0, l1)


def test_euler_keeps_invalid_fixed_and_range():
    fm = MaskedFlowMatching(T=1000, device="cpu")
    m = _mask()

    class Zero(torch.nn.Module):
        def forward(self, xt, cond, mask, t):
            return torch.zeros_like(xt)

    x = fm.ddim_sample(Zero(), None, m, steps=10,
                       generator=torch.Generator().manual_seed(0))
    assert (x * (1 - m)).abs().max() == 0.0
    assert x.min() >= -1.0 - 1e-5 and x.max() <= 1.0 + 1e-5


def test_euler_recovers_memorized_x0():
    """A perfect velocity oracle must transport noise back to x0 — the
    rectified-flow path is straight, so Euler integration is near-exact."""
    fm = MaskedFlowMatching(T=1000, device="cpu")
    m = _mask()
    x0 = (torch.rand(1, 3, 32, 32) * 2 - 1) * m

    class Oracle(torch.nn.Module):
        def forward(self, xt, cond, mask, t):
            tau = (t.float() / fm.T).view(-1, 1, 1, 1).clamp(min=1e-4)
            return (xt - x0) / tau            # v = eps - x0 on the path

    x = fm.ddim_sample(Oracle(), None, m, steps=50,
                       generator=torch.Generator().manual_seed(1))
    assert (x - x0).abs()[m.bool().expand_as(x)].max() < 0.05
