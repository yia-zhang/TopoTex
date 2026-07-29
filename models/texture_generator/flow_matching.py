# -*- coding: utf-8 -*-
"""Masked rectified flow — drop-in alternative texture generator schedule.

Same contract as MaskedDiffusion so the backbone, trainer, and samplers are
untouched:
  - RGB in [-1, 1]; invalid texels FIXED at 0 for every t.
  - integer t in 1..T maps to flow time tau = t/T (tau=1 is pure noise, so
    the recipe's high-t emphasis carries over unchanged).
  - the DiT backbone serves as the velocity network; its timestep embedder
    receives the same 1..T scale it was designed for.

    x_tau = (1 - tau) * x0 + tau * eps        (linear interpolation path)
    v_target = eps - x0                        (constant along the path)
    sampling: Euler ODE from tau=1 to 0, x <- x - dtau * v_hat
"""
import torch


class MaskedFlowMatching:
    def __init__(self, T=1000, device="cuda:0"):
        self.T = T
        self.device = device

    def loss(self, model, x0, cond, mask, t=None, generator=None):
        """Masked velocity MSE. x0 invalid region must already be 0."""
        B = x0.shape[0]
        if t is None:
            t = torch.randint(1, self.T + 1, (B,), device=x0.device,
                              generator=generator)
        tau = (t.float() / self.T).view(-1, 1, 1, 1)
        eps = torch.randn(x0.shape, device=x0.device, generator=generator,
                          dtype=x0.dtype)
        xt = (1 - tau) * x0 + tau * eps
        xt = mask * xt + (1 - mask) * x0
        v_hat = model(xt, cond, mask, t)
        v = eps - x0
        se = (v_hat - v) ** 2 * mask
        return se.sum() / (mask.sum() * x0.shape[1]).clamp(min=1)

    @torch.no_grad()
    def sample(self, model, cond, mask, steps=50, generator=None,
               x_init=None):
        """Euler ODE integration tau: 1 -> 0. Returns x0 estimate in [-1,1];
        invalid region stays 0 throughout."""
        B, _, H, W = mask.shape
        if x_init is not None:
            x = x_init
        else:
            x = torch.randn(B, 3, H, W, device=mask.device,
                            generator=generator)
        x = x * mask
        taus = torch.linspace(1.0, 0.0, steps + 1, device=mask.device)
        for i in range(steps):
            t_int = (taus[i] * self.T).round().clamp(min=1).long()
            v = model(x, cond, mask, t_int.expand(B))
            x = (x - (taus[i] - taus[i + 1]) * v) * mask
        return x.clamp(-1, 1)

    # call-site compatibility with MaskedDiffusion
    ddim_sample = sample
