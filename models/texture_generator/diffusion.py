# -*- coding: utf-8 -*-
"""Cosine noise schedule + masked q_sample + masked DDIM sampler.

Contract (Step 1 spec):
  - RGB lives in [-1, 1]; invalid texels are FIXED at 0 for every t
    (never noised, never denoised, excluded from the loss).
  - epsilon prediction, T=1000 training timesteps, DDIM sampling.
"""
import math

import numpy as np
import torch


def cosine_alphas_cumprod(T=1000, s=0.008):
    t = np.arange(T + 1, dtype=np.float64)
    f = np.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]                       # alpha_bar(0)=1
    betas = 1.0 - ab[1:] / ab[:-1]
    betas = np.clip(betas, 0.0, 0.999)
    alphas = 1.0 - betas
    return np.cumprod(alphas)           # [T], alpha_bar for t=1..T (index t-1)


class MaskedDiffusion:
    """eps-prediction diffusion restricted to the valid-mask interior."""

    def __init__(self, T=1000, device="cuda:0"):
        self.T = T
        ab = torch.as_tensor(cosine_alphas_cumprod(T), dtype=torch.float64,
                             device=device)
        self.alphas_cumprod = ab                       # index t-1 for step t
        self.sqrt_ab = ab.sqrt().float()
        self.sqrt_1mab = (1 - ab).sqrt().float()
        self.device = device

    def q_sample(self, x0, t, mask, noise=None):
        """x0 [B,3,H,W] in [-1,1] with invalid already zeroed; t [B] in 1..T;
        mask [B,1,H,W] in {0,1}. Noise applied ONLY inside mask."""
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_ab[t - 1].view(-1, 1, 1, 1)
        b = self.sqrt_1mab[t - 1].view(-1, 1, 1, 1)
        xt = a * x0 + b * noise
        return mask * xt + (1 - mask) * x0, noise

    def loss(self, model, x0, cond, mask, t=None, generator=None):
        """Masked eps MSE. x0 invalid region must already be 0."""
        B = x0.shape[0]
        if t is None:
            t = torch.randint(1, self.T + 1, (B,), device=x0.device,
                              generator=generator)
        noise = torch.randn(x0.shape, device=x0.device, generator=generator,
                            dtype=x0.dtype)
        xt, noise = self.q_sample(x0, t, mask, noise)
        eps_hat = model(xt, cond, mask, t)
        se = (eps_hat - noise) ** 2 * mask
        return se.sum() / (mask.sum() * x0.shape[1]).clamp(min=1)

    @torch.no_grad()
    def ddim_sample(self, model, cond, mask, steps=50, eta=0.0, generator=None,
                    x_init=None):
        """Returns x0 estimate in [-1,1]; invalid region stays 0 throughout."""
        B, _, H, W = mask.shape
        if x_init is not None:
            x = x_init
        else:
            x = torch.randn(B, 3, H, W, device=mask.device, generator=generator)
        x = x * mask                                      # invalid fixed at 0
        ts = torch.linspace(self.T, 1, steps, device=mask.device).round().long()
        ab = self.alphas_cumprod.float()
        for i in range(steps):
            t = ts[i]
            ab_t = ab[t - 1]
            ab_prev = ab[ts[i + 1] - 1] if i + 1 < steps else torch.tensor(
                1.0, device=mask.device)
            eps = model(x, cond, mask, t.expand(B))
            x0_hat = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            x0_hat = x0_hat.clamp(-1, 1)
            # keep eps consistent with the clamped x0 estimate
            eps = (x - ab_t.sqrt() * x0_hat) / (1 - ab_t).sqrt().clamp(min=1e-8)
            if eta > 0:
                sigma = eta * ((1 - ab_prev) / (1 - ab_t)).sqrt() \
                        * (1 - ab_t / ab_prev).sqrt()
            else:
                sigma = torch.tensor(0.0, device=mask.device)
            dir_xt = (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            x = ab_prev.sqrt() * x0_hat + dir_xt
            if eta > 0:
                x = x + sigma * torch.randn(x.shape, device=x.device,
                                            generator=generator)
            x = x * mask                                  # re-fix background
        return x
