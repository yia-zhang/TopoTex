# -*- coding: utf-8 -*-
"""Shared embedding / (un)patchify helpers.

Used by the flow-matching velocity network (MiniDiT) and the Global
UV Query Attention decoder. Pure tensor functions - no file IO.
"""

import math

import numpy as np
import torch
import torch.nn as nn


def sincos_2d_pos_embed(dim, grid):
    """[grid*grid, dim] fixed 2D sin-cos embedding (DiT/MAE convention)."""
    assert dim % 4 == 0
    ys, xs = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")

    def embed_1d(pos, d):
        omega = 1.0 / (
            10000 ** (np.arange(d // 2, dtype=np.float64) / (d // 2))
        )
        out = pos.reshape(-1)[:, None] * omega[None]
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    emb = np.concatenate(
        [embed_1d(ys, dim // 2), embed_1d(xs, dim // 2)], axis=1
    )
    return torch.from_numpy(emb).float()


def patchify(x, p):
    """[B,C,H,W] -> [B, (H/p)*(W/p), p*p*C] row-major over the patch grid."""
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // p, p, W // p, p)
    x = x.permute(0, 2, 4, 3, 5, 1)  # B, gh, gw, p, p, C
    return x.reshape(B, (H // p) * (W // p), p * p * C)


def unpatchify(tokens, p, C):
    """[B, g*g, p*p*C] -> [B, C, g*p, g*p] (inverse of patchify)."""
    B, N, _ = tokens.shape
    g = int(math.isqrt(N))
    x = tokens.reshape(B, g, g, p, p, C)
    x = x.permute(0, 5, 1, 3, 2, 4)  # B, C, gh, p, gw, p
    return x.reshape(B, C, g * p, g * p)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def bary_encoding(bary, freqs=4):
    """[K,3] -> [K, 3 + 6*freqs] (raw + sin/cos)."""
    outs = [bary]
    for i in range(freqs):
        w = math.pi * (2**i)
        outs.append(torch.sin(w * bary))
        outs.append(torch.cos(w * bary))
    return torch.cat(outs, dim=-1)
