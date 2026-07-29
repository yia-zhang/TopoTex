# -*- coding: utf-8 -*-
"""Global UV Query Attention decoder (multi-UV research line).

Replaces the per-texel gather decoder with patchified UV query tokens that
attend GLOBALLY to the Face Set latent:

    per-texel address embed = Linear([face_token(face_id), enc(bary)])
    patch token (8x8 texels)  -> Q                        [1024, D]
    face tokens               -> K, V                     [F, D]
    N x { CrossAttention(Q, K, V) + MLP }   (no routing, no masks)
    unpatchify -> uv_condition [C, H, W] (+ rgb head)

The UV layout enters only through which faces each patch references and a
learned 2D atlas position embedding; surface CONTENT flows exclusively
through attention over Z_F.
"""
import math

import torch
import torch.nn as nn


def bary_encoding(bary, freqs=4):
    """[K,3] -> [K, 3 + 6*freqs] (raw + sin/cos)."""
    outs = [bary]
    for i in range(freqs):
        w = math.pi * (2 ** i)
        outs.append(torch.sin(w * bary))
        outs.append(torch.cos(w * bary))
    return torch.cat(outs, dim=-1)


class CrossBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.nq = nn.LayerNorm(dim)
        self.nk = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.nm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                 nn.Linear(4 * dim, dim))

    def forward(self, q, kv):
        a, _ = self.attn(self.nq(q), self.nk(kv), self.nk(kv),
                         need_weights=False)
        q = q + a
        q = q + self.mlp(self.nm(q))
        return q


class UVQueryAttention(nn.Module):
    def __init__(self, dim=256, out_channels=64, patch=8, res=256,
                 heads=8, depth=4, freqs=4, texel_dim=32):
        super().__init__()
        self.patch = patch
        self.res = res
        self.grid = res // patch
        self.out_channels = out_channels
        self.texel = nn.Linear(dim + 3 + 6 * freqs, texel_dim)
        self.bg_texel = nn.Parameter(torch.zeros(texel_dim))
        self.freqs = freqs
        self.embed = nn.Linear(patch * patch * texel_dim, dim)
        self.pos = nn.Parameter(torch.zeros(self.grid * self.grid, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(CrossBlock(dim, heads)
                                    for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, patch * patch * out_channels)
        self.rgb_head = nn.Linear(dim, patch * patch * 3)

    def forward(self, face_tokens, face_id, barycentric, with_rgb=False):
        """face_tokens [F,D]; face_id [H,W] int (-1 bg); bary [H,W,3].
        Returns (uv_condition [C,H,W], rgb [3,H,W] | None)."""
        H, W = face_id.shape
        P, G = self.patch, self.grid
        valid = face_id >= 0
        tex = self.bg_texel.expand(H, W, -1).clone()
        if valid.any():
            z = face_tokens[face_id[valid].long()]
            enc = bary_encoding(barycentric[valid], self.freqs)
            tex[valid] = self.texel(torch.cat([z, enc], dim=-1))
        # [H,W,t] -> patch tokens [G*G, P*P*t]
        t = tex.view(G, P, G, P, -1).permute(0, 2, 1, 3, 4) \
              .reshape(G * G, P * P * tex.shape[-1])
        q = self.embed(t) + self.pos                       # [G*G, D]
        kv = face_tokens.unsqueeze(0)                      # [1, F, D]
        q = q.unsqueeze(0)
        for blk in self.blocks:
            q = blk(q, kv)
        q = self.norm(q.squeeze(0))                        # [G*G, D]

        def unpatch(x, c):
            return x.view(G, G, P, P, c).permute(4, 0, 2, 1, 3) \
                    .reshape(c, H, W)
        cond = unpatch(self.head(q), self.out_channels)
        cond = cond * valid.unsqueeze(0)                   # background -> 0
        rgb = None
        if with_rgb:
            rgb = unpatch(self.rgb_head(q), 3) * valid.unsqueeze(0)
        return cond, rgb
