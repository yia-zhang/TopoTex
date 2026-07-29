# -*- coding: utf-8 -*-
"""Face->MV cross attention: Q = face tokens, K/V = image tokens.

No hard masks, no face-id routing, no handcrafted correspondence -- the
attention learns image-surface correspondence end to end.
"""
import torch.nn as nn


class CrossBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                 nn.Linear(dim * 4, dim))

    def forward(self, x, ctx, need_weights=False):
        h, w = self.attn(self.norm_q(x), self.norm_kv(ctx),
                         self.norm_kv(ctx), need_weights=need_weights,
                         average_attn_weights=True)
        x = x + h
        return x + self.mlp(self.norm2(x)), w


class FaceImageAttention(nn.Module):
    def __init__(self, dim=256, heads=8, depth=2):
        super().__init__()
        self.blocks = nn.ModuleList([CrossBlock(dim, heads)
                                     for _ in range(depth)])

    def forward(self, face_tokens, image_tokens, return_attn=False):
        """face_tokens [B,F,D], image_tokens [B,N,D] -> [B,F,D]
        (+ last block's head-averaged attention [B,F,N] when return_attn)."""
        x = face_tokens
        attn = None
        for i, blk in enumerate(self.blocks):
            x, w = blk(x, image_tokens,
                       need_weights=return_attn and i == len(self.blocks) - 1)
            if w is not None:
                attn = w
        if return_attn:
            return x, attn
        return x
