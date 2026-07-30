# -*- coding: utf-8 -*-
"""Minimal pixel-space DiT for UV textures (Step 1 spec).

res 256, patch 8 -> 1024 tokens, hidden 384, depth 8, heads 6, mlp 4x,
fixed 2D sin-cos pos embed, AdaLN-Zero timestep conditioning, eps prediction.
Separate patch embeddings for noisy RGB / 64-d LTM condition / valid mask;
tokens = noisy + cond + mask + pos. No cross-attention. Linear head +
unpatchify back to 3-channel eps.
"""

import torch.nn as nn

from topotex.layers.embeddings import (
    TimestepEmbedder,
    modulate,
    sincos_2d_pos_embed,
    unpatchify,
)
from topotex.layers.flow import MaskedFlowMatching  # noqa: F401


class DiTBlock(nn.Module):
    def __init__(self, hidden, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        mh = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mh),
            nn.GELU(approximate="tanh"),
            nn.Linear(mh, hidden),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))
        nn.init.zeros_(self.adaLN[1].weight)
        nn.init.zeros_(self.adaLN[1].bias)

    def forward(self, x, c):
        sh1, sc1, g1, sh2, sc2, g2 = self.adaLN(c).chunk(6, dim=1)
        h = modulate(self.norm1(x), sh1, sc1)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1.unsqueeze(1) * h
        h = self.mlp(modulate(self.norm2(x), sh2, sc2))
        x = x + g2.unsqueeze(1) * h
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden, patch, out_ch):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden, patch * patch * out_ch)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        nn.init.zeros_(self.adaLN[1].weight)
        nn.init.zeros_(self.adaLN[1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


class MiniDiT(nn.Module):
    def __init__(
        self,
        resolution=256,
        patch=8,
        hidden=384,
        depth=8,
        heads=6,
        mlp_ratio=4.0,
        cond_channels=64,
    ):
        super().__init__()
        assert resolution % patch == 0
        self.patch = patch
        self.grid = resolution // patch
        self.embed_x = nn.Conv2d(3, hidden, patch, stride=patch)
        self.embed_c = nn.Conv2d(cond_channels, hidden, patch, stride=patch)
        self.embed_m = nn.Conv2d(1, hidden, patch, stride=patch)
        self.register_buffer(
            "pos",
            sincos_2d_pos_embed(hidden, self.grid)[None],
            persistent=False,
        )
        self.t_embed = TimestepEmbedder(hidden)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden, heads, mlp_ratio) for _ in range(depth)]
        )
        self.final = FinalLayer(hidden, patch, 3)
        self._init_weights()

    def _init_weights(self):
        def basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(basic)
        for e in (self.embed_x, self.embed_c, self.embed_m):
            nn.init.xavier_uniform_(e.weight.view(e.weight.shape[0], -1))
            nn.init.zeros_(e.bias)
        # re-zero the AdaLN-Zero + head projections (self.apply overwrote them)
        for blk in self.blocks:
            nn.init.zeros_(blk.adaLN[1].weight)
            nn.init.zeros_(blk.adaLN[1].bias)
        nn.init.zeros_(self.final.adaLN[1].weight)
        nn.init.zeros_(self.final.adaLN[1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)

    def forward(self, x_noisy, cond, mask, t):
        """x_noisy [B,3,H,W]; cond [B,64,H,W] normalized (0 on invalid);
        mask [B,1,H,W]; t [B] int in 1..T. Returns eps [B,3,H,W]."""
        tok = self.embed_x(x_noisy) + self.embed_c(cond) + self.embed_m(mask)
        tok = tok.flatten(2).transpose(1, 2) + self.pos  # [B,N,h]
        c = self.t_embed(t)
        for blk in self.blocks:
            tok = blk(tok, c)
        out = self.final(tok, c)  # [B,N,p*p*3]
        return unpatchify(out, self.patch, 3)
