# -*- coding: utf-8 -*-
"""Global UV Query Attention decoder with the factorized dense UV query
encoder.

Per-texel query features are the SUM of two factorized addresses instead
of a concat-and-project bottleneck:

    face_address_table = Linear(D -> Dq)(Z_F)          [F, Dq]  (once)
    e_face[p]  = face_address_table[face_id[p]]        gather at Dq width
    e_bary[p]  = BarycentricMLP(enc(bary[p]))          valid texels only
    e_texel[p] = LayerNorm(e_face[p] + e_bary[p])      [Dq]
    background = learned bg embedding (never indexes Z_F)

    dense texel query [1, Dq, H, W]
      -> Conv2d(Dq, D, kernel=patch, stride=patch)     [1, D, G, G]
      -> + learned 2D atlas position embedding         [1, G*G, D]
      -> N x { CrossAttention(Q, K=V=Z_F) + MLP }      (no routing/masks)
      -> unpatchify -> uv_condition [C, H, W] (+ rgb head)

Face ids are pointers, not semantic embeddings: permuting Z_F rows while
remapping face_id leaves the output unchanged. The UV layout enters only
through which faces each texel references and the atlas position
embedding; surface CONTENT flows exclusively through attention over Z_F.
"""

import torch
import torch.nn as nn

from topotex.config import MIN_TEXEL_DIM
from topotex.layers.embeddings import bary_encoding


class CrossBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.nq = nn.LayerNorm(dim)
        self.nk = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.nm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, q, kv):
        a, _ = self.attn(
            self.nq(q), self.nk(kv), self.nk(kv), need_weights=False
        )
        q = q + a
        q = q + self.mlp(self.nm(q))
        return q


class UVQueryAttention(nn.Module):
    def __init__(
        self,
        dim=256,
        out_channels=64,
        patch=8,
        res=256,
        heads=8,
        depth=4,
        freqs=4,
        texel_dim=None,
    ):
        super().__init__()
        self.patch = patch
        self.res = res
        self.grid = res // patch
        self.out_channels = out_channels
        if texel_dim is None:
            texel_dim = dim // 4
        if texel_dim < MIN_TEXEL_DIM:
            raise ValueError(
                f"texel_dim {texel_dim} < minimum {MIN_TEXEL_DIM}"
            )
        self.texel_dim = texel_dim
        self.freqs = freqs
        # factorized dense texel query: face address + barycentric address
        self.face_proj = nn.Linear(dim, texel_dim)
        self.bary_mlp = nn.Sequential(
            nn.Linear(3 + 6 * freqs, texel_dim),
            nn.GELU(),
            nn.Linear(texel_dim, texel_dim),
        )
        self.texel_norm = nn.LayerNorm(texel_dim)
        self.bg_texel = nn.Parameter(torch.zeros(texel_dim))
        self.patch_embed = nn.Conv2d(
            texel_dim, dim, kernel_size=patch, stride=patch
        )
        self.pos = nn.Parameter(torch.zeros(self.grid * self.grid, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            CrossBlock(dim, heads) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, patch * patch * out_channels)
        self.rgb_head = nn.Linear(dim, patch * patch * 3)

    def forward(self, face_tokens, face_id, barycentric, with_rgb=False):
        """face_tokens [F,D]; face_id [H,W] int (-1 bg); bary [H,W,3].
        Returns (uv_condition [C,H,W], rgb [3,H,W] | None)."""
        H, W = face_id.shape
        P, G = self.patch, self.grid
        valid = face_id >= 0
        # project the full face table ONCE at [F,D]->[F,Dq], then gather
        # at Dq width (never materialize a gathered [H,W,D] tensor)
        table = self.face_proj(face_tokens)  # [F, Dq]
        flat_id = face_id.reshape(-1)
        idx = (flat_id >= 0).nonzero(as_tuple=True)[0]
        tex = self.bg_texel.expand(H * W, -1)  # background never gathers
        if idx.numel():
            e_face = table.index_select(0, flat_id.index_select(0, idx).long())
            enc = bary_encoding(
                barycentric.reshape(-1, 3).index_select(0, idx), self.freqs
            )
            e_texel = self.texel_norm(e_face + self.bary_mlp(enc))
            tex = tex.to(e_texel.dtype).clone()
            tex.index_copy_(0, idx, e_texel)
        else:
            tex = tex.clone()
        dense = tex.view(H, W, -1).permute(2, 0, 1)[None].contiguous()
        q = self.patch_embed(dense)  # [1, D, G, G]
        q = q.flatten(2).transpose(1, 2) + self.pos  # [1, G*G, D]
        kv = face_tokens.unsqueeze(0)  # [1, F, D]
        for blk in self.blocks:
            q = blk(q, kv)
        q = self.norm(q.squeeze(0))  # [G*G, D]

        def unpatch(x, c):
            return (
                x.view(G, G, P, P, c).permute(4, 0, 2, 1, 3).reshape(c, H, W)
            )

        cond = unpatch(self.head(q), self.out_channels)
        cond = cond * valid.unsqueeze(0)  # background -> 0
        rgb = None
        if with_rgb:
            rgb = unpatch(self.rgb_head(q), 3) * valid.unsqueeze(0)
        return cond, rgb
