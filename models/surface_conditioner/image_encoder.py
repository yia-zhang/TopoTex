# -*- coding: utf-8 -*-
"""Multi-view image encoder (from scratch: no CLIP/DINO, no pretrained).

ViT-style: conv patch embedding -> +2D pos embed +learnable view embedding ->
joint self-attention Transformer blocks over ALL view tokens -> [B, Nv*T, D].
No camera-xyz embedding, no mesh-xyz embedding (view identity only).
"""
import torch
import torch.nn as nn


class ViTBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                 nn.Linear(dim * 4, dim))

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.mlp(self.norm2(x))


class MultiViewEncoder(nn.Module):
    def __init__(self, dim=256, num_views=6, image_size=256, patch=16,
                 depth=4, heads=8):
        super().__init__()
        self.num_views = num_views
        self.grid = image_size // patch
        self.patch_embed = nn.Conv2d(3, dim, patch, stride=patch)
        self.view_embed = nn.Parameter(torch.zeros(num_views, dim))
        self.pos = nn.Parameter(torch.zeros(1, self.grid * self.grid, dim))
        nn.init.normal_(self.view_embed, std=0.02)
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([ViTBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    @property
    def tokens_per_view(self):
        return self.grid * self.grid

    def forward(self, images, view_ids=None):
        """images [B,Nv,3,H,W] in [0,1] -> tokens [B, Nv*T, D].
        view_ids: LongTensor [Nv] canonical view indices for arbitrary view
        SUBSETS (e.g. front+top = [0,4]); default = first Nv views. Without
        this, dropped views would silently shift the view embeddings."""
        B, Nv, C, H, W = images.shape
        assert Nv <= self.num_views, f"got {Nv} views > {self.num_views}"
        if view_ids is None:
            view_ids = torch.arange(Nv, device=images.device)
        assert len(view_ids) == Nv
        x = self.patch_embed(images.reshape(B * Nv, C, H, W) * 2 - 1)
        x = x.flatten(2).transpose(1, 2) + self.pos              # [B*Nv,T,D]
        x = x.reshape(B, Nv, -1, x.shape[-1])
        x = x + self.view_embed[view_ids].view(1, Nv, 1, -1)
        x = x.reshape(B, Nv * x.shape[2], x.shape[-1])           # [B,Nv*T,D]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
