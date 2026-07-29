# -*- coding: utf-8 -*-
"""Surface Conditioner: face-set encoder (FaceTokenizer + face-view cross
attention + Topology Transformer) followed by the Global UV Query Attention
decoder.

    Z_F = encode_faces(mesh, six views)         # UV-independent
    uv_condition = UVQueryAttention(UV query, Z_F)
"""
import torch
import torch.nn as nn

from .face_image_attention import FaceImageAttention
from .face_tokenizer import FaceTokenizer, build_face_graph
from .image_encoder import MultiViewEncoder
from .topology_pe import TopologyPE
from .topology_transformer import TopologyTransformer
from .uv_query_attention import UVQueryAttention


class SurfaceConditioner(nn.Module):
    def __init__(self, dim=256, out_channels=64, pe_kind="random_walk",
                 pe_k=16, heads=8, cross_depth=2, topo_depth=4,
                 query_depth=4, num_views=6, image_size=256, patch=8,
                 resolution=256):
        super().__init__()
        self.topo_pe = TopologyPE(pe_kind, pe_k)
        self.tokenizer = FaceTokenizer(dim=dim, pe_dim=pe_k)
        self.image_encoder = MultiViewEncoder(dim, num_views, image_size)
        self.cross = FaceImageAttention(dim, heads, cross_depth)
        self.topo = TopologyTransformer(dim, heads, topo_depth)
        self.decoder = UVQueryAttention(dim, out_channels, patch=patch,
                                        res=resolution, heads=heads,
                                        depth=query_depth)

    def encode_faces(self, mesh, mv_images, graph=None):
        """Z_F: [F, D] face-set latent.

        A cached graph is only valid for the vertices it was built from
        (global_scale / rel edge lengths are vertex-dependent); reusing it
        after a vertex transform silently corrupts the intrinsic features.
        Rebuild whenever the mesh scale no longer matches the cache.
        """
        V, Fc = mesh["vertices"], mesh["faces"]
        if graph is not None:
            tri = V[Fc]
            area2 = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0],
                                dim=1).norm(dim=1)
            gs = torch.sqrt((area2.sum() / 2).clamp(min=1e-20))
            if not torch.isclose(gs, graph["global_scale"], rtol=1e-3):
                graph = None
        if graph is None:
            graph = build_face_graph(V, Fc)
        pe = self.topo_pe(graph, len(Fc))
        x = self.tokenizer(V, Fc, graph, pe)
        x = self.cross(x.unsqueeze(0),
                       self.image_encoder(mv_images)).squeeze(0)
        x = self.topo(x, graph)
        return x, graph

    def forward(self, mesh, mv_images, uv_address, with_rgb=False,
                face_tokens=None):
        """uv_address: dict(face_id [H,W], barycentric [H,W,3], graph?).
        Pass face_tokens to reuse a precomputed Z_F across UV queries."""
        if face_tokens is None:
            face_tokens, graph = self.encode_faces(
                mesh, mv_images, uv_address.get("graph"))
        cond, rgb = self.decoder(face_tokens, uv_address["face_id"],
                                 uv_address["barycentric"], with_rgb=with_rgb)
        out = {"face_tokens": face_tokens,
               "uv_condition": cond.unsqueeze(0)}
        if with_rgb:
            out["uv_rgb"] = rgb.unsqueeze(0)
        return out
