# -*- coding: utf-8 -*-
"""TOPOTEX dataset loader.

Each item:
    sample_id, reference_image, mv_images, mesh {vertices, faces}, graph,
    uv_queries:      TRAIN queries (canonical uv_000 / alternative uv_001 /
                     partial uv_002)
    test_uv_queries: held-out family queries (uv_test) — EVALUATION ONLY,
                     never sampled during training
Each query dict:
    {name, face_id i64 [H,W], barycentric f32 [3,H,W],
     valid_mask bool [H,W], gt_texture f32 [3,H,W],
     uv_vertices f32 [Vuv,2], uv_faces i64 [F,3]}
"""

import json
from pathlib import Path

import numpy as np
import torch

from topotex.data.schema import MeshBatch, TopoTexBatch, UVQueryBatch


class TopoTexDataset(torch.utils.data.Dataset):
    def __init__(self, root, sample_ids=None, device="cpu", build_graphs=True):
        from PIL import Image
        from safetensors.numpy import load_file

        from topotex.layers.topology import build_face_graph

        self.root = Path(root)
        if sample_ids is None:
            sample_ids = [
                json.loads(l)["sample_id"]
                for l in open(self.root / "manifest.jsonl")
            ]
        self.sample_ids = list(sample_ids)
        self.items = []
        for sid in self.sample_ids:
            d = self.root / "samples" / sid
            mesh_np = load_file(str(d / "mesh.safetensors"))
            mv = load_file(str(d / "mv.safetensors"))
            ref = np.asarray(Image.open(d / "reference.png").convert("RGB"))
            mesh = MeshBatch(
                vertices=torch.tensor(mesh_np["vertices"], device=device),
                faces=torch.tensor(
                    mesh_np["faces"].astype(np.int64), device=device
                ),
            )
            queries, test_queries = [], []
            for qd in sorted((d / "uv_queries").iterdir()):
                ua = load_file(str(qd / "uv_address.safetensors"))
                gt = (
                    np.asarray(
                        Image.open(qd / "gt_texture.png").convert("RGB"),
                        np.float32,
                    )
                    / 255.0
                )
                (
                    test_queries if qd.name.startswith("uv_test") else queries
                ).append(
                    UVQueryBatch(
                        name=qd.name,
                        face_id=torch.tensor(
                            ua["face_id"].astype(np.int64), device=device
                        ),
                        barycentric=torch.tensor(
                            ua["barycentric"].astype(np.float32), device=device
                        ),
                        valid_mask=torch.tensor(
                            ua["valid_mask"].astype(bool), device=device
                        ),
                        gt_texture=torch.tensor(
                            gt.transpose(2, 0, 1), device=device
                        ),
                        uv_vertices=ua["uv_vertices"],
                        uv_faces=ua["uv_faces"].astype(np.int64),
                    )
                )
            self.items.append(
                TopoTexBatch(
                    sample_id=sid,
                    reference_image=torch.tensor(
                        ref.transpose(2, 0, 1), device=device
                    ),
                    mv_images=torch.tensor(mv["images"], device=device),
                    mesh=mesh,
                    graph=(
                        build_face_graph(mesh["vertices"], mesh["faces"])
                        if build_graphs
                        else None
                    ),
                    uv_queries=queries,
                    test_uv_queries=test_queries,
                )
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]
