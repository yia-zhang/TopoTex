# -*- coding: utf-8 -*-
"""TopoTexPipeline — the one public inference API.

    pipeline = TopoTexPipeline.from_checkpoint("checkpoints/baseline")
    result = pipeline(mesh=item.mesh, uv_query=item.uv_queries[0],
                      images=item.mv_images, num_steps=50, seed=20260727)
    result.texture   # uint8 [H, W, 3]

The model is conditioned on SIX canonical views (``images``). A single
``reference_image`` cannot be consumed directly at inference — view
generation from one reference is an offline data-construction stage
(UniTEX adapter, :mod:`topotex.data.multiview`); pass it only alongside
``images`` for provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch

from topotex.data.schema import MeshBatch, TopoTexOutput, UVQueryBatch
from topotex.models.topotex import TopoTexModel
from topotex.utils.logging import get_logger

log = get_logger(__name__)


class TopoTexPipeline:
    """Image+Mesh -> Z_F -> UV Query -> Flow Matching -> Texture."""

    def __init__(self, model: TopoTexModel, checkpoint: Optional[dict] = None):
        self.model = model
        self.checkpoint = checkpoint or {}

    @classmethod
    def from_checkpoint(
        cls, path: Union[str, Path], device: str = "cuda:0"
    ) -> "TopoTexPipeline":
        model, ck = TopoTexModel.from_checkpoint(path, device)
        log.info(
            "loaded checkpoint %s (step %s, generator %s)",
            path,
            ck.get("global_step"),
            ck["config"].get("generator"),
        )
        return cls(model, ck)

    @property
    def config(self) -> dict:
        return self.model.config

    def encode(
        self, mesh: MeshBatch, images: torch.Tensor, graph=None
    ) -> torch.Tensor:
        """mesh + six views -> Z_F float32 [F, D] (reusable across
        queries of the same mesh)."""
        return self.model.encode(mesh, images, graph)

    def __call__(
        self,
        mesh: MeshBatch,
        uv_query: UVQueryBatch,
        images: Optional[torch.Tensor] = None,
        face_tokens: Optional[torch.Tensor] = None,
        reference_image: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        seed: int = 20260727,
    ) -> TopoTexOutput:
        """Generate the texture for one UV query.

        mesh:        MeshBatch (vertices [V,3] f32, faces [F,3] i64)
        uv_query:    UVQueryBatch (face_id [H,W] i64, barycentric [3,H,W]
                     f32, valid_mask [H,W] bool)
        images:      uint8 [6, 3, S, S] canonical condition views —
                     required unless face_tokens is given
        face_tokens: precomputed Z_F [F, D] (shared across queries)
        num_steps:   Euler ODE steps (protocol default 50)
        seed:        generation seed (protocol default 20260727)

        Returns TopoTexOutput(texture uint8 [H,W,3], uv_condition,
        face_tokens).
        """
        if face_tokens is None:
            if images is None:
                raise ValueError(
                    "pass `images` (six canonical views) or a precomputed "
                    "`face_tokens`; a lone reference_image must first go "
                    "through offline view generation "
                    "(topotex.data.multiview)"
                )
            face_tokens = self.model.encode(
                mesh, images, uv_query.get("graph")
            )
        out = self.model.condition(face_tokens, uv_query)
        x = self.model.generate(
            out.uv_condition,
            uv_query["valid_mask"],
            num_steps=num_steps,
            seed=seed,
        )
        tex = ((x.clamp(-1, 1) + 1) / 2 * 255).round().byte()
        tex = tex.permute(1, 2, 0).cpu().numpy().copy()
        tex[~uv_query["valid_mask"].cpu().numpy()] = 0
        return TopoTexOutput(
            texture=tex, uv_condition=out.uv_condition, face_tokens=face_tokens
        )


def main():
    """Thin sampling CLI (repository-root ``sample.py``).

    python sample.py --run checkpoints/baseline [--ids id1,id2 | --n 4]
                     [--seed 20260727] [--include-heldout]
    Writes <run>/samples/<sample_id>/{pred,gt}_<query>.png + psnr.json
    """
    import argparse
    import json

    import numpy as np
    from PIL import Image

    from topotex.data.dataset import TopoTexDataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--include-heldout", action="store_true")
    args = ap.parse_args()
    root = Path.cwd()
    pipe = TopoTexPipeline.from_checkpoint(Path(args.run), "cuda:0")
    ck = pipe.checkpoint
    ids = args.ids.split(",") if args.ids else ck["samples"][: args.n]
    ds = TopoTexDataset(
        root / ck["config"]["dataset_root"], ids, device="cuda:0"
    )
    out_root = Path(args.run) / "samples"
    for it in ds.items:
        Z_F = pipe.encode(it["mesh"], it["mv_images"], it["graph"])
        queries = list(it["uv_queries"])
        if args.include_heldout:
            queries += it["test_uv_queries"]
        d = out_root / it["sample_id"]
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        for q in queries:
            pred = pipe(
                mesh=it["mesh"],
                uv_query=q,
                face_tokens=Z_F,
                num_steps=50,
                seed=args.seed,
            ).texture
            v = q["valid_mask"].cpu().numpy()
            gt = (q["gt_texture"].permute(1, 2, 0).cpu().numpy() * 255).astype(
                np.uint8
            )
            Image.fromarray(pred).save(d / f"pred_{q['name']}.png")
            Image.fromarray(gt).save(d / f"gt_{q['name']}.png")
            mse = float(((gt[v] / 255.0 - pred[v] / 255.0) ** 2).mean())
            rows.append(
                {
                    "query": q["name"],
                    "uv_psnr": round(10 * np.log10(1 / max(mse, 1e-12)), 2),
                }
            )
        (d / "psnr.json").write_text(json.dumps(rows, indent=1))
        print(
            it["sample_id"][:12],
            {r["query"]: r["uv_psnr"] for r in rows},
            flush=True,
        )
    print("done:", out_root)


if __name__ == "__main__":
    main()
