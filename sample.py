# -*- coding: utf-8 -*-
"""TOPOTEX sampling — one Z_F per mesh, textures for every UV query.

python sample.py --run checkpoints/baseline [--ids id1,id2 | --n 4]
                 [--seed 20260727] [--include-heldout]

Writes <run>/samples/<sample_id>/{pred,gt}_<query>.png + psnr.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from datasets.dataset import TopoTexDataset
from models.texture_generator import (MaskedDiffusion,
                                      MaskedFlowMatching)
from train import build_models

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--include-heldout", action="store_true")
    args = ap.parse_args()
    device = "cuda:0"
    run = Path(args.run)
    ck = torch.load(run / "ckpt.pt", map_location=device, weights_only=False)
    conditioner, dit = build_models(ck["config"], device)
    conditioner.load_state_dict(ck["conditioner"])
    dit.load_state_dict(ck["dit"])
    conditioner.eval(); dit.eval()
    sched_cls = (MaskedFlowMatching
                 if ck["config"].get("generator") == "fm"
                 else MaskedDiffusion)
    diffusion = sched_cls(T=int(ck["config"]["T"]), device=device)
    ids = args.ids.split(",") if args.ids else ck["samples"][: args.n]
    ds = TopoTexDataset(PROJECT_ROOT / ck["config"]["dataset_root"], ids,
                        device=device)
    out_root = run / "samples"
    for it in ds.items:
        with torch.no_grad():
            Z_F, _ = conditioner.encode_faces(
                it["mesh"], it["mv_images"].float()[None] / 255, it["graph"])
        queries = list(it["uv_queries"])
        if args.include_heldout:
            queries += it["test_uv_queries"]
        d = out_root / it["sample_id"]
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        for q in queries:
            with torch.no_grad():
                o = conditioner(it["mesh"], None,
                                {"face_id": q["face_id"],
                                 "barycentric":
                                     q["barycentric"].permute(1, 2, 0)},
                                face_tokens=Z_F)
                g = torch.Generator(device=device).manual_seed(args.seed)
                x = diffusion.ddim_sample(
                    dit, o["uv_condition"],
                    q["valid_mask"].float()[None, None], steps=50,
                    generator=g)[0]
            v = q["valid_mask"].cpu().numpy()
            pred = ((x.clamp(-1, 1) + 1) / 2 * 255).round().byte() \
                .permute(1, 2, 0).cpu().numpy().copy()
            pred[~v] = 0
            gt = (q["gt_texture"].permute(1, 2, 0).cpu().numpy() * 255
                  ).astype(np.uint8)
            Image.fromarray(pred).save(d / f"pred_{q['name']}.png")
            Image.fromarray(gt).save(d / f"gt_{q['name']}.png")
            mse = float(((gt[v] / 255. - pred[v] / 255.) ** 2).mean())
            rows.append({"query": q["name"],
                         "uv_psnr": round(10 * np.log10(1 / max(mse, 1e-12)),
                                          2)})
        (d / "psnr.json").write_text(json.dumps(rows, indent=1))
        print(it["sample_id"][:12],
              {r["query"]: r["uv_psnr"] for r in rows}, flush=True)
    print("done:", out_root)


if __name__ == "__main__":
    main()
