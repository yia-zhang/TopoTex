# -*- coding: utf-8 -*-
"""TOPOTEX evaluation — closed-loop metrics (DDIM-50, shared Z_F).

python evaluate.py --run checkpoints/baseline [--n 10] [--offset 0] [--out .]

Per mesh:
  uv_000 canonical / uv_001 alternative -> UV PSNR
  uv_002 partial surface query          -> region PSNR + canonical pred on
                                           the same region (partial gap)
  uv_test held-out family               -> UV PSNR (zero-shot)
  render consistency R(M,U0,T0) vs R(M,U1,T1) + GT render fidelity (U1),
  six canonical views each
"""
import argparse
import json
import time
import types
from pathlib import Path

import numpy as np
import torch

from datasets.mesh_utils import (CANONICAL_VIEWS, camera_matrices,
                               dilate_texture, linear_to_srgb_u8,
                               rasterize_view, render_albedo_rebake,
                               seam_error)
from datasets.dataset import TopoTexDataset
from models.texture_generator import (MaskedDiffusion,
                                      MaskedFlowMatching)
from train import build_models

PROJECT_ROOT = Path(__file__).resolve().parent
SEED, DDIM = 20260727, 50


def psnr_region(gt_u8, pred_u8, region):
    mse = float(((gt_u8[region] / 255. - pred_u8[region] / 255.) ** 2).mean())
    return round(10 * np.log10(1 / max(mse, 1e-12)), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--world-size", "--world_size", dest="world_size",
                    type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--ids-file", default=None,
                    help="json list (or {'val': [...]}) of mesh ids to "
                         "evaluate — enables unseen-mesh evaluation")
    ap.add_argument("--merge", action="store_true",
                    help="merge eval_rank_*.json under --run into eval.json")
    args = ap.parse_args()
    device = "cuda:0"
    run = Path(args.run)
    out_path = Path(args.out) if args.out else run / "eval.json"
    if args.merge:
        rows = []
        proto = None
        gstep = None
        for p in sorted(run.glob("eval_rank_*.json")):
            d = json.loads(p.read_text())
            rows.extend(d["per_mesh"])
            proto, gstep = d["protocol"], d.get("global_step")
        seen = set()
        rows = [r for r in rows
                if not (r["mesh_id"] in seen or seen.add(r["mesh_id"]))]
        num_keys = sorted({k for r in rows for k in r
                           if isinstance(r[k], (int, float))
                           and k != "mesh_id"})
        agg = {k: round(float(np.mean(
                   [r[k] for r in rows if r.get(k) is not None])), 2)
               for k in num_keys}
        (run / "eval.json").write_text(json.dumps(
            {"run": str(run), "global_step": gstep, "n_meshes": len(rows),
             "protocol": proto, "aggregate": agg, "per_mesh": rows},
            indent=1))
        print("MERGED", len(rows), "meshes ->", run / "eval.json")
        print("AGG", json.dumps(agg))
        return
    ck = torch.load(run / "ckpt.pt", map_location=device, weights_only=False)
    conditioner, dit = build_models(ck["config"], device)
    conditioner.load_state_dict(ck["conditioner"])
    dit.load_state_dict(ck["dit"])
    conditioner.eval(); dit.eval()
    sched_cls = (MaskedFlowMatching
                 if ck["config"].get("generator") == "fm"
                 else MaskedDiffusion)
    diffusion = sched_cls(T=int(ck["config"]["T"]), device=device)
    if args.ids_file:
        blob = json.loads(Path(args.ids_file).read_text())
        pool = (blob.get("val") or blob.get("unseen") or blob
                if isinstance(blob, dict) else blob)
        ids = list(pool)[args.offset:args.offset + args.n]
    else:
        ids = ck["samples"][args.offset:args.offset + args.n]
    ids = ids[args.rank::args.world_size]
    ds = TopoTexDataset(PROJECT_ROOT / ck["config"]["dataset_root"], ids,
                        device=device)

    gen_times = []

    def sample_query(Z_F, q, mesh):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            o = conditioner(mesh, None,
                            {"face_id": q["face_id"],
                             "barycentric": q["barycentric"].permute(1, 2, 0)},
                            face_tokens=Z_F)
            g = torch.Generator(device=device).manual_seed(SEED)
            x = diffusion.ddim_sample(dit, o["uv_condition"],
                                      q["valid_mask"].float()[None, None],
                                      steps=DDIM, generator=g)[0]
        torch.cuda.synchronize()
        gen_times.append(time.time() - t0)
        img = ((x.clamp(-1, 1) + 1) / 2 * 255).round().byte()
        img = img.permute(1, 2, 0).cpu().numpy().copy()
        img[~q["valid_mask"].cpu().numpy()] = 0
        return img

    def render_q(it, q, tex, vi, res=384):
        V3 = it["mesh"]["vertices"].cpu().numpy().astype(np.float64)
        F3 = it["mesh"]["faces"].cpu().numpy().astype(np.int64)
        canon = types.SimpleNamespace(vertices=V3, faces=F3)
        uvr = types.SimpleNamespace(
            uv_vertices=q["uv_vertices"].astype(np.float64),
            uv_faces=q["uv_faces"], uv_face_to_mesh_face=np.arange(len(F3)))
        _, az, el = CANONICAL_VIEWS[vi]
        gb = rasterize_view(V3, F3,
                            camera_matrices(az, el, V3.min(0), V3.max(0)), res)
        v = q["valid_mask"].cpu().numpy()
        t = tex.copy(); t[~v] = 0
        t = dilate_texture(t, v)
        return (linear_to_srgb_u8(render_albedo_rebake(canon, uvr, t, gb)),
                gb["mask"])

    def render_pair(it, qa, ta, qb, tb):
        vals = []
        for vi in range(6):
            ia, ma = render_q(it, qa, ta, vi)
            ib, mb = render_q(it, qb, tb, vi)
            m = ma & mb
            if m.sum() >= 100:
                vals.append(psnr_region(ia, ib, m))
        return round(float(np.mean(vals)), 2)

    def gt_np(q):
        return (q["gt_texture"].permute(1, 2, 0).cpu().numpy() * 255
                ).astype(np.uint8)

    rows = []
    for it in ds.items:
        with torch.no_grad():
            Z_F, _ = conditioner.encode_faces(
                it["mesh"], it["mv_images"].float()[None] / 255, it["graph"])
        qs = {q["name"]: q for q in it["uv_queries"]}
        qt = it["test_uv_queries"][0]
        preds = {n: sample_query(Z_F, qs[n], it["mesh"]) for n in qs}
        p_test = sample_query(Z_F, qt, it["mesh"])
        region = qs["uv_002"]["valid_mask"].cpu().numpy()
        gt0 = gt_np(qs["uv_000"])
        r = {"mesh_id": it["sample_id"],
             "uv_psnr_canonical": psnr_region(
                 gt0, preds["uv_000"],
                 qs["uv_000"]["valid_mask"].cpu().numpy()),
             "uv_psnr_alternative": psnr_region(
                 gt_np(qs["uv_001"]), preds["uv_001"],
                 qs["uv_001"]["valid_mask"].cpu().numpy()),
             "partial_region_psnr": psnr_region(
                 gt_np(qs["uv_002"]), preds["uv_002"], region),
             "canonical_same_region_psnr": psnr_region(
                 gt0, preds["uv_000"], region),
             "partial_outside_zero": bool(
                 (preds["uv_002"][~region] == 0).all()),
             "uv_psnr_heldout": psnr_region(
                 gt_np(qt), p_test, qt["valid_mask"].cpu().numpy()),
             "render_consistency_U0_U1": render_pair(
                 it, qs["uv_000"], preds["uv_000"],
                 qs["uv_001"], preds["uv_001"]),
             "gt_render_psnr_U1": render_pair(
                 it, qs["uv_001"], preds["uv_001"],
                 qs["uv_001"], gt_np(qs["uv_001"]))}
        # UV seam consistency (alternative layout carries the real seams;
        # the canonical per-vertex layout has none by construction)
        q1 = qs["uv_001"]
        vm1 = q1["valid_mask"].cpu().numpy()
        F_np = it["mesh"]["faces"].cpu().numpy()
        s_gen = seam_error(F_np, q1["uv_vertices"], q1["uv_faces"],
                           preds["uv_001"], vm1)
        s_gt = seam_error(F_np, q1["uv_vertices"], q1["uv_faces"],
                          gt_np(q1), vm1)
        r["seam_consistency"] = s_gen["seam_error"]
        r["seam_consistency_gt_ref"] = s_gt["seam_error"]
        r["n_seam_edges"] = s_gen["n_seam_edges"]
        rows.append(r)
        print(json.dumps(r), flush=True)

    num_keys = sorted({k for r in rows for k in r
                       if isinstance(r[k], (int, float)) and k != "mesh_id"})
    agg = {k: round(float(np.mean(
               [r[k] for r in rows if r.get(k) is not None])), 2)
           for k in num_keys}
    out_path.write_text(json.dumps(
        {"run": str(run), "global_step": ck.get("global_step"),
         "n_meshes": len(rows),
         "protocol": {"generator": ck["config"].get("generator",
                                                     "diffusion"),
                      "sampling_steps": DDIM, "seed": SEED,
                      "shared_Z_F": True,
                      "mean_generation_seconds": round(
                          sum(gen_times) / max(len(gen_times), 1), 3)},
         "aggregate": agg, "per_mesh": rows}, indent=1))
    print("AGG", json.dumps(agg))
    print("done:", out_path)


if __name__ == "__main__":
    main()
