# -*- coding: utf-8 -*-
"""TOPOTEX training (frozen baseline recipe).

python train.py --samples 1   --run-name overfit  --steps 20000
python train.py --samples 10  --run-name smoke
python train.py --samples 265 --run-name baseline [--resume]

Rectified-flow texture generator (velocity MSE, high-t emphasis, aux RGB
L1) over the frozen conditioner. Official efficiency configuration:
face-count-bucketed groups of `group_size` meshes packed into a single
face graph per step (tokenizer + topology transformer run once per group;
per-mesh unit-area normalization keeps intrinsic features numerically
identical), bf16 autocast on the generator loss. One UV query per mesh per
step sampled with config `query_probs`. Budget is counted in mesh
exposures (steps = N * target_mesh_exposures / group_size). Checkpoints
carry model/optimizer/step/RNG; --resume restores them exactly.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from datasets.dataset import TopoTexDataset
from models.surface_conditioner import SurfaceConditioner, build_face_graph
from models.texture_generator import (MaskedDiffusion,
                                      MaskedFlowMatching, MiniDiT)

PROJECT_ROOT = Path(__file__).resolve().parent


def build_models(cfg, device):
    torch.manual_seed(int(cfg["seed"]))
    conditioner = SurfaceConditioner(
        dim=int(cfg["cond_dim"]), out_channels=int(cfg["cond_channels"]),
        pe_kind="random_walk", pe_k=int(cfg["pe_k"]),
        heads=int(cfg["cond_heads"]), cross_depth=int(cfg["cross_depth"]),
        topo_depth=int(cfg["topo_depth"]),
        query_depth=int(cfg["query_depth"]),
        image_size=int(cfg["image_size"]),
        resolution=int(cfg["resolution"]),
        patch=int(cfg["patch"])).to(device)
    dit = MiniDiT(resolution=int(cfg["resolution"]), patch=int(cfg["patch"]),
                  hidden=int(cfg["dit_hidden"]), depth=int(cfg["dit_depth"]),
                  heads=int(cfg["dit_heads"]), mlp_ratio=4.0,
                  cond_channels=int(cfg["cond_channels"])).to(device)
    return conditioner, dit


def to_dev(t, device):
    return t.to(device, non_blocking=True) if t.device.type != device.split(":")[0] else t


def build_groups(ds, k, device):
    """Face-count-bucketed fixed groups + one packed graph per group.
    Per-mesh unit-area normalization + rel rescale keep the tokenizer's
    intrinsic features numerically identical to per-mesh encoding."""
    order = sorted(range(len(ds)), key=lambda i: len(ds[i]["mesh"]["faces"]))
    groups = []
    for i in range(0, len(order), k):
        idxs = order[i:i + k]
        Vs, Fs, voff = [], [], 0
        for j in idxs:
            V = ds[j]["mesh"]["vertices"].to(device)
            F = ds[j]["mesh"]["faces"].to(device)
            tri = V[F]
            area = (torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0],
                                dim=1).norm(dim=1).sum() / 2)
            Vs.append(V / area.clamp(min=1e-20).sqrt())
            Fs.append(F + voff)
            voff += len(V)
        Vp, Fp = torch.cat(Vs), torch.cat(Fs)
        graph = build_face_graph(Vp, Fp)
        graph["rel"][:, 0] *= graph["global_scale"]
        graph["global_scale"] = torch.tensor(1.0, device=device)
        groups.append({"idxs": idxs, "Vp": Vp, "Fp": Fp, "graph": graph,
                       "n_faces": [len(ds[j]["mesh"]["faces"])
                                   for j in idxs]})
    return groups


def group_step(conditioner, group, ds, queries, device, aux_w):
    """Packed conditioner forward for one group. queries: per-mesh query
    dicts (already on device). Returns per-mesh (cond, aux_rgb_loss)."""
    g = group["graph"]
    pe = conditioner.topo_pe(g, len(group["Fp"]))
    x = conditioner.tokenizer(group["Vp"], group["Fp"], g, pe)
    imgs = torch.stack([to_dev(ds[j]["mv_images"], device)
                        for j in group["idxs"]]).float() / 255
    img_tokens = conditioner.image_encoder(imgs)          # [K, Nv*T, D]
    outs, fo = [], 0
    for i, n in enumerate(group["n_faces"]):
        outs.append(conditioner.cross(x[fo:fo + n].unsqueeze(0),
                                      img_tokens[i:i + 1]).squeeze(0))
        fo += n
    x = conditioner.topo(torch.cat(outs), g)
    conds, aux = [], 0.0
    fo = 0
    for q, n in zip(queries, group["n_faces"]):
        c, rgb = conditioner.decoder(x[fo:fo + n], q["face_id"],
                                     q["barycentric"].permute(1, 2, 0),
                                     with_rgb=aux_w > 0)
        conds.append(c.unsqueeze(0))
        if aux_w > 0:
            aux = aux + (rgb - q["gt_texture"]).abs()[
                :, q["valid_mask"]].mean()
        fo += n
    return conds, aux / max(len(queries), 1)


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/topotex_fm_baseline.yaml")
    ap.add_argument("--samples", default="265")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--generator", default=None,
                    choices=["diffusion", "fm"],
                    help="texture generator schedule (default: config value "
                         "or diffusion); recorded in the checkpoint")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.seed is not None:
        cfg["seed"] = args.seed
    device = "cuda:0"
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    query_probs = [float(x) for x in cfg["query_probs"]]
    assert abs(sum(query_probs) - 1) < 1e-6

    root = PROJECT_ROOT / cfg["dataset_root"]
    ids = [json.loads(l)["sample_id"] for l in open(root / "manifest.jsonl")]
    ids = ids[: int(args.samples)]
    group_size = max(1, min(int(cfg.get("group_size", 1)), len(ids)))
    cfg["group_size"] = group_size
    use_bf16 = cfg.get("precision") == "bf16"
    steps = args.steps or math.ceil(
        len(ids) * int(cfg["target_mesh_exposures"]) / group_size)
    cfg["steps"] = steps
    run = PROJECT_ROOT / "runs" / args.run_name
    run.mkdir(parents=True, exist_ok=True)

    if args.generator is not None:
        cfg["generator"] = args.generator
    cfg.setdefault("generator", "diffusion")
    # packed mode stores data on CPU past 300 meshes (moved per step) and
    # skips per-item graphs (group graphs are packed at startup instead)
    data_dev = device if (group_size == 1 or len(ids) <= 300) else "cpu"
    ds = TopoTexDataset(root, ids, device=data_dev,
                        build_graphs=(group_size == 1))
    conditioner, dit = build_models(cfg, device)
    sched_cls = (MaskedFlowMatching if cfg["generator"] == "fm"
                 else MaskedDiffusion)
    diffusion = sched_cls(T=int(cfg["T"]), device=device)
    groups = (build_groups(ds, group_size, device)
              if group_size > 1 else None)
    params = list(conditioner.parameters()) + list(dit.parameters())
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=0.0,
                            betas=(0.9, 0.95))
    n_params = sum(p.numel() for p in params)
    bs = int(cfg["noise_batch"])
    aux_w = float(cfg["aux_rgb_weight"])
    t_high_frac = float(cfg["t_high_frac"])
    t_high_min = int(cfg["t_high_min"])
    lr_final = float(cfg["lr_final_frac"])
    warmup = int(cfg["warmup"])
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = None
    manifest_sha = hashlib.sha256(
        (root / "manifest.jsonl").read_bytes()).hexdigest()
    config_sha = hashlib.sha256(
        Path(args.config).read_bytes()).hexdigest()
    (run / "provenance.json").write_text(json.dumps(
        {"config": cfg, "samples": ids, "n_params": n_params,
         "git_commit": commit, "query_probs": query_probs,
         "dataset_manifest_sha256": manifest_sha,
         "config_sha256": config_sha}, indent=1))

    g = torch.Generator(device=device).manual_seed(seed)
    g_idx = torch.Generator().manual_seed(seed + 1)
    start_step = 0
    loss_ema = None
    if args.resume and (run / "ckpt.pt").exists():
        ck = torch.load(run / "ckpt.pt", map_location=device,
                        weights_only=False)
        conditioner.load_state_dict(ck["conditioner"])
        dit.load_state_dict(ck["dit"])
        opt.load_state_dict(ck["opt"])
        g.set_state(ck["rng_g"].cpu())
        g_idx.set_state(ck["rng_g_idx"].cpu())
        start_step = ck["global_step"]
        loss_ema = ck["loss_ema"]
        print(f"resumed {run} at step {start_step}", flush=True)
    metrics_f = open(run / "metrics.jsonl", "a")
    profile_rows = []
    dev_index = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    last_log = [0, time.time()]
    log_rows, t0 = [], time.time()
    conditioner.train()
    dit.train()
    for step in range(start_step + 1, steps + 1):
        wu = min(1.0, step / max(warmup, 1))
        cos = lr_final + (1 - lr_final) * 0.5 * (
            1 + math.cos(math.pi * step / steps))
        for pg in opt.param_groups:
            pg["lr"] = float(cfg["lr"]) * wu * cos

        def pick_query(item):
            r = float(torch.rand(1, generator=g_idx))
            acc, qi = 0.0, len(query_probs) - 1
            for j, pj in enumerate(query_probs):
                acc += pj
                if r < acc:
                    qi = j
                    break
            return item["uv_queries"][qi]

        if group_size > 1:
            grp = groups[int(torch.randint(0, len(groups), (1,),
                                           generator=g_idx))]
            queries = []
            for j in grp["idxs"]:
                q0 = pick_query(ds[j])
                queries.append({k: (to_dev(v, device) if torch.is_tensor(v)
                                    else v) for k, v in q0.items()})
            conds, aux = group_step(conditioner, grp, ds, queries, device,
                                    aux_w)
            x0s, masks, cs = [], [], []
            for q, c in zip(queries, conds):
                m = q["valid_mask"].float()[None, None]
                x0s.append(((q["gt_texture"][None] * 2 - 1) * m)
                           .expand(bs, -1, -1, -1))
                masks.append(m.expand(bs, -1, -1, -1))
                cs.append(c.expand(bs, -1, -1, -1))
            X0, M, C = torch.cat(x0s), torch.cat(masks), torch.cat(cs)
            B = X0.shape[0]
            t = torch.randint(1, diffusion.T + 1, (B,), device=device,
                              generator=g)
            n_high = int(round(B * t_high_frac))
            if n_high:
                t[:n_high] = torch.randint(t_high_min, diffusion.T + 1,
                                           (n_high,), device=device,
                                           generator=g)
            with torch.autocast("cuda", torch.bfloat16, enabled=use_bf16):
                loss_diff = diffusion.loss(dit, X0, C, M, t=t, generator=g)
            loss_diff = loss_diff.float()
            loss = loss_diff + (aux_w * aux if aux_w > 0 else 0.0)
            aux = aux if aux_w > 0 else None
            q_tag = f"{queries[0]['name']}(+{len(queries) - 1})"
        else:
            it = ds[int(torch.randint(0, len(ds), (1,), generator=g_idx))]
            q = pick_query(it)
            q_tag = q["name"]
            mv = it["mv_images"].float()[None] / 255
            out = conditioner(it["mesh"], mv,
                              {"face_id": q["face_id"],
                               "barycentric":
                                   q["barycentric"].permute(1, 2, 0),
                               "graph": it["graph"]}, with_rgb=aux_w > 0)
            cond = out["uv_condition"]
            mask = q["valid_mask"].float()[None, None]
            x0 = (q["gt_texture"][None] * 2 - 1) * mask
            t = torch.randint(1, diffusion.T + 1, (bs,), device=device,
                              generator=g)
            n_high = int(round(bs * t_high_frac))
            if n_high:
                t[:n_high] = torch.randint(t_high_min, diffusion.T + 1,
                                           (n_high,), device=device,
                                           generator=g)
            with torch.autocast("cuda", torch.bfloat16, enabled=use_bf16):
                loss_diff = diffusion.loss(dit, x0.expand(bs, -1, -1, -1),
                                           cond.expand(bs, -1, -1, -1),
                                           mask.expand(bs, -1, -1, -1), t=t,
                                           generator=g)
            loss_diff = loss_diff.float()
            loss = loss_diff
            aux = None
            if aux_w > 0:
                aux = (out["uv_rgb"][0] - q["gt_texture"]).abs()[
                    :, q["valid_mask"]].mean()
                loss = loss + aux_w * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        l = float(loss)
        if not np.isfinite(l):
            raise RuntimeError(f"NaN/Inf loss at step {step}")
        loss_ema = l if loss_ema is None else 0.99 * loss_ema + 0.01 * l
        if step % int(cfg["log_every"]) == 0 or step == 1:
            row = {"step": step, "loss": l, "loss_ema": loss_ema,
                   "loss_diff": float(loss_diff),
                   "aux_rgb": None if aux is None else float(aux),
                   "uv_query": q_tag,
                   "lr": float(opt.param_groups[0]["lr"]),
                   "elapsed_s": round(time.time() - t0, 1)}
            log_rows.append(row)
            metrics_f.write(json.dumps(row) + "\n")
            metrics_f.flush()
            try:
                util, mem, pw = subprocess.check_output(
                    ["nvidia-smi", "-i", dev_index,
                     "--query-gpu=utilization.gpu,memory.used,power.draw",
                     "--format=csv,noheader,nounits"],
                    text=True, timeout=5).strip().split(", ")
                dsteps = step - last_log[0]
                dtime = max(time.time() - last_log[1], 1e-6)
                profile_rows.append(
                    {"step": step, "gpu_util": float(util),
                     "memory_mb": float(mem), "power_w": float(pw),
                     "steps_per_sec": round(dsteps / dtime, 2),
                     "meshes_per_sec": round(dsteps * group_size / dtime, 2)})
                last_log[:] = [step, time.time()]
            except Exception:
                pass
            print(f"step {step}/{steps} loss {l:.4f} ema {loss_ema:.4f} "
                  f"({q_tag}) ({time.time()-t0:.0f}s)", flush=True)
        if step % int(cfg["ckpt_every"]) == 0 or step == steps:
            torch.save({"conditioner": conditioner.state_dict(),
                        "dit": dit.state_dict(), "config": cfg,
                        "samples": ids, "global_step": step,
                        "loss_ema": loss_ema, "opt": opt.state_dict(),
                        "rng_g": g.get_state(),
                        "rng_g_idx": g_idx.get_state(),
                        "dataset_manifest_sha256": manifest_sha,
                        "config_sha256": config_sha},
                       run / "ckpt.pt.tmp")
            os.replace(run / "ckpt.pt.tmp", run / "ckpt.pt")
            (run / "training_profile.json").write_text(json.dumps(
                {"group_size": group_size, "rows": profile_rows[-500:]},
                indent=1))
    if log_rows:
        with open(run / "train_log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            w.writeheader()
            w.writerows(log_rows)
    print(f"done: {run} | {n_params/1e6:.1f}M params | "
          f"{(time.time()-t0)/60:.1f} min | ema {loss_ema:.4f}")


if __name__ == "__main__":
    main()
