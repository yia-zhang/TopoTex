# -*- coding: utf-8 -*-
"""Training-throughput benchmark over the FROZEN model (no model changes).

python scripts/benchmark_training.py [--samples 10] [--steps 600]
    [--group 4] [--out benchmark.json]

Variants (identical model / recipe semantics, different execution):
  A fp32      the trainer's step as-is: one mesh, noise_batch DiT samples
  B bf16      A + torch.autocast(bfloat16) around forward/loss
  C bucket    face-count-bucketed groups of --group meshes: conditioner
              looped per mesh, ONE batched DiT loss over the group
  D packed    C + the group's meshes packed into a single face graph so
              tokenizer + topology transformer run once per group
              (per-mesh unit-area normalization + graph rel rescale keep
              the intrinsic features numerically identical; face-view
              cross attention stays per mesh — each mesh has its own views)

Records per variant: meshes/sec, steps/sec, peak GPU memory, mean GPU
power (nvidia-smi sampling), loss EMA after the measured steps.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from datasets.dataset import TopoTexDataset               # noqa: E402
from models.surface_conditioner import build_face_graph   # noqa: E402
from models.texture_generator import MaskedDiffusion      # noqa: E402
from train import build_models                            # noqa: E402

DEV = "cuda:0"


class PowerMeter:
    """Samples nvidia-smi power.draw in a subprocess (no threads)."""

    def __init__(self, period_ms=500):
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,"
             "nounits", "-lms", str(period_ms), "-i",
             torch.cuda.current_device().__str__()],
            stdout=subprocess.PIPE, text=True)

    def stop(self):
        self.proc.terminate()
        out, _ = self.proc.communicate(timeout=10)
        vals = [float(x) for x in out.split() if x.strip()]
        return round(sum(vals) / max(len(vals), 1), 1)


def step_single(models, diffusion, cfg, it, q, g, autocast):
    """autocast (bf16) covers the DiT loss only — the graph pathway uses
    index_add/index_reduce kernels that require uniform fp32, and the model
    is frozen; the DiT dominates step time so that is where bf16 pays."""
    conditioner, dit, opt, params = models
    bs = int(cfg["noise_batch"])
    out = conditioner(it["mesh"], it["mv_images"].float()[None] / 255,
                      {"face_id": q["face_id"],
                       "barycentric": q["barycentric"].permute(1, 2, 0),
                       "graph": it["graph"]}, with_rgb=True)
    cond = out["uv_condition"]
    mask = q["valid_mask"].float()[None, None]
    x0 = (q["gt_texture"][None] * 2 - 1) * mask
    with torch.autocast("cuda", torch.bfloat16, enabled=autocast):
        loss = diffusion.loss(dit, x0.expand(bs, -1, -1, -1),
                              cond.expand(bs, -1, -1, -1),
                              mask.expand(bs, -1, -1, -1), generator=g)
    loss = loss.float() + 0.1 * (out["uv_rgb"][0]
                                 - q["gt_texture"]).abs()[
        :, q["valid_mask"]].mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    return float(loss)


def group_cond(conditioner, items, packed):
    """Per-mesh uv_condition for a group. packed=True runs tokenizer+topology
    once over a single disconnected graph (numerically identical features)."""
    conds, rgbs = [], []
    if not packed:
        for it in items:
            q = it["uv_queries"][0]
            out = conditioner(it["mesh"], it["mv_images"].float()[None] / 255,
                              {"face_id": q["face_id"],
                               "barycentric": q["barycentric"].permute(1, 2, 0),
                               "graph": it["graph"]}, with_rgb=True)
            conds.append(out["uv_condition"])
            rgbs.append(out["uv_rgb"])
        return conds, rgbs
    key = "_packed"
    cache = items[0].setdefault(key, {})
    gid = tuple(it["sample_id"] for it in items)
    if gid not in cache:
        Vs, Fs, voff = [], [], 0
        for it in items:
            V, F = it["mesh"]["vertices"], it["mesh"]["faces"]
            tri = V[F]
            area = (torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0],
                                dim=1).norm(dim=1).sum() / 2)
            Vs.append(V / area.clamp(min=1e-20).sqrt())   # unit-area mesh
            Fs.append(F + voff)
            voff += len(V)
        Vp, Fp = torch.cat(Vs), torch.cat(Fs)
        gph = build_face_graph(Vp, Fp)
        gph["rel"][:, 0] *= gph["global_scale"]     # back to per-mesh scale
        gph["global_scale"] = torch.tensor(1.0, device=Vp.device)
        cache[gid] = (Vp, Fp, gph)
    Vp, Fp, gph = cache[gid]
    pe = conditioner.topo_pe(gph, len(Fp))
    x = conditioner.tokenizer(Vp, Fp, gph, pe)
    outs, fo = [], 0
    for it in items:                               # per-mesh views
        n = len(it["mesh"]["faces"])
        img = conditioner.image_encoder(it["mv_images"].float()[None] / 255)
        outs.append(conditioner.cross(x[fo:fo + n].unsqueeze(0),
                                      img).squeeze(0))
        fo += n
    x = conditioner.topo(torch.cat(outs), gph)
    fo = 0
    for it in items:
        n = len(it["mesh"]["faces"])
        q = it["uv_queries"][0]
        c, rgb = conditioner.decoder(x[fo:fo + n], q["face_id"],
                                     q["barycentric"].permute(1, 2, 0),
                                     with_rgb=True)
        conds.append(c.unsqueeze(0))
        rgbs.append(rgb.unsqueeze(0))
        fo += n
    return conds, rgbs


def step_group(models, diffusion, cfg, items, g, packed):
    conditioner, dit, opt, params = models
    bs = int(cfg["noise_batch"])
    conds, rgbs = group_cond(conditioner, items, packed)
    x0s, masks, cs, aux = [], [], [], 0.0
    for it, c, rgb in zip(items, conds, rgbs):
        q = it["uv_queries"][0]
        m = q["valid_mask"].float()[None, None]
        x0s.append(((q["gt_texture"][None] * 2 - 1) * m).expand(bs, -1, -1, -1))
        masks.append(m.expand(bs, -1, -1, -1))
        cs.append(c.expand(bs, -1, -1, -1))
        aux = aux + (rgb[0] - q["gt_texture"]).abs()[:, q["valid_mask"]].mean()
    loss = diffusion.loss(dit, torch.cat(x0s), torch.cat(cs),
                          torch.cat(masks), generator=g) \
        + 0.1 * aux / len(items)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    return float(loss)


def run_variant(name, cfg, ds, steps, group, seed=20260727):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    conditioner, dit = build_models(cfg, DEV)
    params = list(conditioner.parameters()) + list(dit.parameters())
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=0.0,
                            betas=(0.9, 0.95))
    models = (conditioner, dit, opt, params)
    diffusion = MaskedDiffusion(T=int(cfg["T"]), device=DEV)
    g = torch.Generator(device=DEV).manual_seed(seed)
    g_idx = torch.Generator().manual_seed(seed + 1)
    # face-count buckets -> fixed groups (used by C/D)
    order = sorted(range(len(ds)), key=lambda i: len(ds[i]["mesh"]["faces"]))
    groups = [[ds[j] for j in order[i:i + group]]
              for i in range(0, len(order) - group + 1, group)]
    ema, n_meshes, t0 = None, 0, None
    warmup = max(20, steps // 10)
    meter = None
    for s in range(steps):
        if s == warmup:
            torch.cuda.synchronize()
            t0 = time.time()
            n_meshes = 0
            meter = PowerMeter()
        if name in ("A_fp32", "B_bf16"):
            it = ds[int(torch.randint(0, len(ds), (1,), generator=g_idx))]
            l = step_single(models, diffusion, cfg, it, it["uv_queries"][0],
                            g, autocast=(name == "B_bf16"))
            n_meshes += 1
        else:
            grp = groups[int(torch.randint(0, len(groups), (1,),
                                           generator=g_idx))]
            l = step_group(models, diffusion, cfg, grp, g,
                           packed=(name == "D_packed"))
            n_meshes += len(grp)
        ema = l if ema is None else 0.99 * ema + 0.01 * l
    torch.cuda.synchronize()
    dt = time.time() - t0
    power = meter.stop()
    res = {"variant": name,
           "measured_steps": steps - warmup,
           "steps_per_sec": round((steps - warmup) / dt, 2),
           "meshes_per_sec": round(n_meshes / dt, 2),
           "peak_mem_gb": round(
               torch.cuda.max_memory_allocated() / 2 ** 30, 2),
           "mean_power_w": power,
           "loss_ema": round(ema, 4)}
    print(json.dumps(res), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--out", default="benchmark.json")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(PROJECT / "configs" / "topotex_fm_baseline.yaml"))
    root = PROJECT / cfg["dataset_root"]
    ids = [json.loads(l)["sample_id"]
           for l in open(root / "manifest.jsonl")][: args.samples]
    ds = TopoTexDataset(root, ids, device=DEV)
    results = [run_variant(v, cfg, ds, args.steps, args.group)
               for v in ("A_fp32", "B_bf16", "C_bucket", "D_packed")]
    Path(args.out).write_text(json.dumps(
        {"config": {"samples": args.samples, "steps": args.steps,
                    "group": args.group,
                    "gpu": torch.cuda.get_device_name(0)},
         "results": results}, indent=1))
    print("done:", args.out)


if __name__ == "__main__":
    main()
