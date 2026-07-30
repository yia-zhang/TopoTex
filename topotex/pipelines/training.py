# -*- coding: utf-8 -*-
"""TOPOTEX training pipeline (frozen baseline recipe).

CLI wrapper: the repository-root ``train.py``.

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
import torch.distributed as dist

from topotex.data.dataset import TopoTexDataset
from topotex.layers.flow import MaskedFlowMatching
from topotex.layers.topology import build_face_graph
from topotex.models.topotex import build_models
from topotex.paths import data_root
from topotex.utils.distributed import ddp_env

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def to_dev(t, device):
    return (
        t.to(device, non_blocking=True)
        if t.device.type != device.split(":")[0]
        else t
    )


def bucket_group_ids(root, ids, k):
    """Face-count-bucketed groups of ids (meta.json num_faces — identical
    key to the loaded data). Deterministic across ranks."""
    nf = {}
    for sid in ids:
        meta = json.loads((root / "samples" / sid / "meta.json").read_text())
        nf[sid] = meta["num_faces"]
    order = sorted(ids, key=lambda s: (nf[s], s))
    return [order[i : i + k] for i in range(0, len(order), k)]


def build_groups(ds, k, device, topo_pe=None):
    """Face-count-bucketed fixed groups + one packed graph per group.
    Per-mesh unit-area normalization + rel rescale keep the tokenizer's
    intrinsic features numerically identical to per-mesh encoding."""
    order = sorted(range(len(ds)), key=lambda i: len(ds[i]["mesh"]["faces"]))
    groups = []
    for i in range(0, len(order), k):
        idxs = order[i : i + k]
        Vs, Fs, voff = [], [], 0
        for j in idxs:
            V = ds[j]["mesh"]["vertices"].to(device)
            F = ds[j]["mesh"]["faces"].to(device)
            tri = V[F]
            area = (
                torch.cross(
                    tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1
                )
                .norm(dim=1)
                .sum()
                / 2
            )
            Vs.append(V / area.clamp(min=1e-20).sqrt())
            Fs.append(F + voff)
            voff += len(V)
        Vp, Fp = torch.cat(Vs), torch.cat(Fs)
        graph = build_face_graph(Vp, Fp)
        graph["rel"][:, 0] *= graph["global_scale"]
        graph["global_scale"] = torch.tensor(1.0, device=device)
        # the random-walk PE is deterministic per graph — computing it once
        # here removes ~60% of the per-step forward cost
        pe = topo_pe(graph, len(Fp)) if topo_pe is not None else None
        groups.append(
            {
                "idxs": idxs,
                "Vp": Vp,
                "Fp": Fp,
                "graph": graph,
                "pe": pe,
                "n_faces": [len(ds[j]["mesh"]["faces"]) for j in idxs],
            }
        )
    return groups


class PackedLoss(torch.nn.Module):
    """One packed-group training forward: conditioner pathway + generator
    loss in a single module, so DDP gradient hooks cover the whole step."""

    def __init__(self, conditioner, dit, sched, aux_w, bs, use_bf16):
        super().__init__()
        self.conditioner = conditioner
        self.dit = dit
        self.sched = sched
        self.aux_w, self.bs, self.use_bf16 = aux_w, bs, use_bf16

    def forward(self, grp, queries, imgs, t, g):
        c = self.conditioner
        gph = grp["graph"]
        pe = (
            grp["pe"]
            if grp.get("pe") is not None
            else c.topo_pe(gph, len(grp["Fp"]))
        )
        x = c.tokenizer(grp["Vp"], grp["Fp"], gph, pe)
        img_tokens = c.image_encoder(imgs)
        outs, fo = [], 0
        for i, n in enumerate(grp["n_faces"]):
            outs.append(
                c.cross(
                    x[fo : fo + n].unsqueeze(0), img_tokens[i : i + 1]
                ).squeeze(0)
            )
            fo += n
        x = c.topo(torch.cat(outs), gph)
        conds, aux, fo = [], 0.0, 0
        for q, n in zip(queries, grp["n_faces"]):
            cd, rgb = c.decoder(
                x[fo : fo + n],
                q["face_id"],
                q["barycentric"].permute(1, 2, 0),
                with_rgb=self.aux_w > 0,
            )
            conds.append(cd.unsqueeze(0))
            if self.aux_w > 0:
                aux = (
                    aux
                    + (rgb - q["gt_texture"]).abs()[:, q["valid_mask"]].mean()
                )
            fo += n
        x0s, ms, cs = [], [], []
        for q, cd in zip(queries, conds):
            m = q["valid_mask"].float()[None, None]
            x0s.append(
                ((q["gt_texture"][None] * 2 - 1) * m).expand(
                    self.bs, -1, -1, -1
                )
            )
            ms.append(m.expand(self.bs, -1, -1, -1))
            cs.append(cd.expand(self.bs, -1, -1, -1))
        with torch.autocast("cuda", torch.bfloat16, enabled=self.use_bf16):
            loss_diff = self.sched.loss(
                self.dit,
                torch.cat(x0s),
                torch.cat(cs),
                torch.cat(ms),
                t=t,
                generator=g,
            )
        loss_diff = loss_diff.float()
        aux_t = aux / max(len(queries), 1) if self.aux_w > 0 else None
        loss = loss_diff + (self.aux_w * aux_t if aux_t is not None else 0.0)
        return (
            loss,
            loss_diff.detach(),
            aux_t.detach() if aux_t is not None else None,
        )


def main():
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/topotex_fm_baseline.yaml")
    ap.add_argument("--samples", default="265")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--ids-file",
        default=None,
        help="json file with {'train': [...]} or a plain list — "
        "overrides manifest ordering (mesh-level splits)",
    )
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.seed is not None:
        cfg["seed"] = args.seed
    rank, world, local_rank = ddp_env()
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    is_main = rank == 0
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    query_probs = [float(x) for x in cfg["query_probs"]]
    assert abs(sum(query_probs) - 1) < 1e-6

    root = data_root("dataset", PROJECT_ROOT, cfg["dataset_root"])
    if args.ids_file:
        split_bytes = Path(args.ids_file).read_bytes()
        blob = json.loads(split_bytes)
        ids = blob["train"] if isinstance(blob, dict) else blob
        # split provenance: stamped into every checkpoint config
        cfg["split_file"] = str(args.ids_file)
        cfg["split_sha256"] = hashlib.sha256(split_bytes).hexdigest()
    else:
        ids = [
            json.loads(l)["sample_id"] for l in open(root / "manifest.jsonl")
        ]
    ids = ids[: int(args.samples)]
    group_size = max(1, min(int(cfg.get("group_size", 1)), len(ids)))
    cfg["group_size"] = group_size
    cfg["world_size"] = world
    use_bf16 = cfg.get("precision") == "bf16"
    steps = args.steps or math.ceil(
        len(ids) * int(cfg["target_mesh_exposures"]) / (group_size * world)
    )
    cfg["steps"] = steps
    run = data_root("runs", PROJECT_ROOT) / args.run_name
    run.mkdir(parents=True, exist_ok=True)

    cfg.setdefault("generator", "fm")  # flow matching, the only generator
    # packed mode stores data on CPU past 300 meshes (moved per step) and
    # skips per-item graphs (group graphs are packed at startup instead).
    # Under DDP each rank owns a disjoint shard of the bucketed groups and
    # loads only those meshes.
    if world > 1:
        assert group_size > 1, "DDP path uses the packed-group trainer"
        gid_all = bucket_group_ids(root, ids, group_size)
        my_gids = gid_all[rank::world]
        my_ids = [s for g in my_gids for s in g]
    else:
        my_gids, my_ids = None, ids
    data_dev = device if (group_size == 1 or len(my_ids) <= 300) else "cpu"
    ds = TopoTexDataset(
        root, my_ids, device=data_dev, build_graphs=(group_size == 1)
    )
    conditioner, dit = build_models(cfg, device)
    # UV query encoder provenance (stamped into every checkpoint config)
    cfg["uv_query_encoder"] = "factorized_dense"
    cfg["uv_texel_dim"] = int(conditioner.decoder.texel_dim)
    cfg["patch_size"] = int(cfg["patch"])
    cfg["query_heads"] = int(cfg["cond_heads"])
    if is_main:
        print(
            f"uv query encoder: factorized_dense "
            f"Dq={cfg['uv_texel_dim']} (D={cfg['cond_dim']})"
        )
    flow = MaskedFlowMatching(T=int(cfg["T"]), device=device)
    groups = (
        build_groups(ds, group_size, device, topo_pe=conditioner.topo_pe)
        if group_size > 1
        else None
    )
    step_mod = PackedLoss(
        conditioner,
        dit,
        flow,
        float(cfg["aux_rgb_weight"]),
        int(cfg["noise_batch"]),
        cfg.get("precision") == "bf16",
    )
    if world > 1:
        step_mod = torch.nn.parallel.DistributedDataParallel(
            step_mod, device_ids=[local_rank]
        )
    params = list(conditioner.parameters()) + list(dit.parameters())
    opt = torch.optim.AdamW(
        params, lr=float(cfg["lr"]), weight_decay=0.0, betas=(0.9, 0.95)
    )
    n_params = sum(p.numel() for p in params)
    bs = int(cfg["noise_batch"])
    aux_w = float(cfg["aux_rgb_weight"])
    t_high_frac = float(cfg["t_high_frac"])
    t_high_min = int(cfg["t_high_min"])
    lr_final = float(cfg["lr_final_frac"])
    warmup = int(cfg["warmup"])
    try:
        commit = (
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(PROJECT_ROOT),
                    "rev-parse",
                    "--short",
                    "HEAD",
                ],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        commit = None
    manifest_sha = hashlib.sha256(
        (root / "manifest.jsonl").read_bytes()
    ).hexdigest()
    config_sha = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    if is_main:
        (run / "provenance.json").write_text(
            json.dumps(
                {
                    "config": cfg,
                    "samples": ids,
                    "n_params": n_params,
                    "git_commit": commit,
                    "query_probs": query_probs,
                    "dataset_manifest_sha256": manifest_sha,
                    "config_sha256": config_sha,
                },
                indent=1,
            )
        )

    g = torch.Generator(device=device).manual_seed(seed + 1000 * rank)
    g_idx = torch.Generator().manual_seed(seed + 1 + 1000 * rank)
    # group-index generator is SHARED across ranks: shards are stride-slices
    # of the size-sorted group list, so a common index keeps every rank in
    # the same size class per step (otherwise the DDP sync waits on the
    # largest random draw and throughput collapses to the straggler)
    g_grp = torch.Generator().manual_seed(seed + 7)
    start_step = 0
    loss_ema = None
    if args.resume and (run / "ckpt.pt").exists():
        ck = torch.load(
            run / "ckpt.pt", map_location=device, weights_only=False
        )
        assert ck.get("world_size", 1) == world, (
            f"checkpoint world_size {ck.get('world_size', 1)} != {world}"
        )
        conditioner.load_state_dict(ck["conditioner"])
        dit.load_state_dict(ck["dit"])
        opt.load_state_dict(ck["opt"])
        if world > 1 and "rng_all" in ck:
            g.set_state(ck["rng_all"][rank]["g"].cpu())
            g_idx.set_state(ck["rng_all"][rank]["g_idx"].cpu())
        else:
            g.set_state(ck["rng_g"].cpu())
            g_idx.set_state(ck["rng_g_idx"].cpu())
        if "rng_g_grp" in ck:
            g_grp.set_state(ck["rng_g_grp"].cpu())
        start_step = ck["global_step"]
        loss_ema = ck["loss_ema"]
        if is_main:
            print(f"resumed {run} at step {start_step}", flush=True)
    metrics_f = open(run / "metrics.jsonl", "a") if is_main else None
    profile_rows = []
    dev_index = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    last_log = [0, time.time()]
    log_rows, t0 = [], time.time()
    conditioner.train()
    dit.train()
    for step in range(start_step + 1, steps + 1):
        wu = min(1.0, step / max(warmup, 1))
        cos = lr_final + (1 - lr_final) * 0.5 * (
            1 + math.cos(math.pi * step / steps)
        )
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

        do_profile = step % int(cfg["log_every"]) == 0 or step == 1

        def _tick():
            if do_profile:
                torch.cuda.synchronize()
                return time.time()
            return 0.0

        t_p0 = _tick()
        if group_size > 1:
            k = int(torch.randint(0, 1 << 30, (1,), generator=g_grp))
            grp = groups[k % len(groups)]
            queries = []
            for j in grp["idxs"]:
                q0 = pick_query(ds[j])
                queries.append(
                    {
                        k: (to_dev(v, device) if torch.is_tensor(v) else v)
                        for k, v in q0.items()
                    }
                )
            imgs = (
                torch.stack(
                    [to_dev(ds[j]["mv_images"], device) for j in grp["idxs"]]
                ).float()
                / 255
            )
            B = int(cfg["noise_batch"]) * len(queries)
            t = torch.randint(1, flow.T + 1, (B,), device=device, generator=g)
            n_high = int(round(B * t_high_frac))
            if n_high:
                t[:n_high] = torch.randint(
                    t_high_min,
                    flow.T + 1,
                    (n_high,),
                    device=device,
                    generator=g,
                )
            t_p1 = _tick()
            loss, loss_diff, aux = step_mod(grp, queries, imgs, t, g)
            t_p2 = _tick()
            q_tag = f"{queries[0]['name']}(+{len(queries) - 1})"
        else:
            it = ds[int(torch.randint(0, len(ds), (1,), generator=g_idx))]
            q = pick_query(it)
            q_tag = q["name"]
            mv = it["mv_images"].float()[None] / 255
            out = conditioner(
                it["mesh"],
                mv,
                {
                    "face_id": q["face_id"],
                    "barycentric": q["barycentric"].permute(1, 2, 0),
                    "graph": it["graph"],
                },
                with_rgb=aux_w > 0,
            )
            cond = out["uv_condition"]
            mask = q["valid_mask"].float()[None, None]
            x0 = (q["gt_texture"][None] * 2 - 1) * mask
            t = torch.randint(1, flow.T + 1, (bs,), device=device, generator=g)
            n_high = int(round(bs * t_high_frac))
            if n_high:
                t[:n_high] = torch.randint(
                    t_high_min,
                    flow.T + 1,
                    (n_high,),
                    device=device,
                    generator=g,
                )
            with torch.autocast("cuda", torch.bfloat16, enabled=use_bf16):
                loss_diff = flow.loss(
                    dit,
                    x0.expand(bs, -1, -1, -1),
                    cond.expand(bs, -1, -1, -1),
                    mask.expand(bs, -1, -1, -1),
                    t=t,
                    generator=g,
                )
            loss_diff = loss_diff.float()
            loss = loss_diff
            aux = None
            if aux_w > 0:
                aux = (
                    (out["uv_rgb"][0] - q["gt_texture"])
                    .abs()[:, q["valid_mask"]]
                    .mean()
                )
                loss = loss + aux_w * aux
        if group_size == 1:
            t_p1 = t_p2 = _tick()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        t_p3 = _tick()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        t_p4 = _tick()

        l = float(loss)
        if not np.isfinite(l):
            raise RuntimeError(f"NaN/Inf loss at step {step}")
        loss_ema = l if loss_ema is None else 0.99 * loss_ema + 0.01 * l
        if is_main and (step % int(cfg["log_every"]) == 0 or step == 1):
            row = {
                "step": step,
                "loss": l,
                "loss_ema": loss_ema,
                "loss_diff": float(loss_diff),
                "aux_rgb": None if aux is None else float(aux),
                "uv_query": q_tag,
                "lr": float(opt.param_groups[0]["lr"]),
                "elapsed_s": round(time.time() - t0, 1),
            }
            log_rows.append(row)
            metrics_f.write(json.dumps(row) + "\n")
            metrics_f.flush()
            try:
                util, mem, pw = (
                    subprocess.check_output(
                        [
                            "nvidia-smi",
                            "-i",
                            dev_index,
                            "--query-gpu=utilization.gpu,memory.used,power.draw",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        timeout=5,
                    )
                    .strip()
                    .split(", ")
                )
                dsteps = step - last_log[0]
                dtime = max(time.time() - last_log[1], 1e-6)
                profile_rows.append(
                    {
                        "step": step,
                        "gpu_util": float(util),
                        "memory_mb": float(mem),
                        "power_w": float(pw),
                        "steps_per_sec": round(dsteps / dtime, 2),
                        "meshes_per_sec": round(
                            dsteps * group_size / dtime, 2
                        ),
                        "data_ms": round((t_p1 - t_p0) * 1000, 1),
                        "forward_ms": round((t_p2 - t_p1) * 1000, 1),
                        "backward_comm_ms": round((t_p3 - t_p2) * 1000, 1),
                        "optimizer_ms": round((t_p4 - t_p3) * 1000, 1),
                    }
                )
                last_log[:] = [step, time.time()]
            except Exception as e:
                if step == 1:
                    print(f"[profile] sampling failed: {e!r}", flush=True)
            print(
                f"step {step}/{steps} loss {l:.4f} ema {loss_ema:.4f} "
                f"({q_tag}) ({time.time() - t0:.0f}s)",
                flush=True,
            )
        if step % int(cfg["ckpt_every"]) == 0 or step == steps:
            my_rng = {
                "g": g.get_state().cpu(),
                "g_idx": g_idx.get_state().cpu(),
            }
            rng_grp = g_grp.get_state().cpu()
            if world > 1:
                rng_all = [None] * world
                dist.all_gather_object(rng_all, my_rng)
            else:
                rng_all = [my_rng]
            if is_main:
                torch.save(
                    {
                        "conditioner": conditioner.state_dict(),
                        "dit": dit.state_dict(),
                        "config": cfg,
                        "samples": ids,
                        "global_step": step,
                        "loss_ema": loss_ema,
                        "opt": opt.state_dict(),
                        "rng_g": rng_all[0]["g"],
                        "rng_g_idx": rng_all[0]["g_idx"],
                        "rng_all": rng_all,
                        "world_size": world,
                        "rng_g_grp": rng_grp,
                        "dataset_manifest_sha256": manifest_sha,
                        "config_sha256": config_sha,
                    },
                    run / "ckpt.pt.tmp",
                )
                os.replace(run / "ckpt.pt.tmp", run / "ckpt.pt")
                (run / "training_profile.json").write_text(
                    json.dumps(
                        {
                            "group_size": group_size,
                            "world_size": world,
                            "rows": profile_rows[-500:],
                        },
                        indent=1,
                    )
                )
            if world > 1:
                dist.barrier()
    if is_main and log_rows:
        with open(run / "train_log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            w.writeheader()
            w.writerows(log_rows)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if is_main:
        print(
            f"done: {run} | {n_params / 1e6:.1f}M params | "
            f"{(time.time() - t0) / 60:.1f} min | ema {loss_ema:.4f}"
        )


if __name__ == "__main__":
    main()
