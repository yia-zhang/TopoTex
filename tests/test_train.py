# -*- coding: utf-8 -*-
"""Training path: model forward + gradient flow on real data, checkpoint
save/load/resume through the CLI."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT = Path(__file__).resolve().parents[1]
from topotex.paths import data_root  # noqa: E402

DATASET = data_root("dataset", PROJECT)
HAVE_DATA = (DATASET / "manifest.jsonl").exists()
HAVE_CUDA = torch.cuda.is_available()
PY = sys.executable
needs_data = pytest.mark.skipif(
    not (HAVE_DATA and HAVE_CUDA), reason="dataset or GPU unavailable"
)


@needs_data
def test_model_forward_and_gradient_flow():
    import yaml

    from topotex.data.dataset import TopoTexDataset
    from topotex.models import MaskedFlowMatching
    from topotex.models.topotex import build_models

    cfg = yaml.safe_load(
        open(PROJECT / "configs" / "topotex_fm_baseline.yaml")
    )
    ids = [
        json.loads(l)["sample_id"] for l in open(DATASET / "manifest.jsonl")
    ][:1]
    ds = TopoTexDataset(DATASET, ids, device="cuda:0")
    it = ds[0]
    conditioner, dit = build_models(cfg, "cuda:0")
    flow = MaskedFlowMatching(T=1000, device="cuda:0")
    q = it["uv_queries"][0]
    mask = q["valid_mask"].float()[None, None]
    x0 = (q["gt_texture"][None] * 2 - 1) * mask
    params = list(conditioner.parameters()) + list(dit.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)
    # AdaLN-Zero gates are zero-initialized, so block weights legitimately
    # get zero grad on step 1 — check connectivity AFTER the gates open.
    for step in range(3):
        out = conditioner(
            it["mesh"],
            it["mv_images"].float()[None] / 255,
            {
                "face_id": q["face_id"],
                "barycentric": q["barycentric"].permute(1, 2, 0),
                "graph": it["graph"],
            },
            with_rgb=True,
        )
        loss = (
            flow.loss(dit, x0, out["uv_condition"], mask)
            + 0.1
            * (out["uv_rgb"][0] - q["gt_texture"])
            .abs()[:, q["valid_mask"]]
            .mean()
        )
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    for name, model in (("conditioner", conditioner), ("dit", dit)):
        missing = [
            n
            for n, p in model.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert not missing, f"{name} params without gradients: {missing[:5]}"
    n_zero = sum(
        1 for p in params if p.grad is not None and p.grad.abs().sum() == 0
    )
    assert n_zero < 0.2 * len(params), (
        f"{n_zero}/{len(params)} params still have zero grads after warmup"
    )


@needs_data
def test_checkpoint_save_load_resume():
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    }
    run = data_root("runs", PROJECT) / "_test_resume"
    if run.exists():
        shutil.rmtree(run)
    base = [
        PY,
        "train.py",
        "--config",
        "configs/topotex_fm_baseline.yaml",
        "--samples",
        "1",
        "--run-name",
        "_test_resume",
    ]
    r1 = subprocess.run(
        base + ["--steps", "3"],
        cwd=PROJECT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert r1.returncode == 0, r1.stderr[-800:]
    ck = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=False)
    for k in (
        "conditioner",
        "dit",
        "opt",
        "rng_g",
        "rng_g_idx",
        "global_step",
        "loss_ema",
    ):
        assert k in ck, f"checkpoint missing {k}"
    assert ck["global_step"] == 3
    r2 = subprocess.run(
        base + ["--steps", "6", "--resume"],
        cwd=PROJECT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert r2.returncode == 0, r2.stderr[-800:]
    assert "resumed" in r2.stdout
    ck2 = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=False)
    assert ck2["global_step"] == 6
    shutil.rmtree(run)
