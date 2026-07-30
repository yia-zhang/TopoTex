# -*- coding: utf-8 -*-
"""Frozen numerical baseline for the architecture refactor.

The golden artifacts (checkpoints/golden/) were captured with the
pre-refactor code on the frozen baseline checkpoint. The refactored
package must reproduce them: bitwise for the deterministic tokenizer,
within calibrated run-to-run CUDA-nondeterminism tolerances elsewhere
(scatter kernels in the topology transformer are nondeterministic; the
tolerances are 10x the measured same-code run-to-run deltas).
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT = Path(__file__).resolve().parents[1]
GOLDEN = PROJECT / "checkpoints" / "golden"
HAVE = (
    (GOLDEN / "golden.npz").exists()
    and (PROJECT / "checkpoints/baseline/ckpt.pt").exists()
    and torch.cuda.is_available()
)

TOL = {
    "face_features": 0.0,  # deterministic -> bitwise
    "Z_F": 2.5e-4,
    "query_tokens": 2.5e-5,
    "uv_condition": 3e-5,
    "sampled_texture": 2.5e-3,
}


@pytest.mark.skipif(not HAVE, reason="golden artifacts/ckpt/GPU unavailable")
def test_refactor_matches_golden_baseline():
    from topotex import TopoTexDataset, TopoTexPipeline

    meta = json.loads((GOLDEN / "golden_meta.json").read_text())
    gold = np.load(GOLDEN / "golden.npz")
    pipe = TopoTexPipeline.from_checkpoint(
        PROJECT / "checkpoints/baseline", "cuda:0"
    )
    ck = pipe.checkpoint
    # checkpoint compatibility: identical state-dict keys, no migration
    keys_sha = hashlib.sha256(
        json.dumps(
            [
                sorted(ck["conditioner"].keys()),
                sorted(ck["dit"].keys()),
            ]
        ).encode()
    ).hexdigest()
    assert keys_sha == meta["state_dict_keys_sha"]

    it = TopoTexDataset(
        PROJECT / ck["config"]["dataset_root"],
        [meta["sample_id"]],
        device="cuda:0",
    ).items[0]
    q = it.uv_queries[0]
    cond = pipe.model.conditioner

    feats, tok = {}, {}
    h1 = cond.tokenizer.register_forward_hook(
        lambda m, i, o: feats.__setitem__("f", o.detach())
    )
    Z = pipe.encode(it.mesh, it.mv_images, it.graph)
    h1.remove()
    h2 = cond.decoder.norm.register_forward_hook(
        lambda m, i, o: tok.__setitem__("q", o.detach())
    )
    out = pipe.model.condition(Z, q)
    h2.remove()

    g = torch.Generator(device="cuda:0").manual_seed(meta["seed"])
    x0 = (q.gt_texture[None] * 2 - 1) * q.valid_mask.float()[None, None]
    with torch.no_grad():
        loss = pipe.model.schedule.loss(
            pipe.model.dit,
            x0,
            out.uv_condition,
            q.valid_mask.float()[None, None],
            t=torch.tensor([meta["t_fix"]], device="cuda:0"),
            generator=g,
        )
    tex = pipe.model.generate(
        out.uv_condition, q.valid_mask, num_steps=50, seed=meta["seed"]
    )

    new = {
        "face_features": feats["f"].float().cpu().numpy(),
        "Z_F": Z.float().cpu().numpy(),
        "query_tokens": tok["q"].float().cpu().numpy(),
        "uv_condition": out.uv_condition.float().cpu().numpy(),
        "sampled_texture": tex.float().cpu().numpy(),
    }
    for k, arr in new.items():
        d = np.abs(arr.astype(np.float64) - gold[k].astype(np.float64)).max()
        assert d <= TOL[k] or (
            hashlib.sha256(arr.tobytes()).hexdigest() == meta["sha"][k]
        ), f"{k}: maxdiff {d:.3e} > tol {TOL[k]:.1e}"
    rel = abs(float(loss) - meta["fm_loss"]) / meta["fm_loss"]
    assert rel < 1e-6, f"fm loss rel diff {rel:.2e}"
