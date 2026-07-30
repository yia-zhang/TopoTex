# -*- coding: utf-8 -*-
"""Frozen numerical baseline, scoped to what the factorized UV query
encoder does NOT touch.

The golden artifacts (checkpoints/golden/) were captured before the
encoder swap. The decoder golden values (query tokens, uv_condition,
texture, FM loss) are intentionally NOT compared here — the factorized
dense encoder is a different architecture, validated by its own contract
tests and the controlled A/B experiment. Everything upstream of the
decoder (FaceTokenizer, image encoder, face-image cross attention,
topology transformer -> Z_F) is untouched and must still reproduce the
golden capture: bitwise for the deterministic tokenizer, within the
calibrated run-to-run CUDA envelope for Z_F.

Pre-factorized checkpoints must NOT load silently into the new
architecture: loading is required to fail fast (no silent partial load).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT = Path(__file__).resolve().parents[1]
from topotex.paths import data_root  # noqa: E402

GOLDEN = data_root("checkpoints", PROJECT) / "golden"
HAVE = (
    (GOLDEN / "golden.npz").exists()
    and (data_root("checkpoints", PROJECT) / "baseline/ckpt.pt").exists()
    and torch.cuda.is_available()
)

TOL = {"face_features": 0.0, "Z_F": 2.5e-4}


@pytest.mark.skipif(not HAVE, reason="golden artifacts/ckpt/GPU unavailable")
def test_upstream_matches_golden_and_prior_ckpt_fails_fast():
    from topotex import TopoTexDataset, TopoTexModel

    meta = json.loads((GOLDEN / "golden_meta.json").read_text())
    gold = np.load(GOLDEN / "golden.npz")
    ck = torch.load(
        data_root("checkpoints", PROJECT) / "baseline/ckpt.pt",
        map_location="cuda:0",
        weights_only=False,
    )

    # 1) no silent partial load: the pre-factorized checkpoint must be
    #    rejected loudly by the frozen loading path
    with pytest.raises(RuntimeError):
        TopoTexModel.from_checkpoint(
            data_root("checkpoints", PROJECT) / "baseline"
        )

    # 2) the untouched upstream (mesh+views -> Z_F) still reproduces the
    #    golden capture; only decoder.* weights may be absent
    model = TopoTexModel.from_config(ck["config"], "cuda:0").eval()
    upstream = {
        k: v
        for k, v in ck["conditioner"].items()
        if not k.startswith("decoder.")
    }
    missing, unexpected = model.conditioner.load_state_dict(
        upstream, strict=False
    )
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    assert all(k.startswith("decoder.") for k in missing), (
        f"non-decoder keys missing: "
        f"{[k for k in missing if not k.startswith('decoder.')][:5]}"
    )
    model.dit.load_state_dict(ck["dit"])

    it = TopoTexDataset(
        PROJECT / ck["config"]["dataset_root"],
        [meta["sample_id"]],
        device="cuda:0",
    ).items[0]
    feats = {}
    h = model.conditioner.tokenizer.register_forward_hook(
        lambda m, i, o: feats.__setitem__("f", o.detach())
    )
    Z = model.encode(it.mesh, it.mv_images, it.graph)
    h.remove()

    d_feat = np.abs(
        feats["f"].float().cpu().numpy().astype(np.float64)
        - gold["face_features"].astype(np.float64)
    ).max()
    assert d_feat <= TOL["face_features"], f"face_features {d_feat:.3e}"
    d_z = np.abs(
        Z.float().cpu().numpy().astype(np.float64)
        - gold["Z_F"].astype(np.float64)
    ).max()
    assert d_z <= TOL["Z_F"], f"Z_F {d_z:.3e} > {TOL['Z_F']:.1e}"
