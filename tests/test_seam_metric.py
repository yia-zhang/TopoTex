# -*- coding: utf-8 -*-
"""UV seam consistency metric: a texture baked from one surface signal has
near-zero seam error; unrelated content across seams scores high."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.mesh_utils import seam_error

PROJECT = Path(__file__).resolve().parents[1]
DATASET = PROJECT / "output" / "topotex_dataset"
HAVE_DATA = (DATASET / "manifest.jsonl").exists()


def _quad_with_seam():
    """Two faces sharing one mesh edge whose UV images are far apart."""
    F = np.array([[0, 1, 2], [1, 3, 2]])
    uvv = np.array([[.05, .05], [.45, .05], [.05, .45],   # face 0 island
                    [.95, .55], [.55, .55], [.55, .95]])  # face 1 island
    uvf = np.array([[0, 1, 2], [4, 3, 5]])
    return F, uvv, uvf


def test_constant_texture_zero_seam():
    F, uvv, uvf = _quad_with_seam()
    tex = np.full((64, 64, 3), 127, np.uint8)
    r = seam_error(F, uvv, uvf, tex)
    assert r["n_seam_edges"] == 1
    assert r["seam_error"] < 1e-6


def test_mismatched_islands_score_high():
    F, uvv, uvf = _quad_with_seam()
    tex = np.zeros((64, 64, 3), np.uint8)
    tex[:, 32:] = 255                     # island B white, island A black
    r = seam_error(F, uvv, uvf, tex)
    assert r["seam_error"] > 1.0          # sqrt(3) for full RGB flip
    assert r["per_face_error"].max() > 1.0


@pytest.mark.skipif(not HAVE_DATA, reason="dataset not built")
def test_gt_texture_far_below_random():
    """The baked GT is one surface signal -> seams nearly invisible; random
    content scores an order of magnitude higher."""
    from datasets.dataset import TopoTexDataset
    ids = [json.loads(l)["sample_id"]
           for l in open(DATASET / "manifest.jsonl")][:2]
    checked = 0
    for it in TopoTexDataset(DATASET, ids).items:
        q = it["uv_queries"][1]           # alternative layout has the seams
        F = it["mesh"]["faces"].numpy()
        gt = (q["gt_texture"].permute(1, 2, 0).numpy() * 255
              ).astype(np.uint8)
        v = q["valid_mask"].numpy()
        r_gt = seam_error(F, q["uv_vertices"], q["uv_faces"], gt, v)
        if r_gt["n_seam_edges"] < 3:
            continue
        rnd = np.random.default_rng(0).integers(
            0, 255, gt.shape).astype(np.uint8)
        r_rnd = seam_error(F, q["uv_vertices"], q["uv_faces"], rnd, v)
        assert r_gt["seam_error"] < 0.1
        assert r_rnd["seam_error"] > 3 * r_gt["seam_error"]
        checked += 1
    assert checked >= 1
