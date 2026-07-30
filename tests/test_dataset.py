# -*- coding: utf-8 -*-
"""Dataset pipeline: GLB extraction gates, source schema, loader contract,
and partial-query invariants on built data."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topotex.data.builder import (
    SkipSample,
    check_sample_schema,
    extract_glb,
    sample_is_valid,
    uv_address,
)

PROJECT = Path(__file__).resolve().parents[1]
from topotex.paths import data_root  # noqa: E402

SOURCE = PROJECT / "output" / "topotex_source"
DATASET = data_root("dataset", PROJECT)
HAVE_SOURCE = (SOURCE / "manifest.jsonl").exists()
HAVE_DATASET = (DATASET / "manifest.jsonl").exists()


def _make_glb(tmp_path, two_textures=False, overlap=False):
    """Tiny textured GLB: a UV-mapped quad (two triangles)."""
    import trimesh
    from PIL import Image

    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float64)
    f = np.array([[0, 1, 2], [0, 2, 3]])
    if overlap:
        uv = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.8, 0.2]])
    else:
        uv = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
    tex = Image.fromarray(
        (np.random.default_rng(0).integers(0, 255, (64, 64, 3))).astype(
            np.uint8
        )
    )
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=tex)
    m = trimesh.Trimesh(
        v,
        f,
        visual=trimesh.visual.TextureVisuals(uv=uv, material=mat),
        process=False,
    )
    if two_textures:
        tex2 = Image.fromarray(np.full((32, 32, 3), 200, np.uint8))
        mat2 = trimesh.visual.material.PBRMaterial(baseColorTexture=tex2)
        m2 = trimesh.Trimesh(
            v + 2.0,
            f,
            visual=trimesh.visual.TextureVisuals(uv=uv, material=mat2),
            process=False,
        )
        scene = trimesh.Scene({"a": m, "b": m2})
    else:
        scene = trimesh.Scene({"a": m})
    p = tmp_path / "asset.glb"
    scene.export(p)
    return p


def test_glb_extraction_roundtrip(tmp_path):
    p = _make_glb(tmp_path)
    vertices, faces, uv, tex = extract_glb(p)
    assert vertices.dtype == np.float32 and faces.dtype == np.int32
    assert len(faces) == 2 and len(uv) == len(vertices)
    assert 0 <= uv.min() and uv.max() <= 1
    assert tex.ndim == 3 and tex.dtype == np.uint8


def test_multiple_textures_rejected(tmp_path):
    p = _make_glb(tmp_path, two_textures=True)
    with pytest.raises(SkipSample, match="multiple textures"):
        extract_glb(p)


def test_uv_overlap_rejected(tmp_path):
    p = _make_glb(tmp_path, overlap=True)
    vertices, faces, uv, tex = extract_glb(p)
    with pytest.raises(SkipSample, match="overlapping uv"):
        uv_address(uv, faces)


@pytest.mark.skipif(not HAVE_SOURCE, reason="source dataset not built")
def test_source_schema_and_view_order():
    sid = json.loads(open(SOURCE / "manifest.jsonl").readline())["sample_id"]
    d = SOURCE / "samples" / sid
    check_sample_schema(d)
    meta = json.loads((d / "meta.json").read_text())
    assert meta["view_order"] == "front,back,left,right,top,bottom"
    assert meta["generator"]["grid_to_ours"] == [0, 3, 1, 4, 2, 5]


@pytest.mark.skipif(not HAVE_SOURCE, reason="source dataset not built")
def test_resume_skips_valid_sample():
    sid = json.loads(open(SOURCE / "manifest.jsonl").readline())["sample_id"]
    assert sample_is_valid(SOURCE, sid)
    assert not sample_is_valid(SOURCE, "nonexistent_sample")


@pytest.mark.skipif(not HAVE_DATASET, reason="dataset not built")
def test_loader_contract():
    import torch

    from topotex.data.dataset import TopoTexDataset

    ids = [
        json.loads(l)["sample_id"] for l in open(DATASET / "manifest.jsonl")
    ][:2]
    ds = TopoTexDataset(DATASET, ids)
    it = ds[0]
    assert it["mv_images"].shape == (6, 3, 256, 256)
    assert it["mv_images"].dtype == torch.uint8
    names = [q["name"] for q in it["uv_queries"]]
    assert names == ["uv_000", "uv_001", "uv_002"]
    assert [q["name"] for q in it["test_uv_queries"]] == ["uv_test"]
    n_faces = len(it["mesh"]["faces"])
    for q in it["uv_queries"] + it["test_uv_queries"]:
        assert q["face_id"].dtype == torch.int64
        assert q["valid_mask"].dtype == torch.bool
        assert q["gt_texture"].shape == (3, 256, 256)
        v = q["valid_mask"]
        fid = q["face_id"][v]
        assert fid.min() >= 0 and fid.max() < n_faces
        assert (q["face_id"][~v] == -1).all()
        b = q["barycentric"].permute(1, 2, 0)[v]
        assert float((b.sum(-1) - 1).abs().max()) < 2e-3


@pytest.mark.skipif(not HAVE_DATASET, reason="dataset not built")
def test_partial_query_is_subset_of_canonical():
    """uv_002 addresses a face subset of the canonical layout: wherever both
    are valid the face ids must agree, and it must reference strictly fewer
    distinct faces than the full canonical query."""
    from topotex.data.dataset import TopoTexDataset

    ids = [
        json.loads(l)["sample_id"] for l in open(DATASET / "manifest.jsonl")
    ][:2]
    ds = TopoTexDataset(DATASET, ids)
    for it in ds.items:
        q0 = it["uv_queries"][0]
        qp = it["uv_queries"][2]
        shared = q0["valid_mask"] & qp["valid_mask"]
        assert bool(shared.any())
        assert bool((q0["face_id"][shared] == qp["face_id"][shared]).all())
        n0 = len(q0["face_id"][q0["valid_mask"]].unique())
        n_p = len(qp["face_id"][qp["valid_mask"]].unique())
        assert n_p < n0
