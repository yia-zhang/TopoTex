# -*- coding: utf-8 -*-
"""The integrity gate must catch every corruption class it claims to:
dangling symlinks at depth, unreadable required files, manifest/sample
mismatch, missing provenance, and deep hash drift. A gate that passes
corrupt data is worse than no gate."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topotex.data.integrity import run_gate, sha256_bytes

SCHEMA = "topotex_dataset@1"
QUERIES = ("uv_000", "uv_001", "uv_002", "uv_test")


def make_dataset(root, n=2):
    """Minimal but complete synthetic dataset matching the real layout."""
    from PIL import Image
    from safetensors.numpy import save_file

    src = root / "src"
    ds = root / "ds"
    ids = [f"{i:032x}" for i in range(n)]
    for sid in ids:
        s = src / "samples" / sid
        s.mkdir(parents=True)
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(
            s / "gt_texture.png"
        )
        d = ds / "samples" / sid
        (d / "uv_queries").mkdir(parents=True)
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(
            d / "reference.png"
        )
        save_file({"x": np.zeros(3, np.float32)}, str(d / "mv.safetensors"))
        save_file(
            {
                "vertices": np.zeros((3, 3), np.float32),
                "faces": np.zeros((1, 3), np.int64),
            },
            str(d / "mesh.safetensors"),
        )
        queries = []
        for q in QUERIES:
            qd = d / "uv_queries" / q
            qd.mkdir()
            fid = np.full((8, 8), -1, np.int32)
            bary = np.zeros((3, 8, 8), np.float16)
            save_file(
                {
                    "uv_vertices": np.zeros((3, 2), np.float32),
                    "uv_faces": np.zeros((1, 3), np.int32),
                    "face_id": fid,
                    "barycentric": bary,
                    "valid_mask": np.zeros((8, 8), np.uint8),
                },
                str(qd / "uv_address.safetensors"),
            )
            Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(
                qd / "gt_texture.png"
            )
            queries.append(
                {
                    "name": q,
                    "type": "canonical",
                    "held_out": q == "uv_test",
                    "valid_texels": 0,
                    "face_id_sha256": sha256_bytes(fid.tobytes()),
                    "barycentric_sha256": sha256_bytes(bary.tobytes()),
                }
            )
        (d / "meta.json").write_text(
            json.dumps(
                {
                    "sample_id": sid,
                    "schema": SCHEMA,
                    "num_faces": 1,
                    "uv_queries": queries,
                    "query_schema_version": SCHEMA,
                    "query_builder_commit": "test000",
                    "source_texture_sha256": sha256_bytes(
                        (s / "gt_texture.png").read_bytes()
                    ),
                }
            )
        )
    with open(ds / "manifest.jsonl", "w") as f:
        for sid in ids:
            f.write(json.dumps({"sample_id": sid}) + "\n")
    return ds, src, ids


def gate(ds, src, **kw):
    return run_gate(ds, kind="dataset", workers=2, source_root=src, **kw)


def test_clean_dataset_passes(tmp_path):
    ds, src, _ = make_dataset(tmp_path)
    rep = gate(ds, src, deep=True)
    assert rep["n_failures"] == 0, rep["failures"]


def test_dangling_symlink_at_depth_fails(tmp_path):
    """The corruption class from the real incident: a symlinked query
    directory whose target is gone — two levels below the sample dir."""
    ds, src, ids = make_dataset(tmp_path)
    qd = ds / "samples" / ids[0] / "uv_queries" / "uv_001"
    target = tmp_path / "elsewhere"
    target.mkdir()
    import shutil

    shutil.move(str(qd), str(target / "uv_001"))
    os.symlink(target / "uv_001", qd)
    assert gate(ds, src)["n_failures"] == 0  # valid link passes
    shutil.rmtree(target)
    rep = gate(ds, src)
    assert rep["n_failures"] > 0
    assert any("dangling" in f or "missing" in f for f in rep["failures"])


def test_unreadable_required_file_fails(tmp_path):
    ds, src, ids = make_dataset(tmp_path)
    p = (
        ds
        / "samples"
        / ids[1]
        / "uv_queries"
        / "uv_000"
        / "uv_address.safetensors"
    )
    p.write_bytes(b"not a safetensors file")
    rep = gate(ds, src)
    assert any("unreadable" in f for f in rep["failures"])


def test_manifest_sample_mismatch_fails(tmp_path):
    ds, src, ids = make_dataset(tmp_path)
    with open(ds / "manifest.jsonl", "a") as f:
        f.write(json.dumps({"sample_id": "f" * 32}) + "\n")
    rep = gate(ds, src)
    assert any("missing" in f for f in rep["failures"])


def test_provenance_fields_required(tmp_path):
    ds, src, ids = make_dataset(tmp_path)
    mf = ds / "samples" / ids[0] / "meta.json"
    meta = json.loads(mf.read_text())
    del meta["query_builder_commit"]
    meta["query_schema_version"] = "something@9"
    mf.write_text(json.dumps(meta))
    rep = gate(ds, src)
    assert any("query_schema_version" in f for f in rep["failures"])
    assert any("query_builder_commit" in f for f in rep["failures"])


def test_deep_hash_drift_fails(tmp_path):
    """Silent content corruption must be caught by --deep even when every
    file is present and readable."""
    from safetensors.numpy import load_file, save_file

    ds, src, ids = make_dataset(tmp_path)
    p = (
        ds
        / "samples"
        / ids[0]
        / "uv_queries"
        / "uv_002"
        / "uv_address.safetensors"
    )
    arr = dict(load_file(str(p)))
    arr["face_id"] = arr["face_id"] + 1
    save_file(arr, str(p))
    assert gate(ds, src)["n_failures"] == 0  # shallow cannot see it
    rep = gate(ds, src, deep=True)
    assert any("face_id_sha256 MISMATCH" in f for f in rep["failures"])


@pytest.mark.skipif(
    not (
        Path(__file__).resolve().parents[1]
        / "output"
        / "topotex_dataset"
        / "manifest.jsonl"
    ).exists(),
    reason="real dataset unavailable",
)
def test_real_dataset_first_samples_pass_deep():
    root = Path(__file__).resolve().parents[1] / "output" / "topotex_dataset"
    rep = run_gate(root, kind="dataset", deep=True, limit=4, workers=4)
    assert rep["n_failures"] == 0, rep["failures"]
