# -*- coding: utf-8 -*-
"""Recursive dataset integrity gate — run after source-build finalize and
after UV-query-build finalize (and standalone at any time).

python -m topotex.data.integrity --root output/topotex_dataset
    [--kind auto|source|dataset] [--deep] [--limit N] [--workers 16]
    [--out integrity_report.json]

Per official sample (from manifest.jsonl):
  1. broken symlink / missing target — every entry under the sample dir,
     any depth, including through symlinked query directories
  2. unreadable required file — safetensors headers parse, PNGs decode,
     meta.json parses
  3. manifest completeness — duplicate ids, manifest ids without a
     complete sample (FAIL); sample dirs absent from the manifest are
     reported as extras (WARN — the loader reads the manifest only)
  4. query builder provenance — query_schema_version matches the frozen
     schema and query_builder_commit is recorded (dataset kind)
  5. --deep — recompute face_id / barycentric / source-texture hashes and
     compare against meta.json

Exit code 0 only if no failures. Report written next to the manifest.
"""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "topotex_dataset@1"
SOURCE_SCHEMA = "topotex_source"
QUERIES = ("uv_000", "uv_001", "uv_002", "uv_test")
SOURCE_FILES = (
    "reference.png",
    "mv.safetensors",
    "mesh.safetensors",
    "uv_address.safetensors",
    "gt_texture.png",
    "meta.json",
)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _dangling(sample_dir):
    """Symlinks that do not resolve, at any depth (symlinked directories
    are checked but not descended into — their required contents are
    verified explicitly by path, which follows links)."""
    bad = []
    for dirpath, dirnames, filenames in os.walk(sample_dir, followlinks=False):
        for name in dirnames + filenames:
            p = Path(dirpath) / name
            if p.is_symlink() and not p.exists():
                bad.append(str(p))
    return bad


def _readable_safetensors(path, keys=()):
    from safetensors import safe_open

    with safe_open(str(path), framework="np") as f:
        have = set(f.keys())
    missing = [k for k in keys if k not in have]
    if missing:
        raise ValueError(f"missing arrays {missing}")


def _readable_png(path):
    from PIL import Image

    with Image.open(path) as im:
        im.verify()


def check_sample(root, sid, kind, deep, source_root=None):
    """Returns a list of failure strings for one sample (empty = ok)."""
    fails = []
    d = root / "samples" / sid
    if not d.is_dir():
        return [f"{sid}: sample directory missing"]
    for p in _dangling(d):
        fails.append(f"{sid}: dangling symlink {p}")

    def need(rel, checker, *a):
        p = d / rel
        if not p.exists():  # follows symlinks
            fails.append(f"{sid}: missing {rel}")
            return
        try:
            checker(p, *a)
        except Exception as e:
            fails.append(f"{sid}: unreadable {rel} ({e})")

    if kind == "source":
        for f in SOURCE_FILES:
            if f.endswith(".safetensors"):
                need(f, _readable_safetensors)
            elif f.endswith(".png"):
                need(f, _readable_png)
        try:
            meta = json.loads((d / "meta.json").read_text())
            if meta.get("dataset_schema") != SOURCE_SCHEMA:
                fails.append(
                    f"{sid}: dataset_schema {meta.get('dataset_schema')!r}"
                )
        except Exception as e:
            fails.append(f"{sid}: unreadable meta.json ({e})")
        return fails

    for f in ("reference.png", "mv.safetensors", "mesh.safetensors"):
        need(f, _readable_png if f.endswith(".png") else _readable_safetensors)
    for q in QUERIES:
        need(
            f"uv_queries/{q}/uv_address.safetensors",
            _readable_safetensors,
            (
                "uv_vertices",
                "uv_faces",
                "face_id",
                "barycentric",
                "valid_mask",
            ),
        )
        need(f"uv_queries/{q}/gt_texture.png", _readable_png)
    try:
        meta = json.loads((d / "meta.json").read_text())
    except Exception as e:
        fails.append(f"{sid}: unreadable meta.json ({e})")
        return fails
    if meta.get("query_schema_version") != SCHEMA:
        fails.append(
            f"{sid}: query_schema_version "
            f"{meta.get('query_schema_version')!r} != {SCHEMA!r}"
        )
    if "query_builder_commit" not in meta:
        fails.append(f"{sid}: query_builder_commit not recorded")
    qmeta = {q.get("name"): q for q in meta.get("uv_queries", [])}
    if sorted(qmeta) != sorted(QUERIES):
        fails.append(f"{sid}: meta lists queries {sorted(qmeta)}")

    if deep and not fails:
        from safetensors.numpy import load_file

        for q in QUERIES:
            m = qmeta[q]
            arr = load_file(
                str(d / "uv_queries" / q / "uv_address.safetensors")
            )
            for key, field in (
                ("face_id", "face_id_sha256"),
                ("barycentric", "barycentric_sha256"),
            ):
                want = m.get(field)
                if want is None:
                    fails.append(f"{sid}: {q} {field} not recorded")
                elif sha256_bytes(arr[key].tobytes()) != want:
                    fails.append(f"{sid}: {q} {field} MISMATCH")
        want = meta.get("source_texture_sha256")
        src_tex = (
            (source_root or (PROJECT_ROOT / "output" / "topotex_source"))
            / "samples"
            / sid
            / "gt_texture.png"
        )
        if want is None:
            fails.append(f"{sid}: source_texture_sha256 not recorded")
        elif not src_tex.exists():
            fails.append(f"{sid}: source texture missing ({src_tex})")
        elif sha256_bytes(src_tex.read_bytes()) != want:
            fails.append(f"{sid}: source_texture_sha256 MISMATCH")
    return fails


def run_gate(
    root, kind="auto", deep=False, limit=None, workers=16, source_root=None
):
    root = Path(root).resolve()
    ids = [json.loads(l)["sample_id"] for l in open(root / "manifest.jsonl")]
    dup = len(ids) - len(set(ids))
    if limit:
        ids = ids[:limit]
    if kind == "auto":
        meta = json.loads(
            (root / "samples" / ids[0] / "meta.json").read_text()
        )
        kind = (
            "source"
            if meta.get("dataset_schema") == SOURCE_SCHEMA
            else "dataset"
        )
    failures = [f"manifest: {dup} duplicate ids"] if dup else []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fs in ex.map(
            lambda s: check_sample(root, s, kind, deep, source_root), ids
        ):
            failures.extend(fs)
    on_disk = {p.name for p in (root / "samples").iterdir() if p.is_dir()}
    extras = sorted(
        on_disk
        - set(
            json.loads(l)["sample_id"] for l in open(root / "manifest.jsonl")
        )
    )
    return {
        "root": str(root),
        "kind": kind,
        "deep": deep,
        "n_checked": len(ids),
        "n_failures": len(failures),
        "failures": failures[:200],
        "extra_sample_dirs_not_in_manifest": len(extras),
        "extras_preview": extras[:10],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument(
        "--kind", default="auto", choices=["auto", "source", "dataset"]
    )
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--source-root",
        default=None,
        help="source dataset root for --deep source-texture "
        "checks (default output/topotex_source)",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rep = run_gate(
        args.root,
        args.kind,
        args.deep,
        args.limit,
        args.workers,
        Path(args.source_root) if args.source_root else None,
    )
    out = (
        Path(args.out)
        if args.out
        else Path(args.root) / "integrity_report.json"
    )
    out.write_text(json.dumps(rep, indent=1))
    print(
        f"integrity[{rep['kind']}{'/deep' if rep['deep'] else ''}] "
        f"{rep['root']}: {rep['n_checked']} checked, "
        f"{rep['n_failures']} failures, "
        f"{rep['extra_sample_dirs_not_in_manifest']} extra dirs (warn)"
    )
    for f in rep["failures"][:20]:
        print("  FAIL:", f)
    print("report:", out)
    raise SystemExit(1 if rep["n_failures"] else 0)


if __name__ == "__main__":
    main()
