# -*- coding: utf-8 -*-
"""Dataset quality statistics — monitoring only, no filtering.

python -m datasets.dataset_statistics [--source output/topotex_source]
    [--dataset output/topotex_dataset] [--out dataset_statistics.json]

Reports: totals, success rate, failure reasons, face/vertex histograms,
UV occupancy distribution, texture resolution distribution, and MV
generation status across the built dataset.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _hist(values, bins):
    values = np.asarray(values)
    counts, edges = np.histogram(values, bins=bins)
    return {"edges": [round(float(e), 4) for e in edges],
            "counts": [int(c) for c in counts]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="output/topotex_source")
    ap.add_argument("--dataset", default="output/topotex_dataset")
    ap.add_argument("--out", default=None)
    ap.add_argument("--input-manifest", default=None,
                    help="build input manifest: enables attempted-assets "
                         "accounting for the final report")
    args = ap.parse_args()
    src = (PROJECT_ROOT / args.source).resolve()
    dst = (PROJECT_ROOT / args.dataset).resolve()
    out_path = Path(args.out) if args.out else dst / "dataset_statistics.json"

    src_rows = [json.loads(l) for l in open(src / "manifest.jsonl")]
    dup_src = len(src_rows) - len({r["sample_id"] for r in src_rows})
    manifest_sha = hashlib.sha256(
        (src / "manifest.jsonl").read_bytes()).hexdigest()
    attempted = None
    if args.input_manifest:
        attempted = sum(1 for _ in open(args.input_manifest))
    fails = []
    fail_file = src / "dataset_failures.jsonl"
    if fail_file.exists():
        fails = [json.loads(l) for l in open(fail_file)]
    fail_reasons = Counter()
    for r in fails:
        reason = r.get("reason", "unknown")
        fail_reasons[reason.split("(")[0].strip()[:60]] += 1

    faces, verts, tex_res, mv_ok = [], [], Counter(), 0
    for r in src_rows:
        meta = json.loads(
            (src / "samples" / r["sample_id"] / "meta.json").read_text())
        faces.append(meta["num_faces"])
        verts.append(meta["num_vertices"])
        tex_res[meta["texture_resolution"]] += 1
        if meta.get("generator"):
            mv_ok += 1

    occ = {}
    ds_rows = []
    if (dst / "manifest.jsonl").exists():
        ds_rows = [json.loads(l) for l in open(dst / "manifest.jsonl")]
        names = ["canonical", "alternative", "partial", "heldout"]
        res2 = 256 ** 2
        for i, nm in enumerate(names):
            vals = [r["valid_texels"][i] / res2 for r in ds_rows
                    if len(r["valid_texels"]) > i]
            if vals:
                occ[nm] = {"mean": round(float(np.mean(vals)), 4),
                           "p5": round(float(np.percentile(vals, 5)), 4),
                           "p50": round(float(np.percentile(vals, 50)), 4),
                           "p95": round(float(np.percentile(vals, 95)), 4),
                           "hist": _hist(vals, bins=20)}

    n_attempted = len(src_rows) + len(fails)
    ds_dup = len(ds_rows) - len({r["sample_id"] for r in ds_rows})
    coverage = (sum(1 for r in ds_rows if len(r["valid_texels"]) == 4
                    and min(r["valid_texels"]) > 0) / max(len(ds_rows), 1))
    stats = {
        "attempted_assets": attempted,
        "duplicate_check": {"source_duplicates": dup_src,
                            "uv_query_duplicates": ds_dup},
        "manifest_sha256": manifest_sha,
        "source": {
            "total_samples": len(src_rows),
            "failed": len(fails),
            "success_rate": round(len(src_rows) / max(n_attempted, 1), 4),
            "failed_reasons": dict(fail_reasons.most_common()),
            "mv_generation_ok": mv_ok,
            "face_count": {"min": int(min(faces)), "max": int(max(faces)),
                           "median": int(np.median(faces)),
                           "hist": _hist(faces, bins=20)},
            "vertex_count": {"min": int(min(verts)), "max": int(max(verts)),
                             "median": int(np.median(verts)),
                             "hist": _hist(verts, bins=20)},
            "texture_resolution": dict(tex_res),
        },
        "uv_query_dataset": {
            "total_samples": len(ds_rows),
            "full_query_coverage": round(coverage, 4),
            "uv_occupancy": occ,
        },
    }
    out_path.write_text(json.dumps(stats, indent=1))
    print(f"total {len(src_rows)} | failed {len(fails)} "
          f"(rate {stats['source']['success_rate']}) | "
          f"faces median {stats['source']['face_count']['median']} | "
          f"uv-query samples {len(ds_rows)}")
    print("done:", out_path)


if __name__ == "__main__":
    main()
