# -*- coding: utf-8 -*-
"""Merge per-rank build manifests into the final dataset manifest.

python -m datasets.merge_manifest --output output/topotex_source \
    [--input-manifest glbs_eligible.jsonl]

Checks:
  no duplicate  -- the same sample_id recorded by two ranks must agree
                   (re-recording after a resharded resume is legal; a
                   conflicting record is an error)
  no missing    -- with --input-manifest, every input id must be accounted
                   for: built (manifest) or failed (dataset_failures)
  schema valid  -- every published sample passes check_sample_schema
"""
import argparse
import json
import time
from pathlib import Path


def merge(out_dir, input_manifest=None):
    from .build_dataset import DATASET_SCHEMA, RES, VIEW_ORDER, \
        check_sample_schema
    out_dir = Path(out_dir)
    parts = sorted(out_dir.glob("manifest_rank_*.jsonl"))
    assert parts, f"no manifest_rank_*.jsonl under {out_dir}"
    rows, by_id = [], {}
    for p in parts:
        for l in open(p):
            r = json.loads(l)
            sid = r["sample_id"]
            if sid in by_id:
                assert by_id[sid] == r, \
                    f"conflicting duplicate for {sid}: {by_id[sid]} vs {r}"
                continue
            by_id[sid] = r
            rows.append(r)
    for r in rows:
        check_sample_schema(out_dir / "samples" / r["sample_id"])
    with open(out_dir / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    fails = [json.loads(l)
             for p in sorted((out_dir / "failures").glob("rank_*.jsonl"))
             for l in open(p)]
    with open(out_dir / "dataset_failures.jsonl", "w") as f:
        for r in fails:
            f.write(json.dumps(r) + "\n")
    missing = []
    if input_manifest:
        wanted = [json.loads(l)["sample_id"] for l in open(input_manifest)]
        accounted = set(by_id) | {r["sample_id"] for r in fails}
        missing = [sid for sid in wanted if sid not in accounted]
        assert not missing, (f"{len(missing)} input ids unaccounted for "
                             f"(neither built nor failed): {missing[:5]}")
    (out_dir / "dataset_meta.json").write_text(json.dumps(
        {"dataset_schema": DATASET_SCHEMA, "resolution": RES,
         "view_order": ",".join(VIEW_ORDER), "n_samples": len(rows),
         "n_failures": len(fails),
         "finalized_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
    print(f"merged: {len(rows)} samples, {len(fails)} failures, "
          f"0 duplicates conflicting, {len(missing)} missing")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--input-manifest", default=None,
                    help="optional: verify every input id is accounted for")
    args = ap.parse_args()
    merge(Path(args.output), input_manifest=args.input_manifest)


if __name__ == "__main__":
    main()
