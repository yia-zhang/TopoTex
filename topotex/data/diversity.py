# -*- coding: utf-8 -*-
"""Dataset diversity analysis — geometry / UV / appearance distributions.

python -m topotex.data.diversity [--dataset output/topotex_dataset]
    [--limit N] [--out dataset_diversity.json]

Geometry: face count, vertex count, connected-component count, component
size distribution, face degree distribution, boundary edge ratio.
UV: island count (alternative layout), occupancy per query.
Appearance: texture mean/std luminance, colorfulness, entropy.

Monitoring only — no filtering. The companion report interprets whether a
scale-up widens the distribution rather than just adding samples.
"""

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _summ(vals, bins=20):
    v = np.asarray(vals, dtype=np.float64)
    counts, edges = np.histogram(v, bins=bins)
    return {
        "min": round(float(v.min()), 4),
        "p5": round(float(np.percentile(v, 5)), 4),
        "p50": round(float(np.percentile(v, 50)), 4),
        "p95": round(float(np.percentile(v, 95)), 4),
        "max": round(float(v.max()), 4),
        "mean": round(float(v.mean()), 4),
        "hist": {
            "edges": [round(float(e), 4) for e in edges],
            "counts": [int(c) for c in counts],
        },
    }


def topology_stats(faces):
    """Component count + faces-per-component + face degree (shared-edge
    neighbors per face, 0..3) + boundary edge ratio."""
    f = np.asarray(faces, dtype=np.int64)
    n = int(f.max()) + 1
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for tri in f:
        r = find(tri[0])
        for b in tri[1:]:
            rb = find(b)
            if rb != r:
                parent[rb] = r
    from collections import Counter

    comp_of_face = [find(tri[0]) for tri in f]
    comp_sizes = list(Counter(comp_of_face).values())
    e2f = {}
    for fi, tri in enumerate(f):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            e = (min(tri[a], tri[b]), max(tri[a], tri[b]))
            e2f.setdefault(e, []).append(fi)
    degree = np.zeros(len(f), np.int32)
    n_boundary = 0
    for fs in e2f.values():
        if len(fs) == 1:
            n_boundary += 1
        else:
            for a in fs:
                degree[a] += len(fs) - 1
    degree = np.clip(degree, 0, 3)
    return {
        "n_components": len(comp_sizes),
        "comp_sizes": comp_sizes,
        "degree_hist": np.bincount(degree, minlength=4).tolist(),
        "boundary_ratio": n_boundary / max(len(e2f), 1),
    }


def n_islands(uv_faces):
    from topotex.data.uv import face_adjacency  # noqa: F401  (import check)

    f = np.asarray(uv_faces, dtype=np.int64)
    n = int(f.max()) + 1
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for tri in f:
        r = find(tri[0])
        for b in tri[1:]:
            rb = find(b)
            if rb != r:
                parent[rb] = r
    roots = {find(v) for v in np.unique(f)}
    return len(roots)


def main():
    from PIL import Image
    from safetensors.numpy import load_file

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="output/topotex_dataset")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = (PROJECT_ROOT / args.dataset).resolve()
    out_path = Path(args.out) if args.out else root / "dataset_diversity.json"
    ids = [json.loads(l)["sample_id"] for l in open(root / "manifest.jsonl")]
    if args.limit:
        ids = ids[: args.limit]

    faces, verts, comps, islands = [], [], [], []
    comp_sizes_all, degree_hist_total = [], np.zeros(4, np.int64)
    boundary_ratios = []
    occ = {"canonical": [], "alternative": [], "partial": [], "heldout": []}
    lum_mean, lum_std, colorful, entropy = [], [], [], []
    res2 = 256**2
    qnames = [
        ("uv_000", "canonical"),
        ("uv_001", "alternative"),
        ("uv_002", "partial"),
        ("uv_test", "heldout"),
    ]
    for i, sid in enumerate(ids):
        d = root / "samples" / sid
        mesh = load_file(str(d / "mesh.safetensors"))
        F = mesh["faces"].astype(np.int64)
        faces.append(len(F))
        verts.append(len(mesh["vertices"]))
        ts = topology_stats(F)
        comps.append(ts["n_components"])
        comp_sizes_all.extend(ts["comp_sizes"])
        degree_hist_total += np.array(ts["degree_hist"])
        boundary_ratios.append(ts["boundary_ratio"])
        ua = load_file(
            str(d / "uv_queries" / "uv_001" / "uv_address.safetensors")
        )
        islands.append(n_islands(ua["uv_faces"]))
        meta = json.loads((d / "meta.json").read_text())
        qmeta = {q["name"]: q for q in meta["uv_queries"]}
        for qn, label in qnames:
            if qn in qmeta:
                occ[label].append(qmeta[qn]["valid_texels"] / res2)
        tex = np.asarray(
            Image.open(d / "uv_queries" / "uv_000" / "gt_texture.png").convert(
                "RGB"
            ),
            np.float32,
        )
        lum = tex.mean(-1)
        lum_mean.append(float(lum.mean()) / 255)
        lum_std.append(float(lum.std()) / 255)
        rg = tex[..., 0] - tex[..., 1]
        yb = 0.5 * (tex[..., 0] + tex[..., 1]) - tex[..., 2]
        colorful.append(
            float(
                np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
            )
            / 255
        )
        h, _ = np.histogram(lum, bins=64, range=(0, 255))
        p = h / max(h.sum(), 1)
        p = p[p > 0]
        entropy.append(float(-(p * np.log2(p)).sum()))
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(ids)}", flush=True)

    report = {
        "n_samples": len(ids),
        "geometry": {
            "face_count": _summ(faces),
            "vertex_count": _summ(verts),
            "component_count": _summ(comps),
            "multi_component_fraction": round(
                float(np.mean(np.array(comps) > 1)), 4
            ),
            "component_size_faces": _summ(comp_sizes_all),
            "face_degree_hist_0123": [int(x) for x in degree_hist_total],
            "boundary_edge_ratio": _summ(boundary_ratios),
        },
        "uv": {
            "island_count_alternative": _summ(islands),
            "occupancy": {k: _summ(v) for k, v in occ.items() if v},
        },
        "appearance": {
            "luminance_mean": _summ(lum_mean),
            "luminance_std": _summ(lum_std),
            "colorfulness": _summ(colorful),
            "luminance_entropy_bits": _summ(entropy),
        },
    }
    out_path.write_text(json.dumps(report, indent=1))
    print(
        f"n={len(ids)} | faces p50 {report['geometry']['face_count']['p50']}"
        f" | components p50 {report['geometry']['component_count']['p50']}"
        f" | islands p50 {report['uv']['island_count_alternative']['p50']}"
        f" | entropy p50 "
        f"{report['appearance']['luminance_entropy_bits']['p50']}"
    )
    print("done:", out_path)


if __name__ == "__main__":
    main()
