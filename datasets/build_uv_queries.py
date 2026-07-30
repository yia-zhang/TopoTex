# -*- coding: utf-8 -*-
"""TOPOTEX dataset builder: source sample -> UV query set.

python -m datasets.build_uv_queries --limit 265
# 8-way sharded (ids[rank::world_size], per-rank manifest, then merge):
#   python -m datasets.build_uv_queries --rank K --world_size 8 ...
#   python -m datasets.build_uv_queries --finalize [--limit N]

Same Face Set, four queries per mesh:
    uv_000   canonical      native GLB parameterization
    uv_001   alternative    xatlas re-unwrap (different family)
    uv_002   partial        deterministic connected face subset (25/50/75%),
                            re-rasterized address maps, re-baked GT
    uv_test  held-out       Blender Smart UV — EVALUATION ONLY

Face identity is preserved exactly (xatlas and Blender both asserted to
return face row i for input face row i). Every query is rasterized fresh
(face_id / barycentric / valid_mask) — never resampled from another address
map — and its GT texture is baked per texel through
(face, bary) -> native UV -> bilinear sample of the original texture.

samples/<id>/
    reference.png / mv.safetensors / mesh.safetensors -> symlinks to source
    uv_queries/<query>/{uv_address.safetensors, gt_texture.png}
    meta.json
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from .uv_query import connected_subset, face_adjacency

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "output" / "topotex_source"
RES = 256
SCHEMA = "topotex_dataset@1"
PARTIAL_SEED = 20260729
PARTIAL_FRACS = (0.25, 0.5, 0.75)


def builder_commit():
    """Short commit of the builder code writing this sample (provenance,
    recorded in meta.json and checked by datasets.verify_integrity).
    A modified working tree is stamped '-dirty' — a clean commit id must
    never be recorded for code that commit does not contain."""
    try:
        c = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain",
             "--untracked-files=no"],
            text=True, stderr=subprocess.DEVNULL).strip()
        return c + ("-dirty" if dirty else "")
    except Exception:
        return None


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def bilinear_u8(tex_u8, uv):
    """Plain sRGB-space bilinear sample. tex [h,w,3] u8; uv [K,2] v-down."""
    h, w = tex_u8.shape[:2]
    x = np.clip(uv[:, 0] * w - 0.5, 0, w - 1)
    y = np.clip(uv[:, 1] * h - 0.5, 0, h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    tx = (x - x0)[:, None]
    ty = (y - y0)[:, None]
    t = tex_u8.astype(np.float32)
    out = (t[y0, x0] * (1 - tx) * (1 - ty) + t[y0, x1] * tx * (1 - ty)
           + t[y1, x0] * (1 - tx) * ty + t[y1, x1] * tx * ty)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def blender_smart_uv(V, F):
    """Held-out UV family: Blender Smart UV Project (angle-based projection
    clustering) — a genuinely different unwrap family from native/xatlas.
    Per-corner UVs; face order verified preserved."""
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    me = bpy.data.meshes.new("m")
    me.from_pydata(V.astype(np.float64).tolist(), [],
                   F.astype(np.int64).tolist())
    me.update()
    obj = bpy.data.objects.new("o", me)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=1.15192, island_margin=0.003)
    bpy.ops.object.mode_set(mode="OBJECT")
    n_loops = len(me.uv_layers.active.data)
    assert n_loops == 3 * len(F)
    loop_verts = np.array([me.loops[i].vertex_index
                           for i in range(n_loops)]).reshape(len(F), 3)
    assert (loop_verts == F.astype(np.int64)).all(), \
        "blender changed face identity/order"
    uv = np.array([me.uv_layers.active.data[i].uv[:]
                   for i in range(n_loops)], np.float32)
    uv[:, 1] = 1.0 - uv[:, 1]          # v-up -> top-down
    uv_faces = np.arange(3 * len(F), dtype=np.int32).reshape(len(F), 3)
    return np.clip(uv, 0, 1), uv_faces


def xatlas_unwrap(V, F):
    """Re-parameterize the SAME face set (default chart/pack options).
    Returns (uv_vertices, uv_faces); asserts face-row identity."""
    import xatlas
    atlas = xatlas.Atlas()
    atlas.add_mesh(V.astype(np.float32), F.astype(np.uint32))
    atlas.generate(xatlas.ChartOptions(), xatlas.PackOptions())
    vmap, idx, uvs = atlas[0]
    back = vmap[idx].astype(np.int64)
    assert (back == F.astype(np.int64)).all(), \
        "xatlas changed face identity/order"
    uvs = uvs.astype(np.float32)
    uvs[:, 1] = 1.0 - uvs[:, 1]        # v-up -> our top-down texture rows
    return np.clip(uvs, 0, 1), idx.astype(np.int32)


def rasterize_query(uv_vertices, uv_faces, n_faces, uv_face_to_mesh_face=None,
                    min_occupancy=0.01):
    from .rasterizer import rasterize_uv
    if uv_face_to_mesh_face is None:
        uv_face_to_mesh_face = np.arange(n_faces)
    am = rasterize_uv(uv_vertices.astype(np.float64),
                      uv_faces.astype(np.int64), RES,
                      uv_face_to_mesh_face=uv_face_to_mesh_face)
    valid = am.valid_mask.astype(bool)
    if valid.mean() < min_occupancy:
        raise RuntimeError("uv query occupancy below threshold")
    return (am.face_id.astype(np.int32),
            am.barycentric.astype(np.float16).transpose(2, 0, 1),
            valid.astype(np.uint8))


def bake_gt(face_id, bary, valid, native_uv_verts, native_uv_faces, gt_u8):
    """(face, bary) -> native UV -> bilinear sample of the original texture."""
    out = np.zeros((RES, RES, 3), np.uint8)
    v = valid.astype(bool)
    fid = face_id[v].astype(np.int64)
    b = bary.astype(np.float32).transpose(1, 2, 0)[v]          # [K,3]
    uvc = native_uv_verts[native_uv_faces[fid].astype(np.int64)]  # [K,3,2]
    uv = (uvc * b[..., None]).sum(1)
    out[v] = bilinear_u8(gt_u8, uv)
    return out


def partial_query(F, native_uv, native_uvf, sid):
    """Deterministic connected face subset on the canonical layout.
    Address maps come from the SUBSET's own rasterization; uv_vertices /
    uv_faces stay the full layout so rebake tooling is unchanged."""
    rng = np.random.default_rng(PARTIAL_SEED + int(sid[:8], 16))
    adj = face_adjacency(F)
    frac = PARTIAL_FRACS[int(rng.integers(len(PARTIAL_FRACS)))]
    for _ in range(5):                     # retry on degenerate patches
        keep = connected_subset(adj, len(F), frac, rng)
        try:
            fid, bary, valid = rasterize_query(
                native_uv.astype(np.float32), native_uvf[keep], len(F),
                uv_face_to_mesh_face=keep, min_occupancy=64 / RES ** 2)
            return fid, bary, valid, frac, len(keep)
        except RuntimeError:
            continue
    raise RuntimeError("partial query kept too few texels after 5 tries")


def build_sample(sid, out_root):
    from PIL import Image
    from safetensors.numpy import load_file, save_file
    src = SOURCE_ROOT / "samples" / sid
    dst = out_root / "samples" / sid
    q_root = dst / "uv_queries"
    q_root.mkdir(parents=True, exist_ok=True)
    for f in ("reference.png", "mv.safetensors", "mesh.safetensors"):
        link = dst / f
        if not link.exists():
            os.symlink(os.path.realpath(src / f), link)
    mesh = load_file(str(src / "mesh.safetensors"))
    V = mesh["vertices"]
    F = mesh["faces"].astype(np.int64)
    native_uv = mesh["uv_vertices"].astype(np.float64)
    native_uvf = mesh["uv_faces"].astype(np.int64)
    gt = np.asarray(Image.open(src / "gt_texture.png").convert("RGB"))
    t0 = time.time()
    queries = []
    for name, qtype in (("uv_000", "canonical"), ("uv_001", "alternative"),
                        ("uv_002", "partial"), ("uv_test", "heldout")):
        extra = {}
        if qtype == "canonical":
            uvv = native_uv.astype(np.float32)
            uvf = native_uvf.astype(np.int32)
            fid, bary, valid = rasterize_query(uvv, uvf, len(F))
            gt_k = gt
        elif qtype == "alternative":
            uvv, uvf = xatlas_unwrap(V, F)
            fid, bary, valid = rasterize_query(uvv, uvf, len(F))
            gt_k = bake_gt(fid, bary, valid, native_uv, native_uvf, gt)
        elif qtype == "partial":
            uvv = native_uv.astype(np.float32)
            uvf = native_uvf.astype(np.int32)
            fid, bary, valid, frac, n_kept = partial_query(
                F, native_uv, native_uvf, sid)
            gt_k = bake_gt(fid, bary, valid, native_uv, native_uvf, gt)
            extra = {"partial_frac": frac, "n_kept_faces": n_kept}
        else:                              # heldout
            uvv, uvf = blender_smart_uv(V, F)
            fid, bary, valid = rasterize_query(uvv, uvf, len(F))
            gt_k = bake_gt(fid, bary, valid, native_uv, native_uvf, gt)
        qd = q_root / name
        qd.mkdir(exist_ok=True)
        save_file({"uv_vertices": uvv.astype(np.float32),
                   "uv_faces": uvf.astype(np.int32),
                   "face_id": fid, "barycentric": bary,
                   "valid_mask": valid},
                  str(qd / "uv_address.safetensors"))
        Image.fromarray(gt_k).save(qd / "gt_texture.png")
        queries.append({"name": name, "type": qtype,
                        "held_out": qtype == "heldout",
                        "valid_texels": int(valid.sum()),
                        "n_uv_vertices": int(len(uvv)),
                        "face_id_sha256": sha256_bytes(fid.tobytes()),
                        "barycentric_sha256": sha256_bytes(bary.tobytes()),
                        **extra})
    meta_tmp = dst / "meta.json.part"
    meta_tmp.write_text(json.dumps(
        {"sample_id": sid, "schema": SCHEMA, "source": "topotex_source",
         "num_faces": int(len(F)), "uv_queries": queries,
         "query_schema_version": SCHEMA,
         "query_builder_commit": builder_commit(),
         "source_texture_sha256": sha256_bytes(
             (src / "gt_texture.png").read_bytes()),
         "build_seconds": round(time.time() - t0, 1)}, indent=1))
    os.replace(meta_tmp, dst / "meta.json")   # atomic: no truncated metas
    return queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=265)
    ap.add_argument("--ids", default=None, help="comma-separated sample ids")
    ap.add_argument("--output", default="output/topotex_dataset")
    ap.add_argument("--world-size", "--world_size", dest="world_size",
                    type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--finalize", action="store_true",
                    help="merge per-rank manifests, validate, write "
                         "manifest.jsonl + dataset_meta.json")
    args = ap.parse_args()
    out_root = Path(args.output).resolve()
    if args.finalize:
        finalize(out_root, args.limit)
        return
    if args.ids:
        ids = args.ids.split(",")
    else:
        ids = [json.loads(l)["sample_id"]
               for l in open(SOURCE_ROOT / "manifest.jsonl")][: args.limit]
    ids = ids[args.rank::args.world_size]
    rows = []
    rank_fail = []
    for sid in ids:
        meta_f = out_root / "samples" / sid / "meta.json"
        if meta_f.exists():
            try:                       # corrupt meta -> fall through, rebuild
                qs = json.loads(meta_f.read_text())["uv_queries"]
            except Exception:
                qs = []
            if len(qs) == 4:                      # complete -> skip rebuild
                rows.append({"sample_id": sid,
                             "valid_texels": [q["valid_texels"] for q in qs]})
                continue
        try:
            qs = build_sample(sid, out_root)
            rows.append({"sample_id": sid,
                         "valid_texels": [q["valid_texels"] for q in qs]})
            print(f"{sid[:12]} ok | texels {[q['valid_texels'] for q in qs]}",
                  flush=True)
        except Exception as e:
            rank_fail.append({"sample_id": sid, "reason": str(e)})
            print(f"{sid[:12]} FAIL: {e}", flush=True)
    with open(out_root / f"manifest_rank_{args.rank}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(out_root / f"failures_rank_{args.rank}.jsonl", "w") as f:
        for r in rank_fail:
            f.write(json.dumps(r) + "\n")
    print(f"rank {args.rank} done: {len(rows)}/{len(ids)}")
    if args.world_size == 1:
        finalize(out_root, args.limit)


def finalize(out_root, limit=None):
    """Merge per-rank manifests: unique ids, complete 4-query samples."""
    rows, seen = [], set()
    for p in sorted(out_root.glob("manifest_rank_*.jsonl")):
        for l in open(p):
            r = json.loads(l)
            if r["sample_id"] in seen:
                continue
            seen.add(r["sample_id"])
            assert len(r["valid_texels"]) == 4 and min(r["valid_texels"]) > 0
            rows.append(r)
    fails = [json.loads(l)
             for p in sorted(out_root.glob("failures_rank_*.jsonl"))
             for l in open(p)]
    with open(out_root / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(out_root / "dataset_failures.jsonl", "w") as f:
        for r in fails:
            f.write(json.dumps(r) + "\n")
    (out_root / "dataset_meta.json").write_text(json.dumps(
        {"schema": SCHEMA, "n_samples": len(rows),
         "queries": ["canonical", "alternative", "partial", "heldout"]},
        indent=1))
    print(f"finalized: {len(rows)} samples, {len(fails)} failures")


if __name__ == "__main__":
    main()
