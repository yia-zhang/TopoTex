# -*- coding: utf-8 -*-
"""TOPOTEX source dataset builder: textured GLB -> one training sample.

python -m topotex.data.builder source \
    --input-manifest output/source_manifests/glbs.jsonl \
    --output output/topotex_source --limit 10
# 8-GPU sharding: --world_size 8 --rank K --device cuda:0 (one process/GPU,
#   ids[rank::world_size], one UniTEX generator load per worker)
# finalize (merge manifest parts + schema check):
python -m topotex.data.builder source --output output/topotex_source --finalize

Per sample: reference.png, mesh.safetensors (vertices/faces/uv_vertices/
uv_faces), mv.safetensors (frozen UniTEX stage-1 six views, uint8
[6,3,256,256]), uv_address.safetensors (face_id/barycentric/valid_mask @256),
gt_texture.png, meta.json. Assets failing the gate (triangle mesh,
per-vertex non-overlapping UV, single base-color texture) are skipped with
a one-line reason. Atomic publish; key-matched samples resume-skip.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_SCHEMA = "topotex_source"
VIEW_ORDER = ["front", "back", "left", "right", "top", "bottom"]
RES = 256
REFERENCE_CAM = (30.0, 20.0)  # deterministic 3/4 view (az, el)
OVERLAP_TOL = 1.05  # UV triangle area vs covered texels


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class SkipSample(Exception):
    """Asset fails the gate; .args[0] is the one-line reason."""


def extract_glb(glb_path, uv_tol=1e-3):
    """GLB -> (vertices f32 [V,3], faces i32 [F,3], uv f32 [V,2] top-down v,
    texture u8 [T,T,3]). Raises SkipSample with a reason otherwise.
    uv_tol: accepted UV overshoot beyond [0,1] before clipping — the strict
    dataset default rejects tiling textures; interactive callers may relax
    it for slight boundary overshoot."""
    import trimesh

    scene = trimesh.load(str(glb_path), process=False)
    # (transform, mesh) per scene-graph instance so geometry matches the GLB
    parts = []
    if isinstance(scene, trimesh.Scene):
        for node in scene.graph.nodes_geometry:
            T, geom_name = scene.graph.get(node)
            m = scene.geometry.get(geom_name)
            if isinstance(m, trimesh.Trimesh):
                parts.append((T, m))
    else:
        parts = [(np.eye(4), scene)]
    if not parts:
        raise SkipSample("no triangle mesh")
    textured = []
    for T, m in parts:
        vis = getattr(m, "visual", None)
        uv_attr = getattr(vis, "uv", None) if vis is not None else None
        if (
            uv_attr is not None
            and len(uv_attr) == len(m.vertices)
            and getattr(
                getattr(vis, "material", None), "baseColorTexture", None
            )
            is not None
        ):
            textured.append((T, m))
    if not textured:
        raise SkipSample("no uv+baseColorTexture")
    imgs = {
        np.asarray(m.visual.material.baseColorTexture.convert("RGB")).tobytes()
        for _, m in textured
    }
    if len(imgs) > 1:
        raise SkipSample(f"multiple textures ({len(imgs)})")
    tex = np.asarray(
        textured[0][1].visual.material.baseColorTexture.convert("RGB")
    )
    V, F, UV = [], [], []
    off = 0
    for T, m in textured:
        v = (np.c_[m.vertices, np.ones(len(m.vertices))] @ np.asarray(T).T)[
            :, :3
        ]
        V.append(v.astype(np.float32))
        F.append(m.faces.astype(np.int64) + off)
        UV.append(np.asarray(m.visual.uv, np.float32))
        off += len(v)
    vertices = np.concatenate(V)
    faces = np.concatenate(F)
    uv = np.concatenate(UV)
    if faces.shape[1] != 3:
        raise SkipSample("non-triangle faces")
    if len(uv) != len(vertices) or not np.isfinite(uv).all():
        raise SkipSample("invalid per-vertex uv")
    uv = uv.copy()
    uv[:, 1] = 1.0 - uv[:, 1]  # GLB v-up -> our top-down texture rows
    if uv.min() < -uv_tol or uv.max() > 1 + uv_tol:
        raise SkipSample("uv outside [0,1] (tiling texture)")
    return (
        vertices,
        faces.astype(np.int32),
        np.clip(uv, 0, 1),
        tex.astype(np.uint8),
    )


def uv_address(uv, faces):
    """Deterministic UV rasterization -> (face_id i32, bary f16 [3,H,W],
    valid u8). Also gates on UV overlap."""
    from topotex.data.uv import rasterize_uv

    am = rasterize_uv(uv, faces.astype(np.int64), RES)
    valid = am.valid_mask.astype(bool)
    tri = uv[faces]
    area = 0.5 * np.abs(
        (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
        - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1])
    )
    covered = valid.mean()
    if covered < 0.01:
        raise SkipSample("uv occupancy < 1%")
    if area.sum() / max(covered, 1e-9) > OVERLAP_TOL:
        raise SkipSample(
            f"overlapping uv (area {area.sum():.3f} vs cover {covered:.3f})"
        )
    bary = am.barycentric.astype(np.float16)  # [H,W,3]
    return (
        am.face_id.astype(np.int32),
        bary.transpose(2, 0, 1),
        valid.astype(np.uint8),
    )


def render_reference(vertices, faces, uv, gt_texture, out_png):
    """One deterministic textured render (same conventions as evaluation)."""
    import types

    from PIL import Image

    from topotex.data.mesh import (
        camera_matrices,
        linear_to_srgb_u8,
        rasterize_view,
        render_albedo_rebake,
    )

    canon = types.SimpleNamespace(
        vertices=vertices.astype(np.float64), faces=faces.astype(np.int64)
    )
    uvr = types.SimpleNamespace(
        uv_vertices=uv,
        uv_faces=faces.astype(np.int64),
        uv_face_to_mesh_face=np.arange(len(faces)),
    )
    bmin, bmax = canon.vertices.min(0), canon.vertices.max(0)
    cam = camera_matrices(*REFERENCE_CAM, bmin, bmax)
    gb = rasterize_view(canon.vertices, canon.faces, cam, 512)
    if gb["mask"].sum() < 100:
        raise SkipSample("reference render empty (degenerate geometry)")
    img = linear_to_srgb_u8(render_albedo_rebake(canon, uvr, gt_texture, gb))
    img[~gb["mask"]] = 235
    Image.fromarray(img).save(out_png)


def build_source_sample(provider, sid, glb_path, out_dir, scratch):
    """Build one sample into scratch, then publish atomically."""
    from PIL import Image
    from safetensors.numpy import save_file

    t0 = time.time()
    vertices, faces, uv, tex = extract_glb(glb_path)
    face_id, bary, valid = uv_address(uv, faces)
    work = scratch / sid
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    gt = np.asarray(Image.fromarray(tex).resize((RES, RES), Image.LANCZOS))
    Image.fromarray(gt).save(work / "gt_texture.png")
    render_reference(vertices, faces, uv, gt, work / "reference.png")
    obj = work / "textureless.obj"
    from topotex.data.mesh import export_textureless_obj

    export_textureless_obj(vertices, faces, obj)
    inputs_dir = work / "mv_inputs"
    provider.prepare_inputs(work / "reference.png", obj, sid, inputs_dir)
    images, gmeta = provider.generate(inputs_dir, sid, work / "mv_work")
    save_file(
        {
            "vertices": vertices,
            "faces": faces,
            "uv_vertices": uv,
            "uv_faces": faces,
        },
        str(work / "mesh.safetensors"),
    )
    save_file(
        {"images": images, "view_id": np.arange(6, dtype=np.int32)},
        str(work / "mv.safetensors"),
    )
    save_file(
        {"face_id": face_id, "barycentric": bary, "valid_mask": valid},
        str(work / "uv_address.safetensors"),
    )
    meta = {
        "sample_id": sid,
        "dataset_schema": DATASET_SCHEMA,
        "glb_sha256": sha256_file(glb_path),
        "num_vertices": int(len(vertices)),
        "num_faces": int(len(faces)),
        "texture_resolution": RES,
        "address_resolution": RES,
        "valid_texels": int(valid.sum()),
        "view_order": ",".join(VIEW_ORDER),
        "reference_camera": {
            "azimuth": REFERENCE_CAM[0],
            "elevation": REFERENCE_CAM[1],
        },
        "generator": gmeta,
        "build_seconds": round(time.time() - t0, 1),
    }
    (work / "meta.json").write_text(json.dumps(meta, indent=1))
    # strip build intermediates, publish atomically
    shutil.rmtree(inputs_dir)
    shutil.rmtree(work / "mv_work")
    obj.unlink()
    final = out_dir / "samples" / sid
    tmp = final.with_name(final.name + ".tmp_publish")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(work, tmp)
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    shutil.rmtree(work)
    return meta


SAMPLE_FILES = [
    "reference.png",
    "mesh.safetensors",
    "mv.safetensors",
    "uv_address.safetensors",
    "gt_texture.png",
    "meta.json",
]


def sample_is_valid(out_dir, sid, glb_sha=None):
    d = out_dir / "samples" / sid
    if not all((d / f).exists() for f in SAMPLE_FILES):
        return False
    try:
        meta = json.loads((d / "meta.json").read_text())
        if meta.get("dataset_schema") != DATASET_SCHEMA:
            return False
        return glb_sha is None or meta.get("glb_sha256") == glb_sha
    except Exception:
        return False


def check_sample_schema(d):
    from safetensors.numpy import load_file

    mesh = load_file(str(d / "mesh.safetensors"))
    mv = load_file(str(d / "mv.safetensors"))
    ua = load_file(str(d / "uv_address.safetensors"))
    F = mesh["faces"].shape[0]
    assert (
        mesh["vertices"].dtype == np.float32 and mesh["vertices"].shape[1] == 3
    )
    assert mesh["faces"].dtype == np.int32 and mesh["uv_faces"].shape == (F, 3)
    assert mesh["uv_vertices"].dtype == np.float32
    assert (
        mv["images"].shape == (6, 3, RES, RES)
        and mv["images"].dtype == np.uint8
    )
    assert list(mv["view_id"]) == list(range(6))
    assert (
        ua["face_id"].shape == (RES, RES) and ua["face_id"].dtype == np.int32
    )
    assert ua["barycentric"].shape == (3, RES, RES)
    assert ua["barycentric"].dtype == np.float16
    assert ua["valid_mask"].shape == (RES, RES)
    v = ua["valid_mask"].astype(bool)
    assert v.any()
    fid = ua["face_id"][v]
    assert fid.min() >= 0 and fid.max() < F
    s = ua["barycentric"].astype(np.float32).transpose(1, 2, 0)[v].sum(-1)
    assert np.abs(s - 1).max() < 2e-2


def finalize_source(out_dir, input_manifest=None):
    """Merge per-rank manifests; verify uniqueness, coverage and schema."""
    merge(out_dir, input_manifest=input_manifest)


def main_source():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-manifest",
        default=None,
        help="jsonl: {sample_id, glb_path} per line",
    )
    ap.add_argument("--output", default="output/topotex_source")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--world-size", "--world_size", dest="world_size", type=int, default=1
    )
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--scratch-root", default="/tmp/topotex_source")
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument(
        "--gate-only",
        action="store_true",
        help="run extraction/UV gates only (NO FLUX, no sample "
        "publish); writes eligible/rejected manifest parts",
    )
    args = ap.parse_args()
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[-1]
    out_dir = resolve_cli_root(
        "source", PROJECT_ROOT, args.output, "output/topotex_source"
    ).resolve()
    if args.finalize:
        finalize_source(out_dir)
        return
    assert args.input_manifest, "--input-manifest required to build"
    entries = [json.loads(l) for l in open(args.input_manifest)]
    for e in entries:  # absolutize BEFORE the provider chdirs into UniTEX
        e["glb_path"] = str(Path(e["glb_path"]).resolve())
    if args.limit:
        entries = entries[: args.limit]
    entries = entries[args.rank :: args.world_size]
    if args.gate_only:
        gate_dir = out_dir / "gate_scan"
        gate_dir.mkdir(parents=True, exist_ok=True)
        elig = open(gate_dir / f"eligible_rank_{args.rank}.jsonl", "w")
        rej = open(gate_dir / f"rejected_rank_{args.rank}.jsonl", "w")
        t0 = time.time()
        n_e = n_r = 0
        for e in entries:
            try:
                vertices, faces, uv, tex = extract_glb(e["glb_path"])
                uv_address(uv, faces)
                elig.write(
                    json.dumps({**e, "num_faces": int(len(faces))}) + "\n"
                )
                n_e += 1
            except SkipSample as s:
                rej.write(json.dumps({**e, "reason": str(s)}) + "\n")
                n_r += 1
            except Exception as ex:
                rej.write(
                    json.dumps({**e, "reason": f"exception: {ex}"}) + "\n"
                )
                n_r += 1
            if (n_e + n_r) % 500 == 0:
                print(
                    f"[rank {args.rank}] {n_e + n_r}/{len(entries)} "
                    f"({n_e} eligible, {time.time() - t0:.0f}s)",
                    flush=True,
                )
        print(
            f"rank {args.rank} gate scan: {n_e} eligible / {n_r} rejected "
            f"({time.time() - t0:.0f}s)"
        )
        return
    for sub in ("samples", "failures"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    scratch = Path(args.scratch_root) / f"rank_{args.rank}"
    scratch.mkdir(parents=True, exist_ok=True)
    from topotex.data.multiview import UniTexMV

    provider = UniTexMV(seed=args.seed, resolution=RES)
    man_p = out_dir / f"manifest_rank_{args.rank}.jsonl"
    fail_p = out_dir / "failures" / f"rank_{args.rank}.jsonl"
    man_f = open(man_p, "a")
    fail_f = open(fail_p, "a")
    done = (
        {json.loads(l)["sample_id"] for l in open(man_p)}
        if man_p.stat().st_size
        else set()
    )
    n_ok = n_skip = n_fail = 0
    t_all = time.time()
    for e in entries:
        sid, glb = e["sample_id"], Path(e["glb_path"]).resolve()
        if sid in done or sample_is_valid(out_dir, sid):
            if sid not in done:
                meta = json.loads(
                    (out_dir / "samples" / sid / "meta.json").read_text()
                )
                man_f.write(
                    json.dumps(
                        {
                            "sample_id": sid,
                            **{
                                k: meta[k]
                                for k in (
                                    "num_faces",
                                    "valid_texels",
                                    "build_seconds",
                                )
                            },
                        }
                    )
                    + "\n"
                )
                man_f.flush()
            n_skip += 1
            continue
        try:
            meta = build_source_sample(provider, sid, glb, out_dir, scratch)
            man_f.write(
                json.dumps(
                    {
                        "sample_id": sid,
                        "num_faces": meta["num_faces"],
                        "valid_texels": meta["valid_texels"],
                        "build_seconds": meta["build_seconds"],
                    }
                )
                + "\n"
            )
            man_f.flush()
            n_ok += 1
            print(
                f"[rank {args.rank}] [{n_ok}] {sid[:12]} ok "
                f"({meta['build_seconds']:.0f}s, F={meta['num_faces']})",
                flush=True,
            )
        except SkipSample as e_skip:
            n_fail += 1
            fail_f.write(
                json.dumps({"sample_id": sid, "reason": str(e_skip)}) + "\n"
            )
            fail_f.flush()
            print(f"[rank {args.rank}] skip {sid[:12]}: {e_skip}", flush=True)
        except Exception:
            n_fail += 1
            (out_dir / "failures" / f"{sid}.error.log").write_text(
                traceback.format_exc()
            )
            fail_f.write(
                json.dumps(
                    {"sample_id": sid, "reason": "exception (see error log)"}
                )
                + "\n"
            )
            fail_f.flush()
            print(f"[rank {args.rank}] FAIL {sid[:12]}", flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    print(
        f"rank {args.rank} done: {n_ok} built / {n_skip} resumed / "
        f"{n_fail} skip+fail ({time.time() - t_all:.0f}s)"
    )


# ==================================================================
# UV query set builder (canonical/alternative/partial/held-out)
# ==================================================================
"""TOPOTEX dataset builder: source sample -> UV query set.

python -m topotex.data.builder queries --limit 265
# 8-way sharded (ids[rank::world_size], per-rank manifest, then merge):
#   python -m topotex.data.builder queries --rank K --world_size 8
#   python -m topotex.data.builder queries --finalize [--limit N]

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
import subprocess

from topotex.data.uv import connected_subset, face_adjacency
from topotex.paths import data_root, resolve_cli_root

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = data_root("source", PROJECT_ROOT)
RES = 256
SCHEMA = "topotex_dataset@1"
PARTIAL_SEED = 20260729
PARTIAL_FRACS = (0.25, 0.5, 0.75)


def builder_commit():
    """Short commit of the builder code writing this sample (provenance,
    recorded in meta.json and checked by topotex.data.integrity).
    A modified working tree is stamped '-dirty' — a clean commit id must
    never be recorded for code that commit does not contain."""
    try:
        c = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            [
                "git",
                "-C",
                str(PROJECT_ROOT),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
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
    out = (
        t[y0, x0] * (1 - tx) * (1 - ty)
        + t[y0, x1] * tx * (1 - ty)
        + t[y1, x0] * (1 - tx) * ty
        + t[y1, x1] * tx * ty
    )
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def blender_smart_uv(V, F):
    """Held-out UV family: Blender Smart UV Project (angle-based projection
    clustering) — a genuinely different unwrap family from native/xatlas.
    Per-corner UVs; face order verified preserved."""
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    me = bpy.data.meshes.new("m")
    me.from_pydata(
        V.astype(np.float64).tolist(), [], F.astype(np.int64).tolist()
    )
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
    loop_verts = np.array(
        [me.loops[i].vertex_index for i in range(n_loops)]
    ).reshape(len(F), 3)
    assert (loop_verts == F.astype(np.int64)).all(), (
        "blender changed face identity/order"
    )
    uv = np.array(
        [me.uv_layers.active.data[i].uv[:] for i in range(n_loops)], np.float32
    )
    uv[:, 1] = 1.0 - uv[:, 1]  # v-up -> top-down
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
    assert (back == F.astype(np.int64)).all(), (
        "xatlas changed face identity/order"
    )
    uvs = uvs.astype(np.float32)
    uvs[:, 1] = 1.0 - uvs[:, 1]  # v-up -> our top-down texture rows
    return np.clip(uvs, 0, 1), idx.astype(np.int32)


def rasterize_query(
    uv_vertices,
    uv_faces,
    n_faces,
    uv_face_to_mesh_face=None,
    min_occupancy=0.01,
):
    from topotex.data.uv import rasterize_uv

    if uv_face_to_mesh_face is None:
        uv_face_to_mesh_face = np.arange(n_faces)
    am = rasterize_uv(
        uv_vertices.astype(np.float64),
        uv_faces.astype(np.int64),
        RES,
        uv_face_to_mesh_face=uv_face_to_mesh_face,
    )
    valid = am.valid_mask.astype(bool)
    if valid.mean() < min_occupancy:
        raise RuntimeError("uv query occupancy below threshold")
    return (
        am.face_id.astype(np.int32),
        am.barycentric.astype(np.float16).transpose(2, 0, 1),
        valid.astype(np.uint8),
    )


def bake_gt(face_id, bary, valid, native_uv_verts, native_uv_faces, gt_u8):
    """(face, bary) -> native UV -> bilinear sample of the original texture."""
    out = np.zeros((RES, RES, 3), np.uint8)
    v = valid.astype(bool)
    fid = face_id[v].astype(np.int64)
    b = bary.astype(np.float32).transpose(1, 2, 0)[v]  # [K,3]
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
    for _ in range(5):  # retry on degenerate patches
        keep = connected_subset(adj, len(F), frac, rng)
        try:
            fid, bary, valid = rasterize_query(
                native_uv.astype(np.float32),
                native_uvf[keep],
                len(F),
                uv_face_to_mesh_face=keep,
                min_occupancy=64 / RES**2,
            )
            return fid, bary, valid, frac, len(keep)
        except RuntimeError:
            continue
    raise RuntimeError("partial query kept too few texels after 5 tries")


def build_query_sample(sid, out_root):
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
    for name, qtype in (
        ("uv_000", "canonical"),
        ("uv_001", "alternative"),
        ("uv_002", "partial"),
        ("uv_test", "heldout"),
    ):
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
                F, native_uv, native_uvf, sid
            )
            gt_k = bake_gt(fid, bary, valid, native_uv, native_uvf, gt)
            extra = {"partial_frac": frac, "n_kept_faces": n_kept}
        else:  # heldout
            uvv, uvf = blender_smart_uv(V, F)
            fid, bary, valid = rasterize_query(uvv, uvf, len(F))
            gt_k = bake_gt(fid, bary, valid, native_uv, native_uvf, gt)
        qd = q_root / name
        qd.mkdir(exist_ok=True)
        save_file(
            {
                "uv_vertices": uvv.astype(np.float32),
                "uv_faces": uvf.astype(np.int32),
                "face_id": fid,
                "barycentric": bary,
                "valid_mask": valid,
            },
            str(qd / "uv_address.safetensors"),
        )
        Image.fromarray(gt_k).save(qd / "gt_texture.png")
        queries.append(
            {
                "name": name,
                "type": qtype,
                "held_out": qtype == "heldout",
                "valid_texels": int(valid.sum()),
                "n_uv_vertices": int(len(uvv)),
                "face_id_sha256": sha256_bytes(fid.tobytes()),
                "barycentric_sha256": sha256_bytes(bary.tobytes()),
                **extra,
            }
        )
    meta_tmp = dst / "meta.json.part"
    meta_tmp.write_text(
        json.dumps(
            {
                "sample_id": sid,
                "schema": SCHEMA,
                "source": "topotex_source",
                "num_faces": int(len(F)),
                "uv_queries": queries,
                "query_schema_version": SCHEMA,
                "query_builder_commit": builder_commit(),
                "source_texture_sha256": sha256_bytes(
                    (src / "gt_texture.png").read_bytes()
                ),
                "build_seconds": round(time.time() - t0, 1),
            },
            indent=1,
        )
    )
    os.replace(meta_tmp, dst / "meta.json")  # atomic: no truncated metas
    return queries


def main_queries():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=265)
    ap.add_argument("--ids", default=None, help="comma-separated sample ids")
    ap.add_argument("--output", default="output/topotex_dataset")
    ap.add_argument(
        "--world-size", "--world_size", dest="world_size", type=int, default=1
    )
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument(
        "--finalize",
        action="store_true",
        help="merge per-rank manifests, validate, write "
        "manifest.jsonl + dataset_meta.json",
    )
    args = ap.parse_args()
    out_root = resolve_cli_root(
        "dataset", PROJECT_ROOT, args.output, "output/topotex_dataset"
    ).resolve()
    if args.finalize:
        finalize_queries(out_root, args.limit)
        return
    if args.ids:
        ids = args.ids.split(",")
    else:
        ids = [
            json.loads(l)["sample_id"]
            for l in open(SOURCE_ROOT / "manifest.jsonl")
        ][: args.limit]
    ids = ids[args.rank :: args.world_size]
    rows = []
    rank_fail = []
    for sid in ids:
        meta_f = out_root / "samples" / sid / "meta.json"
        if meta_f.exists():
            try:  # corrupt meta -> fall through, rebuild
                qs = json.loads(meta_f.read_text())["uv_queries"]
            except Exception:
                qs = []
            if len(qs) == 4:  # complete -> skip rebuild
                rows.append(
                    {
                        "sample_id": sid,
                        "valid_texels": [q["valid_texels"] for q in qs],
                    }
                )
                continue
        try:
            qs = build_query_sample(sid, out_root)
            rows.append(
                {
                    "sample_id": sid,
                    "valid_texels": [q["valid_texels"] for q in qs],
                }
            )
            print(
                f"{sid[:12]} ok | texels {[q['valid_texels'] for q in qs]}",
                flush=True,
            )
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
        finalize_queries(out_root, args.limit)


def finalize_queries(out_root, limit=None):
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
    fails = [
        json.loads(l)
        for p in sorted(out_root.glob("failures_rank_*.jsonl"))
        for l in open(p)
    ]
    with open(out_root / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(out_root / "dataset_failures.jsonl", "w") as f:
        for r in fails:
            f.write(json.dumps(r) + "\n")
    (out_root / "dataset_meta.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "n_samples": len(rows),
                "queries": ["canonical", "alternative", "partial", "heldout"],
            },
            indent=1,
        )
    )
    print(f"finalized: {len(rows)} samples, {len(fails)} failures")


# ==================================================================
# per-rank manifest merge
# ==================================================================
"""Merge per-rank build manifests into the final dataset manifest.

python -m topotex.data.builder merge --output output/topotex_source \
    [--input-manifest output/source_manifests/glbs_eligible.jsonl]

Checks:
  no duplicate  -- the same sample_id recorded by two ranks must agree
                   (re-recording after a resharded resume is legal; a
                   conflicting record is an error)
  no missing    -- with --input-manifest, every input id must be accounted
                   for: built (manifest) or failed (dataset_failures)
  schema valid  -- every published sample passes check_sample_schema
"""


def merge(out_dir, input_manifest=None):
    out_dir = Path(out_dir)
    parts = sorted(out_dir.glob("manifest_rank_*.jsonl"))
    assert parts, f"no manifest_rank_*.jsonl under {out_dir}"
    rows, by_id = [], {}
    for p in parts:
        for l in open(p):
            r = json.loads(l)
            sid = r["sample_id"]
            if sid in by_id:
                assert by_id[sid] == r, (
                    f"conflicting duplicate for {sid}: {by_id[sid]} vs {r}"
                )
                continue
            by_id[sid] = r
            rows.append(r)
    for r in rows:
        check_sample_schema(out_dir / "samples" / r["sample_id"])
    with open(out_dir / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    fails = [
        json.loads(l)
        for p in sorted((out_dir / "failures").glob("rank_*.jsonl"))
        for l in open(p)
    ]
    with open(out_dir / "dataset_failures.jsonl", "w") as f:
        for r in fails:
            f.write(json.dumps(r) + "\n")
    missing = []
    if input_manifest:
        wanted = [json.loads(l)["sample_id"] for l in open(input_manifest)]
        accounted = set(by_id) | {r["sample_id"] for r in fails}
        missing = [sid for sid in wanted if sid not in accounted]
        assert not missing, (
            f"{len(missing)} input ids unaccounted for "
            f"(neither built nor failed): {missing[:5]}"
        )
    (out_dir / "dataset_meta.json").write_text(
        json.dumps(
            {
                "dataset_schema": DATASET_SCHEMA,
                "resolution": RES,
                "view_order": ",".join(VIEW_ORDER),
                "n_samples": len(rows),
                "n_failures": len(fails),
                "finalized_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=1,
        )
    )
    print(
        f"merged: {len(rows)} samples, {len(fails)} failures, "
        f"0 duplicates conflicting, {len(missing)} missing"
    )
    return rows


def main_merge():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--input-manifest",
        default=None,
        help="optional: verify every input id is accounted for",
    )
    args = ap.parse_args()
    merge(
        resolve_cli_root(
            "source", PROJECT_ROOT, args.output, "output/topotex_source"
        ),
        input_manifest=args.input_manifest,
    )


# ================================================================== CLI
def main():
    """python -m topotex.data.builder {source|queries|merge} [args...]"""

    cmds = {
        "source": main_source,
        "queries": main_queries,
        "merge": main_merge,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        raise SystemExit(
            "usage: python -m topotex.data.builder "
            "{source|queries|merge} [args...]"
        )
    cmd = sys.argv.pop(1)
    cmds[cmd]()


if __name__ == "__main__":
    main()
