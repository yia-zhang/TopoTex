# -*- coding: utf-8 -*-
"""TOPOTEX source dataset builder: textured GLB -> one training sample.

python -m datasets.build_dataset --input-manifest output/source_manifests/glbs.jsonl \
    --output output/topotex_source --limit 10
# 8-GPU sharding: --world_size 8 --rank K --device cuda:0 (one process/GPU,
#   ids[rank::world_size], one UniTEX generator load per worker)
# finalize (merge manifest parts + schema check):
python -m datasets.build_dataset --output output/topotex_source --finalize

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_SCHEMA = "topotex_source"
VIEW_ORDER = ["front", "back", "left", "right", "top", "bottom"]
RES = 256
REFERENCE_CAM = (30.0, 20.0)          # deterministic 3/4 view (az, el)
OVERLAP_TOL = 1.05                    # UV triangle area vs covered texels


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
        if (uv_attr is not None and len(uv_attr) == len(m.vertices)
                and getattr(getattr(vis, "material", None),
                            "baseColorTexture", None) is not None):
            textured.append((T, m))
    if not textured:
        raise SkipSample("no uv+baseColorTexture")
    imgs = {np.asarray(m.visual.material.baseColorTexture.convert("RGB"))
            .tobytes() for _, m in textured}
    if len(imgs) > 1:
        raise SkipSample(f"multiple textures ({len(imgs)})")
    tex = np.asarray(
        textured[0][1].visual.material.baseColorTexture.convert("RGB"))
    V, F, UV = [], [], []
    off = 0
    for T, m in textured:
        v = (np.c_[m.vertices, np.ones(len(m.vertices))] @ np.asarray(T).T)[:, :3]
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
    uv[:, 1] = 1.0 - uv[:, 1]         # GLB v-up -> our top-down texture rows
    if uv.min() < -uv_tol or uv.max() > 1 + uv_tol:
        raise SkipSample("uv outside [0,1] (tiling texture)")
    return (vertices, faces.astype(np.int32), np.clip(uv, 0, 1),
            tex.astype(np.uint8))


def uv_address(uv, faces):
    """Deterministic UV rasterization -> (face_id i32, bary f16 [3,H,W],
    valid u8). Also gates on UV overlap."""
    from .rasterizer import rasterize_uv
    am = rasterize_uv(uv, faces.astype(np.int64), RES)
    valid = am.valid_mask.astype(bool)
    tri = uv[faces]
    area = 0.5 * np.abs(
        (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
        - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))
    covered = valid.mean()
    if covered < 0.01:
        raise SkipSample("uv occupancy < 1%")
    if area.sum() / max(covered, 1e-9) > OVERLAP_TOL:
        raise SkipSample(
            f"overlapping uv (area {area.sum():.3f} vs cover {covered:.3f})")
    bary = am.barycentric.astype(np.float16)          # [H,W,3]
    return (am.face_id.astype(np.int32),
            bary.transpose(2, 0, 1), valid.astype(np.uint8))


def render_reference(vertices, faces, uv, gt_texture, out_png):
    """One deterministic textured render (same conventions as evaluation)."""
    import types
    from PIL import Image
    from .mesh_utils import (camera_matrices, linear_to_srgb_u8,
                           rasterize_view, render_albedo_rebake)
    canon = types.SimpleNamespace(vertices=vertices.astype(np.float64),
                                  faces=faces.astype(np.int64))
    uvr = types.SimpleNamespace(uv_vertices=uv, uv_faces=faces.astype(np.int64),
                                uv_face_to_mesh_face=np.arange(len(faces)))
    bmin, bmax = canon.vertices.min(0), canon.vertices.max(0)
    cam = camera_matrices(*REFERENCE_CAM, bmin, bmax)
    gb = rasterize_view(canon.vertices, canon.faces, cam, 512)
    if gb["mask"].sum() < 100:
        raise SkipSample("reference render empty (degenerate geometry)")
    img = linear_to_srgb_u8(render_albedo_rebake(canon, uvr, gt_texture, gb))
    img[~gb["mask"]] = 235
    Image.fromarray(img).save(out_png)


def build_sample(provider, sid, glb_path, out_dir, scratch):
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
    from .mesh_utils import export_textureless_obj
    export_textureless_obj(vertices, faces, obj)
    inputs_dir = work / "mv_inputs"
    provider.prepare_inputs(work / "reference.png", obj, sid, inputs_dir)
    images, gmeta = provider.generate(inputs_dir, sid, work / "mv_work")
    save_file({"vertices": vertices, "faces": faces,
               "uv_vertices": uv, "uv_faces": faces},
              str(work / "mesh.safetensors"))
    save_file({"images": images,
               "view_id": np.arange(6, dtype=np.int32)},
              str(work / "mv.safetensors"))
    save_file({"face_id": face_id, "barycentric": bary, "valid_mask": valid},
              str(work / "uv_address.safetensors"))
    meta = {"sample_id": sid, "dataset_schema": DATASET_SCHEMA,
            "glb_sha256": sha256_file(glb_path),
            "num_vertices": int(len(vertices)), "num_faces": int(len(faces)),
            "texture_resolution": RES, "address_resolution": RES,
            "valid_texels": int(valid.sum()),
            "view_order": ",".join(VIEW_ORDER),
            "reference_camera": {"azimuth": REFERENCE_CAM[0],
                                 "elevation": REFERENCE_CAM[1]},
            "generator": gmeta,
            "build_seconds": round(time.time() - t0, 1)}
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


SAMPLE_FILES = ["reference.png", "mesh.safetensors", "mv.safetensors",
                "uv_address.safetensors", "gt_texture.png", "meta.json"]


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
    assert mesh["vertices"].dtype == np.float32 and mesh["vertices"].shape[1] == 3
    assert mesh["faces"].dtype == np.int32 and mesh["uv_faces"].shape == (F, 3)
    assert mesh["uv_vertices"].dtype == np.float32
    assert mv["images"].shape == (6, 3, RES, RES) and mv["images"].dtype == np.uint8
    assert list(mv["view_id"]) == list(range(6))
    assert ua["face_id"].shape == (RES, RES) and ua["face_id"].dtype == np.int32
    assert ua["barycentric"].shape == (3, RES, RES)
    assert ua["barycentric"].dtype == np.float16
    assert ua["valid_mask"].shape == (RES, RES)
    v = ua["valid_mask"].astype(bool)
    assert v.any()
    fid = ua["face_id"][v]
    assert fid.min() >= 0 and fid.max() < F
    s = ua["barycentric"].astype(np.float32).transpose(1, 2, 0)[v].sum(-1)
    assert np.abs(s - 1).max() < 2e-2


def finalize(out_dir, input_manifest=None):
    """Merge per-rank manifests; verify uniqueness, coverage and schema."""
    from .merge_manifest import merge
    merge(out_dir, input_manifest=input_manifest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-manifest", default=None,
                    help="jsonl: {sample_id, glb_path} per line")
    ap.add_argument("--output", default="output/topotex_source")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--world-size", "--world_size", dest="world_size",
                    type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--scratch-root", default="/tmp/topotex_source")
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--gate-only", action="store_true",
                    help="run extraction/UV gates only (NO FLUX, no sample "
                         "publish); writes eligible/rejected manifest parts")
    args = ap.parse_args()
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[-1]
    out_dir = Path(args.output).resolve()
    if args.finalize:
        finalize(out_dir)
        return
    assert args.input_manifest, "--input-manifest required to build"
    entries = [json.loads(l) for l in open(args.input_manifest)]
    for e in entries:   # absolutize BEFORE the provider chdirs into UniTEX
        e["glb_path"] = str(Path(e["glb_path"]).resolve())
    if args.limit:
        entries = entries[: args.limit]
    entries = entries[args.rank::args.world_size]
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
                elig.write(json.dumps({**e, "num_faces": int(len(faces))})
                           + "\n")
                n_e += 1
            except SkipSample as s:
                rej.write(json.dumps({**e, "reason": str(s)}) + "\n")
                n_r += 1
            except Exception as ex:
                rej.write(json.dumps({**e, "reason": f"exception: {ex}"})
                          + "\n")
                n_r += 1
            if (n_e + n_r) % 500 == 0:
                print(f"[rank {args.rank}] {n_e + n_r}/{len(entries)} "
                      f"({n_e} eligible, {time.time()-t0:.0f}s)", flush=True)
        print(f"rank {args.rank} gate scan: {n_e} eligible / {n_r} rejected "
              f"({time.time()-t0:.0f}s)")
        return
    for sub in ("samples", "failures"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    scratch = Path(args.scratch_root) / f"rank_{args.rank}"
    scratch.mkdir(parents=True, exist_ok=True)
    from .mv_generator import UniTexMV
    provider = UniTexMV(seed=args.seed, resolution=RES)
    man_p = out_dir / f"manifest_rank_{args.rank}.jsonl"
    fail_p = out_dir / "failures" / f"rank_{args.rank}.jsonl"
    man_f = open(man_p, "a")
    fail_f = open(fail_p, "a")
    done = {json.loads(l)["sample_id"] for l in open(man_p)} \
        if man_p.stat().st_size else set()
    n_ok = n_skip = n_fail = 0
    t_all = time.time()
    for e in entries:
        sid, glb = e["sample_id"], Path(e["glb_path"]).resolve()
        if sid in done or sample_is_valid(out_dir, sid):
            if sid not in done:
                meta = json.loads(
                    (out_dir / "samples" / sid / "meta.json").read_text())
                man_f.write(json.dumps(
                    {"sample_id": sid, **{k: meta[k] for k in
                     ("num_faces", "valid_texels", "build_seconds")}}) + "\n")
                man_f.flush()
            n_skip += 1
            continue
        try:
            meta = build_sample(provider, sid, glb, out_dir, scratch)
            man_f.write(json.dumps(
                {"sample_id": sid, "num_faces": meta["num_faces"],
                 "valid_texels": meta["valid_texels"],
                 "build_seconds": meta["build_seconds"]}) + "\n")
            man_f.flush()
            n_ok += 1
            print(f"[rank {args.rank}] [{n_ok}] {sid[:12]} ok "
                  f"({meta['build_seconds']:.0f}s, F={meta['num_faces']})",
                  flush=True)
        except SkipSample as e_skip:
            n_fail += 1
            fail_f.write(json.dumps({"sample_id": sid,
                                     "reason": str(e_skip)}) + "\n")
            fail_f.flush()
            print(f"[rank {args.rank}] skip {sid[:12]}: {e_skip}", flush=True)
        except Exception:
            n_fail += 1
            (out_dir / "failures" / f"{sid}.error.log").write_text(
                traceback.format_exc())
            fail_f.write(json.dumps({"sample_id": sid,
                                     "reason": "exception (see error log)"})
                         + "\n")
            fail_f.flush()
            print(f"[rank {args.rank}] FAIL {sid[:12]}", flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"rank {args.rank} done: {n_ok} built / {n_skip} resumed / "
          f"{n_fail} skip+fail ({time.time() - t_all:.0f}s)")


if __name__ == "__main__":
    main()
