# -*- coding: utf-8 -*-
"""Minimal Orient-Anything-V2 GLB canonicalizer.

    raw.glb -> canonical.glb + transform.json + preview.png

Pipeline: load GLB -> flatten scene graph -> center + unit-scale -> render
ONE image with a FIXED known camera (az 30, el 20 — matches the TOPOTEX
reference camera; 3/4 view gives OA-V2 more shape cues than dead-front)
-> OA-V2 (isolated env, subprocess) -> convert the predicted (azimuth,
polar, in-plane) to a world-space correction rotation -> rotate the full
mesh -> export.

COORDINATE CONVENTION (frozen by tests/test_orientation_conventions.py —
calibrated against known rotations, never by intuition):
  * Our world: Y up; camera azimuth a has eye direction [sin a, 0, cos a];
    canonical front = +Z (faces the az=0 camera).
  * OA-V2 head output (official decode): az in [0,360) — horizontal angle
    of the CAMERA around the object measured from the object's front
    (az=0 <=> camera looks at the front); el in [-90,90) — camera height
    angle; ro in [-180,180) — in-plane roll of the object in the image.
  * Official object-rotation matrix (utils/app_utils.py):
        R_obj = Rx(ro) @ Ry(el) @ Rz(-az)   (their gizmo frame)
  * The calibrated change of basis THEIR gizmo frame -> OUR world and the
    camera-relative composition live in `predicted_world_rotation()`; the
    BASIS matrix below is selected by the calibration test from the full
    candidate set and then frozen.

Camera metadata is used ONLY for this offline canonicalization step.
It is never an input to TOPOTEX.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

TOPOTEX = "/root/youjiaZhang/TopoTex"
sys.path.insert(0, TOPOTEX)

OA2_ENV_PY = next(
    (p for p in ("/root/youjiaZhang/envs/orient_anything_v2/bin/python",
                 "/root/youjiaZhang/oa2_env/bin/python")
     if __import__("os").path.exists(p)
     and __import__("subprocess").run(
         [p, "-c", "import torch"], capture_output=True).returncode == 0),
    "/root/youjiaZhang/envs/orient_anything_v2/bin/python",
)
OA2_WRAPPER = ("/root/youjiaZhang/topotex_OA_study/orientation_validation/"
               "oav2_infer.py")
OA2_REPO = "/root/youjiaZhang/topotex_OA_study/code/OrientAnything"
OA2_CKPT = ("/root/youjiaZhang/hf_cache/checkpoints/OriAnyV2/"
            "rotmod_realrotaug_best.pt")
RENDER_CAM = (30.0, 20.0)  # fixed known extrinsic (az, el) for the OA-V2 input


def rot(axis, deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def oa2_gizmo_rotation(az, el, ro):
    """Official R_obj = Rx(ro) @ Ry(el) @ Rz(-az) (column-vector form)."""
    return rot("x", ro) @ rot("y", el) @ rot("z", -az)


# Spherical-registration composition (calibrated by
# orientation_validation/calibrate_basis.py over known rotations):
# OA-V2's (az, el) is the CAMERA's spherical position in the object's
# canonical frame (el equals the camera elevation for an upright object —
# verified empirically), ro is the in-plane roll about the viewing axis.
# Given the known world camera (az_c, el_c), the object orientation R
# satisfies  cam_dir_world = R @ cam_dir_object,  with the remaining DOF
# pinned by ro about the viewing axis. RO_SIGN / AZ_MIRROR are the only
# discrete calibration constants (selected against known perturbations).
RO_SIGN = 1.0
AZ_MIRROR = 1.0
CALIB_FROZEN = True  # frozen 2026-08-01: ro_sign=1, az_mirror=1 (calibration.json)


def geodesic_deg(Ra, Rb):
    tr = np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(tr)))


def _cam_dir(az, el):
    a, e = np.deg2rad(az), np.deg2rad(el)
    return np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)])


def _cam_basis(az, el):
    """Right-handed camera-anchored basis with world/object up reference:
    z = direction object->camera, x = normalize(up x z), y = z x x."""
    z = _cam_dir(az, el)
    up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    n = np.linalg.norm(x)
    if n < 1e-8:  # camera straight above/below: fall back to +X reference
        x = np.array([1.0, 0.0, 0.0])
    else:
        x = x / n
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)  # columns


def _axis_rot(axis, deg):
    axis = axis / np.linalg.norm(axis)
    a = np.deg2rad(deg)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def predicted_world_rotation(az, el, ro, cam_az, cam_el=None):
    """Object orientation R (world <- object frame) implied by an OA-V2
    prediction taken from the known world camera (cam_az, cam_el)."""
    if cam_el is None:
        cam_el = RENDER_CAM[1]
    B_obj = _cam_basis(AZ_MIRROR * az, el)
    B_wld = _cam_basis(cam_az, cam_el)
    R0 = B_wld @ B_obj.T
    z_wld = B_wld[:, 2]
    return _axis_rot(z_wld, RO_SIGN * ro) @ R0


def correction_rotation(az, el, ro, cam_az, yaw_only=False):
    """World-space rotation that maps the observed pose back to canonical."""
    Rw = predicted_world_rotation(az, el, ro, cam_az)
    Rc = Rw.T  # undo the estimated orientation
    if yaw_only:
        f = Rc @ np.array([0.0, 0.0, 1.0])
        yaw = np.degrees(np.arctan2(f[0], f[2]))
        return rot("y", yaw)
    return Rc


def run_oa2(image_paths, gpu="0", rm_bkg=True):
    """Batch OA-V2 inference via the isolated env. Returns list of dicts.
    rm_bkg=True replicates the official demo preprocessing (rembg +
    foreground recrop) — the model's intended input distribution."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(str(p) for p in image_paths))
        lst = f.name
    out = subprocess.run(
        [OA2_ENV_PY, OA2_WRAPPER, lst, "--ckpt", OA2_CKPT]
        + (["--rm-bkg"] if rm_bkg else []),
        capture_output=True, text=True,
        env={"CUDA_VISIBLE_DEVICES": gpu, "PATH": "/usr/bin:/bin",
             "HF_HOME": "/root/youjiaZhang/hf_cache"},
        cwd=OA2_REPO, timeout=3600,
    )
    if out.returncode != 0:
        raise RuntimeError(f"oav2_infer failed: {out.stderr[-800:]}")
    rows = [json.loads(l) for l in out.stdout.splitlines() if l.startswith("{")]
    return rows, out.stderr


def render_fixed(V, F, uv, tex, out_png, cam=RENDER_CAM, res=512):
    from PIL import Image

    from topotex.data.mesh import (
        camera_matrices,
        linear_to_srgb_u8,
        rasterize_view,
        render_albedo_rebake,
    )
    V64 = V.astype(np.float64)
    F64 = F.astype(np.int64)
    canon = types.SimpleNamespace(vertices=V64, faces=F64)
    uvr = types.SimpleNamespace(uv_vertices=uv, uv_faces=F64,
                                uv_face_to_mesh_face=np.arange(len(F)))
    gb = rasterize_view(V64, F64,
                        camera_matrices(cam[0], cam[1], V64.min(0), V64.max(0)),
                        res)
    if gb["mask"].sum() < 100:
        raise RuntimeError("render empty")
    img = linear_to_srgb_u8(render_albedo_rebake(canon, uvr, tex, gb))
    img[~gb["mask"]] = 255
    Image.fromarray(img).save(out_png)


def canonicalize(raw_glb, out_dir, gpu="0"):
    """Full single-object pipeline. Returns the transform record."""
    from PIL import Image

    from topotex.data.builder import extract_glb, sha256_file
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    V, F, uv, tex = extract_glb(str(raw_glb))
    center = (V.max(0) + V.min(0)) / 2
    scale = float(np.abs(V - center).max())
    Vn = (V - center) / max(scale, 1e-9)

    prev = out_dir / "oa2_input.png"
    render_fixed(Vn, F, uv, tex, prev)
    rows, _ = run_oa2([prev], gpu=gpu)
    r = rows[0]
    az, el, ro = r["azimuth"], r["polar"], r["rotation"]
    Rc = correction_rotation(az, el, ro, cam_az=RENDER_CAM[0])
    Vc = (Vn @ Rc.T).astype(np.float32)

    import trimesh
    mesh = trimesh.Trimesh(vertices=Vc.astype(np.float64),
                           faces=F.astype(np.int64), process=False)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.c_[uv[:, 0], 1.0 - uv[:, 1]], image=Image.fromarray(tex))
    mesh.export(out_dir / "canonical.glb")
    render_fixed(Vc, F, uv, tex, out_dir / "preview.png", cam=(0.0, 10.0))

    rec = {
        "source_id": Path(raw_glb).stem,
        "source_glb_sha256": sha256_file(raw_glb),
        "render_camera": {"azimuth": RENDER_CAM[0], "elevation": RENDER_CAM[1],
                          "note": "offline canonicalization only; never a "
                                  "TOPOTEX input"},
        "oa_v2_commit": "73b11c9dc83e84daeb563d0c766831f2c66b0a18",
        "oa_v2_checkpoint_sha": OA2_CKPT_SHA,
        "predicted_orientation": {"azimuth": az, "polar": el, "rotation": ro},
        "predicted_symmetry": r["raw"].get("alpha_num_directions"),
        "confidence": r["confidence"],
        "canonical_rotation_matrix": Rc.tolist(),
        "normalization_translation": (-center).tolist(),
        "normalization_scale": float(1.0 / max(scale, 1e-9)),
        "runtime_seconds": round(time.time() - t0, 2),
        "basis_frozen": BASIS_FROZEN,
    }
    (out_dir / "transform.json").write_text(json.dumps(rec, indent=1))
    return rec


OA2_CKPT_SHA = "7b6b7f258d32b95123b9d023005ecca357d8ab944fb83476f532d3cf7a2295eb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_glb")
    ap.add_argument("out_dir")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    rec = canonicalize(args.raw_glb, args.out_dir, gpu=args.gpu)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
