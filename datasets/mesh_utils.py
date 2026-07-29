# -*- coding: utf-8 -*-
"""Minimal geometry/rendering utilities for the dataset pipeline and evaluation.

Self-contained (migrated from the retired uvdata framework): cameras,
nvdiffrast rasterization, texture-rebake rendering, sRGB helpers, texture
dilation, textureless OBJ export. Nothing else.
"""
import math

import numpy as np
import torch

CANONICAL_VIEWS = [("front", 0, 0), ("back", 180, 0), ("left", 90, 0),
                   ("right", 270, 0), ("top", 0, 89), ("bottom", 0, -89)]
REFERENCE_VIEW = (30.0, 20.0)   # deterministic 3/4 reference camera (az, el)

_CTX = None
_SRGB_LUT = None


def _ctx():
    global _CTX
    if _CTX is None:
        import nvdiffrast.torch as dr
        _CTX = dr.RasterizeCudaContext(device="cuda:0")
    return _CTX


def camera_matrices(azim_deg, elev_deg, bbox_min, bbox_max, fov_deg=40.0,
                    margin=1.15, znear=None, zfar=None):
    center = (np.asarray(bbox_min) + np.asarray(bbox_max)) / 2.0
    radius = float(np.linalg.norm(
        np.asarray(bbox_max) - np.asarray(bbox_min)) / 2.0)
    radius = max(radius, 1e-6)
    dist = radius * margin / math.tan(math.radians(fov_deg) / 2.0)
    # clip planes scale with the object (raw GLB scales vary wildly)
    if znear is None:
        znear = max(dist - 2.0 * radius, dist * 1e-3)
    if zfar is None:
        zfar = dist + 2.0 * radius
    az, el = math.radians(azim_deg), math.radians(elev_deg)
    eye = center + dist * np.array([math.cos(el) * math.sin(az),
                                    math.sin(el),
                                    math.cos(el) * math.cos(az)])
    up = np.array([0.0, 1.0, 0.0])
    f = center - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = s, u, -f
    view[:3, 3] = -view[:3, :3] @ eye
    t = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    proj = np.zeros((4, 4))
    proj[0, 0] = t
    proj[1, 1] = t
    proj[2, 2] = (zfar + znear) / (znear - zfar)
    proj[2, 3] = 2 * zfar * znear / (znear - zfar)
    proj[3, 2] = -1.0
    return {"view": view, "proj": proj, "eye": eye, "fov_deg": fov_deg,
            "azimuth_deg": azim_deg, "elevation_deg": elev_deg}


def rasterize_view(vertices, faces, cam, res=512):
    """Returns dict with face_id [H,W] (-1 bg), bary [H,W,3], mask, zw.
    nvdiffrast bary convention (verified vs dr.interpolate): u -> vertex0,
    v -> vertex1, 1-u-v -> vertex2. Row 0 = top (PNG convention)."""
    import nvdiffrast.torch as dr
    V = torch.as_tensor(np.ascontiguousarray(vertices), dtype=torch.float32,
                        device="cuda:0")
    F = torch.as_tensor(np.ascontiguousarray(faces), dtype=torch.int32,
                        device="cuda:0")
    mvp = torch.as_tensor(cam["proj"] @ cam["view"], dtype=torch.float32,
                          device="cuda:0")
    Vh = torch.cat([V, torch.ones_like(V[:, :1])], dim=1) @ mvp.T
    rast, _ = dr.rasterize(_ctx(), Vh[None], F, resolution=[res, res])
    rast = rast[0]
    fid = rast[..., 3].long() - 1                     # -1 = background
    u = rast[..., 0].double()
    v = rast[..., 1].double()
    bary = torch.stack([u, v, 1 - u - v], dim=-1)
    fid = torch.flip(fid, dims=[0])
    bary = torch.flip(bary, dims=[0])
    zw = torch.flip(rast[..., 2], dims=[0])
    return {"face_id": fid.cpu().numpy().astype(np.int64),
            "bary": bary.cpu().numpy(),
            "mask": (fid >= 0).cpu().numpy(),
            "zw": zw.cpu().numpy()}


def _srgb_lut():
    global _SRGB_LUT
    if _SRGB_LUT is None:
        x = np.arange(256, dtype=np.float64) / 255.0
        _SRGB_LUT = np.where(x <= 0.04045, x / 12.92,
                             ((x + 0.055) / 1.055) ** 2.4)
    return _SRGB_LUT


def sample_texture_bilinear(tex_u8, uv):
    """tex [h,w,3] uint8 sRGB; uv [K,2] (v down), REPEAT wrap -> linear [K,3]."""
    h, w = tex_u8.shape[:2]
    tex = _srgb_lut()[tex_u8]
    x = uv[:, 0] * w - 0.5
    y = uv[:, 1] * h - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    tx = (x - x0)[:, None]
    ty = (y - y0)[:, None]
    x0m, x1m = x0 % w, (x0 + 1) % w
    y0m, y1m = y0 % h, (y0 + 1) % h
    return (tex[y0m, x0m] * (1 - tx) * (1 - ty)
            + tex[y0m, x1m] * tx * (1 - ty)
            + tex[y1m, x0m] * (1 - tx) * ty
            + tex[y1m, x1m] * tx * ty)


def render_albedo_rebake(canon, uvr, atlas_u8, gb):
    """Interpolate UVs at rasterized pixels, bilinear-sample the atlas."""
    mask = gb["mask"]
    H, W = mask.shape
    m2u = np.empty(len(canon.faces), dtype=np.int64)
    m2u[np.asarray(uvr.uv_face_to_mesh_face, dtype=np.int64)] = \
        np.arange(len(uvr.uv_faces))
    lin = np.zeros((H, W, 3), dtype=np.float64)
    if mask.any():
        fid = gb["face_id"][mask]
        uvf = np.asarray(uvr.uv_faces, dtype=np.int64)[m2u[fid]]
        uvc = np.asarray(uvr.uv_vertices, dtype=np.float64)[uvf]
        uv = (uvc * gb["bary"][mask][..., None]).sum(axis=1)
        lin[mask] = sample_texture_bilinear(atlas_u8, uv)
    return lin


def linear_to_srgb_u8(rgb_linear):
    x = np.clip(rgb_linear, 0.0, 1.0)
    s = np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)
    return np.round(s * 255.0).astype(np.uint8)


def dilate_texture(rgb_u8, valid_mask):
    """Fill invalid texels with nearest valid texel color (anti-bleeding)."""
    from scipy import ndimage
    invalid = ~valid_mask.astype(bool)
    if invalid.all() or not invalid.any():
        return rgb_u8.copy()
    _, (iy, ix) = ndimage.distance_transform_edt(invalid, return_indices=True)
    return rgb_u8[iy, ix]


def export_textureless_obj(vertices, faces, out_obj):
    with open(out_obj, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for a, b, c in np.asarray(faces) + 1:
            f.write(f"f {a} {b} {c}\n")


def seam_error(faces, uv_vertices, uv_faces, texture_u8, valid_mask=None,
               samples_per_edge=8, inset=0.03, seam_px_threshold=1.0):
    """UV seam consistency of a texture on a mesh.

    Every shared mesh edge is one surface curve with (up to) two UV images.
    Sample the same 3D points on both sides (barycentric mix of the edge
    corners, inset slightly toward each face's interior so bilinear taps
    stay inside the chart), map through each face's UV, and compare colors.
    Edges whose two UV images coincide (< seam_px_threshold texels apart)
    are atlas-contiguous and skipped — the metric targets real seams.

    Returns dict:
      seam_error       mean ||RGB_A - RGB_B|| over seam samples ([0,1] RGB)
      n_seam_edges / n_shared_edges
      per_face_error   float [F] (max seam error of the face's seam edges)

    Note: native per-vertex-UV layouts (canonical query) encode their seams
    as duplicated vertices — such edges are topological boundaries, not
    shared edges, so this metric reports 0 seams there by construction. It
    targets re-parameterized layouts (per-corner uv_faces, e.g. xatlas).
    """
    F = np.asarray(faces, dtype=np.int64)
    uvv = np.asarray(uv_vertices, dtype=np.float64)
    uvf = np.asarray(uv_faces, dtype=np.int64)
    H, W = texture_u8.shape[:2]
    if valid_mask is not None:
        # rendering convention: dilate chart borders so bilinear taps near
        # a seam do not blend with the empty background
        texture_u8 = dilate_texture(texture_u8, valid_mask.astype(bool))
    tex = texture_u8.astype(np.float32) / 255.0

    def bilinear(uv):
        x = np.clip(uv[:, 0] * W - 0.5, 0, W - 1)
        y = np.clip(uv[:, 1] * H - 0.5, 0, H - 1)
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        x1 = np.minimum(x0 + 1, W - 1)
        y1 = np.minimum(y0 + 1, H - 1)
        tx = (x - x0)[:, None]
        ty = (y - y0)[:, None]
        return (tex[y0, x0] * (1 - tx) * (1 - ty)
                + tex[y0, x1] * tx * (1 - ty)
                + tex[y1, x0] * (1 - tx) * ty
                + tex[y1, x1] * tx * ty)

    e2f = {}
    for fi, tri in enumerate(F):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            e = (min(tri[a], tri[b]), max(tri[a], tri[b]))
            e2f.setdefault(e, []).append(fi)

    t = (np.arange(samples_per_edge) + 0.5) / samples_per_edge
    per_face = np.zeros(len(F), np.float32)
    diffs, n_seam, n_shared = [], 0, 0

    def edge_uv(fi, u, v):
        """UV samples along edge (u,v) inside face fi, inset to interior."""
        corners = list(F[fi])
        iu, iv = corners.index(u), corners.index(v)
        iw = 3 - iu - iv
        bary = np.zeros((samples_per_edge, 3))
        bary[:, iu] = (1 - t) * (1 - inset)
        bary[:, iv] = t * (1 - inset)
        bary[:, iw] = inset
        return bary @ uvv[uvf[fi]]

    for (u, v), fs in e2f.items():
        if len(fs) != 2:
            continue
        n_shared += 1
        fa, fb = fs
        uva = edge_uv(fa, u, v)
        uvb = edge_uv(fb, u, v)
        px = np.abs(uva - uvb) * [W, H]
        if px.max() < seam_px_threshold + 2 * inset * max(W, H) * 0.5:
            continue                       # atlas-contiguous edge
        n_seam += 1
        d = np.linalg.norm(bilinear(uva) - bilinear(uvb), axis=1)
        diffs.append(d)
        m = float(d.mean())
        per_face[fa] = max(per_face[fa], m)
        per_face[fb] = max(per_face[fb], m)

    return {"seam_error": (round(float(np.concatenate(diffs).mean()), 4)
                           if diffs else 0.0),
            "n_seam_edges": n_seam, "n_shared_edges": n_shared,
            "per_face_error": per_face}
