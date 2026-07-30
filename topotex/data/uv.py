# -*- coding: utf-8 -*-
"""Deterministic UV surface-address rasterizer.

Convention (documented in IMPLEMENTATION_PLAN.md §2):
  texel (row j, col i) center <-> uv = ((i+0.5)/W, (j+0.5)/H), v runs top->down
  (glTF image convention). Address tensors share memory layout with saved PNGs.

Fill rule: top-left rule in y-down pixel space so adjacent triangles never
double-cover a shared edge. Edge functions evaluated in float64. Triangles with
negative signed area (flipped in UV) are rasterized with normalized winding but
counted; zero-area triangles are skipped and counted.

Overlap: per-texel coverage count; coverage > 1 => texel excluded from valid
(no last-write-wins) and reported. Canonical atlases must have overlap == 0.
"""

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class AddressMaps:
    valid_mask: np.ndarray  # uint8 [H,W]  1 = uniquely covered interior
    face_id: np.ndarray  # int32 [H,W]  row index into canonical faces, -1 bg
    barycentric: np.ndarray  # float32 [H,W,3] (order = mesh face vertex order)
    chart_id: np.ndarray  # int32 [H,W] -1 bg
    coverage: np.ndarray  # int32 [H,W] raw coverage count (diagnostic)
    xyz: np.ndarray  # float32 [H,W,3] canonical xyz (0 where invalid)
    normal: np.ndarray  # float32 [H,W,3]
    texel_density: np.ndarray  # float32 [H,W] 3d_area/uv_area (relative)
    stats: dict = field(default_factory=dict)


def _as_t(a, dtype, device):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)


def rasterize_uv(
    uv_vertices,
    uv_faces,
    res,
    mesh_vertices=None,
    mesh_faces=None,
    uv_face_to_mesh_face=None,
    chart_id_per_face=None,
    vertex_normals=None,
    device=None,
):
    """Rasterize a UV atlas at res x res. Returns AddressMaps.

    uv_vertices: [Nuv,2] float in [0,1]; uv_faces: [F,3] int; mesh_* give 3D
    attributes; uv_face_to_mesh_face maps uv face row -> canonical face row.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    H = W = int(res)
    uvv = _as_t(uv_vertices, torch.float64, device)
    fcs = _as_t(uv_faces, torch.long, device)
    F = fcs.shape[0]
    if uv_face_to_mesh_face is None:
        uv_face_to_mesh_face = np.arange(F, dtype=np.int64)
    u2m = _as_t(uv_face_to_mesh_face, torch.long, device)

    # pixel space (y down; uv v is already top-down per our convention)
    P = uvv * torch.tensor(
        [W, H], dtype=torch.float64, device=device
    )  # [Nuv,2]
    tri = P[fcs]  # [F,3,2]

    # signed area in y-down space (CW on screen == CCW in v-up math)
    e01 = tri[:, 1] - tri[:, 0]
    e02 = tri[:, 2] - tri[:, 0]
    area2 = e01[:, 0] * e02[:, 1] - e01[:, 1] * e02[:, 0]  # [F]
    zero_area = area2 == 0
    flipped = area2 < 0
    # normalize winding: make area positive by swapping vertices 1/2
    tri_n = tri.clone()
    tri_n[flipped] = tri[flipped][:, [0, 2, 1], :]
    perm = torch.tensor([0, 1, 2], device=device).repeat(F, 1)
    perm[flipped] = torch.tensor([0, 2, 1], device=device)

    # candidate texels from bboxes
    mn = tri_n.min(dim=1).values  # [F,2]
    mx = tri_n.max(dim=1).values
    x0 = torch.clamp(torch.ceil(mn[:, 0] - 0.5).long(), 0, W - 1)
    x1 = torch.clamp(torch.floor(mx[:, 0] - 0.5).long(), 0, W - 1)
    y0 = torch.clamp(torch.ceil(mn[:, 1] - 0.5).long(), 0, H - 1)
    y1 = torch.clamp(torch.floor(mx[:, 1] - 0.5).long(), 0, H - 1)
    bw = torch.clamp(x1 - x0 + 1, min=0)
    bh = torch.clamp(y1 - y0 + 1, min=0)
    ncand = bw * bh
    ncand[zero_area] = 0
    keep = ncand > 0
    fidx_all = torch.nonzero(keep, as_tuple=False).squeeze(1)
    if fidx_all.numel() == 0:
        z = np.zeros((H, W), np.int32)
        return AddressMaps(
            np.zeros((H, W), np.uint8),
            z - 1,
            np.zeros((H, W, 3), np.float32),
            z.copy() - 1,
            z.copy(),
            np.zeros((H, W, 3), np.float32),
            np.zeros((H, W, 3), np.float32),
            np.zeros((H, W), np.float32),
            {
                "num_zero_area": int(zero_area.sum()),
                "num_flipped": int(flipped.sum()),
            },
        )

    counts = ncand[fidx_all]
    face_of_cand = torch.repeat_interleave(fidx_all, counts)  # [C]
    # local offsets within each bbox
    total = int(counts.sum())
    start = torch.cumsum(counts, 0) - counts
    local = torch.arange(total, device=device) - torch.repeat_interleave(
        start, counts
    )
    bwf = bw[face_of_cand]
    lx = local % bwf
    ly = local // bwf
    px = x0[face_of_cand] + lx
    py = y0[face_of_cand] + ly
    cx = px.double() + 0.5
    cy = py.double() + 0.5

    t = tri_n[face_of_cand]  # [C,3,2]

    def edge(a, b):
        return (b[:, 0] - a[:, 0]) * (cy - a[:, 1]) - (b[:, 1] - a[:, 1]) * (
            cx - a[:, 0]
        )

    # CCW-in-ydown triangles: interior has positive edge functions
    # for edges (0->1, 1->2, 2->0)
    E0 = edge(t[:, 1], t[:, 2])  # opposite vertex 0
    E1 = edge(t[:, 2], t[:, 0])  # opposite vertex 1
    E2 = edge(t[:, 0], t[:, 1])  # opposite vertex 2

    def topleft(a, b):
        # Tie-break for pixel centers exactly on an edge. Adjacent triangles
        # traverse a shared edge in opposite directions, so a rule that depends
        # only on the lexicographic sign of the edge direction accepts
        # the pixel
        # in exactly one of the two triangles (top-left rule family).
        dy = b[:, 1] - a[:, 1]
        dx = b[:, 0] - a[:, 0]
        return (dy < 0) | ((dy == 0) & (dx > 0))

    tl0 = topleft(t[:, 1], t[:, 2])
    tl1 = topleft(t[:, 2], t[:, 0])
    tl2 = topleft(t[:, 0], t[:, 1])
    inside = (
        ((E0 > 0) | ((E0 == 0) & tl0))
        & ((E1 > 0) | ((E1 == 0) & tl1))
        & ((E2 > 0) | ((E2 == 0) & tl2))
    )

    face_in = face_of_cand[inside]
    pix = py[inside] * W + px[inside]
    Es = torch.stack([E0[inside], E1[inside], E2[inside]], dim=1)
    bary_n = Es / Es.sum(
        dim=1, keepdim=True
    )  # sums to exactly 1 (normalized winding order)
    # undo winding permutation -> barycentric in ORIGINAL uv_faces vertex order
    p = perm[face_in]  # [K,3] maps normalized slot -> original slot
    bary = torch.zeros_like(bary_n)
    bary.scatter_(1, p, bary_n)

    coverage = torch.zeros(H * W, dtype=torch.int32, device=device)
    coverage.scatter_add_(0, pix, torch.ones_like(pix, dtype=torch.int32))
    unique = coverage[pix] == 1

    face_map = torch.full((H * W,), -1, dtype=torch.long, device=device)
    face_map[pix[unique]] = face_in[unique]
    bary_map = torch.zeros(H * W, 3, dtype=torch.float64, device=device)
    bary_map[pix[unique]] = bary[unique]

    valid = face_map >= 0
    mesh_face_map = torch.full((H * W,), -1, dtype=torch.long, device=device)
    mesh_face_map[valid] = u2m[face_map[valid]]

    # chart ids
    chart_map = torch.full((H * W,), -1, dtype=torch.long, device=device)
    if chart_id_per_face is not None:
        cpf = _as_t(chart_id_per_face, torch.long, device)
        chart_map[valid] = cpf[face_map[valid]]

    # 3D attributes
    xyz_map = torch.zeros(H * W, 3, dtype=torch.float64, device=device)
    nrm_map = torch.zeros(H * W, 3, dtype=torch.float64, device=device)
    dens_map = torch.zeros(H * W, dtype=torch.float64, device=device)
    if mesh_vertices is not None and mesh_faces is not None:
        V3 = _as_t(mesh_vertices, torch.float64, device)
        F3 = _as_t(mesh_faces, torch.long, device)
        vf = valid.nonzero(as_tuple=False).squeeze(1)
        mf = mesh_face_map[vf]
        corners = V3[F3[mf]]  # [K,3,3]
        b = bary_map[vf].unsqueeze(-1)  # [K,3,1]
        xyz_map[vf] = (corners * b).sum(dim=1)
        if vertex_normals is not None:
            N3 = _as_t(vertex_normals, torch.float64, device)
            n = (N3[F3[mf]] * b).sum(dim=1)
            n = n / torch.clamp(n.norm(dim=1, keepdim=True), min=1e-12)
            nrm_map[vf] = n
        # texel density: 3d face area / uv face area (in uv units)
        fa3 = (
            torch.cross(
                V3[F3[:, 1]] - V3[F3[:, 0]], V3[F3[:, 2]] - V3[F3[:, 0]], dim=1
            ).norm(dim=1)
            * 0.5
        )
        fa2 = area2.abs() * 0.5 / (W * H)  # uv area in [0,1]^2 units
        ratio = fa3[u2m] / torch.clamp(fa2, min=1e-20)
        dens_map[vf] = ratio[face_map[vf]]

    HW = (H, W)
    covered = int((coverage > 0).sum())
    overlap_px = int((coverage > 1).sum())
    stats = {
        "num_faces": int(F),
        "num_zero_area": int(zero_area.sum()),
        "num_flipped": int(flipped.sum()),
        "covered_px": covered,
        "overlap_px": overlap_px,
        "overlap_ratio": overlap_px / max(covered, 1),
        "valid_px": int(valid.sum()),
        "occupancy": int(valid.sum()) / (H * W),
    }
    return AddressMaps(
        valid_mask=valid.reshape(HW).to(torch.uint8).cpu().numpy(),
        face_id=mesh_face_map.reshape(HW).to(torch.int32).cpu().numpy(),
        barycentric=bary_map.reshape(H, W, 3).to(torch.float32).cpu().numpy(),
        chart_id=chart_map.reshape(HW).to(torch.int32).cpu().numpy(),
        coverage=coverage.reshape(HW).cpu().numpy(),
        xyz=xyz_map.reshape(H, W, 3).to(torch.float32).cpu().numpy(),
        normal=nrm_map.reshape(H, W, 3).to(torch.float32).cpu().numpy(),
        texel_density=dens_map.reshape(HW).to(torch.float32).cpu().numpy(),
        stats=stats,
    )


def verify_address(
    am,
    uv_vertices,
    uv_faces,
    mesh_vertices,
    mesh_faces,
    uv_face_to_mesh_face,
    res,
    bary_tol=1e-6,
    uv_px_tol=0.5,
    xyz_tol=1e-5,
    bary_sum_tol=1e-5,
):
    # bary_sum_tol: 1e-5 fits float32 in-memory maps; pass ~2e-3 when
    # verifying fp16-quantized tensors loaded from disk.
    """Self-checks from the spec; returns dict of measured errors
    (raises nothing)."""
    valid = am.valid_mask.astype(bool)
    out = {"n_valid": int(valid.sum())}
    if out["n_valid"] == 0:
        out["empty"] = True
        return out
    b = am.barycentric[valid].astype(np.float64)
    out["bary_sum_max_err"] = float(np.abs(b.sum(1) - 1).max())
    out["bary_min"] = float(b.min())
    fid = am.face_id[valid]
    out["face_id_ok"] = bool(
        (fid >= 0).all() and (fid < len(mesh_faces)).all()
    )
    # uv round trip: mesh_face -> uv face row (need inverse map)
    m2u = {int(m): i for i, m in enumerate(np.asarray(uv_face_to_mesh_face))}
    ur = np.array([m2u[int(m)] for m in fid], dtype=np.int64)
    uvf = np.asarray(uv_faces)[ur]  # [K,3]
    uvc = np.asarray(uv_vertices, dtype=np.float64)[uvf]  # [K,3,2]
    uv_rec = (uvc * b[..., None]).sum(1)
    jj, ii = np.nonzero(valid)
    centers = np.stack([(ii + 0.5) / res, (jj + 0.5) / res], axis=1)
    out["uv_roundtrip_max_px"] = float((np.abs(uv_rec - centers) * res).max())
    # xyz reconstruction
    Vc = np.asarray(mesh_vertices, dtype=np.float64)[
        np.asarray(mesh_faces)[fid]
    ]
    xyz_rec = (Vc * b[..., None]).sum(1)
    out["xyz_recon_max_err"] = float(np.abs(xyz_rec - am.xyz[valid]).max())
    # barycentric is stored float32 -> sum deviates by O(1e-7); 1e-5 is the
    # fp32-quantization gate (fp64 math itself is exact to <1e-12)
    out["pass"] = bool(
        out["bary_sum_max_err"] < bary_sum_tol
        and out["bary_min"] > -bary_tol
        and out["face_id_ok"]
        and out["uv_roundtrip_max_px"] <= uv_px_tol
        and out["xyz_recon_max_err"] < xyz_tol
    )
    return out


# ==================================================================
# face-subset utilities (connected partial queries)
# ==================================================================
"""UV query helpers: face adjacency and connected surface subsets.

A partial surface query addresses a connected subset of faces; the subset's
own rasterization produces its face_id / barycentric / valid_mask (see
datasets/build_uv_queries.py).
"""
from collections import deque  # noqa: E402  (second concatenated stage)


def face_adjacency(faces):
    """Shared-edge neighbor lists (mesh faces). Returns list[list[int]]."""
    f = np.asarray(faces, dtype=np.int64)
    e2f = {}
    for fi, tri in enumerate(f):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            e = (min(tri[a], tri[b]), max(tri[a], tri[b]))
            e2f.setdefault(e, []).append(fi)
    adj = [[] for _ in range(len(f))]
    for fs in e2f.values():
        for a in fs:
            for b in fs:
                if a != b:
                    adj[a].append(b)
    return adj


def connected_subset(adj, n_faces, frac, rng):
    """BFS patch of ~frac*n_faces from a random seed face. On meshes with
    several connected components the frontier can exhaust early; reseed on an
    unvisited face until the target is met (patch = union of a few connected
    patches). Returns int64 face indices."""
    target = max(1, int(round(n_faces * frac)))
    seen = set()
    order = rng.permutation(n_faces)
    oi = 0
    while len(seen) < target and oi < n_faces:
        while oi < n_faces and int(order[oi]) in seen:
            oi += 1
        if oi >= n_faces:
            break
        seed = int(order[oi])
        seen.add(seed)
        qd = deque([seed])
        while qd and len(seen) < target:
            for nb in adj[qd.popleft()]:
                if nb not in seen:
                    seen.add(nb)
                    qd.append(nb)
                    if len(seen) >= target:
                        break
    return np.fromiter(seen, dtype=np.int64)
