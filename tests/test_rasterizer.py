# -*- coding: utf-8 -*-
"""UV rasterizer: edge convention, overlap exclusion, bary validity,
coincident-face distinct addressing, determinism."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.rasterizer import rasterize_uv, verify_address


def fixture_adjacent():
    """Two UV triangles sharing an edge (quad split): every interior texel
    covered exactly once — protects the top-left fill convention."""
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float64)
    f = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    uv = np.array([[.1, .1], [.9, .1], [.9, .9], [.1, .9]], np.float64)
    return v, f, uv, f.copy()


def fixture_coincident():
    """Two faces with IDENTICAL xyz but separate UV islands — the address
    must keep them distinct (the core TOPOTEX identity property)."""
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]] * 2, np.float64)
    f = np.array([[0, 1, 2], [3, 4, 5]], np.int32)
    uv = np.array([[.05, .05], [.45, .05], [.05, .45],
                   [.55, .55], [.95, .55], [.55, .95]], np.float64)
    return v, f, uv, f.copy()


def fixture_uv_overlap():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [0, 1, 1]], np.float64)
    f = np.array([[0, 1, 2], [3, 4, 5]], np.int32)
    uv = np.array([[.10, .10], [.90, .10], [.10, .90],
                   [.20, .20], [.95, .25], [.25, .95]], np.float64)
    return v, f, uv, f.copy()


def test_adjacent_edge_no_double_cover():
    v, f, uv, uvf = fixture_adjacent()
    am = rasterize_uv(uv, uvf, 128)
    assert (am.coverage <= 1).all(), "shared edge double-covered"
    assert am.valid_mask.sum() > 0.5 * 128 * 128


def test_overlap_detection_excludes_texels():
    v, f, uv, uvf = fixture_uv_overlap()
    am = rasterize_uv(uv, uvf, 128)
    assert (am.coverage > 1).any()
    assert not am.valid_mask[am.coverage > 1].any(), \
        "overlapping texels must be excluded from valid"


def test_coincident_faces_distinct_address():
    v, f, uv, uvf = fixture_coincident()
    am = rasterize_uv(uv, uvf, 128)
    fid = am.face_id[am.valid_mask.astype(bool)]
    assert set(np.unique(fid)) == {0, 1}


def test_bary_sum_range_and_roundtrip():
    v, f, uv, uvf = fixture_adjacent()
    am = rasterize_uv(uv, uvf, 256, mesh_vertices=v, mesh_faces=f,
                      uv_face_to_mesh_face=np.arange(2))
    rep = verify_address(am, uv, uvf, v, f, np.arange(2), 256)
    assert rep["pass"], rep


def test_deterministic():
    v, f, uv, uvf = fixture_adjacent()
    a = rasterize_uv(uv, uvf, 128)
    b = rasterize_uv(uv, uvf, 128)
    assert np.array_equal(a.face_id, b.face_id)
    assert np.array_equal(a.barycentric, b.barycentric)
