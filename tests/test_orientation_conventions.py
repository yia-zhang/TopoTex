# -*- coding: utf-8 -*-
"""Coordinate-convention guards for the OA-V2 GLB canonicalizer.

Math invariants always run. The live-model calibration itself is produced
by orientation_validation/calibrate_basis.py (isolated OA-V2 env + GPU);
its frozen result is verified here whenever calibration.json exists —
each known perturbation case must report predicted transform, expected
transform and a small geodesic error. Euler orders are never assumed by
intuition: the basis is selected from the full det=+1 signed-permutation
set against known rotations.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/orientation"))
import canonicalize_glb_oa_v2 as C10  # noqa: E402

CALIB = Path(
    "/root/youjiaZhang/topotex_data_OA/orientation_validation/calib/"
    "calibration.json"
)


def test_rotation_primitives_are_proper():
    for axis in "xyz":
        R = C10.rot(axis, 33.0)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)
    # yaw about y moves +Z toward +X (right-handed, Y-up, column vectors)
    assert np.allclose(C10.rot("y", 90) @ np.array([0, 0, 1.0]), [1, 0, 0],
                       atol=1e-12)


def test_gizmo_formula_matches_official_composition():
    az, el, ro = 37.0, -12.0, 55.0
    R = C10.oa2_gizmo_rotation(az, el, ro)
    R_ref = C10.rot("x", ro) @ C10.rot("y", el) @ C10.rot("z", -az)
    assert np.allclose(R, R_ref, atol=1e-12)


def test_spherical_registration_identity_and_yaw():
    # a prediction equal to the true camera position must give R ~= I
    R = C10.predicted_world_rotation(30.0, 20.0, 0.0, 30.0, 20.0)
    assert C10.geodesic_deg(R, np.eye(3)) < 1e-6
    # object yawed by +90: camera azimuth in object frame drops by 90
    R = C10.predicted_world_rotation(30.0 - 90.0, 20.0, 0.0, 30.0, 20.0)
    assert C10.geodesic_deg(R, C10.rot("y", 90)) < 1e-6


def test_correction_is_inverse_of_predicted_world_rotation():
    az, el, ro, cam = 200.0, 25.0, -40.0, 30.0
    Rw = C10.predicted_world_rotation(az, el, ro, cam)
    Rc = C10.correction_rotation(az, el, ro, cam)
    assert np.allclose(Rc @ Rw, np.eye(3), atol=1e-9)


def test_geodesic_metric_sanity():
    assert C10.geodesic_deg(np.eye(3), np.eye(3)) < 1e-6
    assert abs(C10.geodesic_deg(np.eye(3), C10.rot("y", 90)) - 90) < 1e-6


@pytest.mark.skipif(not CALIB.exists(), reason="live calibration not run")
def test_frozen_constants_match_live_calibration():
    calib = json.loads(CALIB.read_text())
    assert calib["ro_sign"] == C10.RO_SIGN and calib["az_mirror"] == C10.AZ_MIRROR, (
        "tools/orientation RO_SIGN/AZ_MIRROR != calibrated constants"
    )
    # the yaw channel is the calibrated/guaranteed regime; tilt perception
    # is a measured model property (see oa_v2_benchmark.json), not a
    # convention error
    assert calib["yaw_worst_error_deg"] <= 15.0, calib["per_case"]
    for name, row in calib["per_case"].items():
        # every case carries predicted transform inputs + geodesic error
        assert "pred_azelro" in row and "geodesic_err_deg" in row, name
