"""Tests for src/models/base_pose_tracker.py distortion-parameter handling.

Viam distortion models list radial coefficients first and tangential last;
OpenCV expects (k1, k2, p1, p2, k3, ...). These tests pin the mapping using
real values observed from a brown_conrady_k6 camera, where passing parameters
through verbatim put k3=0.25 in the p1 slot and biased PnP depth by ~90mm.
"""

import cv2
import numpy as np
import pytest

from models.base_pose_tracker import (
    distortion_candidates,
    distortion_to_opencv,
    select_distortion_by_reprojection,
)


def test_brown_conrady_reorders_to_opencv():
    # Viam order: (k1, k2, k3, p1, p2)
    out = distortion_to_opencv("brown_conrady", [0.1, -0.3, 0.25, -1e-4, 8e-5])
    np.testing.assert_allclose(out, [0.1, -0.3, -1e-4, 8e-5, 0.25], rtol=1e-6)


def test_brown_conrady_k6_reorders_to_opencv():
    # Viam order: (k1, k2, k3, k4, k5, k6, p1, p2) — values from the real
    # sensing-camera that surfaced the bug.
    raw = [0.11790542, -0.3221269, 0.25121865, 0.0, 0.0, 0.0, -1.3837375e-4, 8.1919257e-5]
    out = distortion_to_opencv("brown_conrady_k6", raw)
    np.testing.assert_allclose(
        out,
        # OpenCV order: (k1, k2, p1, p2, k3, k4, k5, k6)
        [0.11790542, -0.3221269, -1.3837375e-4, 8.1919257e-5, 0.25121865, 0.0, 0.0, 0.0],
        rtol=1e-6,
    )


def test_model_name_is_case_insensitive():
    out = distortion_to_opencv("Brown_Conrady", [1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(out, [1.0, 2.0, 4.0, 5.0, 3.0])


def test_empty_params_mean_rectified_image():
    np.testing.assert_allclose(distortion_to_opencv("", []), np.zeros(5))


def test_known_model_with_wrong_param_count_raises():
    with pytest.raises(Exception, match="8 parameters"):
        distortion_to_opencv("brown_conrady_k6", [0.1, 0.2, 0.3])


def test_unknown_model_with_opencv_length_passes_through():
    out = distortion_to_opencv("some_module_specific", [0.1, -0.2, 1e-4, -1e-4, 0.05])
    np.testing.assert_allclose(out, [0.1, -0.2, 1e-4, -1e-4, 0.05])


def test_unknown_model_with_bad_length_raises():
    with pytest.raises(Exception, match="unrecognized distortion model"):
        distortion_to_opencv("mystery", [0.1, 0.2, 0.3])


# ---- candidate enumeration -------------------------------------------------


def test_candidates_prior_first_and_deduped():
    raw = [0.1, -0.3, 0.25, 0.0, 0.0, 0.0, -1e-4, 8e-5]
    cands = distortion_candidates("brown_conrady_k6", raw)
    labels = [label for label, _ in cands]
    # Prior (== radial-first reorder) first, then the two other distinct reads.
    assert labels[0] == "prior for model 'brown_conrady_k6'"
    np.testing.assert_allclose(cands[0][1], [0.1, -0.3, -1e-4, 8e-5, 0.25, 0, 0, 0], rtol=1e-6)
    # The radial-first ordering duplicates the prior and must be dropped.
    assert not any("(k1..k6,p1,p2)" in l for l in labels[1:])
    assert any(l.startswith("opencv") for l in labels)


def test_candidates_collapse_when_orderings_coincide():
    # No tangential/k3 terms -> every interpretation is numerically identical.
    cands = distortion_candidates("whatever", [0.1, -0.2, 0.0, 0.0, 0.0])
    assert len(cands) == 1


def test_candidates_empty_params():
    cands = distortion_candidates("", [])
    assert len(cands) == 1
    np.testing.assert_allclose(cands[0][1], np.zeros(5))


# ---- empirical ordering selection -------------------------------------------


K_SYNTH = np.array([
    [1200.0, 0.0, 960.0],
    [0.0, 1200.0, 540.0],
    [0.0, 0.0, 1.0],
])
# OpenCV order (k1, k2, p1, p2, k3) — meaningful k3 so orderings differ.
DIST_TRUE_CV = np.array([0.118, -0.322, -1.4e-4, 8.2e-5, 0.251])


def _synthetic_view():
    """A large tilted board filling much of the frame (r^6 term needs corners
    far from the principal point to be observable)."""
    cols, rows, square = 14, 8, 60.0
    objp = np.zeros((rows * cols, 3))
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    objp[:, :2] -= [cols * square / 2, rows * square / 2]
    from scipy.spatial.transform import Rotation
    rvec = Rotation.from_euler("xyz", [15, -10, 5], degrees=True).as_rotvec()
    tvec = np.array([30.0, -20.0, 900.0])
    imgp, _ = cv2.projectPoints(objp, rvec, tvec, K_SYNTH, DIST_TRUE_CV)
    return objp, imgp.reshape(-1, 2)


def test_selection_overrides_prior_when_pixels_disagree():
    # The camera reports radial-first (k1, k2, k3, p1, p2) under an unknown
    # model name, so the prior (pass-through) reads it as OpenCV order — wrong.
    objp, imgp = _synthetic_view()
    raw_radial_first = [0.118, -0.322, 0.251, -1.4e-4, 8.2e-5]
    cands = distortion_candidates("vendor_specific", raw_radial_first)

    label, dist, scores = select_distortion_by_reprojection(K_SYNTH, cands, objp, imgp)

    assert label == "radial-first (k1,k2,k3,p1,p2)"
    np.testing.assert_allclose(dist, DIST_TRUE_CV, rtol=1e-5, atol=1e-8)
    assert scores[label] < 0.05
    assert scores["prior for model 'vendor_specific'"] > 0.5


def test_selection_overrides_lying_model_name():
    # The module labels its output brown_conrady_k6 (radial-first per rdk)
    # but actually emits OpenCV order — the user-reported ecosystem hazard.
    objp, imgp = _synthetic_view()
    raw_opencv_order = list(DIST_TRUE_CV) + [0.0, 0.0, 0.0]
    cands = distortion_candidates("brown_conrady_k6", raw_opencv_order)

    label, dist, scores = select_distortion_by_reprojection(K_SYNTH, cands, objp, imgp)

    assert label.startswith("opencv")
    np.testing.assert_allclose(dist[:5], DIST_TRUE_CV, rtol=1e-5, atol=1e-8)


def test_selection_keeps_prior_when_correct():
    # Camera honors its model name; the prior must win, not be overridden.
    objp, imgp = _synthetic_view()
    raw_radial_first = [0.118, -0.322, 0.251, 0.0, 0.0, 0.0, -1.4e-4, 8.2e-5]
    cands = distortion_candidates("brown_conrady_k6", raw_radial_first)

    label, dist, scores = select_distortion_by_reprojection(K_SYNTH, cands, objp, imgp)

    assert label == "prior for model 'brown_conrady_k6'"
    np.testing.assert_allclose(dist[:5], DIST_TRUE_CV, rtol=1e-5, atol=1e-8)


def test_selection_keeps_prior_on_ties():
    # Negligible distortion: every ordering fits equally; keep the prior
    # rather than flip-flopping on noise.
    objp, imgp = _synthetic_view()
    tiny = [1e-6, -1e-6, 1e-7, 1e-7, 1e-6]
    # Re-project with effectively-zero distortion for consistent pixels.
    imgp, _ = cv2.projectPoints(
        objp, np.array([0.26, -0.17, 0.09]), np.array([30.0, -20.0, 900.0]),
        K_SYNTH, np.asarray(tiny))
    cands = distortion_candidates("vendor_specific", tiny)

    label, _, _ = select_distortion_by_reprojection(K_SYNTH, cands, objp, imgp.reshape(-1, 2))

    assert label == "prior for model 'vendor_specific'"
