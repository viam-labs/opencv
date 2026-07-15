"""Tests for src/solvers/robust_init.py.

Synthetic ground-truth scenarios: build a known true (X, Y) and a set of
stations, derive each station's true target-in-camera pose from the chain,
then corrupt a subset (flipped PnP branches, outlier stations) and verify the
robust bootstrap recovers X, picks the chain-consistent branches, and rejects
the bad stations.
"""

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from solvers.robust_init import robust_bootstrap_handeye


def _se3(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def _euler_se3(rx_deg, ry_deg, rz_deg, t):
    R = Rotation.from_euler("xyz", [rx_deg, ry_deg, rz_deg], degrees=True).as_matrix()
    return _se3(R, t)


def _se3_inverse(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def _rotation_angle_deg(R):
    cos_angle = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


# True camera-in-gripper: a small rotation plus a mounting offset.
X_TRUE = _euler_se3(5.0, -3.0, 90.0, [50.0, -20.0, 30.0])

# True target-in-base: board flat-ish in front of the robot.
Y_TRUE = _euler_se3(180.0, 0.0, 15.0, [500.0, 50.0, 20.0])


def _stations(n=12):
    """Diverse gripper-in-base poses (hand-eye needs rotation diversity)."""
    rng = np.random.default_rng(7)
    T_be_list = []
    for k in range(n):
        rx = 150.0 + 20.0 * np.sin(1.3 * k)
        ry = 25.0 * np.cos(0.9 * k + 1.0)
        rz = 40.0 * np.sin(0.7 * k + 2.0)
        t = np.array([450.0, 30.0, 350.0]) + rng.uniform(-80.0, 80.0, size=3)
        T_be_list.append(_euler_se3(rx, ry, rz, t))
    return T_be_list


def _true_T_cw(T_be):
    return _se3_inverse(X_TRUE) @ _se3_inverse(T_be) @ Y_TRUE


def _flipped(T_cw, angle_deg=160.0):
    """A plausible wrong PnP branch: board rotated about an in-plane axis."""
    flip = _euler_se3(angle_deg, 0.0, 0.0, [0.0, 0.0, 0.0])
    return T_cw @ flip


def _assert_close_to_truth(X, rot_tol_deg=0.5, trans_tol_mm=2.0):
    r_err = _rotation_angle_deg(X[:3, :3] @ X_TRUE[:3, :3].T)
    t_err = float(np.linalg.norm(X[:3, 3] - X_TRUE[:3, 3]))
    assert r_err < rot_tol_deg, f"rotation error {r_err:.3f}deg"
    assert t_err < trans_tol_mm, f"translation error {t_err:.3f}mm"


def test_clean_data_recovers_x_and_y():
    T_be_list = _stations()
    cands = [[_true_T_cw(T)] for T in T_be_list]

    result = robust_bootstrap_handeye(T_be_list, cands)

    _assert_close_to_truth(result["X_init"])
    assert result["kept_indices"] == list(range(len(T_be_list)))
    assert result["rejected"] == []
    assert result["ambiguous_indices"] == []
    assert result["branch_corrections"] == []

    y_r_err = _rotation_angle_deg(result["Y_init"][:3, :3] @ Y_TRUE[:3, :3].T)
    y_t_err = float(np.linalg.norm(result["Y_init"][:3, 3] - Y_TRUE[:3, 3]))
    assert y_r_err < 0.5
    assert y_t_err < 2.0


def test_ambiguous_flipped_branches_are_corrected():
    T_be_list = _stations()
    flipped_stations = {1, 4, 6, 9}

    cands, errs = [], []
    for i, T_be in enumerate(T_be_list):
        T_cw = _true_T_cw(T_be)
        if i in flipped_stations:
            # Ambiguous view: the WRONG branch has marginally better
            # reprojection error, so error alone picks the flip.
            cands.append([_flipped(T_cw), T_cw])
            errs.append([0.30, 0.31])
        else:
            cands.append([T_cw, _flipped(T_cw)])
            errs.append([0.25, 4.0])

    result = robust_bootstrap_handeye(T_be_list, cands, errs)

    _assert_close_to_truth(result["X_init"])
    assert set(result["ambiguous_indices"]) == flipped_stations
    assert set(result["branch_corrections"]) == flipped_stations
    for i in flipped_stations:
        assert result["selected_branch"][i] == 1
    # With the branches fixed the stations are consistent — nothing rejected.
    assert result["kept_indices"] == list(range(len(T_be_list)))


def test_outlier_station_is_rejected():
    T_be_list = _stations()
    cands = [[_true_T_cw(T)] for T in T_be_list]
    # Station 5's board pose is wrong in a way no branch can fix (the target
    # was bumped / the robot pose was misreported).
    bad = 5
    cands[bad] = [_se3(cands[bad][0][:3, :3], cands[bad][0][:3, 3] + [80.0, 0.0, 0.0])]

    result = robust_bootstrap_handeye(T_be_list, cands)

    assert bad not in result["kept_indices"]
    assert [r["station"] for r in result["rejected"]] == [bad]
    assert result["rejected"][0]["translation_deviation_mm"] > 10.0
    _assert_close_to_truth(result["X_init"])


def test_flip_without_alternative_branch_is_rejected():
    # A flipped station that only reports one branch (e.g. legacy tracker)
    # can't be corrected — it must be rejected instead.
    T_be_list = _stations()
    cands = [[_true_T_cw(T)] for T in T_be_list]
    bad = 2
    cands[bad] = [_flipped(cands[bad][0])]

    result = robust_bootstrap_handeye(T_be_list, cands)

    assert bad not in result["kept_indices"]
    _assert_close_to_truth(result["X_init"])


def test_noisy_but_healthy_run_keeps_all_stations():
    # Realistic measurement noise well under the rejection floors must not
    # cause spurious rejections.
    rng = np.random.default_rng(3)
    T_be_list = _stations()
    cands = []
    for T_be in T_be_list:
        T_cw = _true_T_cw(T_be)
        jitter = _euler_se3(
            *rng.normal(0.0, 0.1, size=3), rng.normal(0.0, 0.5, size=3)
        )
        cands.append([T_cw @ jitter])

    result = robust_bootstrap_handeye(T_be_list, cands)

    assert result["kept_indices"] == list(range(len(T_be_list)))
    _assert_close_to_truth(result["X_init"], rot_tol_deg=1.0, trans_tol_mm=5.0)


def test_too_few_stations_raises():
    T_be_list = _stations(2)
    cands = [[_true_T_cw(T)] for T in T_be_list]
    with pytest.raises(ValueError):
        robust_bootstrap_handeye(T_be_list, cands)
