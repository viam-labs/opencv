"""Robust bootstrap for hand-eye calibration.

``cv2.calibrateHandEye`` is a least-squares solve with no outlier handling: a
single station whose board pose is wrong — most commonly the mirror branch of
the planar-PnP two-fold ambiguity on a near-frontal view — drags the whole
solution off, and the reprojection refinement then starts (and often stays) in
the wrong basin.

This module makes the bootstrap robust in two ways:

1. Branch selection. Planar PnP (SOLVEPNP_IPPE) returns up to two candidate
   board poses per station. When the candidates have similar reprojection
   error the view is ambiguous and picking by reprojection error alone is a
   coin flip. The arm chain knows more: with the correct branches, the implied
   target-in-base T_be,k · X · T_cw,k is the same for every station k (the
   target is physically fixed). We bootstrap X from the unambiguous stations,
   then re-pick each station's branch as the candidate most consistent with
   the consensus target-in-base, and iterate.

2. Outlier rejection. After branch selection, stations whose implied
   target-in-base still deviates far from the consensus (bad robot pose,
   target bumped, detection error) are dropped before the final solve.

Conventions match the rest of the pipeline: T_be is gripper-in-base, T_cw is
target-in-camera, X is camera-in-gripper (T_ec), all 4x4, translations in mm.
"""

from typing import List, Optional, Sequence

import cv2
import numpy as np

# Rotation/translation equivalence used when scoring a candidate's deviation
# from the consensus: 1 degree of rotation counts the same as this many mm of
# translation. Ambiguity flips are rotation-dominant (tens of degrees), so the
# exact value matters little — it only tie-breaks near-identical candidates.
_MM_PER_DEG = 10.0


def _rotation_angle_deg(R: np.ndarray) -> float:
    cos_angle = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _chordal_mean_rotation(R_list: Sequence[np.ndarray]) -> np.ndarray:
    """Frobenius average projected back onto SO(3) via SVD."""
    R_avg = np.mean(np.stack(R_list, axis=0), axis=0)
    U, _, Vt = np.linalg.svd(R_avg)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:
        Vt[-1] *= -1
        R_mean = U @ Vt
    return R_mean


def _consensus_transform(T_list: Sequence[np.ndarray]) -> np.ndarray:
    """Robust center of a set of 4x4 transforms: per-axis median translation,
    chordal-mean rotation. The median keeps a minority of flipped/outlier
    stations from dragging the translation; rotation flips are handled by the
    selection loop, not the mean."""
    T = np.eye(4)
    T[:3, :3] = _chordal_mean_rotation([Ti[:3, :3] for Ti in T_list])
    T[:3, 3] = np.median(np.stack([Ti[:3, 3] for Ti in T_list], axis=0), axis=0)
    return T


def _deviation(T: np.ndarray, center: np.ndarray) -> tuple:
    """(translation mm, rotation deg) deviation of T from center."""
    t_dev = float(np.linalg.norm(T[:3, 3] - center[:3, 3]))
    r_dev = _rotation_angle_deg(T[:3, :3] @ center[:3, :3].T)
    return t_dev, r_dev


def _target_in_base(T_be: np.ndarray, X: np.ndarray, T_cw: np.ndarray) -> np.ndarray:
    return T_be @ X @ T_cw


def _solve_X(
    T_be_list: Sequence[np.ndarray],
    T_cw_selected: Sequence[np.ndarray],
    indices: Sequence[int],
    method: int,
) -> np.ndarray:
    R_c2g, t_c2g = cv2.calibrateHandEye(
        R_gripper2base=[T_be_list[i][:3, :3] for i in indices],
        t_gripper2base=[T_be_list[i][:3, 3].reshape(3, 1) for i in indices],
        R_target2cam=[T_cw_selected[i][:3, :3] for i in indices],
        t_target2cam=[T_cw_selected[i][:3, 3].reshape(3, 1) for i in indices],
        method=method,
    )
    if R_c2g is None or t_c2g is None:
        raise Exception("cv2.calibrateHandEye failed during robust bootstrap")
    X = np.eye(4)
    X[:3, :3] = R_c2g
    X[:3, 3] = t_c2g.flatten()
    return X


def robust_bootstrap_handeye(
    T_be_list: Sequence[np.ndarray],
    T_cw_candidates_list: Sequence[Sequence[np.ndarray]],
    reproj_errs_list: Optional[Sequence[Optional[Sequence[float]]]] = None,
    method: int = cv2.CALIB_HAND_EYE_TSAI,
    ambiguity_ratio: float = 1.5,
    n_rounds: int = 2,
    reject_trans_floor_mm: float = 10.0,
    reject_rot_floor_deg: float = 5.0,
    reject_mad_scale: float = 3.0,
) -> dict:
    """Bootstrap hand-eye calibration with PnP-branch disambiguation and
    station outlier rejection.

    Args:
        T_be_list: per-station 4x4 gripper-in-base transforms.
        T_cw_candidates_list: per-station list of candidate 4x4
            target-in-camera transforms, best-reprojection-error first (as
            returned by SOLVEPNP_IPPE). A station with a single candidate is
            treated as unambiguous.
        reproj_errs_list: per-station reprojection errors (px) parallel to the
            candidates, or None per station when unavailable. Without errors a
            multi-candidate station is always treated as ambiguous.
        method: cv2.CALIB_HAND_EYE_* constant for the bootstrap solves.
        ambiguity_ratio: a station is *unambiguous* when its second-best
            candidate's reprojection error is at least this multiple of the
            best's — i.e. the pixels alone clearly prefer one branch.
        n_rounds: branch-selection iterations (solve X → re-pick branches).
        reject_trans_floor_mm / reject_rot_floor_deg: outlier thresholds never
            drop below these floors, so tight, healthy runs don't reject
            stations over measurement noise.
        reject_mad_scale: threshold = median + this × 1.4826 × MAD.

    Returns dict with:
        X_init: 4x4 camera-in-gripper from the final (kept-stations) solve.
        Y_init: 4x4 consensus target-in-base of the kept stations — a direct
            initial value for the reprojection refinement's board pose.
        selected_T_cw: per-station chosen candidate (all stations).
        selected_branch: per-station chosen candidate index.
        kept_indices: stations that survived outlier rejection.
        rejected: list of {station, translation_deviation_mm,
            rotation_deviation_deg} for dropped stations.
        ambiguous_indices: stations whose view could not distinguish branches.
        branch_corrections: stations where the chain-consistent branch was NOT
            the best-reprojection-error one.
        deviations: per-station (mm, deg) deviation from the final consensus.
    """
    n = len(T_be_list)
    if n < 3:
        raise ValueError(f"need at least 3 stations, got {n}")
    if len(T_cw_candidates_list) != n:
        raise ValueError("T_be_list and T_cw_candidates_list must be the same length")

    T_be = [np.asarray(T, dtype=np.float64) for T in T_be_list]
    candidates = [
        [np.asarray(T, dtype=np.float64) for T in cands]
        for cands in T_cw_candidates_list
    ]
    if any(len(c) == 0 for c in candidates):
        raise ValueError("every station needs at least one T_cw candidate")

    # Initial selection: best reprojection error (candidates arrive sorted,
    # but don't rely on it when errors are provided).
    selection: List[int] = []
    ambiguous: List[int] = []
    for i, cands in enumerate(candidates):
        errs = reproj_errs_list[i] if reproj_errs_list is not None else None
        if errs is not None and len(errs) == len(cands):
            best = int(np.argmin(errs))
            selection.append(best)
            if len(cands) > 1:
                sorted_errs = sorted(float(e) for e in errs)
                # Guard the ratio against a ~0px best fit.
                if sorted_errs[1] < ambiguity_ratio * max(sorted_errs[0], 0.05):
                    ambiguous.append(i)
        else:
            selection.append(0)
            if len(cands) > 1:
                ambiguous.append(i)

    def selected_T_cw() -> List[np.ndarray]:
        return [candidates[i][selection[i]] for i in range(n)]

    unambiguous = [i for i in range(n) if i not in ambiguous]
    seed_indices = unambiguous if len(unambiguous) >= 3 else list(range(n))
    X = _solve_X(T_be, selected_T_cw(), seed_indices, method)

    def score(T_tb: np.ndarray, center: np.ndarray) -> float:
        t_dev, r_dev = _deviation(T_tb, center)
        return r_dev + t_dev / _MM_PER_DEG

    for _ in range(n_rounds):
        T_cw_sel = selected_T_cw()
        # Consensus from the stations whose branch we trust most; fall back to
        # everything when too few views are unambiguous.
        center = _consensus_transform(
            [_target_in_base(T_be[i], X, T_cw_sel[i]) for i in seed_indices]
        )
        new_selection = [
            int(np.argmin([
                score(_target_in_base(T_be[i], X, cand), center)
                for cand in candidates[i]
            ]))
            for i in range(n)
        ]
        changed = new_selection != selection
        selection = new_selection
        X = _solve_X(T_be, selected_T_cw(), list(range(n)), method)
        if not changed:
            break

    # Outlier rejection against the final consensus.
    T_cw_sel = selected_T_cw()
    T_tb = [_target_in_base(T_be[i], X, T_cw_sel[i]) for i in range(n)]
    center = _consensus_transform(T_tb)
    deviations = [_deviation(T, center) for T in T_tb]
    t_devs = np.array([d[0] for d in deviations])
    r_devs = np.array([d[1] for d in deviations])

    def mad_threshold(values: np.ndarray, floor: float) -> float:
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)))
        return max(floor, med + reject_mad_scale * 1.4826 * mad)

    t_thr = mad_threshold(t_devs, reject_trans_floor_mm)
    r_thr = mad_threshold(r_devs, reject_rot_floor_deg)
    kept = [i for i in range(n) if t_devs[i] <= t_thr and r_devs[i] <= r_thr]

    if len(kept) < 3:
        # Not enough consistent stations for a solve — keep the 3 most
        # consistent so the caller still gets a result plus diagnostics
        # showing how bad the run was.
        kept = sorted(
            range(n), key=lambda i: r_devs[i] + t_devs[i] / _MM_PER_DEG
        )[:3]
        kept.sort()

    rejected = [
        {
            "station": i,
            "translation_deviation_mm": float(t_devs[i]),
            "rotation_deviation_deg": float(r_devs[i]),
        }
        for i in range(n)
        if i not in kept
    ]

    if len(kept) < n:
        X = _solve_X(T_be, T_cw_sel, kept, method)
        T_tb_kept = [_target_in_base(T_be[i], X, T_cw_sel[i]) for i in kept]
    else:
        T_tb_kept = T_tb
    Y_init = _consensus_transform(T_tb_kept)

    branch_corrections = []
    for i in range(n):
        errs = reproj_errs_list[i] if reproj_errs_list is not None else None
        best_by_err = int(np.argmin(errs)) if errs is not None and len(errs) == len(candidates[i]) else 0
        if selection[i] != best_by_err:
            branch_corrections.append(i)

    return {
        "X_init": X,
        "Y_init": Y_init,
        "selected_T_cw": T_cw_sel,
        "selected_branch": selection,
        "kept_indices": kept,
        "rejected": rejected,
        "ambiguous_indices": ambiguous,
        "branch_corrections": branch_corrections,
        "deviations": deviations,
    }
