"""Offline diagnostics for captured hand-eye calibration data.

Loads a directory produced by capture_calibration_data.py and reconstructs the
calibration exactly as the hand_eye_calibration service does, then localizes
where the data is inconsistent. Run as many times as you like with no rebuild.

  PYTHONPATH=src python tools/analyze_calibration_data.py ./calib_capture

What it reports, per station and in summary:
  1. PnP self-reprojection error -- projects the observed 3D corners through the
     observation's own rvec/tvec. LOW (sub-pixel to a few px) => the corners,
     K and dist are mutually consistent. HIGH => intrinsics/distortion or corner
     detection are off.
  2. Board-in-base consistency -- using the calibrateHandEye bootstrap X, where
     does each station place the (physically fixed) board in the base frame? A
     fixed board should give the SAME pose every station. Large spread =>
     gripper poses are wrong/inaccurate, or the board moved. THIS is the usual
     smoking gun when (1) is low but the hand-eye solve still fails.
  3. The reprojection refinement result (with the x_scale conditioning fix).
  4. Whether arm.get_end_position() and motion.get_pose() agree.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from utils.utils import call_go_ov2mat
from solvers.reprojection_solver import refine_handeye


def _se3(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def _rot_geo_deg(Ra, Rb):
    dR = Ra @ Rb.T
    return float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))


def _T_be_from_arm_pose(p):
    """Reconstruct end-effector-in-base EXACTLY as the service does."""
    R_eb = call_go_ov2mat(p["o_x"], p["o_y"], p["o_z"], p["theta"])
    if R_eb is None:
        raise RuntimeError("call_go_ov2mat failed (is the go_utils binary built?)")
    T = np.eye(4)
    T[:3, :3] = R_eb.T          # service: R_be = R_eb.T (gripper-in-base)
    T[:3, 3] = [p["x"], p["y"], p["z"]]
    return T


def _K_dist_from_obs(obs):
    K = np.asarray(obs["K"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(obs["dist"], dtype=np.float64).reshape(-1)
    return K, dist


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: analyze_calibration_data.py <capture_dir>")
    d = Path(sys.argv[1])
    stations = json.loads((d / "stations.json").read_text())

    T_be_list, T_cw_list, c2d_list, c3d_list = [], [], [], []
    pnp_err, K, dist = [], None, None
    use_motion_disagreement = []

    print(f"{'stn':>3} {'corners':>7} {'pnp_px':>8}  arm(x,y,z)")
    for s in stations:
        obs = s.get("charuco_observation")
        if not obs:
            print(f"{s['index']:>3}   (no observation)")
            continue
        c2d = np.asarray(obs["corners_2d"], dtype=np.float64).reshape(-1, 2)
        c3d = np.asarray(obs["corners_3d"], dtype=np.float64).reshape(-1, 3)
        rvec = np.asarray(obs["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(obs["tvec"], dtype=np.float64).reshape(3)
        Ki, di = _K_dist_from_obs(obs)
        if K is None:
            K, dist = Ki, di

        proj, _ = cv2.projectPoints(c3d, rvec, tvec.reshape(3, 1), Ki, di)
        e = float(np.sqrt(np.mean((proj.reshape(-1, 2) - c2d) ** 2)))
        pnp_err.append(e)

        R_ct, _ = cv2.Rodrigues(rvec)
        T_be_list.append(_T_be_from_arm_pose(s["arm_pose_end_position"]))
        T_cw_list.append(_se3(R_ct, tvec))
        c2d_list.append(c2d)
        c3d_list.append(c3d)

        ap = s["arm_pose_end_position"]
        mp = s.get("arm_pose_motion_service")
        if mp is not None:
            dt = np.linalg.norm(np.array([ap["x"], ap["y"], ap["z"]]) -
                                np.array([mp["x"], mp["y"], mp["z"]]))
            use_motion_disagreement.append(dt)

        print(f"{s['index']:>3} {c3d.shape[0]:>7} {e:>8.2f}  "
              f"({ap['x']:.0f},{ap['y']:.0f},{ap['z']:.0f})")

    n = len(T_be_list)
    if n < 3:
        raise SystemExit(f"need >=3 valid stations, got {n}")

    # --- (1) PnP self-reprojection summary ---
    print(f"\n[1] PnP self-reproj px: mean={np.mean(pnp_err):.2f} "
          f"max={np.max(pnp_err):.2f}  "
          f"({'LOW -> corners/K/dist consistent' if np.max(pnp_err) < 3 else 'HIGH -> intrinsics/detection suspect'})")

    if use_motion_disagreement:
        print(f"[*] arm.get_end_position vs motion.get_pose: "
              f"max translation diff = {max(use_motion_disagreement):.1f} mm")

    # --- bootstrap X via calibrateHandEye ---
    R_g2b = [T[:3, :3] for T in T_be_list]
    t_g2b = [T[:3, 3].reshape(3, 1) for T in T_be_list]
    R_t2c = [T[:3, :3] for T in T_cw_list]
    t_t2c = [T[:3, 3].reshape(3, 1) for T in T_cw_list]
    R_c2g, t_c2g = cv2.calibrateHandEye(
        R_g2b, t_g2b, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_TSAI)
    X_boot = _se3(R_c2g, t_c2g.reshape(3))

    # --- (2) board-in-base consistency under bootstrap X ---
    T_t2b = []
    for i in range(n):
        R_t2b = R_g2b[i] @ R_c2g @ R_t2c[i]
        t_t2b = (R_g2b[i] @ R_c2g @ t_t2c[i].reshape(3)
                 + R_g2b[i] @ t_c2g.reshape(3) + t_g2b[i].reshape(3))
        T_t2b.append(_se3(R_t2b, t_t2b))
    t_mean = np.mean([T[:3, 3] for T in T_t2b], axis=0)
    Rs = np.stack([T[:3, :3] for T in T_t2b])
    U, _, Vt = np.linalg.svd(Rs.mean(0))
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:
        Vt[-1] *= -1
        R_mean = U @ Vt
    rot_resid = [_rot_geo_deg(T[:3, :3], R_mean) for T in T_t2b]
    trans_resid = [float(np.linalg.norm(T[:3, 3] - t_mean)) for T in T_t2b]
    print(f"[2] board-in-base spread (should be ~0 for a fixed board): "
          f"rot mean={np.mean(rot_resid):.1f}deg max={np.max(rot_resid):.1f}deg; "
          f"trans mean={np.mean(trans_resid):.1f}mm max={np.max(trans_resid):.1f}mm")
    worst = int(np.argmax(rot_resid))
    print(f"    worst station: idx={worst} rot={rot_resid[worst]:.1f}deg "
          f"trans={trans_resid[worst]:.1f}mm")

    # --- (3) reprojection refinement (with conditioning fix) ---
    out = refine_handeye(T_be_list, c2d_list, c3d_list, K, dist,
                         X_init=X_boot, T_cw_list=T_cw_list)
    X = out["X_refined"]
    print(f"[3] refine: rmse={out['rmse_pixels']:.2f}px success={out['success']} "
          f"X_translation(mm)={np.round(X[:3, 3], 1).tolist()}")

    # --- verdict ---
    print("\nVERDICT:")
    if np.max(pnp_err) >= 3:
        print("  Per-pose PnP doesn't fit -> corner detection or K/dist are wrong.")
    elif np.max(rot_resid) > 10 or np.max(trans_resid) > 30:
        print("  Corners are self-consistent but the board does NOT look fixed in")
        print("  the base frame -> the GRIPPER POSES are inconsistent with the")
        print("  camera observations. Likely: arm not settled before capture,")
        print("  inaccurate kinematics at sampled poses, the board physically")
        print("  moved, or a wrong arm/camera frame. Check the worst stations.")
    else:
        print("  Inputs look consistent; if the live run still failed, compare")
        print("  this capture's poses to the live auto-sampled ones.")


if __name__ == "__main__":
    main()
