import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must come after backend selection)

from viam.proto.common import Pose

from utils.pose_utils import pose_to_matrix


def rotation_error(R1: np.ndarray, R2: np.ndarray) -> float:
    """Compute rotation error in degrees between two rotation matrices."""
    R_error = R1.T @ R2
    angle_rad = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1.0, 1.0))
    return float(np.degrees(angle_rad))


def compare_poses(pose1: Pose, pose2: Pose) -> Dict[str, Any]:
    """Compare two poses and return translation/rotation deltas."""
    T1 = pose_to_matrix(pose1)
    T2 = pose_to_matrix(pose2)

    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    t1 = T1[:3, 3]
    t2 = T2[:3, 3]

    translation_diff = t2 - t1
    translation_error = float(np.linalg.norm(translation_diff))
    rot_error_deg = rotation_error(R1, R2)

    position_diff = {
        "x": pose2.x - pose1.x,
        "y": pose2.y - pose1.y,
        "z": pose2.z - pose1.z,
    }

    orientation_diff = {
        "o_x": pose2.o_x - pose1.o_x,
        "o_y": pose2.o_y - pose1.o_y,
        "o_z": pose2.o_z - pose1.o_z,
        "theta": pose2.theta - pose1.theta,
    }

    return {
        "translation_error_mm": translation_error,
        "rotation_error_deg": rot_error_deg,
        "position_diff_mm": position_diff,
        "orientation_diff": orientation_diff,
        "pose1": {
            "x": pose1.x,
            "y": pose1.y,
            "z": pose1.z,
            "o_x": pose1.o_x,
            "o_y": pose1.o_y,
            "o_z": pose1.o_z,
            "theta": pose1.theta,
        },
        "pose2": {
            "x": pose2.x,
            "y": pose2.y,
            "z": pose2.z,
            "o_x": pose2.o_x,
            "o_y": pose2.o_y,
            "o_z": pose2.o_z,
            "theta": pose2.theta,
        },
    }


def draw_marker_debug(
    image: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_type: str = "chessboard",
    chessboard_size: Sequence[int] = (11, 8),
    square_size: float = 30.0,
    aruco_size: float = 200.0,
    validation_info: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Draw debugging overlays for detected calibration target."""
    debug_image = image.copy()

    if marker_type == "chessboard":
        cv2.drawFrameAxes(debug_image, camera_matrix, dist_coeffs, rvec, tvec, length=50)

        if validation_info and "reprojected_points" in validation_info:
            reprojected_points = validation_info["reprojected_points"]
            mean_error = validation_info.get("mean_reprojection_error", 0.0)
            max_error = validation_info.get("max_reprojection_error", 0.0)

            for point in reprojected_points:
                cv2.circle(debug_image, tuple(point.astype(int)), 2, (0, 255, 0), -1)

            cv2.putText(
                debug_image,
                f"Reprojection Error: {mean_error:.1f}px",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                debug_image,
                f"Max Error: {max_error:.1f}px",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            if "sharpness" in validation_info:
                sharpness = validation_info["sharpness"]
                if sharpness != float("inf"):
                    sharpness_color = (0, 255, 0)
                    if sharpness >= 5.0:
                        sharpness_color = (0, 0, 255)
                    elif sharpness >= 3.0:
                        sharpness_color = (0, 165, 255)
                    cv2.putText(
                        debug_image,
                        f"Sharpness: {sharpness:.1f}px",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        sharpness_color,
                        2,
                    )
    elif marker_type == "aruco":
        cv2.drawFrameAxes(debug_image, camera_matrix, dist_coeffs, rvec, tvec, aruco_size / 2)

    return debug_image


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def calculate_measurement_statistics(measurements: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not measurements:
        return None

    valid = [m for m in measurements if m.get("success")]
    if not valid:
        return None

    stats: Dict[str, Any] = {
        "num_measurements": len(valid),
        "success_rate": len(valid) / len(measurements),
    }

    def add_stats(key: str, values: List[float]) -> None:
        if values:
            stats.update(
                {
                    f"{key}_avg": float(np.mean(values)),
                    f"{key}_std": float(np.std(values)),
                    f"{key}_min": float(np.min(values)),
                    f"{key}_max": float(np.max(values)),
                }
            )

    add_stats("mean_reprojection_error", [m.get("mean_reprojection_error", 0.0) for m in valid])
    add_stats("max_reprojection_error", [m.get("max_reprojection_error", 0.0) for m in valid])
    add_stats(
        "sharpness",
        [m.get("sharpness") for m in valid if m.get("sharpness") not in (None, float("inf"))],
    )
    add_stats("corners_count", [m.get("corners_count", 0) for m in valid])

    add_stats(
        "rotation_error",
        [m.get("hand_eye_errors", {}).get("rotation_error", 0.0) for m in valid],
    )
    add_stats(
        "translation_error",
        [m.get("hand_eye_errors", {}).get("translation_error", 0.0) for m in valid],
    )

    return stats


def generate_comprehensive_statistics(rotation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rotation_data:
        return {
            "total_poses": 0,
            "successful_poses": 0,
            "success_rate": 0.0,
            "hand_eye": {},
            "reprojection": {},
            "detection": {},
        }

    total_poses = len(rotation_data)
    successful_poses = len([p for p in rotation_data if p.get("hand_eye_errors")])
    success_rate = successful_poses / total_poses if total_poses > 0 else 0.0

    rotation_errors: List[float] = []
    translation_errors: List[float] = []
    mean_reprojection_errors: List[float] = []
    max_reprojection_errors: List[float] = []
    sharpness_values: List[float] = []
    corner_counts: List[float] = []

    for pose_data in rotation_data:
        for measurement in pose_data.get("measurements", []):
            if not measurement.get("success"):
                continue
            he = measurement.get("hand_eye_errors", {})
            rotation_errors.append(he.get("rotation_error", 0.0))
            translation_errors.append(he.get("translation_error", 0.0))
            mean_reprojection_errors.append(measurement.get("mean_reprojection_error", 0.0))
            max_reprojection_errors.append(measurement.get("max_reprojection_error", 0.0))
            sharp = measurement.get("sharpness")
            if sharp is not None and sharp != float("inf"):
                sharpness_values.append(sharp)
            corner_counts.append(measurement.get("corners_count", 0.0))

    def calc(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return {
        "total_poses": total_poses,
        "successful_poses": successful_poses,
        "success_rate": success_rate,
        "hand_eye": {
            "rotation_error": calc(rotation_errors),
            "translation_error": calc(translation_errors),
        },
        "reprojection": {
            "mean_error": calc(mean_reprojection_errors),
            "max_error": calc(max_reprojection_errors),
        },
        "detection": {
            "sharpness": calc(sharpness_values),
            "corners": calc(corner_counts),
        },
    }


def _ensure_plot_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def create_comprehensive_statistics_plot(rotation_data: List[Dict[str, Any]], data_dir: str, tag: Optional[str] = None) -> None:
    stats = generate_comprehensive_statistics(rotation_data)
    if not stats["total_poses"]:
        return

    plot_dir = data_dir
    _ensure_plot_dir(plot_dir)

    values = stats["hand_eye"]["rotation_error"]
    translations = stats["hand_eye"]["translation_error"]
    repro_mean = stats["reprojection"]["mean_error"]
    repro_max = stats["reprojection"]["max_error"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].bar(["Rotation"], [values["mean"]], yerr=[values["std"]])
    axes[0, 0].set_title("Rotation Error (°)")

    axes[0, 1].bar(["Translation"], [translations["mean"]], yerr=[translations["std"]])
    axes[0, 1].set_title("Translation Error (mm)")

    axes[1, 0].bar(["Mean reproj"], [repro_mean["mean"]], yerr=[repro_mean["std"]])
    axes[1, 0].set_title("Mean Reprojection Error (px)")

    axes[1, 1].bar(["Max reproj"], [repro_max["mean"]], yerr=[repro_max["std"]])
    axes[1, 1].set_title("Max Reprojection Error (px)")

    if tag:
        fig.suptitle(f"Calibration statistics - {tag}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "comprehensive_statistics.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_pose_error_plot(rotation_data: List[Dict[str, Any]], data_dir: str, tag: Optional[str] = None) -> None:
    if not rotation_data:
        return

    pose_indices: List[int] = []
    rotation_errors: List[float] = []
    translation_errors: List[float] = []

    for i, pose_data in enumerate(rotation_data):
        pose_index = pose_data.get("pose_index", i)
        stats = pose_data.get("measurement_statistics") or {}
        rot = stats.get("rotation_error_avg")
        trans = stats.get("translation_error_avg")

        if rot is None or trans is None:
            continue
        pose_indices.append(pose_index)
        rotation_errors.append(rot)
        translation_errors.append(trans)

    if not pose_indices:
        return

    _ensure_plot_dir(data_dir)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(pose_indices, rotation_errors, "o-", color="tab:blue", label="Rotation (°)")
    ax1.set_xlabel("Pose Index")
    ax1.set_ylabel("Rotation Error (°)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(pose_indices, translation_errors, "s-", color="tab:red", label="Translation (mm)")
    ax2.set_ylabel("Translation Error (mm)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    title = "Pose-by-Pose Hand-Eye Errors"
    if tag:
        title += f" - {tag}"
    plt.title(title)
    fig.tight_layout()
    plt.savefig(os.path.join(data_dir, "pose_error_analysis.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_summary_table_plot(rotation_data: List[Dict[str, Any]], data_dir: str, tag: Optional[str] = None) -> None:
    stats = generate_comprehensive_statistics(rotation_data)
    if not stats["total_poses"]:
        return

    table_data = [
        ["Metric", "Mean", "Std Dev", "Min", "Max"],
        [
            "Rotation Error (°)",
            f"{stats['hand_eye']['rotation_error']['mean']:.3f}",
            f"{stats['hand_eye']['rotation_error']['std']:.3f}",
            f"{stats['hand_eye']['rotation_error']['min']:.3f}",
            f"{stats['hand_eye']['rotation_error']['max']:.3f}",
        ],
        [
            "Translation Error (mm)",
            f"{stats['hand_eye']['translation_error']['mean']:.3f}",
            f"{stats['hand_eye']['translation_error']['std']:.3f}",
            f"{stats['hand_eye']['translation_error']['min']:.3f}",
            f"{stats['hand_eye']['translation_error']['max']:.3f}",
        ],
        [
            "Mean Reprojection (px)",
            f"{stats['reprojection']['mean_error']['mean']:.3f}",
            f"{stats['reprojection']['mean_error']['std']:.3f}",
            f"{stats['reprojection']['mean_error']['min']:.3f}",
            f"{stats['reprojection']['mean_error']['max']:.3f}",
        ],
        [
            "Max Reprojection (px)",
            f"{stats['reprojection']['max_error']['mean']:.3f}",
            f"{stats['reprojection']['max_error']['std']:.3f}",
            f"{stats['reprojection']['max_error']['min']:.3f}",
            f"{stats['reprojection']['max_error']['max']:.3f}",
        ],
        [
            "Sharpness (px)",
            f"{stats['detection']['sharpness']['mean']:.2f}",
            f"{stats['detection']['sharpness']['std']:.2f}",
            f"{stats['detection']['sharpness']['min']:.2f}",
            f"{stats['detection']['sharpness']['max']:.2f}",
        ],
        [
            "Corner Count",
            f"{stats['detection']['corners']['mean']:.1f}",
            f"{stats['detection']['corners']['std']:.1f}",
            f"{stats['detection']['corners']['min']:.1f}",
            f"{stats['detection']['corners']['max']:.1f}",
        ],
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    title = "Calibration Summary"
    if tag:
        title += f" - {tag}"
    plt.title(title, fontsize=14, fontweight="bold", pad=20)

    plt.savefig(os.path.join(data_dir, "statistics_summary_table.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_hand_eye_verification_errors(
    T_hand_eye: np.ndarray,
    T_delta_A_world_frame: np.ndarray,
    T_delta_B_camera_frame: np.ndarray,
) -> Dict[str, float]:
    T_eye_hand = np.linalg.inv(T_hand_eye)
    T_A_predicted = T_hand_eye @ T_delta_B_camera_frame @ T_eye_hand

    R_A_actual = T_delta_A_world_frame[:3, :3]
    R_A_predicted = T_A_predicted[:3, :3]
    t_A_actual = T_delta_A_world_frame[:3, 3]
    t_A_predicted = T_A_predicted[:3, 3]

    rot_error = rotation_error(R_A_predicted, R_A_actual)
    trans_error = float(np.linalg.norm(t_A_predicted - t_A_actual))

    return {"rotation_error": rot_error, "translation_error": trans_error}


def update_hand_eye_errors_for_run(
    rotation_data: List[Dict[str, Any]],
    T_A_world_frames: List[np.ndarray],
    T_B_camera_frames: List[np.ndarray],
    T_hand_eye: np.ndarray,
) -> None:
    if not rotation_data or not T_A_world_frames or not T_B_camera_frames:
        return

    T_A_0 = T_A_world_frames[0]
    T_B_0 = T_B_camera_frames[0]

    for idx, pose_data in enumerate(rotation_data):
        if idx >= len(T_A_world_frames) or idx >= len(T_B_camera_frames):
            break

        T_A_i = T_A_world_frames[idx]
        T_B_i = T_B_camera_frames[idx]

        T_delta_A = np.linalg.inv(T_A_i) @ T_A_0
        T_delta_B = T_B_i @ np.linalg.inv(T_B_0)

        errors = compute_hand_eye_verification_errors(T_hand_eye, T_delta_A, T_delta_B)
        pose_data["hand_eye_errors"] = errors

        for measurement in pose_data.get("measurements", []):
            measurement.setdefault("hand_eye_errors", errors.copy())

        stats = calculate_measurement_statistics(pose_data.get("measurements", []))
        if stats:
            pose_data["measurement_statistics"] = stats


def save_run_outputs(
    tracking_dir: str,
    rotation_data: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    os.makedirs(tracking_dir, exist_ok=True)

    pose_data_path = os.path.join(tracking_dir, "pose_data.json")
    save_json(pose_data_path, rotation_data)

    stats = generate_comprehensive_statistics(rotation_data)

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "metadata": metadata or {},
        "statistics": stats,
        "pose_count": len(rotation_data),
        "pose_data_file": pose_data_path,
    }

    summary_path = os.path.join(tracking_dir, "summary.json")
    save_json(summary_path, summary)

    create_summary_table_plot(rotation_data, tracking_dir, tag=tag)
    create_pose_error_plot(rotation_data, tracking_dir, tag=tag)

    return summary

