import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from utils.utils import call_go_ov2mat
except ModuleNotFoundError:
    from ..utils.utils import call_go_ov2mat


def load_calibrations_as_matrices(filepath):
    """Loads a calibration JSON file and converts each frame to a homogeneous matrix."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    matrices = []
    for item in data:
        frame = item["run_calibration"]["frame"]
        translation = frame["translation"]
        orientation = frame["orientation"]["value"]

        t_vec = np.array([translation["x"], translation["y"], translation["z"]])
        
        o_x = orientation.get("x", 0.0)
        o_y = orientation.get("y", 0.0)
        o_z = orientation.get("z", 0.0)
        theta = orientation.get("th", 0.0)

        r_mat = call_go_ov2mat(o_x, o_y, o_z, theta)

        if r_mat is None:
            print(f"Warning: Failed to convert orientation for an item in {filepath}")
            continue

        h_mat = np.eye(4)
        h_mat[:3, :3] = r_mat
        h_mat[:3, 3] = t_vec
        matrices.append(h_mat)

    return matrices


def calculate_mean_transform(matrices):
    """Calculates the mean of a list of homogeneous transformation matrices."""
    if not matrices:
        return None

    translations = [m[:3, 3] for m in matrices]
    mean_translation = np.mean(translations, axis=0)

    rotations = [m[:3, :3] for m in matrices]
    m_sum = np.sum(rotations, axis=0)
    u, _, v_t = np.linalg.svd(m_sum)
    mean_rotation = u @ v_t

    if np.linalg.det(mean_rotation) < 0:
        u_prime = u.copy()
        u_prime[:, -1] *= -1
        mean_rotation = u_prime @ v_t

    mean_h_mat = np.eye(4)
    mean_h_mat[:3, :3] = mean_rotation
    mean_h_mat[:3, 3] = mean_translation

    return mean_h_mat


def calculate_variance(matrices, mean_matrix):
    """Calculates the variance of translation and rotation."""
    if not matrices:
        return None, None, None

    translations = [m[:3, 3] for m in matrices]
    mean_translation = mean_matrix[:3, 3]
    trans_variance = np.var(translations, axis=0)

    distances = [np.linalg.norm(t - mean_translation) for t in translations]
    avg_euclidean_dist = np.mean(distances)

    rotations = [m[:3, :3] for m in matrices]
    mean_rotation = mean_matrix[:3, :3]
    
    angular_distances_deg = []
    for r_mat in rotations:
        r_relative = mean_rotation.T @ r_mat
        
        trace = np.trace(r_relative)
        clipped_trace = np.clip((trace - 1) / 2, -1.0, 1.0)
        angle_rad = np.arccos(clipped_trace)
        angle_deg = np.rad2deg(angle_rad)
        angular_distances_deg.append(angle_deg)
        
    rot_variance_deg = np.var(angular_distances_deg)

    return trans_variance, rot_variance_deg, avg_euclidean_dist


def print_stats(name, matrices):
    """Calculates and prints statistics for a set of matrices."""
    print(f"\n--- Statistics for {name} ---")
    if len(matrices) < 2:
        print("Not enough data to compute variance.")
        return None

    mean_transform = calculate_mean_transform(matrices)
    
    print("Mean Transformation Matrix:")
    with np.printoptions(precision=4, suppress=True):
        print(mean_transform)

    trans_variance, rot_variance_deg, avg_euclidean_dist = calculate_variance(
        matrices, mean_transform
    )

    trans_std_dev = np.sqrt(trans_variance)
    rot_std_dev_deg = np.sqrt(rot_variance_deg)

    print("\nStandard Deviation:")
    print(
        f"  Translation (x, y, z) [mm]: {trans_std_dev[0]:.4f}, {trans_std_dev[1]:.4f}, {trans_std_dev[2]:.4f}"
    )
    print(f"  Average Euclidean Translation Deviation [mm]: {avg_euclidean_dist:.4f}")
    print(f"  Rotation [degrees]: {rot_std_dev_deg:.4f}")
    
    return mean_transform


def print_transform_diff(name1, mean1, name2, mean2):
    """Prints the difference between the mean of the two transformation matrices"""
    print(f"\n--- Diff between the mean of matrices {name1} and {name2} ---")
    
    trans1 = mean1[:3, 3]
    trans2 = mean2[:3, 3]
    trans_diff = np.linalg.norm(trans1 - trans2)
    
    rot1 = mean1[:3, :3]
    rot2 = mean2[:3, :3]
    
    r_relative = rot1.T @ rot2
    trace = np.trace(r_relative)
    clipped_trace = np.clip((trace - 1) / 2, -1.0, 1.0)
    angle_rad = np.arccos(clipped_trace)
    angle_deg = np.rad2deg(angle_rad)
    
    print(f"  Translational Distance [mm]: {trans_diff:.4f}")
    print(f"  Rotational Distance [degrees]: {angle_deg:.4f}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Analyze variance between hand-eye calibration datasets."
    )
    parser.add_argument(
        "calibration_files",
        nargs=2,
        type=str,
        help="Paths to the two calibration JSON files to compare.",
    )
    args = parser.parse_args()

    file1, file2 = args.calibration_files
    print(f"Analyzing {file1} and {file2}")

    matrices1 = load_calibrations_as_matrices(file1)
    matrices2 = load_calibrations_as_matrices(file2)

    name1 = os.path.basename(file1)
    name2 = os.path.basename(file2)

    mean1 = print_stats(name1, matrices1)
    mean2 = print_stats(name2, matrices2)

    if mean1 is not None and mean2 is not None:
        print_transform_diff(name1, mean1, name2, mean2)


if __name__ == "__main__":
    main()
