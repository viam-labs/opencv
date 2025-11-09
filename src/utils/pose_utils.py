import cv2
import numpy as np

from viam.proto.common import Pose

try:
    from utils.utils import call_go_ov2mat, call_go_mat2ov
except ModuleNotFoundError:
    from ..utils.utils import call_go_ov2mat, call_go_mat2ov

def pose_to_matrix(pose: Pose) -> np.ndarray:
    """Convert a Viam Pose to a 4x4 homogeneous transformation matrix."""
    # Get 3x3 rotation matrix from orientation vector
    R = call_go_ov2mat(pose.o_x, pose.o_y, pose.o_z, pose.theta)
    if R is None:
        raise Exception("Failed to convert orientation vector to rotation matrix")

    # Build 4x4 homogeneous transformation matrix
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = [pose.x, pose.y, pose.z]

    return T

def matrix_to_pose(T: np.ndarray) -> Pose:
    """Convert a 4x4 homogeneous transformation matrix to a Viam Pose."""
    # Extract rotation matrix (top-left 3x3)
    R = T[0:3, 0:3]

    # Extract translation vector
    t = T[0:3, 3]

    # Convert rotation matrix to orientation vector
    ov = call_go_mat2ov(R)
    if ov is None:
        raise Exception("Failed to convert rotation matrix to orientation vector")

    ox, oy, oz, theta = ov

    return Pose(
        x=float(t[0]),
        y=float(t[1]),
        z=float(t[2]),
        o_x=ox,
        o_y=oy,
        o_z=oz,
        theta=theta
    )

def invert_pose_rotation_only(pose: Pose) -> Pose:
    """Invert only the rotation of a pose, keeping translation unchanged."""
    # Convert pose to matrix
    T = pose_to_matrix(pose)

    # Extract rotation and translation
    R = T[0:3, 0:3]
    t = T[0:3, 3]

    # Invert rotation (transpose for rotation matrices)
    R_inv = R.T

    # Build new transformation with inverted rotation but same translation
    T_inv_rot = np.eye(4)
    T_inv_rot[0:3, 0:3] = R_inv
    T_inv_rot[0:3, 3] = t  # Keep original translation

    # Convert back to pose
    return matrix_to_pose(T_inv_rot)

def validate_chessboard_detection(corners, rvec, tvec, camera_matrix, dist_coeffs, 
                                   chessboard_size, square_size=30.0, objp=None):
    # Generate 3D object points ONLY if not provided
    if objp is None:
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size
    
    # CRITICAL FIX: Reshape corners to (N, 2) to match reprojected_points
    # OpenCV returns corners as (N, 1, 2), but we need (N, 2) for calculations
    corners_2d = corners.reshape(-1, 2)
    
    # Verify dimensions match
    if len(corners_2d) != len(objp):
        raise ValueError(f"Dimension mismatch: {len(corners_2d)} corners but {len(objp)} object points")
    
    # Calculate reprojection errors using the same method as camera_calibration.py
    # Project 3D points back to image space
    reprojected_points, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
    reprojected_points = reprojected_points.reshape(-1, 2)
    
    # Calculate reprojection error using camera_calibration.py method (OpenCV standard)
    imgpoints2, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
    mean_error = cv2.norm(corners, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    
    # Calculate per-point errors
    errors = np.linalg.norm(corners_2d - reprojected_points, axis=1)
    
    # Calculate statistics
    max_error = np.max(errors)
    
    return mean_error, max_error, reprojected_points, errors



def get_chessboard_pose_in_camera_frame(image, camera_matrix, dist_coeffs, chessboard_size, 
                                       square_size=30.0, pnp_method=cv2.SOLVEPNP_IPPE, 
                                       use_sb_detection=True, verbose=False):
    """
    Get chessboard pose in camera frame using PnP with improved outlier filtering.
    
    Args:
        data_dir: Directory to save histogram. If None, saves in current directory.
    
    Note: cv2.solvePnP returns the transformation from chessboard coordinates to camera coordinates.
    This is T_chessboard_to_camera, NOT T_camera_to_chessboard.
    
    Args:
        use_sb_detection: If True, use findChessboardCornersSB (more robust), otherwise use findChessboardCorners
    
    Returns: (success, rotation_vector, translation_vector, corners, marker_info)
        - rvec, tvec represent the chessboard's pose in the camera's coordinate system
        - marker_info contains validation info (reprojection error, sharpness)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Prepare 3D object points (chessboard corners in world coordinates)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)
    objp *= square_size  # Scale by square size (e.g., 25mm)
    
    # Find chessboard corners using selected method
    corners = None
    if use_sb_detection:
        if verbose:
            print("Using findChessboardCornersSB (subpixel detection)")
        flags = (cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
        ret, corners = cv2.findChessboardCornersSB(gray, chessboard_size, flags=flags)
    else:
        if verbose:
            print("Using findChessboardCorners (traditional detection)")
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if ret:
        # Enhanced corner refinement for higher accuracy (only for traditional method)
        if not use_sb_detection:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)  # More iterations, stricter convergence
            corners = cv2.cornerSubPix(gray, corners, (15,15), (-1,-1), criteria)  # Larger refinement window

        # Estimate chessboard sharpness (only when detection succeeds)
        sharpness = float('inf')
        try:
            sharpness_result = cv2.estimateChessboardSharpness(image, chessboard_size, corners)
            # The function returns a tuple ((sharpness_value, ...), sharpness_map)
            sharpness = sharpness_result[0][0]  # First element of first tuple
            if verbose:
                print(f"Chessboard sharpness: {sharpness:.2f} pixels")
        except Exception as e:
            if verbose:
                print(f"Could not estimate sharpness: {e}")
            sharpness = float('inf')  # Mark as unknown
        
        
        # Filter outliers before solvePnP with IMPROVED thresholds
        if verbose:
            print(f"Original corners: {len(corners)} points")
        
        # Method 1: Use iterative outlier filtering with tighter threshold
        try:
            # First, try to get an initial pose estimate with all points
            success_init, rvec_init, tvec_init = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs, flags=pnp_method)
            
            if success_init:
                # Project points back and calculate errors
                projected_points, _ = cv2.projectPoints(objp, rvec_init, tvec_init, camera_matrix, dist_coeffs)
                projected_points = projected_points.reshape(-1, 2)
                corners_2d = corners.reshape(-1, 2)
                
                # Calculate reprojection errors
                errors = np.linalg.norm(corners_2d - projected_points, axis=1)
                
                # IMPROVED: Use statistical outlier detection
                # Filter using median absolute deviation (more robust than std)
                median_error = np.median(errors)
                mad = np.median(np.abs(errors - median_error))
                
                # Modified z-score using MAD
                modified_z_scores = 0.6745 * (errors - median_error) / (mad + 1e-10)
                
                # Keep points with modified z-score < 3.5 (equivalent to ~3 sigma)
                # OR use absolute threshold of 2 pixels (whichever is stricter)
                threshold_statistical = median_error + 3.5 * mad
                threshold_absolute = 2.0
                threshold = min(threshold_statistical, threshold_absolute)
                
                good_indices = errors < threshold
                
                n_filtered = len(corners) - np.sum(good_indices)
                if n_filtered > 0 and verbose:
                    print(f"Filtering {n_filtered} outliers (threshold: {threshold:.2f}px)")
                    print(f"  Error range: {np.min(errors):.3f} to {np.max(errors):.3f}px")
                    print(f"  Median: {median_error:.3f}px, MAD: {mad:.3f}px")
                
                if np.sum(good_indices) >= 20:  # Need at least 20 points for reliable PnP
                    filtered_corners = corners[good_indices]
                    filtered_objp = objp[good_indices]
                    if verbose:
                        print(f"Filtered corners: {len(filtered_corners)}/{len(corners)} points (error < {threshold:.2f}px)")
                    
                    # Use filtered points for final solvePnP
                    success, rvec, tvec = cv2.solvePnP(filtered_objp, filtered_corners, camera_matrix, dist_coeffs, flags=pnp_method)
                    corners = filtered_corners  # Update corners for validation
                    objp = filtered_objp  # Update objp for validation
                else:
                    if verbose:
                        print(f"Not enough good points ({np.sum(good_indices)}), using all points")
                    success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs, flags=pnp_method)
            else:
                if verbose:
                    print("Initial solvePnP failed, using all points")
                success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs, flags=pnp_method)
                
        except Exception as e:
            if verbose:
                print(f"Outlier filtering failed: {e}, using all points")
            success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs, flags=pnp_method)

        if not success:
            if verbose:
                print("Failed to solve PnP")
            return False, None, None, None, None, None, None

        # Enhanced refinement with VVS
        rvec, tvec = cv2.solvePnPRefineVVS(objp, corners, camera_matrix, dist_coeffs, rvec, tvec)
        
        # Additional refinement with LM method
        rvec, tvec = cv2.solvePnPRefineLM(objp, corners, camera_matrix, dist_coeffs, rvec, tvec)
        
        # Validate detection quality
        mean_error, max_error, reprojected_points, errors = validate_chessboard_detection(
            corners, rvec, tvec, camera_matrix, dist_coeffs, chessboard_size, square_size, objp)

        R, _ = cv2.Rodrigues(rvec)
        tvec_reshaped = tvec.reshape(3, 1)

        # solvePnP returns: R (object->camera), tvec (position of object origin in camera frame)
        # We need: R_cam2target, t_cam2target (position of target origin in camera frame)
        R_cam2target = R.T  # Inverse rotation: camera->target
        t_cam2target = tvec_reshaped  # tvec is already the position of target origin in camera frame

        
        if verbose:
            print(f"Chessboard detection quality: mean={mean_error:.3f}px, max={max_error:.3f}px")
        
        # Return validation info as marker_info
        validation_info = {
            'mean_reprojection_error': mean_error,
            'max_reprojection_error': max_error,
            'reprojected_points': reprojected_points,
            'sharpness': sharpness,
            'errors': errors
        }
        
        return True, rvec, tvec, R_cam2target, t_cam2target, corners, validation_info
    else:
        return False, None, None, None, None, None, None


def get_aruco_pose_in_camera_frame(image, camera_matrix, dist_coeffs, marker_id=0, marker_size=300.0, aruco_dict=cv2.aruco.DICT_6X6_250, pnp_method=cv2.SOLVEPNP_IPPE_SQUARE):
    """
    Get ArUco marker pose in camera frame using PnP.
    
    Args:
        image: Input image
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        marker_id: ID of the marker to detect (if None, uses first detected marker)
        marker_size: Size of the marker in mm
        aruco_dict: ArUco dictionary to use
        pnp_method: PnP method to use (cv2.SOLVEPNP_* constant)
        
    Returns: (success, rotation_vector, translation_vector, corners, detected_id)
        - rvec, tvec represent the marker's pose in the camera's coordinate system
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Create ArUco detector with optimized parameters
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
    parameters = cv2.aruco.DetectorParameters()
    
    # Optimize corner refinement for higher accuracy
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementMaxIterations = 50
    parameters.cornerRefinementMinAccuracy = 0.01
    parameters.cornerRefinementWinSize = 7
    parameters.relativeCornerRefinmentWinSize = 0.4
    
    # Optimize detection sensitivity
    parameters.minMarkerPerimeterRate = 0.02
    parameters.maxMarkerPerimeterRate = 3.0
    parameters.minCornerDistanceRate = 0.03
    
    # Improve error correction
    parameters.errorCorrectionRate = 0.8
    parameters.maxErroneousBitsInBorderRate = 0.2
    
    # Optimize adaptive thresholding
    parameters.adaptiveThreshWinSizeMin = 5
    parameters.adaptiveThreshWinSizeMax = 25
    parameters.adaptiveThreshWinSizeStep = 8
    parameters.adaptiveThreshConstant = 5
    
    detector = cv2.aruco.ArucoDetector(aruco_dictionary, parameters)
    
    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if ids is not None and len(ids) > 0:
        # Find the requested marker (or use first one if marker_id is None)
        marker_idx = None
        if marker_id is None:
            marker_idx = 0
            detected_id = ids[0][0]
        else:
            for i, id_val in enumerate(ids):
                if id_val[0] == marker_id:
                    marker_idx = i
                    detected_id = marker_id
                    break
        
        if marker_idx is None:
            print(f"Marker ID {marker_id} not found. Detected IDs: {ids.flatten()}")
            return False, None, None, None, None
        
        # Get corners for the detected marker
        marker_corners = corners[marker_idx][0]  # Shape: (4, 2)
        
        # Define 3D points of the marker (standard ArUco coordinate system)
        # Origin at top-left, X-right, Y-down, Z-out
        objp = np.array([
            [0, 0, 0],                    # Top-left
            [marker_size, 0, 0],          # Top-right
            [marker_size, marker_size, 0], # Bottom-right
            [0, marker_size, 0]           # Bottom-left
        ], dtype=np.float32)
        
        # Solve PnP for this marker using specified method
        success, rvec, tvec = cv2.solvePnP(
            objp, marker_corners, camera_matrix, dist_coeffs,
            flags=pnp_method
        )
        
        if not success:
            print("Failed to solve PnP for ArUco marker")
            return False, None, None, None, None
        
        # Refine with VVS
        rvec, tvec = cv2.solvePnPRefineVVS(objp, marker_corners, camera_matrix, dist_coeffs, rvec, tvec)
        
        return True, rvec, tvec, marker_corners, detected_id
    else:
        return False, None, None, None, None
