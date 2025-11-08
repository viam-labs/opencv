import numpy as np
import subprocess
import sys
import os

from viam.components.arm import Pose


def _get_binary_path():
    """Get the path to the go_utils binary"""
    if getattr(sys, 'frozen', False):
        # Running in PyInstaller bundle
        binary_path = os.path.join(sys._MEIPASS, 'go_utils')
    else:
        # Running in development
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        binary_path = os.path.join(script_dir, 'go_utils', 'go_utils')
    
    return binary_path


def call_go_ov2mat(ox: float, oy: float, oz: float, theta: float) -> np.ndarray:
    """
    Call Go binary to convert Viam orientation vector to rotation matrix
    
    Args:
        ox, oy, oz, theta: Viam orientation vector components
        
    Returns:
        3x3 rotation matrix as numpy array
    """
    try:
        binary_path = _get_binary_path()
        
        # Call Go binary with orientation vector parameters
        result = subprocess.run([
            binary_path, 'ov2mat',
            str(ox), str(oy), str(oz), str(theta)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Go binary error: {result.stderr}")
            return None
            
        # Parse the 9 values returned by Go binary (3x3 matrix flattened)
        values = [float(x) for x in result.stdout.strip().split()]
        if len(values) != 9:
            print(f"Expected 9 values from Go binary, got {len(values)}")
            return None
            
        # Reshape to 3x3 matrix
        return np.array(values).reshape(3, 3)
        
    except Exception as e:
        print(f"Failed to call Go orientation converter: {e}")
        return None
    

def call_go_mat2ov(R: np.ndarray) -> tuple:
    """
    Call Go binary to convert rotation matrix to Viam orientation vector

    Args:
        R: 3x3 rotation matrix as numpy array

    Returns:
        (ox, oy, oz, theta): Orientation vector components, or None if failed
    """
    try:
        binary_path = _get_binary_path()

        # Flatten the rotation matrix to get 9 elements
        flat_matrix = R.flatten()
        matrix_args = [str(val) for val in flat_matrix]

        # Call Go binary with mat2ov command and 9 matrix elements
        result = subprocess.run([
            binary_path, 'mat2ov'
        ] + matrix_args, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Go binary error: {result.stderr}")
            return None

        # Parse the 4 values returned by Go binary (ox, oy, oz, theta)
        values = [float(x) for x in result.stdout.strip().split()]
        if len(values) != 4:
            print(f"Expected 4 values from Go binary, got {len(values)}")
            return None

        # Return orientation vector components
        return values[0], values[1], values[2], values[3]

    except Exception as e:
        print(f"Failed to call Go mat2ov converter: {e}")
        return None