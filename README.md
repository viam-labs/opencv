# Module opencv

A Viam module that provides OpenCV-based computer vision components for robotics applications. This module includes pose tracking capabilities using chessboard patterns and hand-eye calibration services for robotic arm calibration.

## Model viam:opencv:chessboard

A pose tracker component that detects and tracks the pose of chessboard calibration patterns in camera images. This model uses OpenCV's chessboard detection algorithms to provide accurate 6-DOF pose estimation of chessboard targets, making it ideal for camera calibration and pose tracking applications.

### Post Tracker Configuration

The following attribute template can be used to configure this model:

```json
{
"camera_name": <string>,
"pattern_size": <list>,
"square_size_mm": <int>
}
```

#### Pose Tracker Attributes

The following attributes are available for this model:

| Name             | Type   | Inclusion | Description                                             |
|------------------|--------|-----------|---------------------------------------------------------|
| `camera_name`    | string | Required  | Name of the camera used for checking pose of chessboard.|
| `pattern_size`   | list   | Required  | Dimensions of the chessboard pattern (rows x columns of inner corner squares).|
| `square_size_mm` | int    | Required  | Physical size of a square in the chessboard pattern.    |

#### Pose Tracker Example Configuration

```json
{
  "camera_name": "cam",
  "pattern_size": [9, 6],
  "square_size_mm": 21
}
```

## Model viam:opencv:handeyecalibration

A calibration service that performs hand-eye calibration for robotic arms with mounted or fixed cameras. This service automates the process of determining the transformation between the robot's end-effector and camera coordinate frames by moving the arm through predefined poses while tracking tags. The resulting calibration enables accurate coordination between robot motion and visual perception.

### Hand Eye Calibration Configuration

The following attribute template can be used to configure this model:

```json
{
"arm_name": <string>,
"calibration_type": <string>,
"camera_name": <string>,
"joint_positions": <list>,
"method": <string>,
"pose_tracker": <string>,
"motion": <string>,
"sleep_seconds": <float>
}
```

#### Hand Eye Calibration Attributes

The following attributes are available for this model:

| Name              | Type     | Inclusion  | Description                                                                        |
|-------------------|----------|------------|------------------------------------------------------------------------------------|
| `arm_name`        | `string` | `Required` | Name of the arm component used for calibration.                                    |
| `calibration_type`| `string` | `Required` | Name of the type of calibration to perform.                                        |
| `camera_name`     | `string` | `Required` | Name of the camera component used for calibration.                                 |
| `joint_positions` | `list`   | `Required` | List of joint positions for calibration poses.                                     |
| `method`          | `string` | `Required` | Method to use for calibration.                                                     |
| `pose_tracker`    | `string` | `Required` | Name of the pose tracker component to detect markers.                              |
| `motion`          | `string` | `Optional` | Name of the motion service for coordinated movement.                               |
| `sleep_seconds`   | `float`  | `Optional` | Sleep time between movements to allow for arm to settle (defaults to 2.0 seconds). |

Available calibrations are:

- "eye-in-hand"
- "eye-to-hand"

Available methods are:

- "CALIB_HAND_EYE_TSAI"
- "CALIB_HAND_EYE_PARK"
- "CALIB_HAND_EYE_HORAUD"
- "CALIB_HAND_EYE_ANDREFF"
- "CALIB_HAND_EYE_DANIILIDIS"

#### Hand Eye Calibration Example Configuration

```json
{
  "arm_name": "my_arm",
  "calibration_type": "eye-in-hand",
  "camera_name": "cam",
  "joint_positions": [[0, 0, 0, 0, 0, 0], [0.1, 0.2, 0.3, 0, 0, 0]],
  "method": "CALIB_HAND_EYE_TSAI",
  "pose_tracker": "tag_tracker",
  "motion": "motion_service",
  "sleep_seconds": 2.0
}
```

## Model viam:opencv:camera-calibration

A generic service that provides camera calibration functionality through the `do_command` interface. This service uses chessboard patterns to determine camera intrinsic parameters (focal lengths, principal point, and distortion coefficients) from multiple images. Unlike the chessboard pose tracker, this service is dedicated solely to calibration and does not track poses.

### Camera Calibration Configuration

The following attribute template can be used to configure this model:

```json
{
  "pattern_size": <list>,
  "square_size_mm": <int>
}
```

#### Camera Calibration Attributes

The following attributes are available for this model:

| Name             | Type   | Inclusion | Description                                             |
|------------------|--------|-----------|---------------------------------------------------------|
| `pattern_size`   | list   | Required  | Dimensions of the chessboard pattern (rows x columns of inner corner squares).|
| `square_size_mm` | int    | Required  | Physical size of a square in the chessboard pattern in millimeters.|

#### Camera Calibration Example Configuration

```json
{
  "pattern_size": [9, 6],
  "square_size_mm": 21
}
```

### Calibrate Camera Command

Use the `calibrate_camera` command via `do_command` to compute camera intrinsics:

See `src/scripts/camera_calibration_script.py` for the Python script to do so..

**Parameters:**
- `images` (required): List of base64 encoded image strings containing chessboard patterns

**Returns:**
The command returns a dictionary with the following structure:

```json
{
  "success": true,
  "rms_error": 0.234,
  "reprojection_error": 0.156,
  "num_images": 10,
  "image_size": {
    "width": 1920,
    "height": 1080
  },
  "camera_matrix": {
    "fx": 1234.56,
    "fy": 1235.67,
    "cx": 960.12,
    "cy": 540.34
  },
  "distortion_coefficients": {
    "k1": -0.123,
    "k2": 0.045,
    "p1": -0.001,
    "p2": 0.002,
    "k3": -0.012
  }
}
```

**How it works:**
1. You capture images of the chessboard pattern at your own pace (the user controls when each image is captured)
2. Encode the images as base64 strings and pass them to the command
3. For each image, the system detects and refines the chessboard corners
4. Once all images are processed, it uses OpenCV's `calibrateCamera` function to compute:
   - **Camera matrix (intrinsics)**: Focal lengths (fx, fy) and principal point (cx, cy)
   - **Distortion coefficients**: Radial (k1, k2, k3) and tangential (p1, p2) distortion parameters
   - **RMS error**: Root mean square reprojection error from calibration (lower is better)
   - **Re-projection error**: Mean error when projecting 3D points back to 2D using calibrated parameters (lower is better, closer to zero indicates more accurate calibration)
   - **Number of images**: Number of images successfully used for calibration

**Tips for best results:**
- Capture 10-20 images of the chessboard in different positions and orientations
- Cover different areas of the camera's field of view
- Ensure the chessboard is well-lit and in focus in each image
- Tilt and rotate the chessboard or the camera between captures for better calibration
- At least 3 valid images are required for calibration to succeed
