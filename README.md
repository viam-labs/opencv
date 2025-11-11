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

A calibration service that performs hand-eye calibration for robotic arms with mounted or fixed cameras. This service automates the process of determining the transformation between the robot's end-effector and camera coordinate frames by moving the arm through predefined positions while tracking bodies. The resulting calibration enables accurate coordination between robot motion and visual perception.

This service supports three operation modes:

1. **Joint Position Mode**: Uses direct joint control to move the arm through predefined joint positions
2. **Direct Pose Mode**: Uses `arm.move_to_position` to move the arm through predefined poses without motion planning, assuming the arm implements that method.
3. **Motion Planning Mode**: Uses the motion service to move the arm through predefined poses with obstacle avoidance and motion planning

### Hand Eye Calibration Configuration

The following attribute template can be used to configure this model:

```json
{
"arm_name": <string>,
"body_name": <string>,
"calibration_type": <string>,
"joint_positions": <list>,
"poses": <list>,
"method": <string>,
"pose_tracker": <string>,
"motion": <string>,
"sleep_seconds": <float>,
"use_motion_service_for_poses": <bool>
}
```

#### Hand Eye Calibration Attributes

The following attributes are available for this model:

| Name              | Type     | Inclusion  | Description                                                                        |
|-------------------|----------|------------|------------------------------------------------------------------------------------|
| `arm_name`        | `string` | `Required` | Name of the arm component used for calibration.                                    |
| `calibration_type`| `string` | `Required` | Type of calibration to perform (see available calibrations below).                 |
| `joint_positions` | `list`   | `Required*` | List of joint positions (in radians) for calibration poses. Required if `poses` is not provided. |
| `poses`           | `list`   | `Required*` | List of poses (with x, y, z, o_x, o_y, o_z, theta) for calibration. Required if `joint_positions` is not provided. Can be used with or without the `motion` service. If both `joint_positions` and `poses` are provided, `poses` will be used. |
| `method`          | `string` | `Required` | Calibration method to use (see available methods below).                           |
| `pose_tracker`    | `string` | `Required` | Name of the pose tracker component to detect tracked bodies.                       |
| `motion`          | `string` | `Optional` | Name of the motion service for motion planning with obstacle avoidance. When provided with `poses`, uses motion planning. When not provided with `poses`, uses direct `arm.move_to_position`. |
| `sleep_seconds`   | `float`  | `Optional` | Sleep time between movements to allow for arm to settle (defaults to 2.0 seconds). |
| `use_motion_service_for_poses` | `bool` | `Optional` | Whether to use the motion service's `get_pose()` method to retrieve arm poses during calibration (defaults to false). When true, uses `motion.get_pose()` with the arm's origin frame. When false, uses `arm.get_end_position()`. Requires `motion` service to be configured when true. |
| `body_name`       | `string` | `Optional` | Name of the specific tracked body to use (e.g., AprilTag ID like "tag36h11:0" or chessboard corner like "corner_0"). Calibration expects exactly one pose, so if the pose tracker's `get_poses` returns more than one pose, this attribute will be necessary to specify. **Important**: When using chessboard corners, ensure the chessboard maintains consistent orientation across all calibration poses to ensure the same corner is tracked. |

**Note**: Either `joint_positions` or `poses` must be provided. If both are provided, `poses` will take precedence.

Available calibrations are:

- "eye-in-hand"
- "eye-to-hand"

Available methods are:

- "CALIB_HAND_EYE_TSAI"
- "CALIB_HAND_EYE_PARK"
- "CALIB_HAND_EYE_HORAUD"
- "CALIB_HAND_EYE_ANDREFF"
- "CALIB_HAND_EYE_DANIILIDIS"

#### Hand Eye Calibration Example Configurations

**Joint Position Mode:**

```json
{
  "arm_name": "my_arm",
  "body_name": "corner_1",
  "calibration_type": "eye-in-hand",
  "joint_positions": [[0, 0, 0, 0, 0, 0], [0.1, 0.2, 0.3, 0, 0, 0]],
  "method": "CALIB_HAND_EYE_TSAI",
  "pose_tracker": "pose_tracker_opencv",
  "sleep_seconds": 2.0
}
```

**Direct Pose Mode (using arm.move_to_position):**

```json
{
  "arm_name": "my_arm",
  "body_name": "corner_1",
  "calibration_type": "eye-in-hand",
  "poses": [
    {"x": 100, "y": 200, "z": 300, "o_x": 0, "o_y": 0, "o_z": 1, "theta": 0},
    {"x": 150, "y": 200, "z": 350, "o_x": 0, "o_y": 0, "o_z": 1, "theta": 45}
  ],
  "method": "CALIB_HAND_EYE_TSAI",
  "pose_tracker": "pose_tracker_opencv",
  "sleep_seconds": 2.0
}
```

**Motion Planning Mode (with motion service):**

```json
{
  "arm_name": "my_arm",
  "body_name": "corner_1",
  "calibration_type": "eye-in-hand",
  "poses": [
    {"x": 100, "y": 200, "z": 300, "o_x": 0, "o_y": 0, "o_z": 1, "theta": 0},
    {"x": 150, "y": 200, "z": 350, "o_x": 0, "o_y": 0, "o_z": 1, "theta": 45}
  ],
  "method": "CALIB_HAND_EYE_TSAI",
  "pose_tracker": "pose_tracker_opencv",
  "motion": "motion",
  "sleep_seconds": 2.0,
  "use_motion_service_for_poses": true
}
```

### Available Commands

The hand-eye calibration service provides the following commands via `do_command`:

#### `run_calibration`

Runs the hand-eye calibration procedure by moving the arm through all configured positions and computing the camera-to-gripper transformation.

**Example:**
```python
result = await hand_eye_service.do_command({"run_calibration": True})
```

**Returns:** The calibration result in frame system compatible format.

#### `get_current_arm_pose`

Returns the current end-effector pose of the arm. Use this to build up a list of poses for the configuration.

**Example:**
```python
pose = await hand_eye_service.do_command({"get_current_arm_pose": True})
# Returns: {"x": 100, "y": 200, "z": 300, "o_x": 0, "o_y": 0, "o_z": 1, "theta": 0}
```

#### `move_arm_to_position`

Moves the arm to a configured position by index. Automatically uses the appropriate mode based on configuration:
- **Joint Position Mode**: Uses `arm.move_to_joint_positions`
- **Direct Pose Mode**: Uses `arm.move_to_position` (when poses are provided without motion service)
- **Motion Planning Mode**: Uses motion planning (when poses are provided with motion service)

**Parameters:**
- `index`: Index of the position to move to.

**Example:**
```python
result = await hand_eye_service.do_command({"move_arm_to_position": 0})
```

#### `check_bodies`

Checks how many tracked bodies are currently visible to the pose tracker.

**Example:**
```python
result = await hand_eye_service.do_command({"check_bodies": True})
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

See `src/scripts/camera_calibration_script.py` for the Python script to do so.

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

## Utility Scripts

### Touch Test Script

The `touch_test.py` script is a utility for measuring hand-eye calibration accuracy by physically touching tracked targets (such as chessboard corners) with a calibrated touch probe. By comparing the pose reported by the camera system against the arm's physical position reached by the touch probe, you can quantify the error in the derived hand-eye transformation.

To use it you must create a `.env` file in the same directory as the script, and set `VIAM_MACHINE_ADDRESS`, `VIAM_MACHINE_API_KEY_ID` and `VIAM_MACHINE_API_KEY`.

**Location:** `src/scripts/touch_test.py`

**Example usage:**
```bash
python3 src/scripts/touch_test.py \
  --arm-name ur20-modular \
  --pose-tracker-name pose-tracker-opencv \
  --motion-service-name motion \
  --body-names corner_0 corner_1 corner_2 corner_3 corner_4 corner_5 corner_6 corner_7 corner_8 \
  --probe-collision-frame touch-probe \
  --allowed-collision-frames pedestal-ur5e apriltags-obstacle chessboard-obstacle \
  --scanning-pose 100 200 300 0 0 1 0
```

**Required Arguments:**
- `--arm-name`: Name of the arm component
- `--pose-tracker-name`: Name of the pose tracker resource
- `--motion-service-name`: Name of the motion service
- `--body-names`: List of body names to track (e.g., `corner_0 corner_1`)
- `--probe-collision-frame`: Collision frame name for the touch probe
- `--allowed-collision-frames`: List of collision frames that the probe is allowed to collide with

**Optional Arguments:**
- `--touch-probe-length-mm`: Length of the touch probe in mm (default: 113)
- `--pretouch-offset-mm`: Additional offset beyond touch probe length in mm (default: 30)
- `--velocity-normal`: Normal velocity setting for moving between scanning and various pretouch poses (default: 25)
- `--velocity-slow`: Slow velocity setting for touching (default: 10)
- `--world-frame`: Name of the world reference frame (default: "world")
- `--scanning-pose`: Initial scanning pose as 7 values (x y z o_x o_y o_z theta)
