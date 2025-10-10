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

| Name              | Type     | Inclusion  | Description                                             |
|-------------------|----------|------------|---------------------------------------------------------|
| `arm_name`        | `string` | `Required` | Name of the arm component used for calibration.         |
| `calibration_type`| `string` | `Required` | Name of the type of calibration to perform.             |
| `camera_name`     | `string` | `Required` | Name of the camera component used for calibration.      |
| `joint_positions` | `list`   | `Required` | List of joint positions for calibration poses.          |
| `method`          | `string` | `Required` | Method to use for calibration.                          |
| `pose_tracker`    | `string` | `Required` | Name of the pose tracker component to detect markers.   |
| `motion`          | `string` | `Optional` | Name of the motion service for coordinated movement.    |
| `sleep_seconds`   | `float`  | `Optional` | Sleep time between movements (defaults to 1.0 seconds). |

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
  "calibration": "eye-in-hand",
  "camera_name": "cam",
  "joint_positions": [[0, 0, 0, 0, 0, 0], [0.1, 0.2, 0.3, 0, 0, 0]],
  "method": "CALIB_HAND_EYE_TSAI",
  "pose_tracker": "tag_tracker",
  "motion": "motion_service",
  "sleep_seconds": 2.0
}
```
