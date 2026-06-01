# Module opencv

A Viam module that provides OpenCV-based computer vision components for robotics applications. This module includes pose tracking capabilities using chessboard patterns and hand-eye calibration services for robotic arm calibration.

## Model viam:opencv:chessboard

A pose tracker component that detects and tracks the pose of chessboard calibration patterns in camera images. This model uses OpenCV's chessboard detection algorithms to provide accurate 6-DOF pose estimation of chessboard targets, making it ideal for camera calibration and pose tracking applications.

Each inner corner of the chessboard is exposed as a separate tracked body named `corner_0`, `corner_1`, … through `corner_{n-1}` (numbered left-to-right, top-to-bottom in pattern order), with the chessboard's reported pose (used by hand-eye calibration) at `corner_0`.

### Pose Tracker Configuration

The following attribute template can be used to configure this model:

```json
{
  "camera_name": <string>,
  "pattern_size": <list>,
  "square_size_mm": <int>,
  "camera_intrinsics": {
    "K":    {"fx": <float>, "fy": <float>, "cx": <float>, "cy": <float>},
    "dist": {"k1": <float>, "k2": <float>, "k3": <float>, "p1": <float>, "p2": <float>}
  }
}
```

#### Pose Tracker Attributes

The following attributes are available for this model:

| Name                | Type   | Inclusion | Description                                                                                                                                                                                                       |
|---------------------|--------|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `camera_name`       | string | Required  | Name of the camera used for checking pose of chessboard.                                                                                                                                                          |
| `pattern_size`      | list   | Required  | Dimensions of the chessboard pattern (rows x columns of inner corner squares).                                                                                                                                    |
| `square_size_mm`    | int    | Required  | Physical size of a square in the chessboard pattern, in mm.                                                                                                                                                       |
| `camera_intrinsics` | dict   | Optional  | Override the camera's intrinsics rather than fetching them from `camera.do_command({"get_camera_params": None})`. Requires both `K` (with `fx`, `fy`, `cx`, `cy`) and `dist` (with `k1`, `k2`, `k3`, `p1`, `p2`). |

#### Pose Tracker Example Configuration

```json
{
  "camera_name": "cam",
  "pattern_size": [9, 6],
  "square_size_mm": 21
}
```

### Pose Tracker Commands

#### `get_chessboard_observation`

Returns the raw chessboard observation at the current camera frame: detected 2D corner pixels, 3D corner coordinates in the board frame, camera intrinsics, and the board's pose-in-camera estimate (`rvec`, `tvec`) from `cv2.solvePnP`. This is what the hand-eye calibration service consumes when running its reprojection-based solver.

**Example:**

```python
result = await pose_tracker.do_command({"get_chessboard_observation": True})
obs = result["get_chessboard_observation"]
# obs["corners_2d"]:   (N, 2) pixel coordinates
# obs["corners_3d"]:   (N, 3) board-frame points
# obs["K"], obs["dist"]: intrinsics
# obs["rvec"], obs["tvec"]: board pose in camera frame
# obs["ids"]: per-corner stable ids (0..N-1 for a chessboard)
# obs["pattern_size"], obs["square_size_mm"]
```

## Model viam:opencv:charuco

A pose tracker component that detects and tracks the pose of [ChArUco boards](https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html) (a chessboard with ArUco markers in the white squares). Compared to the plain `chessboard` model it tolerates **partial views** (the board need not be fully in frame), is immune to the chessboard's 180° orientation ambiguity, and still measures high-accuracy chessboard saddle corners — making it the recommended target for hand-eye calibration.

Each detected interior corner is exposed as a tracked body named `corner_{id}`, where `id` is the corner's stable index on the board. Because the ids come from the ArUco markers, a given corner keeps the same name across frames even when only part of the board is visible.

### Pose Tracker Configuration

The following attribute template can be used to configure this model:

```json
{
  "camera_name": <string>,
  "squares_x": <int>,
  "squares_y": <int>,
  "square_size_mm": <float>,
  "marker_size_mm": <float>,
  "dictionary": <string>,
  "camera_intrinsics": {
    "K":    {"fx": <float>, "fy": <float>, "cx": <float>, "cy": <float>},
    "dist": {"k1": <float>, "k2": <float>, "k3": <float>, "p1": <float>, "p2": <float>}
  }
}
```

#### Pose Tracker Attributes

The following attributes are available for this model:

| Name                | Type   | Inclusion | Description                                                                                                                                                                                                       |
|---------------------|--------|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `camera_name`       | string | Required  | Name of the camera used for checking pose of the ChArUco board.                                                                                                                                                   |
| `squares_x`         | int    | Required  | Number of chessboard squares along the X axis (columns).                                                                                                                                                          |
| `squares_y`         | int    | Required  | Number of chessboard squares along the Y axis (rows).                                                                                                                                                             |
| `square_size_mm`    | float  | Required  | Physical side length of a chessboard square, in mm.                                                                                                                                                               |
| `marker_size_mm`    | float  | Required  | Physical side length of an ArUco marker, in mm. Must be smaller than `square_size_mm`.                                                                                                                             |
| `dictionary`        | string | Optional  | Predefined ArUco dictionary used by the board (e.g. `DICT_4X4_50`, `DICT_5X5_100`). Defaults to `DICT_4X4_50`. Must match the dictionary your printed/manufactured board uses.                                     |
| `camera_intrinsics` | dict   | Optional  | Override the camera's intrinsics rather than fetching them from `camera.do_command({"get_camera_params": None})`. Requires both `K` (with `fx`, `fy`, `cx`, `cy`) and `dist` (with `k1`, `k2`, `k3`, `p1`, `p2`). |

#### Pose Tracker Example Configuration

```json
{
  "camera_name": "cam",
  "squares_x": 9,
  "squares_y": 13,
  "square_size_mm": 30.0,
  "marker_size_mm": 22.0,
  "dictionary": "DICT_4X4_50"
}
```

### Pose Tracker Commands

#### `get_charuco_observation`

Returns the raw ChArUco observation at the current camera frame, in the same shape as the chessboard model's `get_chessboard_observation` so the hand-eye calibration service consumes either interchangeably. Because the board may be partially visible, the returned corner set is a subset of the full board and varies frame to frame — `ids` identifies which corners were detected.

**Example:**

```python
result = await pose_tracker.do_command({"get_charuco_observation": True})
obs = result["get_charuco_observation"]
# obs["corners_2d"]:   (N, 2) pixel coordinates
# obs["corners_3d"]:   (N, 3) board-frame points
# obs["ids"]:          (N,) stable corner ids for the detected subset
# obs["K"], obs["dist"]: intrinsics
# obs["rvec"], obs["tvec"]: board pose in camera frame
# obs["squares_x"], obs["squares_y"], obs["square_size_mm"], obs["marker_size_mm"], obs["dictionary"]
```

## Model viam:opencv:hand_eye_calibration

A calibration service that performs hand-eye calibration for robotic arms with mounted or fixed cameras. This service automates the process of determining the transformation between the robot's end-effector and camera coordinate frames by moving the arm through positions while tracking bodies. The resulting calibration enables accurate coordination between robot motion and visual perception.

### Operation modes

The service composes two orthogonal choices:

**How poses are produced** (`pose_selection`):

1. **Manual** (default): you supply the pose list yourself via `joint_positions` or `poses`.
2. **Auto**: the service randomly samples poses inside a rectangular workspace volume, aims the end-effector at a `look_at_point`, and rolls about the optical axis. Each candidate is filtered by reachability (the motion plan succeeds) and by chessboard visibility (the pose tracker actually detects the target). See `pose_sampling` below.

**How poses are reached** (only relevant in manual mode):

1. **Joint Position Mode**: direct joint control via `arm.move_to_joint_positions` (when `joint_positions` is configured).
2. **Direct Pose Mode**: `arm.move_to_position` (when `poses` is configured without a motion service). Requires the arm driver to implement `move_to_position`.
3. **Motion Planning Mode**: the motion service plans collision-aware paths to each pose (when `poses` is configured *with* `motion`). Auto mode always uses motion planning and so requires a motion service.

**Which calibration solver runs** (`solver`):

1. **`opencv`** (default): the original behavior — `cv2.calibrateHandEye` with the chosen `method`.
2. **`hybrid`**: bootstraps with `cv2.calibrateHandEye(method=…)`, then refines the result by minimizing per-corner pixel reprojection error with a Levenberg–Marquardt + Huber solver. More accurate on noisy real-world data; reports per-pose pixel RMSE so you can spot outlier observations. Requires a corner-observation pose tracker — either `viam:opencv:chessboard` or `viam:opencv:charuco` (the corner observations come from it).
3. **`reprojection`**: same refinement step as `hybrid` but without the OpenCV bootstrap exposed in the response. In practice, prefer `hybrid`.

### Hand Eye Calibration Configuration

The following attribute template can be used to configure this model:

```json
{
  "arm_name": <string>,
  "body_name": <string>,
  "calibration_type": <string>,
  "method": <string>,
  "solver": <string>,
  "joint_positions": <list>,
  "poses": <list>,
  "pose_selection": <string>,
  "pose_sampling": {
    "workspace_bounds": {
      "x": {"min": <float>, "max": <float>},
      "y": {"min": <float>, "max": <float>},
      "z": {"min": <float>, "max": <float>}
    },
    "look_at_point": [<float>, <float>, <float>],
    "n_poses": <int>,
    "max_attempts": <int>,
    "roll_range_deg": [<float>, <float>],
    "roll_reference": [<float>, <float>, <float>],
    "seed": <int>
  },
  "pose_tracker": <string>,
  "motion": <string>,
  "sleep_seconds": <float>,
  "use_motion_service_for_poses": <bool>
}
```

#### Hand Eye Calibration Attributes

The following attributes are available for this model:

| Name              | Type     | Inclusion   | Description |
|-------------------|----------|-------------|-------------|
| `arm_name`        | `string` | `Required`  | Name of the arm component used for calibration. |
| `calibration_type`| `string` | `Required`  | Type of calibration to perform (see available calibrations below). |
| `method`          | `string` | `Required`  | OpenCV calibration method (see available methods below). In `hybrid`/`reprojection` solver modes this is used to bootstrap the iterative solver. |
| `pose_tracker`    | `string` | `Required`  | Name of the pose tracker component to detect tracked bodies. The `hybrid` and `reprojection` solvers additionally require this be a corner-observation model — `viam:opencv:chessboard` or `viam:opencv:charuco`. |
| `joint_positions` | `list`   | `Required*` | List of joint positions (in radians) for calibration poses. Required when `pose_selection` is `"manual"` and `poses` is not provided. |
| `poses`           | `list`   | `Required*` | List of poses (with `x`, `y`, `z`, `o_x`, `o_y`, `o_z`, `theta`) for calibration. Required when `pose_selection` is `"manual"` and `joint_positions` is not provided. If both are present, `poses` wins. |
| `solver`          | `string` | `Optional`  | Which calibration solver to run: `"opencv"` (default), `"hybrid"`, or `"reprojection"`. See *Operation modes* above. |
| `pose_selection`  | `string` | `Optional`  | `"manual"` (default — use the configured `joint_positions` / `poses`) or `"auto"` (sample poses inside `pose_sampling.workspace_bounds`). |
| `pose_sampling`   | `dict`   | `Required**`| Configuration for the auto pose sampler. Required when `pose_selection="auto"`. See *Auto pose sampling* below. |
| `motion`          | `string` | `Optional`  | Name of the motion service used for motion planning. Required when `pose_selection="auto"`. When used in manual mode with `poses`, enables motion planning; when omitted, manual `poses` use `arm.move_to_position` directly. |
| `sleep_seconds`   | `float`  | `Optional`  | Sleep time between movements to allow the arm to settle (defaults to 2.0 seconds). |
| `use_motion_service_for_poses` | `bool` | `Optional` | Use `motion.get_pose()` (with the arm's origin frame) instead of `arm.get_end_position()` to read the achieved end-effector pose. Defaults to false. Requires `motion` when true. |
| `body_name`       | `string` | `Optional`  | Name of the specific tracked body to use (e.g., AprilTag ID like `"tag36h11:0"` or chessboard corner like `"corner_0"`). Calibration expects exactly one tracked pose; set this when the pose tracker returns multiple. **Important**: when using chessboard corners, ensure the board's orientation is stable across all poses so the same corner is tracked. |

**Note**: in `pose_selection="manual"` mode, either `joint_positions` or `poses` must be provided. In `pose_selection="auto"` mode, both are ignored and `pose_sampling` is required.

Available calibrations are:

- "eye-in-hand"
- "eye-to-hand" *(not yet implemented end-to-end; the new corner-based solvers and auto sampler currently assume eye-in-hand)*

Available methods are:

- "CALIB_HAND_EYE_TSAI"
- "CALIB_HAND_EYE_PARK"
- "CALIB_HAND_EYE_HORAUD"
- "CALIB_HAND_EYE_ANDREFF"
- "CALIB_HAND_EYE_DANIILIDIS"

#### Auto pose sampling

When `pose_selection` is `"auto"`, the service randomly generates and visits poses inside a rectangular workspace volume. Each candidate has:

- a position drawn uniformly from `workspace_bounds`,
- an orientation that aims the end-effector's +Z axis at `look_at_point`,
- a random *roll* about that axis drawn from `roll_range_deg`.

The roll randomization is what gives the resulting pose set good rotation-axis diversity. Pure look-at without roll would cluster rotation axes in a plane and produce ill-conditioned calibrations (see `compute_pose_diversity` below).

After moving to each candidate the service verifies the chessboard is actually detected before counting the pose; unreachable poses, planning failures, and poses where the board is out of view are silently skipped and resampled, up to `max_attempts` total attempts.

**Assumption**: the camera's optical axis is roughly aligned with the end-effector +Z. The chessboard tracker is forgiving as long as corners are in frame, but if your mount is at a wild angle you may need to widen `workspace_bounds` so enough candidates land with the target in view.

| Field                          | Type    | Required | Description |
|--------------------------------|---------|----------|-------------|
| `workspace_bounds.x.min/max`   | float   | yes      | Sampling range for end-effector position along base-frame X, in mm. (Likewise `y`, `z`.) |
| `look_at_point`                | `[x,y,z]` | yes    | Chessboard center in robot base frame, in mm. Easiest way to find: touch the board with the TCP and read `arm.get_end_position()`. |
| `n_poses`                      | int     | no       | Number of successful poses to collect. Defaults to 20. |
| `max_attempts`                 | int     | no       | Cap on total sample attempts (including skips). Defaults to 60. |
| `roll_range_deg`               | `[lo, hi]` | no    | Range for the random roll about the optical axis, in degrees. Defaults to `[-180, 180]`. |
| `roll_reference`               | `[x, y, z]` | no   | World-frame vector that anchors what `roll = 0` means. See *Roll reference* below. Omit for the default per-pose reference. |
| `seed`                         | int     | no       | Optional seed for reproducible sampling. |

##### Roll reference

The `roll_range_deg` restriction is measured relative to a "roll = 0" X-axis that the sampler picks at each pose. There are two ways to pick it:

- **Per-pose (default)** — `roll_reference` omitted. The sampler computes `x_ref = world_up × z / ||…||` at each pose. This reference depends on the look-at direction `z`, so it can flip 180° in world frame when the position crosses to the other side of the look-at point. The numeric roll bound is still respected, but the *world-frame* orientation of the gripper can jump between consecutive poses.
- **Anchored** — `roll_reference` set to a world-frame vector (e.g., `[1, 0, 0]`). The sampler takes that vector, projects it onto the plane perpendicular to `z`, and uses that as `x_ref`. The reference no longer depends on which side of the look-at point you sampled, so a tight `roll_range_deg` actually means a tight bound on wrist twist in world coordinates.

Edge case for anchored mode: when the optical axis becomes (nearly) parallel to `roll_reference`, the perpendicular projection has near-zero length and the orientation is ill-defined. The sampler rejects those positions and resamples. If `roll_reference` is nearly parallel to most of your workspace's look-at directions, the sampler can exhaust `max_attempts` — pick a reference that's mostly perpendicular to the dominant look-at direction.

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

**Reprojection-refined Solver (`hybrid`):**

```json
{
  "arm_name": "my_arm",
  "body_name": "corner_0",
  "calibration_type": "eye-in-hand",
  "poses": [
    {"x": 100, "y": 200, "z": 300, "o_x": 0, "o_y": 0, "o_z": 1, "theta": 0},
    {"x": 150, "y": 200, "z": 350, "o_x": 0, "o_y": 0, "o_z": 1, "theta": 45}
  ],
  "method": "CALIB_HAND_EYE_TSAI",
  "solver": "hybrid",
  "pose_tracker": "pose_tracker_opencv",
  "motion": "motion"
}
```

The response gains a `refinement` block with `rmse_pixels`, `per_pose_rmse_pixels`, and solver diagnostics. Per-pose RMSEs greater than 3× the median are flagged in logs as probable outliers.

**Auto Pose Generation Mode:**

```json
{
  "arm_name": "my_arm",
  "body_name": "corner_0",
  "calibration_type": "eye-in-hand",
  "method": "CALIB_HAND_EYE_TSAI",
  "solver": "hybrid",
  "pose_tracker": "pose_tracker_opencv",
  "motion": "motion",
  "pose_selection": "auto",
  "pose_sampling": {
    "workspace_bounds": {
      "x": {"min": 200, "max": 600},
      "y": {"min": -300, "max": 300},
      "z": {"min": 200, "max": 500}
    },
    "look_at_point": [400, 0, 0],
    "n_poses": 20,
    "max_attempts": 80,
    "roll_range_deg": [-180, 180]
  }
}
```

No `joint_positions` or `poses` are needed in auto mode. The response includes an `auto_sampling` block reporting `requested_n_poses` vs `captured_n_poses`.

### Available Commands

The hand-eye calibration service provides the following commands via `do_command`:

#### `run_calibration`

Runs the hand-eye calibration procedure. In manual mode, moves the arm through all configured positions; in auto mode, samples and visits poses until `n_poses` reachable observations are collected. Then runs the chosen `solver` and computes the camera-to-gripper transformation.

**Example:**

```python
result = await hand_eye_service.do_command({"run_calibration": True})
```

**Returns** a dict with the following keys:

| Key             | Present when      | Contents |
|-----------------|-------------------|----------|
| `frame`         | always            | Frame-system-compatible transform (`translation`, `orientation`, `parent`). |
| `residuals`     | always            | Per-pose translation/rotation residuals against the mean board pose in base frame, plus summary stats. Lets you spot outlier poses without re-running calibration. |
| `solver`        | always            | The solver that ran (`"opencv"`, `"hybrid"`, or `"reprojection"`). |
| `refinement`    | `hybrid`/`reprojection` only | Reprojection solver diagnostics: `rmse_pixels`, `per_pose_rmse_pixels`, `iterations`, `success`, `message`. |
| `auto_sampling` | `pose_selection="auto"` only | `requested_n_poses` and `captured_n_poses`. |

#### `compute_pose_diversity`

Computes diagnostics on a pose set without moving the arm: pairwise Tsai-rule scores, the translation condition number `c_t` (Horn et al. 2023), rotation-axis density on the unit sphere, and an actionable feedback string. Catches ill-conditioned pose sets (e.g., rotation axes clustered along one direction) before running calibration.

By default operates on the service's configured `poses`. You can also pass an explicit pose list to diagnose a hypothetical set:

```python
# Diagnose the configured pose set
result = await hand_eye_service.do_command({"compute_pose_diversity": True})

# Diagnose an arbitrary pose set
result = await hand_eye_service.do_command({
    "compute_pose_diversity": {
        "poses": [
            {"x": 100, "y": 200, "z": 300, "o_x": 0, "o_y": 0, "o_z": 1, "theta": 0},
            ...
        ]
    }
})
```

Returns a dict with `n_poses`, `n_pairs`, `mean_rotation_angle_deg`, `translation_condition_number`, `clustered_axis_direction`, `axis_density_ratio`, `warnings`, and a `feedback` string. `c_t` values close to 1 are well-conditioned; values above ~100 indicate clustered rotation axes.

#### `generate_poses`

Standalone pose sampler — returns a list of poses without moving the arm. Same sampling logic as `pose_selection="auto"` but useful as a preview: paste the result into your `poses` config and run a manual calibration, or just sanity-check the diversity numbers before kicking off an auto run.

**Example:**

```python
result = await hand_eye_service.do_command({
    "generate_poses": {
        "workspace_bounds": {
            "x": {"min": 200, "max": 600},
            "y": {"min": -300, "max": 300},
            "z": {"min": 200, "max": 500}
        },
        "look_at_point": [400, 0, 0],
        "n_poses": 12,
        "roll_range_deg": [-180, 180],
        "seed": 42
    }
})
# result["generate_poses"]["poses"]      -> list of pose dicts (paste-ready)
# result["generate_poses"]["diversity"]  -> compute_pose_diversity report on the generated set
```

#### `get_current_arm_pose`

Returns the current end-effector pose of the arm. Use this to build up a list of poses for the configuration, or to find a `look_at_point` (point the TCP at the chessboard center, then read the pose).

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

## Model viam:opencv:camera_calibration

A generic service that provides camera calibration functionality through the `do_command` interface. This service uses a chessboard or ChArUco target to determine camera intrinsic parameters (focal lengths, principal point, and distortion coefficients) from multiple images. Unlike the pose trackers, this service is dedicated solely to calibration and does not track poses.

Select the target with the `target_type` attribute (defaults to `chessboard`). A ChArUco target is recommended: images in which the board is only partially visible still contribute their detected corners, so more of your captures are usable.

### Camera Calibration Configuration

The following attribute template can be used to configure this model:

```json
{
  "target_type": <string>,
  "pattern_size": <list>,
  "square_size_mm": <float>,
  "squares_x": <int>,
  "squares_y": <int>,
  "marker_size_mm": <float>,
  "dictionary": <string>
}
```

#### Camera Calibration Attributes

The following attributes are available for this model:

| Name             | Type   | Inclusion              | Description                                             |
|------------------|--------|------------------------|---------------------------------------------------------|
| `target_type`    | string | Optional               | `chessboard` (default) or `charuco`. Selects which attributes below are required. |
| `pattern_size`   | list   | Required (chessboard)  | Dimensions of the chessboard pattern (rows x columns of inner corner squares). |
| `square_size_mm` | float  | Required               | Physical side length of a square, in mm (both target types). |
| `squares_x`      | int    | Required (charuco)     | Number of chessboard squares along the X axis (columns). |
| `squares_y`      | int    | Required (charuco)     | Number of chessboard squares along the Y axis (rows). |
| `marker_size_mm` | float  | Required (charuco)     | Physical side length of an ArUco marker, in mm. Must be smaller than `square_size_mm`. |
| `dictionary`     | string | Optional (charuco)     | Predefined ArUco dictionary (e.g. `DICT_4X4_50`). Defaults to `DICT_4X4_50`. Must match your board. |

#### Camera Calibration Example Configurations

Chessboard:

```json
{
  "pattern_size": [9, 6],
  "square_size_mm": 21
}
```

ChArUco:

```json
{
  "target_type": "charuco",
  "squares_x": 9,
  "squares_y": 13,
  "square_size_mm": 30.0,
  "marker_size_mm": 22.0,
  "dictionary": "DICT_4X4_50"
}
```

### Calibrate Camera Command

Use the `calibrate_camera` command via `do_command` to compute camera intrinsics:

See `src/scripts/camera_calibration_script.py` for the Python script to do so.

**Parameters:**
- `images` (required): List of base64 encoded image strings containing the configured target (chessboard or ChArUco)

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
