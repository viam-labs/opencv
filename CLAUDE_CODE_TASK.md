# Status checkpoint — 2026-05-21

Random pose sampler added on top of the 2026-05-07 work. Phase 0 and Phase 1
are still awaiting real-robot validation; this new piece is independent and
also synthetic-test-only so far.

## What's new

### Anchored roll reference (follow-up to the sampler)

The original sampler picked `x_ref = world_up × z` per-pose, which flips 180°
when the position crosses to the other side of the look-at point — so a tight
`roll_range_deg` was respected numerically but did NOT bound the world-frame
wrist twist between poses. Added an optional `roll_reference` (length-3 vector
in `pose_sampling`). When set, the sampler projects that vector onto the
plane perpendicular to the optical axis and uses it as the "roll = 0" X axis.
That projection is consistent across positions, so a tight roll range now
bounds actual world-frame twist.

Edge case: when the reference becomes (nearly) parallel to a sampled
position's optical axis (`|ref · z| > 0.99`), the projection is ill-defined.
`look_at_rotation` raises `ValueError`; `sample_transform` catches it and
resamples (still capped at `max_retries`, default 32 inside the sampler;
the auto loop separately caps at `max_attempts`).

6 new tests covering: anchored consistency across the look-at point, per-pose
flip (documents the behavior the anchored mode fixes), parallel-rejection,
zero-vector rejection, retry-then-succeed in a mostly-parallel workspace,
and roll-range bound being respected against the anchored reference.

### Automatic pose generation (rectangular workspace + look-at + random roll)

The user was tired of hand-authoring pose lists. New sampler generates poses
inside a user-specified rectangular prism with orientation aimed at a
configured look-at point and random roll about the optical axis. The roll
randomization is the load-bearing part — pure look-at without roll clusters
rotation axes in a plane and produces a higher `c_t` (verified in
`test_random_roll_improves_conditioning_vs_zero_roll`).

**Mount assumption**: the sampler aligns end-effector +Z with the look-at
direction. This works cleanly when the camera optical axis is roughly
aligned with gripper +Z (common). If the camera is mounted at a wild angle
relative to the gripper, the look-at aiming will be off — but the chessboard
tracker is forgiving as long as corners are in frame.

**Files added**
- `src/active_calibration/__init__.py`
- `src/active_calibration/pose_sampler.py` — pure numpy. `look_at_rotation`,
  `sample_transform`, `generate_pose_set`. No Viam or Go-binary deps so it's
  unit-testable in isolation.
- `tests/active_calibration/test_pose_sampler.py` — 13 tests.

**Files modified**
- `src/models/hand_eye_calibration.py`:
  - New config attrs `pose_selection` (`"manual"` default, or `"auto"`) and
    `pose_sampling` block (workspace_bounds, look_at_point, n_poses,
    max_attempts, roll_range_deg, seed).
  - `validate_config` relaxes the "must have joint_positions or poses"
    requirement when `pose_selection='auto'`; instead requires
    `pose_sampling` and a motion service.
  - New `_validate_sampling_attrs` and `_parse_sampling_attrs` helpers at
    module scope, shared by `validate_config` and the `generate_poses`
    do_command.
  - New `_transform_to_viam_pose` helper that takes a 4x4 transform from the
    sampler and produces a Viam `Pose` via `call_go_mat2ov(R.T)` (the
    transpose because Viam OV's underlying matrix is body-from-parent,
    while the sampler returns columns-are-gripper-axes-in-base).
  - New `_sample_and_move_loop` — sample → `motion.move()` → verify the
    chessboard is visible via `pose_tracker.get_poses` → record the
    **measured** end-effector pose (`get_end_position` or
    `motion.get_pose`), not the sampled candidate. On any failure (unreachable,
    chessboard not visible, multiple bodies and no `body_name` to
    disambiguate), skip and resample. Caps at `max_attempts` total sample
    attempts. Requires the motion service.
  - `run_calibration` runs the sample-and-move loop first when
    `pose_selection='auto'`, then feeds the resulting poses into the
    existing solver paths. Response now includes an `auto_sampling` block
    with `requested_n_poses` and `captured_n_poses`.
  - New `do_command({"generate_poses": {...}})` — standalone sampler that
    returns the pose list plus a `compute_pose_diversity` report. Does
    **not** move the arm; useful for previewing what auto mode would
    generate.

## How to try it

### Standalone preview (no arm movement)
```python
result = await hand_eye_service.do_command({
    "generate_poses": {
        "workspace_bounds": {
            "x": {"min": 200, "max": 600},
            "y": {"min": -300, "max": 300},
            "z": {"min": 200, "max": 500},
        },
        "look_at_point": [400, 0, 0],   # chessboard center in robot base frame, mm
        "n_poses": 12,
        "roll_range_deg": [-180, 180],  # optional, default ±180
        "seed": 42,                       # optional, for reproducibility
    }
})
# result["generate_poses"]["poses"] is a list ready to paste into the `poses` config
# result["generate_poses"]["diversity"] is the Phase 1 diagnostic report
```

### Auto mode wired into run_calibration
Config:
```json
{
  "pose_selection": "auto",
  "pose_sampling": {
    "workspace_bounds": {
      "x": {"min": 200, "max": 600},
      "y": {"min": -300, "max": 300},
      "z": {"min": 200, "max": 500}
    },
    "look_at_point": [400, 0, 0],
    "n_poses": 12,
    "max_attempts": 60,
    "roll_range_deg": [-180, 180]
  },
  "motion": "motion",
  ...
}
```
Then:
```python
result = await hand_eye_service.do_command({"run_calibration": True})
# result["run_calibration"]["auto_sampling"] reports requested vs captured
```

`pose_selection: "manual"` (or omitting it) is the unchanged behavior.
`joint_positions` / `poses` config is still required in manual mode.

## How to find a look-at point

The user has to provide `look_at_point` in robot base frame. Easy ways:
- Touch the chessboard center with the TCP and read the arm's reported
  position.
- Place the board at a known offset from the arm base and measure with a
  ruler.

We could auto-discover it by triangulating a chessboard observation against
a rough initial X estimate, but that's a follow-up.

## What's NOT done (still)

- Real-robot validation for Phase 0 / Phase 1 / the new sampler.
- Phase 2 / Phase 3 (heuristic greedy and full NBV from the original spec).
  The auto sampler is intentionally simpler — random sampling with the roll
  trick should give enough diversity that NBV's added complexity isn't worth
  it for most setups.
- Eye-to-hand mode (still eye-in-hand only).
- A `discover_look_at_point` do_command.

---

# Status checkpoint — 2026-05-07

This task was started in a Claude Code session on 2026-05-07. Phase 1 and Phase 0
have landed. Phases 2 and 3 are not started. Pausing here for the user to
validate Phase 0 and Phase 1 on the real robot before continuing.

## What's done

### Phase 1 — Pose-set diagnostics (shipped first; spec ordering reversed deliberately)

**Files added**
- `src/diagnostics/__init__.py`
- `src/diagnostics/pose_diversity.py` — `compute_pose_diversity(transforms)`. Pure
  numpy, no scipy. Computes Tsai-rule stats, translation condition number `c_t`
  (via `H = Σ 2(1−cos θ)·(I − n n^T)`, no Jacobian needed), rotation-axis density
  on the unit sphere, warnings, and an actionable feedback string.
- `tests/diagnostics/test_pose_diversity.py` — 13 synthetic tests.
- `conftest.py` at repo root so pytest can find `src/`.

**File modified**
- `src/models/hand_eye_calibration.py` — added `compute_pose_diversity` case to
  `do_command`. Accepts `{"compute_pose_diversity": true}` (uses `self.poses`)
  or `{"compute_pose_diversity": {"poses": [...]}}` (uses provided poses).
  Logs feedback and warnings.

**Spec deviation**: the original spec in this file says `c_t`'s smallest-eigenvalue
eigenvector is the *missing* axis the user should rotate about. The math actually
points to the over-represented (clustered) direction — translation can't be
estimated along that direction precisely because rotations don't span away from
it. The implemented field is named `clustered_axis_direction` and the feedback
string says "axes are clustered along [v]; add motions whose rotation axis is
perpendicular to this." Opposite of the spec's literal wording but mathematically
correct.

### Phase 0 — Reprojection-error solver

**Files added**
- `src/solvers/__init__.py`
- `src/solvers/reprojection_solver.py` — `refine_handeye(T_be_list,
  corners_2d_list, corners_3d, K, dist, X_init, ...)` minimizes per-corner pixel
  reprojection error via `scipy.optimize.least_squares` (method='trf',
  loss='huber', f_scale=1.0). Returns dict with `X_refined`, `Y_refined`,
  `rmse_pixels`, `per_pose_rmse_pixels`, `residuals`, `jacobian`, `success`,
  `n_iterations`, `message`. Eye-in-hand only.
- `tests/solvers/test_reprojection_solver.py` — 7 synthetic ground-truth tests:
  noise-free recovery (machine precision), 0.5px-noise tolerance, 1px-noise
  refinement-beats-bootstrap, Jacobian shape, per-pose RMSE shape, n<3 raises,
  T_bw bootstrap correctness.

**Files modified**
- `src/models/chessboard.py` — factored corner detection into
  `_detect_chessboard_observation()`. New `do_command`
  `get_chessboard_observation` returns `corners_2d`, `corners_3d`, `K`, `dist`,
  `rvec`, `tvec`. `get_poses` was refactored to reuse the helper.
- `src/models/hand_eye_calibration.py` — new `solver` config attribute:
  `"opencv"` (default, unchanged behavior), `"hybrid"`, `"reprojection"`.
  New helper `_collect_calibration_data_with_corners()` collects T_be, T_cw,
  corners in OpenCV-standard convention. `run_calibration` branches: `opencv`
  mode is exactly the old code path; `hybrid`/`reprojection` bootstraps with
  `cv2.calibrateHandEye` and refines via `refine_handeye`, returns the refined
  frame plus a `refinement` block with rmse_pixels, per_pose_rmse_pixels, etc.
- `requirements.txt` — `scipy>=1.10` added.

**Convention discovered (important for any future work)**: the existing service
applies `.T` to `call_go_ov2mat(...)` outputs before feeding `cv2.calibrateHandEye`.
I empirically tested this on synthetic data: `call_go_ov2mat` returns the
**body-from-parent** rotation (`R_eb` for an arm pose), and the `.T` puts it
in OpenCV-standard parent-from-body (`R_be`). The new `_collect_calibration_data_with_corners`
path uses standard convention throughout. The legacy `_collect_calibration_data`
path is untouched and still uses the existing convention.

## How to run tests

Pytest's config picks up a `pyproject.toml` from `~`, so the invocation has to
override that:

```bash
PYTHONPATH=src ./venv/bin/python -m pytest tests/ -v -c /dev/null --rootdir=.
```

Should pass 20 tests (13 diagnostics + 7 solver) in under a second.

## How to try the new features

### Pose-set diagnostics
Existing config — no changes required. Just call `do_command`:

```python
result = await hand_eye_service.do_command({"compute_pose_diversity": True})
# returns dict with n_poses, n_pairs, mean_rotation_angle_deg,
# translation_condition_number, clustered_axis_direction, axis_density_ratio,
# warnings, feedback, etc.
```

Or pass an explicit pose list to diagnose without modifying config:

```python
result = await hand_eye_service.do_command({
    "compute_pose_diversity": {"poses": [{"x":..., "o_x":..., ...}, ...]}
})
```

### Reprojection-based solver
Add `"solver": "hybrid"` to the hand-eye-calibration config. The pose tracker
must be the `viam:opencv:chessboard` model with intrinsics either configured on
the chessboard or fetchable from the camera. Then run as before:

```python
result = await hand_eye_service.do_command({"run_calibration": True})
# returned dict now includes "solver" and "refinement" sub-blocks.
```

`solver: "opencv"` (or omitting the field) is the original behavior, untouched.

## What's NOT done

- **Phase 2 (heuristic active selection)** — not started.
- **Phase 3 (NBV with information gain)** — not started.
- **Eye-to-hand mode** — same TODO that was in the repo before. The new solver
  path also assumes eye-in-hand.
- **Real-robot validation against `touch_test.py`** — synthetic tests prove the
  math is right and refinement beats bootstrap under simulated pixel noise. The
  accuracy claim on actual sensor data is unverified.
- **`camera_intrinsics_source` config attribute** — not added. Intrinsics flow
  through the chessboard tracker exactly the way they already did. Adding a
  separate "service-name lookup" for intrinsics on the hand-eye service is
  marginal value since the tracker already exposes them.

## Suggested validation before continuing

1. Configure a robot with the existing `solver: "opencv"` setting — confirm
   nothing regressed (the existing code path was not modified).
2. Run `compute_pose_diversity` on the user's existing pose lists — verify the
   feedback strings make sense for known-good and known-bad sets.
3. Set `solver: "hybrid"` and run a calibration — compare the resulting
   `frame` and the new `refinement.rmse_pixels` to a calibration done with
   `opencv` mode on the same poses.
4. Run `touch_test.py` with the hybrid-mode result vs the opencv-mode result.
   The hybrid result should match or beat opencv on physical-error numbers.

If hybrid mode actually improves accuracy on the real robot, that's the green
light to continue with Phases 2 and 3. If it doesn't (or makes things worse),
that's a signal to debug the solver wiring before adding more on top.

## Quick orientation for a future session

- Original task spec is below this checkpoint, unchanged.
- All new code is gated behind opt-in config — `solver: "opencv"` and
  manual pose lists give exactly the pre-existing behavior.
- `git status` should show: modified `requirements.txt`, modified
  `src/models/chessboard.py`, modified `src/models/hand_eye_calibration.py`,
  added `conftest.py`, added `src/diagnostics/`, added `src/solvers/`,
  added `tests/diagnostics/`, added `tests/solvers/`, modified this file.

---

# Task: Add active hand-eye calibration with optimal pose selection and convergence detection

## Context

This repo (`viam-labs/opencv`) currently implements hand-eye calibration as a Viam module that wraps `cv2.calibrateHandEye`. The user provides a fixed list of joint positions or Cartesian poses, the arm visits them, the chessboard pose-tracker reports the target's pose at each, and OpenCV's analytical solver (TSAI, PARK, HORAUD, ANDREFF, or DANIILIDIS) returns the camera→gripper transform.

The problem: the user has to guess how many poses are enough and which poses to pick. Bad pose distributions (e.g., rotation axes clustered in one direction) silently produce inaccurate calibrations. Adding more poses doesn't fix it and can make it worse.

This task adds three capabilities, in order:

1. A reprojection-error-minimization solver that produces a Jacobian as a byproduct (needed by phases 2 and 3, and more accurate than the analytical solvers on its own).
2. Pose-set diagnostics that flag ill-conditioned pose distributions and tell the user which rotation axis direction is missing.
3. Active calibration that picks each next pose to maximize information gain and stops automatically when the calibration has converged.

The existing `cv2.calibrateHandEye` path stays as-is for backward compatibility. Everything new is additive and gated behind config options.

## Reference papers

The two papers driving this design — read these before writing code:

- **Yang, Rebello, Waslander, "Next-Best-View Selection for Robot Eye-in-Hand Calibration"** — arXiv:2303.06766 (2023). Source for the FIM-based information-gain formulation and the active calibration loop.
- **Horn, Wodtko, Buchholz, Dietmayer, "User Feedback and Sample Weighting for Ill-Conditioned Hand-Eye Calibration"** — arXiv:2308.06045 (2023). Source for the translation condition number `c_t` and the rotation-axis density analysis.

Background:

- Tsai & Lenz 1989 ("A New Technique for Fully Autonomous and Efficient 3D Robotics Hand/Eye Calibration") — original "Golden Rules" guidance on inter-station angles.
- Tabb & Yousef 2017 ("Solving the Robot-World Hand-Eye(s) Calibration Problem with Iterative Methods") — reference implementation pattern for reprojection-error refinement.

---

## Phase 0: Reprojection-error solver

### Why
- OpenCV's analytical solvers minimize an algebraic error on relative pose pairs. Reprojection-error minimization minimizes the actual pixel error in image space, which is statistically the right thing and is consistently more accurate on noisy real-world data.
- More importantly: it produces a Jacobian, which phases 2 and 3 need. Constructing a Jacobian by finite-differencing OpenCV's output is awkward and noisy.
- The standard recipe: bootstrap from `cv2.calibrateHandEye` (TSAI), then refine with `scipy.optimize.least_squares`.

### What to build

Create `src/solvers/reprojection_solver.py`.

**Inputs to the solver:**
- A list of measurement sets, each containing:
  - `T_eb` (4×4): end-effector pose in robot base frame, from forward kinematics.
  - `corners_2d` (N×2): pixel coordinates of detected chessboard corners.
  - `corners_3d` (N×3): corresponding 3D positions in world/board frame (a fixed grid based on `pattern_size` and `square_size_mm`).
- Camera intrinsics `K` (3×3) and distortion coefficients (5,) — already produced by the existing `viam:opencv:camera-calibration` service. Wire those in; don't re-implement.
- An initial estimate for `X = T_ce` (camera in end-effector frame). Get this by calling the existing `cv2.calibrateHandEye(method=TSAI)` on the same poses.
- For eye-to-hand mode, also estimate `Y = T_bw` (world in robot base frame). For eye-in-hand, `Y` is fixed by the chessboard's pose in the world.

**Cost function:**

For each measurement set k and each corner j:
```
residual_kj = u_kj − project(K, dist, X · T_eb,k · Y · P_w_j)
```
where `project()` is `cv2.projectPoints` or equivalent.

Stack all residuals into one vector and minimize the sum of squared norms.

**Parameterization:**

Parameterize `X` (and `Y` if applicable) as 6-vectors in se(3): 3 components for translation, 3 for axis-angle rotation. Use `scipy.spatial.transform.Rotation` for the conversions:
```python
from scipy.spatial.transform import Rotation as R
# axis-angle (rotvec) → matrix
R_mat = R.from_rotvec(xi[:3]).as_matrix()
# matrix → axis-angle
xi_rot = R.from_matrix(R_mat).as_rotvec()
```
Don't optimize rotation matrices directly — 9 parameters with 6 orthogonality constraints behaves badly numerically.

**Solver call:**

```python
result = scipy.optimize.least_squares(
    fun=residual_fn,
    x0=initial_params,
    method='lm',           # Levenberg-Marquardt
    loss='huber',          # robust against bad corner detections
    f_scale=1.0,           # huber threshold in pixels
    max_nfev=200,
)
```

Return:
- `X_refined` (4×4 SE(3) matrix)
- `Y_refined` if applicable
- `residuals` (per-corner pixel errors, useful for diagnostics)
- `jacobian` (`result.jac`, shape `[2*N_corners*N_poses, 6]` for eye-in-hand, `[..., 12]` for eye-to-hand) — keep this around, it's needed in later phases
- `rmse_pixels` (sqrt of mean squared residual)
- `per_pose_rmse` (list of per-pose pixel RMSEs — flag any pose whose RMSE is >3× the median; that pose is probably bad)

### Wiring into the existing service

Add a new config attribute `solver` to the `handeyecalibration` model:
- `"opencv"` (default, existing behavior — keep this exactly as it is)
- `"reprojection"` (new — pure reprojection-error minimization, requires intrinsics)
- `"hybrid"` (recommended — runs opencv first as bootstrap, then refines with reprojection)

For `reprojection` and `hybrid`, the service needs camera intrinsics. Add a config attribute `camera_intrinsics_source`:
- A direct dict `{"fx": ..., "fy": ..., "cx": ..., "cy": ..., "dist": [...]}`, or
- The name of a `viam:opencv:camera-calibration` service whose calibration result should be queried.

The pose tracker also needs to expose the raw 2D corner pixel coordinates, not just the chessboard pose. Add a method to the pose tracker that returns them — call it `get_corner_pixels()` — and have the calibration service request these alongside the existing pose query at each station.

### Validation

Use the existing `src/scripts/touch_test.py` as ground truth. The hybrid solver should produce equal or lower physical-error numbers than `opencv` mode on the same set of poses. Add a test that runs both and reports the comparison.

---

## Phase 1: Pose-set diagnostics

### Why

Even before changing how poses are picked, give the user a quality report on whatever pose set they've got. This catches bad calibrations *before* they're trusted, and tells the user what to fix.

### What to build

Create `src/diagnostics/pose_diversity.py`.

**Inputs:**
- A list of arm poses `[T_eb,1, …, T_eb,n]`.
- Optionally, the corresponding camera-frame target poses `[T_tc,1, …, T_tc,n]` if you want to check the target side too.

**Compute and return:**

1. **Rotation axes of inter-pose motions.**
   For each consecutive pair `(i, i+1)` (and optionally all C(n,2) pairs), compute the relative rotation `R_ij = R_j · R_i^T`, extract its axis-angle representation `(n_ij, θ_ij)`. Return the list of `(axis, angle)` pairs.

2. **Tsai-rule scores.** For each pair, report:
   - Rotation angle `θ_ij` in degrees (Golden Rule 2: maximize this; flag if mean is below ~30°).
   - Translation magnitude `||t_ij||` in mm (Golden Rule 3: minimize relative to scene; just report it).
   - Pairwise angles between rotation axes across all pairs (Golden Rule 1: maximize spread). Report mean, min, max.

3. **Translation condition number `c_t`** — the key metric from Horn et al.
   This tells you whether rotation axes are clustered along a single direction.

   Computation (simplified version of Horn et al. §IV-A):
   - Build the cost matrix `Q` from the dual-quaternion formulation, OR (simpler if you've gone the reprojection-error route in Phase 0) use the Jacobian `J` from the reprojection solver and form `H = J^T J`.
   - Take the 3×3 sub-block corresponding to the translation parameters of `X`.
   - Compute its eigenvalues `λ_1 ≤ λ_2 ≤ λ_3`.
   - `c_t = λ_3 / λ_1` (with `c_t ≥ 1`; values close to 1 are well-conditioned, values >>1 are ill-conditioned).
   - Also return the eigenvector `v_t,1` corresponding to the smallest eigenvalue — this is the direction of the missing rotation axis.

4. **Rotation-axis density on the unit sphere** (Horn et al. §IV-B).
   For each rotation axis `n_i`, compute its local density:
   ```
   ρ_i = Σ_j exp(-d(n_i, n_j)² / (2 · σ²))
   ```
   where `d(n_i, n_j) = π/2 − |arccos(n_i · n_j) − π/2|` (treats `n` and `-n` as the same axis), and `σ` is a kernel bandwidth in radians (default ~0.3 rad ≈ 17°).
   High density = clustered axes = bad. Return per-axis densities and a summary (mean, max, ratio of max to mean).

5. **Actionable feedback string.** Based on the metrics:
   - If `c_t > 100`: "Rotation axes are clustered. Add motions that rotate about [direction printed in robot-base coordinates from `v_t,1`]."
   - If mean rotation angle < 20°: "Inter-pose rotations are too small. Increase the angular variation between poses."
   - If max-to-mean density ratio > 3: "Some rotation axes are over-represented. Add poses with rotations about other axes."
   - Otherwise: "Pose set looks well-conditioned."

### Wiring

Expose as a new `do_command` on the existing `handeyecalibration` service:

```python
result = await hand_eye_service.do_command({"compute_pose_diversity": True})
# returns:
# {
#   "n_poses": 8,
#   "n_pairs": 7,
#   "mean_rotation_angle_deg": 24.3,
#   "min_rotation_angle_deg": 4.1,
#   "translation_condition_number": 142.7,
#   "missing_axis_direction": [0.1, -0.07, 0.99],  # unit vector in base frame
#   "max_axis_density": 18.2,
#   "mean_axis_density": 5.1,
#   "feedback": "Rotation axes are clustered. Add motions that rotate about [0.10, -0.07, 0.99] (mostly the Z axis).",
#   "warnings": ["mean_rotation_angle below 30°"]
# }
```

The user can call this on their existing pose list before running the calibration. This alone is high-value — it catches bad setups in seconds.

### Tests

Build synthetic test cases:
- All rotations about the same axis → `c_t` should be huge (>1e3).
- Rotations spread evenly over the sphere → `c_t` should be near 1.
- Half the poses about Z, half about X → `c_t` moderate, density ratio moderate.

---

## Phase 2: Heuristic pose selection (Tsai's Golden Rules)

### Why

This is the simplest active-calibration strategy: from a candidate pool of poses, greedily pick the one that best satisfies Tsai's rules relative to what's already been picked. No optimization required, runs in milliseconds, gives ~80% of the benefit of full NBV.

### What to build

Create `src/active_calibration/heuristic_planner.py`.

**Inputs:**
- A pool of candidate end-effector poses. The user supplies these — either as an explicit list, or as a workspace volume + sampling density (sample a grid of orientations within the volume; reject any where the chessboard would leave the camera's FOV based on the current X estimate).
- The poses already collected.
- The current X estimate (used for FOV checking; can be the bootstrap from the first ~3 poses).

**Algorithm:**

```
already_visited = [first 3 bootstrap poses with non-parallel rotation axes]
while not converged:
    best_score = -inf
    best_pose = None
    for candidate in candidate_pool:
        if candidate in already_visited: continue
        if not chessboard_in_fov(candidate, X_estimate): continue
        score = tsai_score(candidate, already_visited)
        if score > best_score:
            best_score = score
            best_pose = candidate
    move to best_pose, capture, update X_estimate
    check convergence (see Phase 3)
```

**Tsai score:**

```python
def tsai_score(candidate, visited):
    # Compute relative motion from each visited pose to candidate
    axes = []
    angles = []
    translations = []
    for v in visited:
        T_rel = candidate @ inv(v)
        R_rel, t_rel = T_rel[:3,:3], T_rel[:3,3]
        rotvec = Rotation.from_matrix(R_rel).as_rotvec()
        angle = norm(rotvec)
        axis = rotvec / (angle + 1e-9)
        axes.append(axis)
        angles.append(angle)
        translations.append(norm(t_rel))

    # Score components (all higher = better)
    # Rule 1: angles between this candidate's axes and all existing axes
    existing_axes = [axis between consecutive visited poses]
    angular_spread = mean over candidate axes of (
        min angle between candidate_axis and any existing_axis
    )
    # Rule 2: rotation magnitude
    rotation_magnitude = mean(angles)
    # Rule 3: translation small (negative weight)
    translation_penalty = mean(translations)

    return w1 * angular_spread + w2 * rotation_magnitude - w3 * translation_penalty
```

Default weights `w1=2.0, w2=1.0, w3=0.001` (translation in mm, angles in radians). Expose as config but the defaults should work for most setups.

### Bootstrap

The first 3 poses can't be selected by score (no history). Pick them deterministically:
- Pose 1: any candidate near the workspace center.
- Pose 2: candidate with largest rotation about the X axis relative to pose 1.
- Pose 3: candidate with largest rotation about the Y axis relative to pose 1.

This guarantees non-parallel rotation axes, satisfying the minimum requirement for a unique solution.

### Wiring

New config option `pose_selection: "manual" | "heuristic" | "nbv"` (default `"manual"` to preserve current behavior).

If `"heuristic"`, also require a `candidate_pool` config attribute — either a list of poses, or a `{"workspace_bounds": [...], "n_samples": int}` spec for grid sampling.

New `do_command`:
```python
await hand_eye_service.do_command({"start_active_calibration": True})
# Runs the loop, blocks until converged, returns the calibration
```

---

## Phase 3: Next-Best-View (NBV) with information gain

### Why

Replaces the heuristic score with a principled information-theoretic score. Picks the pose that maximally reduces uncertainty in the calibration parameters. Per the Toronto paper, achieves the same accuracy as random sampling with ~half the poses.

### What to build

Create `src/active_calibration/nbv_planner.py`.

This requires the reprojection-error solver from Phase 0 — that's where the Jacobian comes from.

**Algorithm (Toronto paper §IV):**

```
1. Bootstrap with K=3 poses (use heuristic bootstrap from Phase 2).
2. Run reprojection-error solver to get X*, J (current Jacobian).
3. Compute current covariance: Σ = (J^T J)^(-1).
4. Compute current entropy: h(Σ) = 0.5 * log((2πe)^n * det(Σ)).
   (Use trace(Σ) as a robust alternative to det(Σ) — it avoids numerical issues
   with near-singular matrices and gives equivalent ranking for NBV purposes.
   The Toronto paper notes this substitution explicitly.)
5. For each candidate pose θ in the pool:
   a. Predict the Jacobian row J_θ that *would* result from adding this pose,
      using the current X* and the candidate's geometry. Don't move the robot
      yet — this is a prediction step.
   b. Form the augmented Jacobian: J' = [J; J_θ]
   c. Predicted new covariance: Σ' = (J'^T J')^(-1)
   d. Information gain: I(θ) = h(Σ) - h(Σ')
6. Pick θ* = argmax I(θ).
7. Move arm to θ*, capture measurement, append to the pose set.
8. Re-run reprojection solver with the augmented set to get updated X*, J.
9. Check convergence (next section). If not converged, go to 5.
```

### Predicting the Jacobian row without moving the robot

This is the technically subtle part. For a candidate pose `T_eb,candidate`:
- Use the current `X*` to predict where the chessboard corners would project on the image: `P_predicted = π(K, X* · T_eb,candidate · Y · P_w)`.
- Compute the Jacobian of those predicted projections w.r.t. the calibration parameters, exactly the same way as in the optimizer (chain rule through the projection function).
- This gives you `J_θ` without ever moving the arm. The Toronto paper §IV-B figure 3 illustrates this.

**Important constraint:** only score candidates where the predicted projections fall within the image bounds. Out-of-FOV candidates aren't useful and break the prediction.

### Convergence detection

Implement three signals; declare converged when **any** fires (after the minimum-pose floor is met):

**Signal 1 — Information gain plateau.**
```
if max(I(θ) for θ in candidates) < info_gain_threshold:
    converged = True
```
Default threshold: `0.01` (in nats — small entropy reduction). Expose as config.

**Signal 2 — Parameter stability.**
```
if len(X_history) >= window:
    last_window = X_history[-window:]
    translation_changes = [norm(X_i.t - X_{i-1}.t) for ...]
    rotation_changes = [angle(X_i.R @ X_{i-1}.R.T) for ...]
    if max(translation_changes) < t_threshold and max(rotation_changes) < r_threshold:
        converged = True
```
Defaults: window=3, t_threshold=0.5mm, r_threshold=0.1°. Expose as config.

**Signal 3 — Held-out reprojection RMSE.**
Reserve ~20% of poses (or ask user to provide a separate validation set) and compute RMSE on them after each new training pose:
```
if rmse_history[-1] > rmse_history[-2] or
   abs(rmse_history[-1] - rmse_history[-2]) / rmse_history[-2] < 0.01:
    converged = True  # plateau or starting to overfit
```

**Hard guardrails (always applied):**
- `min_poses` (default 5): never declare converged below this.
- `max_poses` (default 30): always stop here.
- `condition_number_max` (default 500): if `c_t` from Phase 1 exceeds this after convergence, return an error rather than the calibration — pose set is degenerate.

### Wiring

If `pose_selection: "nbv"`:
- Use the same `candidate_pool` config as Phase 2.
- Expose all the convergence thresholds as config:
  ```json
  {
    "convergence": {
      "info_gain_threshold": 0.01,
      "param_stability": {
        "translation_mm": 0.5,
        "rotation_deg": 0.1,
        "window": 3
      },
      "min_poses": 5,
      "max_poses": 30,
      "condition_number_max": 500
    }
  }
  ```

Add a `get_calibration_status` do_command that returns the live state during the active calibration loop:
```python
{
  "poses_captured": 7,
  "current_estimate": {...X...},
  "last_info_gain": 0.034,
  "condition_number": 18.2,
  "converged": false,
  "convergence_signals": {
    "info_gain_below_threshold": false,
    "params_stable": false,
    "rmse_plateau": false
  },
  "estimated_remaining_poses": 3   # rough heuristic from info gain trend
}
```

---

## Configuration summary

After all phases land, the full config might look like:

```json
{
  "arm_name": "my_arm",
  "pose_tracker": "pose_tracker_opencv",
  "calibration_type": "eye-in-hand",
  "method": "CALIB_HAND_EYE_TSAI",
  "solver": "hybrid",
  "camera_intrinsics_source": "camera-calibration",
  "pose_selection": "nbv",
  "candidate_pool": {
    "workspace_bounds": {"x": [200, 600], "y": [-300, 300], "z": [200, 500]},
    "n_orientation_samples_per_position": 8,
    "n_position_samples": 50
  },
  "convergence": {
    "info_gain_threshold": 0.01,
    "param_stability": {"translation_mm": 0.5, "rotation_deg": 0.1, "window": 3},
    "min_poses": 5,
    "max_poses": 30,
    "condition_number_max": 500
  },
  "motion": "motion",
  "sleep_seconds": 2.0
}
```

Backward compatibility: if `pose_selection` is absent or `"manual"` and `solver` is absent or `"opencv"`, behavior is exactly as today.

---

## Implementation order

Strict order — each phase builds on the previous:

1. **Phase 0** (reprojection solver) — must come first. It's the foundation, and it's also the only phase that delivers value entirely on its own (better calibration accuracy with no UX changes).
2. **Phase 1** (diagnostics) — independent of pose selection, can be shipped between any two phases. Recommend shipping right after Phase 0 because it's small and high-impact.
3. **Phase 2** (heuristic) — requires Phase 0 only for the reprojection refinement of each iteration's calibration update. Most users will be happy stopping here.
4. **Phase 3** (NBV) — requires Phase 0's Jacobian. The headline feature.

Each phase should be its own PR with its own tests. Don't merge them as one giant change.

---

## Things NOT to change

- The existing `joint_positions`/`poses`-based calibration flow. It stays as-is for backward compatibility. The new behavior is opt-in via `solver` and `pose_selection`.
- The five OpenCV solver method options (TSAI/PARK/HORAUD/ANDREFF/DANIILIDIS). Keep them all; in `hybrid` mode the chosen `method` is what bootstraps the reprojection refinement.
- The pose-tracker's existing API. Just add `get_corner_pixels()` to it; don't modify what's there.
- The `viam:opencv:camera-calibration` service. Phase 0 reads from it but doesn't change it.

---

## Dependencies to add

- `scipy>=1.10` (for `scipy.optimize.least_squares` and `scipy.spatial.transform.Rotation`).
- That's it. No `cvxpy`, no `pylie`, no `MOSEK` — the SDP-based "certifiably optimal" solvers from the recent literature are overkill here and not worth the dep cost.

Update `requirements.txt` accordingly.

---

## Validation plan

For each phase, before declaring done:

- Phase 0: hybrid solver matches or beats `opencv` mode on `touch_test.py` physical-accuracy numbers using the same pose set. Add a synthetic test with known ground-truth X, perturb measurements with Gaussian noise, verify reprojection solver recovers X within tolerance.
- Phase 1: synthetic pose sets (single-axis, well-distributed, mixed) produce expected condition numbers and feedback strings.
- Phase 2: heuristic + manual reach equivalent calibration accuracy on the same number of poses, but heuristic should win when the user's manual pose set is poorly distributed.
- Phase 3: NBV reaches the same accuracy as heuristic with measurably fewer poses (target: 30%+ reduction, matching the Toronto paper's reported numbers). Run on `touch_test.py` for ground truth.

If Phase 3 doesn't beat Phase 2 by a meaningful margin in your setup, ship Phase 2 as the recommended default and keep NBV as an opt-in advanced feature.
