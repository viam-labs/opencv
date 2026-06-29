"""Capture hand-eye calibration data from a live machine for offline analysis.

Records, per station, exactly what the hand_eye_calibration service consumes:
  - the camera image,
  - the arm pose (both arm.get_end_position() and, if a motion service is
    given, motion.get_pose()),
  - the camera intrinsics (get_properties -> K, dist),
  - the pose tracker's raw ChArUco observation
    (do_command get_charuco_observation: corners_2d, corners_3d, ids,
    rvec, tvec, K, dist).

Everything is saved to an output directory (PNG images + stations.json) so the
calibration can be reconstructed and diagnosed offline, repeatedly, without
rebuilding the module.

Two capture modes:
  --mode interactive  (default, SAFE): you move the arm to each calibration
      pose yourself (teach pendant / your own flow) and press Enter to capture.
  --mode poses --poses-file poses.json : the script MOVES the arm to each pose
      in the file (Cartesian Pose list) and captures. This commands real
      hardware -- use only if you trust the poses.

Connection is via env vars (or flags):
  ROBOT_ADDRESS, ROBOT_API_KEY, ROBOT_API_KEY_ID

Example:
  ROBOT_ADDRESS=my-machine.xxxx.viam.cloud \\
  ROBOT_API_KEY=... ROBOT_API_KEY_ID=... \\
  python tools/capture_calibration_data.py \\
      --arm myArm --camera myCam --tracker charuco \\
      --out ./calib_capture --mode interactive
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.pose_tracker import PoseTracker
from viam.media.utils.pil import viam_to_pil_image
from viam.media.video import CameraMimeType

try:
    from viam.services.motion import MotionClient
except Exception:  # motion service optional
    MotionClient = None


def _pose_to_dict(p):
    return {
        "x": p.x, "y": p.y, "z": p.z,
        "o_x": p.o_x, "o_y": p.o_y, "o_z": p.o_z, "theta": p.theta,
    }


async def _connect(args):
    opts = RobotClient.Options.with_api_key(
        api_key=args.api_key, api_key_id=args.api_key_id
    )
    return await RobotClient.at_address(args.address, opts)


async def _grab_image(camera: Camera):
    images, _ = await camera.get_images()
    for img in images:
        if img.mime_type in (
            CameraMimeType.JPEG, CameraMimeType.PNG, CameraMimeType.VIAM_RGBA
        ):
            return np.array(viam_to_pil_image(img))
    raise RuntimeError("no usable color image from camera")


async def _intrinsics(camera: Camera):
    props = await camera.get_properties()
    ip = props.intrinsic_parameters
    dp = props.distortion_parameters
    return {
        "K": {
            "fx": ip.focal_x_px, "fy": ip.focal_y_px,
            "cx": ip.center_x_px, "cy": ip.center_y_px,
            "width": ip.width_px, "height": ip.height_px,
        },
        "distortion_model": getattr(dp, "model", ""),
        "distortion_parameters": list(dp.parameters),
    }


async def _capture_station(idx, out_dir, arm, camera, tracker, motion, arm_name):
    arm_pose = await arm.get_end_position()
    motion_pose = None
    if motion is not None and arm_name is not None:
        try:
            pif = await motion.get_pose(
                component_name=arm_name, destination_frame=arm_name + "_origin"
            )
            motion_pose = _pose_to_dict(pif.pose)
        except Exception as e:
            print(f"  (motion.get_pose failed: {e})")

    image = await _grab_image(camera)
    img_path = out_dir / f"station_{idx:02d}.png"
    Image.fromarray(image).save(img_path)

    obs = None
    try:
        resp = await tracker.do_command({"get_charuco_observation": True})
        obs = resp.get("get_charuco_observation")
    except Exception as e:
        print(f"  (tracker observation failed: {e})")

    intr = await _intrinsics(camera)

    station = {
        "index": idx,
        "image": img_path.name,
        "arm_pose_end_position": _pose_to_dict(arm_pose),
        "arm_pose_motion_service": motion_pose,
        "intrinsics": intr,
        "charuco_observation": obs,
    }
    n_corners = len(obs["corners_2d"]) if obs and "corners_2d" in obs else 0
    print(f"  station {idx}: saved {img_path.name}, {n_corners} corners, "
          f"arm=({arm_pose.x:.0f},{arm_pose.y:.0f},{arm_pose.z:.0f})")
    return station


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", default=os.environ.get("ROBOT_ADDRESS"))
    ap.add_argument("--api-key", default=os.environ.get("ROBOT_API_KEY"))
    ap.add_argument("--api-key-id", default=os.environ.get("ROBOT_API_KEY_ID"))
    ap.add_argument("--arm", required=True, help="arm component name")
    ap.add_argument("--camera", required=True, help="camera component name")
    ap.add_argument("--tracker", required=True, help="charuco pose tracker name")
    ap.add_argument("--motion", default=None, help="motion service name (optional)")
    ap.add_argument("--out", default="./calib_capture")
    ap.add_argument("--mode", choices=["interactive", "poses"], default="interactive")
    ap.add_argument("--poses-file", default=None,
                    help="JSON list of Cartesian poses (move mode)")
    ap.add_argument("--settle-seconds", type=float, default=2.0)
    args = ap.parse_args()

    if not all([args.address, args.api_key, args.api_key_id]):
        raise SystemExit("Set ROBOT_ADDRESS / ROBOT_API_KEY / ROBOT_API_KEY_ID")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    robot = await _connect(args)
    try:
        arm = Arm.from_robot(robot, args.arm)
        camera = Camera.from_robot(robot, args.camera)
        tracker = PoseTracker.from_robot(robot, args.tracker)
        motion = None
        if args.motion and MotionClient is not None:
            motion = MotionClient.from_robot(robot, args.motion)

        stations = []
        if args.mode == "interactive":
            print("Interactive capture. Move the arm to each calibration pose, "
                  "then press Enter to capture. Type 'q' + Enter to finish.")
            idx = 0
            while True:
                cmd = input(f"[station {idx}] Enter=capture, q=quit: ").strip().lower()
                if cmd == "q":
                    break
                await asyncio.sleep(0.2)
                stations.append(await _capture_station(
                    idx, out_dir, arm, camera, tracker, motion, args.arm))
                idx += 1
        else:
            from viam.proto.common import Pose
            if not args.poses_file:
                raise SystemExit("--mode poses requires --poses-file")
            poses = json.loads(Path(args.poses_file).read_text())
            print(f"MOVE MODE: commanding arm through {len(poses)} poses.")
            for idx, pd in enumerate(poses):
                print(f"moving to pose {idx+1}/{len(poses)} ...")
                await arm.move_to_position(Pose(
                    x=pd["x"], y=pd["y"], z=pd["z"],
                    o_x=pd["o_x"], o_y=pd["o_y"], o_z=pd["o_z"], theta=pd["theta"]))
                await asyncio.sleep(args.settle_seconds)
                stations.append(await _capture_station(
                    idx, out_dir, arm, camera, tracker, motion, args.arm))

        (out_dir / "stations.json").write_text(json.dumps(stations, indent=2))
        print(f"\nWrote {len(stations)} stations to {out_dir}/stations.json")
        print("Send me that directory (or stations.json + images) to analyze.")
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
