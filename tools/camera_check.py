"""Read-only sanity check for a camera's image vs. reported intrinsics.

Bad calibration almost always traces to one of:
  - the reported intrinsics (K) are for a DIFFERENT resolution than the image
    actually returned by get_images  (the #1 gotcha with a new camera),
  - the principal point / focal lengths are implausible,
  - distortion coefficients are missing / wrong.

This script connects, grabs an image and the camera properties, and flags any
mismatch. No arm motion, nothing written to the machine. Optionally, if you
pass --tracker, it pulls one ChArUco/chessboard observation and reports the
per-view PnP reprojection error using the camera's own K/dist -- a high value
there means the image and intrinsics are inconsistent (i.e., the camera is the
problem, not the calibration math).

  ROBOT_ADDRESS=... ROBOT_API_KEY=... ROBOT_API_KEY_ID=... \\
  PYTHONPATH=src python tools/camera_check.py --camera <camName> [--tracker <name>]
"""

import argparse
import asyncio
import os

import numpy as np
from PIL import Image

from viam.robot.client import RobotClient
from viam.components.camera import Camera
from viam.components.pose_tracker import PoseTracker
from viam.media.utils.pil import viam_to_pil_image


async def _connect(args):
    opts = RobotClient.Options.with_api_key(
        api_key=args.api_key, api_key_id=args.api_key_id
    )
    return await RobotClient.at_address(args.address, opts)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", default=os.environ.get("ROBOT_ADDRESS"))
    ap.add_argument("--api-key", default=os.environ.get("ROBOT_API_KEY"))
    ap.add_argument("--api-key-id", default=os.environ.get("ROBOT_API_KEY_ID"))
    ap.add_argument("--camera", required=True)
    ap.add_argument("--tracker", default=None,
                    help="optional charuco/chessboard pose tracker name")
    ap.add_argument("--save", default="./camera_check.png")
    args = ap.parse_args()
    if not all([args.address, args.api_key, args.api_key_id]):
        raise SystemExit("Set ROBOT_ADDRESS / ROBOT_API_KEY / ROBOT_API_KEY_ID")

    robot = await _connect(args)
    try:
        camera = Camera.from_robot(robot, args.camera)

        # --- images ---
        images, _ = await camera.get_images()
        print(f"get_images returned {len(images)} frame(s):")
        color = None
        for im in images:
            try:
                pil = viam_to_pil_image(im)
                w, h = pil.size
                print(f"  - mime={im.mime_type} size={w}x{h} mode={pil.mode}")
                if color is None:
                    color = pil
            except Exception as e:
                print(f"  - mime={im.mime_type} (could not decode: {e})")
        if color is None:
            raise SystemExit("no decodable color image")
        img_w, img_h = color.size
        color.save(args.save)
        print(f"saved first color frame -> {args.save}")

        # --- properties / intrinsics ---
        props = await camera.get_properties()
        ip = props.intrinsic_parameters
        dp = props.distortion_parameters
        print("\nreported intrinsic_parameters:")
        print(f"  width_px={ip.width_px} height_px={ip.height_px}")
        print(f"  fx={ip.focal_x_px:.2f} fy={ip.focal_y_px:.2f} "
              f"cx={ip.center_x_px:.2f} cy={ip.center_y_px:.2f}")
        print(f"  distortion model={getattr(dp,'model','?')!r} "
              f"params={list(dp.parameters)}")

        # --- consistency checks ---
        print("\nCHECKS:")
        ok = True
        if ip.width_px and ip.height_px:
            if (int(ip.width_px), int(ip.height_px)) != (img_w, img_h):
                ok = False
                sx = img_w / ip.width_px if ip.width_px else float("nan")
                sy = img_h / ip.height_px if ip.height_px else float("nan")
                print(f"  [FAIL] intrinsics are for {ip.width_px}x{ip.height_px} "
                      f"but image is {img_w}x{img_h} "
                      f"(scale x={sx:.3f}, y={sy:.3f}) "
                      f"-> K must be scaled to the image, or the camera is "
                      f"emitting the wrong resolution. THIS BREAKS PnP.")
            else:
                print(f"  [ok] intrinsic resolution matches image ({img_w}x{img_h})")
        else:
            print("  [warn] intrinsics report no width/height; cannot verify "
                  "resolution match")

        cx_off = abs(ip.center_x_px - img_w / 2) / img_w if img_w else 0
        cy_off = abs(ip.center_y_px - img_h / 2) / img_h if img_h else 0
        if cx_off > 0.2 or cy_off > 0.2:
            ok = False
            print(f"  [FAIL] principal point ({ip.center_x_px:.0f},"
                  f"{ip.center_y_px:.0f}) is far from image center "
                  f"({img_w/2:.0f},{img_h/2:.0f})")
        else:
            print("  [ok] principal point near image center")

        if not (0.3 * img_w < ip.focal_x_px < 5 * img_w):
            ok = False
            print(f"  [FAIL] fx={ip.focal_x_px:.0f} implausible for a "
                  f"{img_w}px-wide image")
        else:
            print("  [ok] focal length in a plausible range")

        if all(abs(p) < 1e-9 for p in dp.parameters):
            print("  [warn] all distortion params are zero — fine if the image "
                  "is truly rectified, suspicious otherwise")

        # --- optional: per-view PnP reprojection through the tracker ---
        if args.tracker:
            tracker = PoseTracker.from_robot(robot, args.tracker)
            obs = None
            for key in ("get_charuco_observation", "get_chessboard_observation"):
                try:
                    r = await tracker.do_command({key: True})
                    if isinstance(r.get(key), dict):
                        obs = r[key]
                        break
                except Exception:
                    pass
            if obs is None:
                print("\n[tracker] no observation (is the target in view?)")
            else:
                import cv2
                c2d = np.asarray(obs["corners_2d"], np.float64).reshape(-1, 2)
                c3d = np.asarray(obs["corners_3d"], np.float64).reshape(-1, 3)
                rvec = np.asarray(obs["rvec"], np.float64).reshape(3, 1)
                tvec = np.asarray(obs["tvec"], np.float64).reshape(3, 1)
                K = np.asarray(obs["K"], np.float64).reshape(3, 3)
                dist = np.asarray(obs["dist"], np.float64).reshape(-1)
                proj, _ = cv2.projectPoints(c3d, rvec, tvec, K, dist)
                err = float(np.sqrt(np.mean((proj.reshape(-1, 2) - c2d) ** 2)))
                print(f"\n[tracker] {c3d.shape[0]} corners, "
                      f"per-view PnP reprojection error = {err:.2f}px")
                if err > 3:
                    ok = False
                    print("  [FAIL] high single-view reprojection error -> the "
                          "image and intrinsics are inconsistent (camera issue), "
                          "not the hand-eye math.")
                else:
                    print("  [ok] single view fits its own pose -> intrinsics & "
                          "corners are self-consistent for this frame.")

        print("\nRESULT:", "looks consistent" if ok else
              ">>> inconsistency found (see [FAIL] lines above) <<<")
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
