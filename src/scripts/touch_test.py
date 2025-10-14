import argparse
import asyncio
import copy
import os

from dotenv import load_dotenv
from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.pose_tracker import PoseTracker
from viam.services.motion import MotionClient, Constraints
from viam.proto.service.motion import CollisionSpecification
from viam.proto.common import PoseInFrame, Pose

from typing import Dict, Optional


# Default values for optional args
DEFAULT_TOUCH_PROBE_LENGTH_MM = 113
DEFAULT_PRETOUCH_OFFSET_MM = 500  # additional offset beyond touch probe length
DEFAULT_VELOCITY_NORMAL = 25
DEFAULT_VELOCITY_SLOW = 10
DEFAULT_WORLD_FRAME = "world"


async def connect():
    load_dotenv()
    opts = RobotClient.Options.with_api_key( 
        api_key=os.getenv('VIAM_MACHINE_API_KEY'),
        api_key_id=os.getenv('VIAM_MACHINE_API_KEY_ID'),
    )
    address = os.getenv('VIAM_MACHINE_ADDRESS')
    return await RobotClient.at_address(address, opts)


async def get_pretouch_poses(
    machine: RobotClient,
    pt: PoseTracker,
    body_names: list[str],
    world_frame: str,
    length_of_touch_tip: float = 0,
    pretouch_offset_mm: float = DEFAULT_PRETOUCH_OFFSET_MM
) -> Dict[str, PoseInFrame]:
    poses_in_camera_frame = await pt.get_poses(body_names=body_names)
    if len(poses_in_camera_frame) == 0:
        raise Exception("No poses found from pose tracker in camera frame")
    if len(poses_in_camera_frame) != len(body_names):
        print(f"poses in camera frame: {poses_in_camera_frame.keys()}")
        print(f"body names: {body_names}")
        raise Exception(
            f"Number of poses found from pose tracker in camera frame "
            f"({len(poses_in_camera_frame)}) does not match number of body names "
            f"({len(body_names)})"
        )
    
    print("Got poses from pose tracker in camera frame:")
    print(poses_in_camera_frame)

    for pose_name, pose_in_camera_frame in poses_in_camera_frame.items():
        pose_in_world_frame_pretouch = await machine.transform_pose(pose_in_camera_frame, world_frame)

        o_x = pose_in_world_frame_pretouch.pose.o_x
        o_y = pose_in_world_frame_pretouch.pose.o_y
        o_z = pose_in_world_frame_pretouch.pose.o_z

        offset_distance = -(length_of_touch_tip + pretouch_offset_mm)
        
        pose_in_world_frame_pretouch.pose.x += o_x * offset_distance
        pose_in_world_frame_pretouch.pose.y += o_y * offset_distance
        pose_in_world_frame_pretouch.pose.z += o_z * offset_distance

        poses_in_camera_frame[pose_name] = pose_in_world_frame_pretouch
    return poses_in_camera_frame


async def move_to_scanning_pose(
    motion_service: MotionClient,
    arm_name: str,
    scanning_pose: list[float],
    world_frame: str
) -> None:
    print(f"Moving to scanning pose: scanning_pose={scanning_pose}")
    scan_pose = Pose(
        x=scanning_pose[0],
        y=scanning_pose[1],
        z=scanning_pose[2],
        o_x=scanning_pose[3],
        o_y=scanning_pose[4],
        o_z=scanning_pose[5],
        theta=scanning_pose[6]
    )
    scan_pose_in_frame = PoseInFrame(
        reference_frame=world_frame,
        pose=scan_pose
    )
    
    await motion_service.move(
        component_name=arm_name,
        destination=scan_pose_in_frame
    )
    print("Arrived at scanning position")


async def start_touching(
    motion_service: MotionClient,
    arm: Arm,
    poses: Dict[str, PoseInFrame],
    probe_collision_frame: str,
    allowed_collision_frames: list[str],
    velocity_normal: float,
    velocity_slow: float
) -> None:
    # Sort poses lexicographically by name
    sorted_poses = sorted(poses.items(), key=lambda item: item[0])
    print(f"Sorted poses: {sorted_poses}")
    
    for pose_name, pose_pretouch in sorted_poses:
        print(f"Moving to pretouch position: {pose_name}")
        await motion_service.move(
            component_name=arm.name,
            destination=pose_pretouch
        )
        
        await arm.do_command({"set_vel": velocity_slow})
        
        current_pose = copy.deepcopy(pose_pretouch)
        
        o_x = pose_pretouch.pose.o_x
        o_y = pose_pretouch.pose.o_y
        o_z = pose_pretouch.pose.o_z
        
        # Incremental forward movement loop (along orientation vector)
        distance_mm_moved = 0
        while True:
            response = input("Keep going? By how many mm (number or press Enter to finish)?: ")
            if not response.strip():
                print(f"Finished touching {pose_name}. Moved {distance_mm_moved}mm along orientation vector before stopping.")
                print("Moving to next pose.")
                break
            
            try:
                distance = float(response)
                current_pose.pose.x += o_x * distance
                current_pose.pose.y += o_y * distance
                current_pose.pose.z += o_z * distance
                print(f"Moving forward {distance}mm along orientation vector")
                print(f"  New position: x={current_pose.pose.x:.2f}, y={current_pose.pose.y:.2f}, z={current_pose.pose.z:.2f}")
                
                collision_spec = CollisionSpecification()
                collision_spec.allows.extend([
                    CollisionSpecification.AllowedFrameCollisions(
                        frame1=probe_collision_frame,
                        frame2=allowed_collision_frame
                    ) for allowed_collision_frame in allowed_collision_frames
                ])
                constraints = Constraints(
                    collision_specification=[collision_spec],
                )
                
                await motion_service.move(
                    component_name=arm.name,
                    destination=current_pose,
                    constraints=constraints
                )
                distance_mm_moved += distance
            except Exception as e:
                print(f"Error: {e}")
                print(f"Error touching {pose_name}. Moved {distance_mm_moved}mm along orientation vector before stopping.")
                print("Moving to next pose")
                break

        await arm.do_command({"set_vel": velocity_normal})
        await motion_service.move(
            component_name=arm.name,
            destination=pose_pretouch,
        )
        await asyncio.sleep(1)

async def main(
    arm_name: str,
    pose_tracker_name: str,
    motion_service_name: str,
    body_names: list[str],
    probe_collision_frame: str,
    allowed_collision_frames: list[str],
    touch_probe_length_mm: float = DEFAULT_TOUCH_PROBE_LENGTH_MM,
    pretouch_offset_mm: float = DEFAULT_PRETOUCH_OFFSET_MM,
    velocity_normal: float = DEFAULT_VELOCITY_NORMAL,
    velocity_slow: float = DEFAULT_VELOCITY_SLOW,
    world_frame: str = DEFAULT_WORLD_FRAME,
    scanning_pose: Optional[list[float]] = None
):
    machine: Optional[RobotClient] = None
    pt: Optional[PoseTracker] = None
    arm: Optional[Arm] = None
    motion_service: Optional[MotionClient] = None

    try:
        machine = await connect()
        arm = Arm.from_robot(machine, arm_name)
        await arm.do_command({"set_vel": velocity_normal})

        # Print initial arm pose for utility and sanity check
        motion_service = MotionClient.from_robot(machine, motion_service_name)
        initial_arm_pose = await motion_service.get_pose(
            component_name=arm.name,
            destination_frame=world_frame
        )
        print(f"Initial, pre-scanning position arm pose: {initial_arm_pose}")

        # Move to scanning pose if provided
        if scanning_pose:
            await move_to_scanning_pose(motion_service, arm.name, scanning_pose, world_frame)
        else:
            print("No scanning position provided, assuming bodies are already visible")

        input("Press Enter to collect poses...")

        # Get poses from pose tracker in camera frame and convert to world frame
        pt = PoseTracker.from_robot(machine, pose_tracker_name)
        poses = await get_pretouch_poses(machine, pt, body_names, world_frame, length_of_touch_tip=touch_probe_length_mm, pretouch_offset_mm=pretouch_offset_mm)
        print("Converted poses to world frame and applied offset:")
        print(poses)

        # Start touches
        input("Press Enter to start touches...")
        await start_touching(motion_service, arm, poses, probe_collision_frame, allowed_collision_frames, velocity_normal, velocity_slow)
    except Exception as e:
        print("Caught exception in script main: ")
        raise e
    finally:
        if pt:
            await pt.close()
        if arm:
            await arm.close()
        if motion_service:
            await motion_service.close()
        if machine:
            await machine.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Touch test script')
    parser.add_argument(
        '--arm-name',
        type=str,
        required=True,
        help='Name of the arm component'
    )
    parser.add_argument(
        '--pose-tracker-name',
        type=str,
        required=True,
        help='Name of the pose tracker resource'
    )
    parser.add_argument(
        '--motion-service-name',
        type=str,
        required=True,
        help='Name of the motion service'
    )
    parser.add_argument(
        '--body-names',
        type=str,
        nargs='+',
        required=True,
        help='List of body names to track (e.g., corner_0 corner_1)'
    )
    parser.add_argument(
        '--probe-collision-frame',
        type=str,
        required=True,
        help='Collision frame name for the touch probe'
    )
    parser.add_argument(
        '--allowed-collision-frames',
        type=str,
        nargs='+',
        required=True,
        help='List of collision frames that the probe is allowed to collide with'
    )
    parser.add_argument(
        '--touch-probe-length-mm',
        type=float,
        default=DEFAULT_TOUCH_PROBE_LENGTH_MM,
        help=f'Length of the touch probe in mm (default: {DEFAULT_TOUCH_PROBE_LENGTH_MM})'
    )
    parser.add_argument(
        '--pretouch-offset-mm',
        type=float,
        default=DEFAULT_PRETOUCH_OFFSET_MM,
        help=f'Additional offset beyond touch probe length in mm (default: {DEFAULT_PRETOUCH_OFFSET_MM})'
    )
    parser.add_argument(
        '--velocity-normal',
        type=float,
        default=DEFAULT_VELOCITY_NORMAL,
        help=f'Normal velocity setting (default: {DEFAULT_VELOCITY_NORMAL})'
    )
    parser.add_argument(
        '--velocity-slow',
        type=float,
        default=DEFAULT_VELOCITY_SLOW,
        help=f'Slow velocity setting for touching (default: {DEFAULT_VELOCITY_SLOW})'
    )
    parser.add_argument(
        '--world-frame',
        type=str,
        default=DEFAULT_WORLD_FRAME,
        help=f'Name of the world reference frame (default: {DEFAULT_WORLD_FRAME})'
    )
    parser.add_argument(
        '--scanning-pose',
        type=float,
        nargs=7,
        metavar=('X', 'Y', 'Z', 'O_X', 'O_Y', 'O_Z', 'THETA'),
        help='Scanning pose in world frame: x y z o_x o_y o_z theta (7 values). If not provided, assumes bodies are already visible from the current pose.'
    )

    args = parser.parse_args()
    asyncio.run(main(
        arm_name=args.arm_name,
        pose_tracker_name=args.pose_tracker_name,
        motion_service_name=args.motion_service_name,
        body_names=args.body_names,
        probe_collision_frame=args.probe_collision_frame,
        allowed_collision_frames=args.allowed_collision_frames,
        touch_probe_length_mm=args.touch_probe_length_mm,
        pretouch_offset_mm=args.pretouch_offset_mm,
        velocity_normal=args.velocity_normal,
        velocity_slow=args.velocity_slow,
        world_frame=args.world_frame,
        scanning_pose=args.scanning_pose
    ))
