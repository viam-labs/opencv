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


# Consts
ARM_NAME = "ur5e"
POSE_TRACKER_NAME = "chessboard-pose-tracker"
MOTION_SERVICE_NAME = "motion-service"

BODY_NAMES = ['corner_53', 'corner_62', 'corner_75']

TOUCH_PROBE_LENGTH_MM = 113
PRETOUCH_OFFSET_MM = 30  # additional offset beyond touch probe length

VELOCITY_NORMAL = 48
VELOCITY_SLOW = 25

WORLD_FRAME = "world"
COLLISION_FRAME_PROBE = "touch_probe"
COLLISION_FRAME_PEDESTAL = "pedestal-ur5e"
COLLISION_FRAME_APRILTAGS = "apriltags-obstacle"
COLLISION_FRAME_CHESSBOARD = "chessboard-obstacle"
ALLOWED_PROBE_COLLISION_FRAMES = [COLLISION_FRAME_PEDESTAL, COLLISION_FRAME_APRILTAGS, COLLISION_FRAME_CHESSBOARD]


async def connect():
    load_dotenv()
    opts = RobotClient.Options.with_api_key( 
        api_key=os.getenv('VIAM_MACHINE_API_KEY'),
        api_key_id=os.getenv('VIAM_MACHINE_API_KEY_ID'),
    )
    address = os.getenv('VIAM_MACHINE_ADDRESS')
    return await RobotClient.at_address(address, opts)


async def get_pretouch_poses(machine: RobotClient, pt: PoseTracker, length_of_touch_tip: float = 0) -> Dict[str, PoseInFrame]:
    poses_in_camera_frame = await pt.get_poses(body_names=BODY_NAMES)
    print("Got poses from pose tracker in camera frame:")
    print(poses_in_camera_frame)

    for pose_name, pose_in_camera_frame in poses_in_camera_frame.items():
        pose_in_world_frame_pretouch = await machine.transform_pose(pose_in_camera_frame, WORLD_FRAME)
        
        o_x = pose_in_world_frame_pretouch.pose.o_x
        o_y = pose_in_world_frame_pretouch.pose.o_y
        o_z = pose_in_world_frame_pretouch.pose.o_z
        
        offset_distance = -(length_of_touch_tip + PRETOUCH_OFFSET_MM)
        
        pose_in_world_frame_pretouch.pose.x += o_x * offset_distance
        pose_in_world_frame_pretouch.pose.y += o_y * offset_distance
        pose_in_world_frame_pretouch.pose.z += o_z * offset_distance

        poses_in_camera_frame[pose_name] = pose_in_world_frame_pretouch
    return poses_in_camera_frame


async def move_to_scanning_pose(motion_service: MotionClient, arm: Arm, scanning_pose: list[float]) -> None:
    print(f"Moving to initial scanning pose: scanning_pose={scanning_pose}")
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
        reference_frame=WORLD_FRAME,
        pose=scan_pose
    )
    await motion_service.move(
        component_name=arm.name,
        destination=scan_pose_in_frame
    )
    print("Arrived at scanning position")


async def start_touching(motion_service: MotionClient, arm: Arm, poses: Dict[str, PoseInFrame]) -> None:
    for pose_name, pose_pretouch in poses.items():
        print(f"Moving to pretouch position: {pose_name}")
        await motion_service.move(
            component_name=arm.name,
            destination=pose_pretouch
        )
        
        await arm.do_command({"set_vel": VELOCITY_SLOW})
        
        current_pose = copy.deepcopy(pose_pretouch)
        
        o_x = pose_pretouch.pose.o_x
        o_y = pose_pretouch.pose.o_y
        o_z = pose_pretouch.pose.o_z
        
        # Incremental forward movement loop (along orientation vector)
        while True:
            response = input("Keep going? By how many mm (number or press Enter to finish)?: ")
            if not response.strip():
                print(f"Finished with {pose_name}, moving to next pose")
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
                        frame1=COLLISION_FRAME_PROBE,
                        frame2=frame
                    ) for frame in ALLOWED_PROBE_COLLISION_FRAMES
                ])
                constraints = Constraints(collision_specification=[collision_spec])
                
                await motion_service.move(
                    component_name=arm.name,
                    destination=current_pose,
                    constraints=constraints
                )
            except ValueError:
                print("Invalid response, please enter a number (e.g., 1, 10, 0.5, 2.5).")
                print(f"Finished with {pose_name}, moving to next pose")
                break
            except Exception as e:
                print(f"Error: {e}")
                print(f"Finished with {pose_name}, moving to next pose")
                break
        
        await arm.do_command({"set_vel": VELOCITY_NORMAL})
        await motion_service.move(
            component_name=arm.name,
            destination=pose_pretouch,
        )
        await asyncio.sleep(1)

async def main(scanning_pose: Optional[list[float]] = None):
    machine: Optional[RobotClient] = None
    pt: Optional[PoseTracker] = None
    arm: Optional[Arm] = None
    motion_service: Optional[MotionClient] = None

    try:
        machine = await connect()
        arm = Arm.from_robot(machine, ARM_NAME)
        await arm.do_command({"set_vel": VELOCITY_NORMAL})

        # Print initial arm pose for utility and sanity check
        motion_service = MotionClient.from_robot(machine, MOTION_SERVICE_NAME)
        initial_arm_pose = await motion_service.get_pose(
            component_name=arm.name,
            destination_frame=WORLD_FRAME
        )
        print(f"Initial arm pose: {initial_arm_pose}")
        
        # Move to scanning pose if provided
        if scanning_pose:
            await move_to_scanning_pose(motion_service, arm, scanning_pose)
        else:
            print("No scanning position provided, assuming bodies re already visible")
        
        input("Press Enter to continue...")
        
        # Get poses from pose tracker in camera frame and convert to world frame
        pt = PoseTracker.from_robot(machine, POSE_TRACKER_NAME)
        poses = await get_pretouch_poses(machine, pt, length_of_touch_tip=TOUCH_PROBE_LENGTH_MM)
        print("Converted poses to world frame and applied offset:")
        print(poses)

        # Start touches
        input("Press Enter to start touches...")
        await start_touching(motion_service, arm, poses)
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
        '--scanning-pose',
        type=float,
        nargs=7,
        metavar=('X', 'Y', 'Z', 'O_X', 'O_Y', 'O_Z', 'THETA'),
        help='Scanning pose in world frame: x y z o_x o_y o_z theta (7 values). If not provided, assumes bodies are already visible from the current pose.'
    )
    
    args = parser.parse_args()
    asyncio.run(main(scanning_pose=args.scanning_pose))
