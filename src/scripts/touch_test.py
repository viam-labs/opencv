import argparse
import asyncio
import copy
import math
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
POSE_TRACKER_NAME = "april-tag-tracker"
MOTION_SERVICE_NAME = "motion-service"

APRILTAG_BODY_NAMES = ['1', '6', '11', '16', '21']

TOUCH_PROBE_LENGTH_MM = 113
PRETOUCH_OFFSET_MM = 10  # additional offset beyond touch probe length

VELOCITY_NORMAL = 48
VELOCITY_SLOW = 25

WORLD_FRAME = "world"
COLLISION_FRAME_PROBE = "touch_probe"
COLLISION_FRAME_PEDESTAL = "pedestal-ur5e"


async def connect():
    load_dotenv()
    opts = RobotClient.Options.with_api_key( 
        api_key=os.getenv('VIAM_MACHINE_API_KEY'),
        api_key_id=os.getenv('VIAM_MACHINE_API_KEY_ID'),
    )
    address = os.getenv('VIAM_MACHINE_ADDRESS')
    return await RobotClient.at_address(address, opts)


async def transform_and_adjust_poses(machine: RobotClient, poses: Dict[str, PoseInFrame], length_of_touch_tip: float = 0) -> Dict[str, PoseInFrame]:
    """Transform poses to world frame and apply offset along orientation vector"""
    for pose_name, pose_in_camera_frame in poses.items():
        pose_in_world_frame_pretouch = await machine.transform_pose(pose_in_camera_frame, WORLD_FRAME)
        
        # Get ov of the tag (pre-normalized)
        o_x = pose_in_world_frame_pretouch.pose.o_x
        o_y = pose_in_world_frame_pretouch.pose.o_y
        o_z = pose_in_world_frame_pretouch.pose.o_z
        
        # Calculate offset distance (negative to move away from tag surface)
        offset_distance = -(length_of_touch_tip + PRETOUCH_OFFSET_MM)
        
        pose_in_world_frame_pretouch.pose.x += o_x * offset_distance
        pose_in_world_frame_pretouch.pose.y += o_y * offset_distance
        pose_in_world_frame_pretouch.pose.z += o_z * offset_distance

        poses[pose_name] = pose_in_world_frame_pretouch
    return poses


async def move_to_poses(motion_service: MotionClient, arm: Arm, poses: Dict[str, PoseInFrame]) -> None:
    for pose_name, pose_pretouch in poses.items():
        print(f"\n--- Moving to AprilTag: {pose_name} ---")
        
        # Move to pretouch pose
        await motion_service.move(
            component_name=arm.name,
            destination=pose_pretouch,
        )
        print(f"Arrived at pretouch pose for {pose_name}")
        
        # Set slower velocity for incremental movements
        await arm.do_command({"set_vel": VELOCITY_SLOW})
        
        # Current pose for incremental movement
        current_pose = copy.deepcopy(pose_pretouch)
        
        # Get the orientation vector for this AprilTag
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
                # Move forward by specified distance along the orientation vector
                current_pose.pose.x += o_x * distance
                current_pose.pose.y += o_y * distance
                current_pose.pose.z += o_z * distance
                print(f"Moving forward {distance}mm along orientation vector")
                print(f"  New position: x={current_pose.pose.x:.2f}, y={current_pose.pose.y:.2f}, z={current_pose.pose.z:.2f}")
                
                collision_spec = CollisionSpecification()
                collision_spec.allows.append(
                    CollisionSpecification.AllowedFrameCollisions(
                        frame1=COLLISION_FRAME_PROBE,
                        frame2=COLLISION_FRAME_PEDESTAL
                    )
                )
                constraints = Constraints(collision_specification=[collision_spec])
                
                await motion_service.move(
                    component_name=arm.name,
                    destination=current_pose,
                    constraints=constraints
                )
            except ValueError:
                print("Invalid response, please enter a number (e.g., 1, 0.5, 2.5).")
                print(f"Finished with {pose_name}, moving to next pose")
                break
            except Exception as e:
                print(f"Error: {e}")
                print(f"Finished with {pose_name}, moving to next pose")
                break
        
        # Restore normal velocity
        await arm.do_command({"set_vel": VELOCITY_NORMAL})

async def main(scanning_pose: Optional[list[float]] = None):
    machine: Optional[RobotClient] = None
    pt: Optional[PoseTracker] = None
    arm: Optional[Arm] = None
    motion_service: Optional[MotionClient] = None

    try:
        machine = await connect()
        arm = Arm.from_robot(machine, ARM_NAME)
        # Set initial speed
        await arm.do_command({"set_vel": VELOCITY_NORMAL})

        motion_service = MotionClient.from_robot(machine, MOTION_SERVICE_NAME)
        initial_arm_pose = await motion_service.get_pose(
            component_name=arm.name,
            destination_frame=WORLD_FRAME
        )
        print(f"Initial arm pose captured: {initial_arm_pose}")
        
        if scanning_pose:
            print(f"Moving to initial AprilTag scanning pose: scanning_pose={scanning_pose}")
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
        else:
            print("No scanning position provided, assuming AprilTags are already visible")
        
        input("Press Enter to continue...")
        
        pt = PoseTracker.from_robot(machine, POSE_TRACKER_NAME)
        poses = await pt.get_poses(body_names=APRILTAG_BODY_NAMES)

        poses = await transform_and_adjust_poses(machine, poses, length_of_touch_tip=TOUCH_PROBE_LENGTH_MM)

        input("Press Enter to start touches...")
        await move_to_poses(motion_service, arm, poses)
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
    parser = argparse.ArgumentParser(description='Touch test script for AprilTag detection')
    parser.add_argument(
        '--scanning-pose',
        type=float,
        nargs=7,
        metavar=('X', 'Y', 'Z', 'O_X', 'O_Y', 'O_Z', 'THETA'),
        help='Scanning pose in world frame: x y z o_x o_y o_z theta (7 values). If not provided, assumes AprilTags are already visible from the current pose.'
    )
    
    args = parser.parse_args()
    asyncio.run(main(scanning_pose=args.scanning_pose))
