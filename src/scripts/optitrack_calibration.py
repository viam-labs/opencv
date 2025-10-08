import asyncio
import os

from dotenv import load_dotenv
from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.pose_tracker import PoseTracker

from typing import Optional


async def connect():
    load_dotenv()
    opts = RobotClient.Options.with_api_key( 
        api_key=os.getenv('VIAM_MACHINE_API_KEY'),
        api_key_id=os.getenv('VIAM_MACHINE_API_KEY_ID'),
    )
    address = os.getenv('VIAM_MACHINE_ADDRESS')
    return await RobotClient.at_address(address, opts)


async def get_t_matrix(arm: Arm, pt: PoseTracker):
    # TODO: 1. SETUP
    # TODO: Define a list of 5-10 robot poses (e.g., as Viam Pose objects or 4x4 numpy matrices).
    # These should be well-distributed in the robot's workspace with varied orientations.

    # TODO: Initialize two empty lists to store the collected poses.
    # e.g., robot_poses = []
    # e.g., optitrack_poses = []

    # TODO: 2. DATA COLLECTION LOOP
    # TODO: Loop through each of the predefined robot poses from the setup step.
        # TODO: Command the arm to move to the current pose in the loop.
        # e.g., await arm.move_to_pose(...)
        # TODO: Add a short sleep/wait to ensure the arm has settled.
        # e.g., await asyncio.sleep(1)

        # TODO: Get the robot's current end-effector pose from the controller.
        # e.g., robot_pose = await arm.get_end_position()
        # TODO: Append the robot_pose to the robot_poses list.

        # TODO: Get the rigid body's pose from the OptiTrack system (PoseTracker).
    poses = await pt.get_poses(body_names=["RigidBody"])
    print(poses)
        # TODO: Extract the pose for your specific rigid body from the 'poses' dictionary.
        # TODO: Append the optitrack_pose to the optitrack_poses list.
    
    # TODO: End of the loop.

    # TODO: 3. CALCULATION
    # TODO: After the loop, use the two lists (robot_poses and optitrack_poses) to calculate the transformation matrix.
    # This involves:
    # TODO: a. Separating the translation vectors and rotation matrices for both lists.
    # TODO: b. Finding the centroid of each set of translation vectors.
    # TODO: c. Using an algorithm like Kabsch or SVD to find the optimal rotation matrix (R).
    # TODO: d. Calculating the translation vector (t).
    # TODO: e. Assembling the final 4x4 transformation matrix T_RobotBase_OptiTrack from R and t.

    # TODO: 4. VERIFICATION (Optional but Recommended)
    # TODO: Move the arm to a new pose that was NOT used in the calibration.
    # TODO: Get the pose from both the arm and the pose tracker.
    # TODO: Apply the calculated transformation matrix to the OptiTrack pose.
    # TODO: Compare the result with the arm's reported pose to check the error.

    # TODO: 5. OUTPUT
    # TODO: Print or save the final calculated transformation matrix.
    pass

async def main():
    machine: Optional[RobotClient] = None
    pt: Optional[PoseTracker] = None
    arm: Optional[Arm] = None
    try:
        machine = await connect()
        pt = PoseTracker.from_robot(machine, "pose_tracker-1")
        arm = Arm.from_robot(machine, "ur5e")

        await get_t_matrix(arm, pt)

        await pt.close()
        await arm.close()
    except Exception as e:
        print("Caught exception in script main: ", e)
    finally:
        if pt:
            await pt.close()
        if arm:
            await arm.close()
        if machine:
            await machine.close()

if __name__ == '__main__':
  asyncio.run(main())
