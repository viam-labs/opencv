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
    poses = await pt.get_poses(body_names=["RigidBody"])
    print(poses)
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
