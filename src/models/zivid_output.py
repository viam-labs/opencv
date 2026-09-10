import asyncio

from viam.robot.client import RobotClient
from viam.components.camera import Camera
from viam.services.generic import Generic

async def connect():
    opts = RobotClient.Options.with_api_key(
         
        api_key='8zi96lxz4bgc1zflrxjoxkeummpnqtlo',
        
        api_key_id='b07b0c0d-8d68-438e-954a-9c523bebbc54'
    )
    
    return await RobotClient.at_address('njsanding2-main.tbr1ap8x6t.viam.cloud', opts)

async def main():
    async with await connect() as machine:
        camera = Generic.from_robot(machine, "zivid-calibration")
        output = await camera.do_command({"command": "capture_and_detect"})
        print(f"capture_and_detect output: {output["detected"]}")

if __name__ == '__main__':
    asyncio.run(main())
