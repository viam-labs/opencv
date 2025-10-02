import asyncio
import base64
import io

from PIL import Image
import numpy as np

from viam.components.camera import Camera
from viam.components.pose_tracker import PoseTracker
from viam.robot.client import RobotClient
from viam.media.video import CameraMimeType
from viam.media.utils.pil import viam_to_pil_image


async def connect():
    """Establish a network connection between Codespace host and the robot."""
    opts = RobotClient.Options.with_api_key( 
        api_key='8onoqogfufm7ftywcxpernv8xq0w6814',
        api_key_id='7ba18f3e-565a-453f-9ef2-4c213897d4d2',
    )
    address = 'sean-framework-main.yznpnnv7bl.viam.cloud'
    return await RobotClient.at_address(address, opts)


def pil_image_to_base64(pil_image: Image.Image) -> str:
    """Convert a PIL Image to base64 encoded string."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG')
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


async def capture_and_validate_image(cam: Camera) -> tuple[Image.Image, bool]:
    """Capture an image and show it to the user for validation.
    
    Returns:
        Tuple of (PIL Image, accepted)
    """
    # Capture image from camera
    images = await cam.get_images()
    pil_image = None
    
    for cam_image in images[0]:
        if cam_image.mime_type in [CameraMimeType.JPEG, CameraMimeType.PNG, CameraMimeType.VIAM_RGBA]:
            pil_image = viam_to_pil_image(cam_image)
            break
    
    if pil_image is None:
        print("Error: Could not capture image from camera")
        return None, False
    
    # Show image to user
    pil_image.show()
    
    # Get user validation
    while True:
        response = input("Accept this image? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return pil_image, True
        elif response in ['n', 'no']:
            return pil_image, False
        else:
            print("Please enter 'y' or 'n'")


async def collect_calibration_images(cam: Camera, num_images: int = 10) -> list[str]:
    """Collect calibration images with user validation.
    
    Args:
        cam: Camera component to capture from
        num_images: Number of images to collect
        
    Returns:
        List of base64 encoded image strings
    """
    base64_images = []
    
    print(f"\n{'='*60}")
    print(f"Camera Calibration - Collecting {num_images} Images")
    print(f"{'='*60}")
    print("\nInstructions:")
    print("- Position the chessboard at different angles and distances")
    print("- Cover different areas of the camera's field of view")
    print("- Ensure good lighting and focus")
    print("- Each captured image will be shown for your approval")
    print(f"{'='*60}\n")
    
    while len(base64_images) < num_images:
        current_count = len(base64_images) + 1
        print(f"\n📸 Capturing image {current_count}/{num_images}...")
        print("Position the chessboard and press Enter when ready...")
        input()
        
        # Keep trying until user accepts an image
        accepted = False
        while not accepted:
            pil_image, accepted = await capture_and_validate_image(cam)
            
            if pil_image is None:
                print("Failed to capture image. Retrying...")
                await asyncio.sleep(1)
                continue
            
            if accepted:
                # Convert to base64 and add to collection
                base64_img = pil_image_to_base64(pil_image)
                base64_images.append(base64_img)
                print(f"✓ Image {current_count}/{num_images} accepted and saved!")
            else:
                print("✗ Image rejected. Retaking...")
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully collected all {num_images} images!")
    print(f"{'='*60}\n")
    
    return base64_images


async def run_calibration(chessboard: PoseTracker, base64_images: list[str]):
    """Run camera calibration using collected images.
    
    Args:
        chessboard: Chessboard pose tracker component
        base64_images: List of base64 encoded images
    """
    print(f"\n{'='*60}")
    print("Running Camera Calibration...")
    print(f"{'='*60}\n")
    
    result = await chessboard.do_command({
        "calibrate_camera": {
            "images": base64_images
        }
    })
    
    if result.get("success"):
        print("✓ Calibration successful!\n")
        print(f"RMS Error: {result['rms_error']:.4f}")
        print(f"Images Used: {result['num_images']}")
        print(f"Image Size: {result['image_size']['width']}x{result['image_size']['height']}")
        print("\nCamera Matrix (Intrinsics):")
        cm = result['camera_matrix']
        print(f"  fx: {cm['fx']:.2f}")
        print(f"  fy: {cm['fy']:.2f}")
        print(f"  cx: {cm['cx']:.2f}")
        print(f"  cy: {cm['cy']:.2f}")
        print("\nDistortion Coefficients:")
        dc = result['distortion_coefficients']
        print(f"  k1: {dc['k1']:.6f}")
        print(f"  k2: {dc['k2']:.6f}")
        print(f"  p1: {dc['p1']:.6f}")
        print(f"  p2: {dc['p2']:.6f}")
        print(f"  k3: {dc['k3']:.6f}")
    else:
        print(f"✗ Calibration failed: {result.get('error', 'Unknown error')}")
    
    print(f"\n{'='*60}\n")
    return result


async def main():
    """Main function to orchestrate the calibration process."""
    machine = None
    try:
        # Connect to robot
        print("Connecting to robot...")
        machine = await connect()
        print("✓ Connected!\n")
        
        # Get camera and chessboard components
        cam = Camera.from_robot(machine, "orbbec-1")
        chessboard = PoseTracker.from_robot(machine, "chessboard")
        
        # Ask user how many images to collect
        while True:
            try:
                num_images_str = input("How many images to collect for calibration? (default: 10): ").strip()
                if num_images_str == "":
                    num_images = 10
                else:
                    num_images = int(num_images_str)
                    if num_images < 3:
                        print("Please enter at least 3 images")
                        continue
                break
            except ValueError:
                print("Please enter a valid number")
        
        # Collect calibration images with user validation
        base64_images = await collect_calibration_images(cam, num_images)
        
        # Run calibration
        result = await run_calibration(chessboard, base64_images)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if machine:
            await machine.close()
            print("Connection closed.")


if __name__ == '__main__':
    asyncio.run(main())
