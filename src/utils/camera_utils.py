from viam.components.camera import Camera
from viam.media.video import CameraMimeType
from viam.media.utils.pil import viam_to_pil_image
import numpy as np

async def get_camera_image(camera: Camera) -> np.ndarray:
    cam_images = await camera.get_images()
    pil_image = None
    for cam_image in cam_images[0]:
        # Accept any standard image format that viam_to_pil_image can handle
        if cam_image.mime_type in [CameraMimeType.JPEG, CameraMimeType.PNG, CameraMimeType.VIAM_RGBA]:
            pil_image = viam_to_pil_image(cam_image)
            break
    if pil_image is None:
        raise Exception("Could not get latest image from camera")        
    image = np.array(pil_image)
    return image

async def get_camera_intrinsics(camera: Camera) -> tuple:
    """Get camera intrinsic parameters"""
    camera_params = await camera.do_command({"get_camera_params": None})
    intrinsics = camera_params["Color"]["intrinsics"]
    dist_params = camera_params["Color"]["distortion"]
    
    K = np.array([
        [intrinsics["fx"], 0, intrinsics["cx"]],
        [0, intrinsics["fy"], intrinsics["cy"]],
        [0, 0, 1]
    ], dtype=np.float32)

    dist = np.array([dist_params["k1"], dist_params["k2"], dist_params["p1"], dist_params["p2"], dist_params["k3"]], dtype=np.float32)
    
    if K is None or dist is None:
        raise Exception("Could not get camera intrinsic parameters")
    
    return K, dist