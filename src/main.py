import asyncio
from viam.module.module import Module
try:
    from models.camera_calibration import CameraCalibration
    from models.chessboard import Chessboard
    from models.hand_eye_calibration import HandEyeCalibration
except ModuleNotFoundError:
    # when running as local module with run.sh
    from .models.camera_calibration import CameraCalibration
    from .models.chessboard import Chessboard
    from .models.hand_eye_calibration import HandEyeCalibration


if __name__ == '__main__':
    asyncio.run(Module.run_from_registry())
