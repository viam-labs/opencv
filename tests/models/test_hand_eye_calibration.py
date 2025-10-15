"""Unit tests for models.hand_eye_calibration module."""

import os
from PIL import Image
import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.pose_tracker import PoseTracker
from viam.proto.common import Pose, PoseInFrame
from viam.media.video import CameraMimeType

from src.models.hand_eye_calibration import (
    ARM_ATTR,
    CALIB_ATTR,
    CAM_ATTR,
    JOINT_POSITIONS_ATTR,
    METHOD_ATTR,
    MOTION_ATTR,
    POSE_TRACKER_ATTR,
    SLEEP_ATTR,
    HandEyeCalibration
)

@pytest.fixture
def test_config(): 
    return {
        ARM_ATTR: "test_arm",
        CALIB_ATTR: "eye-in-hand",
        CAM_ATTR: "test_camera",
        JOINT_POSITIONS_ATTR: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        METHOD_ATTR: "CALIB_HAND_EYE_TSAI",
        MOTION_ATTR: "test_motion",
        POSE_TRACKER_ATTR: "test_pose_tracker",
        SLEEP_ATTR: 1.0
}

@pytest.fixture
def mock_pose():
    return Pose(
        x=0.1,
        y=0.2,
        z=0.3,
        o_x=0.1,
        o_y=0.1,
        o_z=-0.9,
        theta=89.0
    )

class TestHandEyeCalibrationValidate:
    """Test the validate method"""
    def test_validate_config_success(self, test_config):
        """Test that valid config passes validation"""
        with patch("src.models.hand_eye_calibration.struct_to_dict") as mock_struct:
            mock_struct.return_value = test_config
            required, optional = HandEyeCalibration.validate_config(mock_struct)

            assert "test_arm" in required
            assert "test_camera" in required
            assert "test_pose_tracker" in required
            assert optional == []

    def test_validate_config_missing_arm(self, test_config):
        with patch("src.models.hand_eye_calibration.struct_to_dict") as mock_struct:
            test_config.pop(ARM_ATTR)
            mock_struct.return_value = test_config

            with pytest.raises(Exception, match=f"Missing required {ARM_ATTR} attribute."):
                HandEyeCalibration.validate_config(mock_struct)

    def test_validate_config_missing_camera(self, test_config):
        with patch("src.models.hand_eye_calibration.struct_to_dict") as mock_struct:
            test_config.pop(CAM_ATTR)
            mock_struct.return_value = test_config

            with pytest.raises(Exception, match=f"Missing required {CAM_ATTR} attribute."):
                HandEyeCalibration.validate_config(mock_struct)

    def test_validate_config_missing_pose_tracker(self, test_config):
        with patch("src.models.hand_eye_calibration.struct_to_dict") as mock_struct:
            test_config.pop(POSE_TRACKER_ATTR)
            mock_struct.return_value = test_config

            with pytest.raises(Exception, match=f"Missing required {POSE_TRACKER_ATTR} attribute."):
                HandEyeCalibration.validate_config(mock_struct)


class TestHandEyeCalibrationReconfigure:
    """Test the reconfigure method."""
    def test_successful_reconfigure(self, test_config):
        hand_eye_calibration = HandEyeCalibration('test_hand_eye_calibration')

        config = Mock(spec=ComponentConfig)
        config.attributes = Mock()  
        
        mock_arm = Mock(spec=Arm)
        mock_camera = Mock(spec=Camera)
        mock_pose_tracker = Mock(spec=PoseTracker)
        mock_arm.name = 'test_arm'
        mock_camera.name = 'test_camera'
        mock_pose_tracker.name = 'test_pose_tracker'
        
        dependencies = {
            ResourceName(namespace='rdk', type='component', subtype='arm', name='test_arm'): mock_arm,
            ResourceName(namespace='rdk', type='component', subtype='camera', name='test_camera'): mock_camera,
            ResourceName(namespace='rdk', type='component', subtype='pose_tracker', name='test_pose_tracker'): mock_pose_tracker
        }

        with patch('src.models.hand_eye_calibration.struct_to_dict') as mock_struct_to_dict:
            with patch.object(Arm, 'get_resource_name') as mock_arm_get_resource_name:
                with patch.object(Camera, 'get_resource_name') as mock_camera_get_resource_name:
                    with patch.object(PoseTracker, 'get_resource_name') as mock_pose_tracker_get_resource_name:
                        mock_struct_to_dict.return_value = test_config
                        mock_arm_get_resource_name.return_value = ResourceName(
                            namespace='rdk', type='component', subtype='arm', name='test_arm'
                        )
                        mock_camera_get_resource_name.return_value = ResourceName(
                            namespace='rdk', type='component', subtype='camera', name='test_camera'
                        )
                        mock_pose_tracker_get_resource_name.return_value = ResourceName(
                            namespace='rdk', type='component', subtype='pose_tracker', name='test_pose_tracker'
                        )
                        
                        hand_eye_calibration.reconfigure(config, dependencies)
                        
                        assert hand_eye_calibration.arm == mock_arm
                        assert hand_eye_calibration.camera == mock_camera
                        assert hand_eye_calibration.pose_tracker == mock_pose_tracker

    def test_missing_arm_dependency(self, test_config):
        """Test that missing arm dependency raises exception."""
        hand_eye_calibration = HandEyeCalibration('test_hand_eye_calibration')

        config = Mock(spec=ComponentConfig)
        config.attributes = Mock()  
        
        mock_camera = Mock(spec=Camera)
        mock_pose_tracker = Mock(spec=PoseTracker)
        mock_camera.name = 'test_camera'
        mock_pose_tracker.name = 'test_pose_tracker'
        
        dependencies = {
            ResourceName(namespace='rdk', type='component', subtype='camera', name='test_camera'): mock_camera,
            ResourceName(namespace='rdk', type='component', subtype='pose_tracker', name='test_pose_tracker'): mock_pose_tracker
        }

        with patch('src.models.hand_eye_calibration.struct_to_dict') as mock_struct_to_dict:
            with patch.object(Arm, 'get_resource_name') as mock_arm_get_resource_name:
                with patch.object(Camera, 'get_resource_name') as mock_camera_get_resource_name:
                    with patch.object(PoseTracker, 'get_resource_name') as mock_pose_tracker_get_resource_name:
                        mock_struct_to_dict.return_value = test_config
                        mock_arm_get_resource_name.return_value = ResourceName(
                            namespace='rdk', type='component', subtype='arm', name='test_arm'
                        )
                        mock_camera_get_resource_name.return_value = ResourceName(
                            namespace='rdk', type='component', subtype='camera', name='test_camera'
                        )
                        mock_pose_tracker_get_resource_name.return_value = ResourceName(
                            namespace='rdk', type='component', subtype='pose_tracker', name='test_pose_tracker'
                        )
                        
                        hand_eye_calibration.reconfigure(config, dependencies)

                        assert hand_eye_calibration.arm == None
                        assert hand_eye_calibration.camera == mock_camera
                        assert hand_eye_calibration.pose_tracker == mock_pose_tracker

    def test_missing_camera_dependency(self, test_config):
        """Test that missing camera dependency raises exception."""
        hand_eye_calibration = HandEyeCalibration('test_hand_eye_calibration')

        config = Mock(spec=ComponentConfig)
        config.attributes = Mock()  
        
        mock_arm = Mock(spec=Arm)
        mock_pose_tracker = Mock(spec=PoseTracker)
        mock_arm.name = 'test_arm'
        mock_pose_tracker.name = 'test_pose_tracker'
        dependencies = {
            ResourceName(namespace='rdk', type='component', subtype='arm', name='test_arm'): mock_arm,
            ResourceName(namespace='rdk', type='component', subtype='pose_tracker', name='test_pose_tracker'): mock_pose_tracker
        }

        with patch('src.models.hand_eye_calibration.struct_to_dict') as mock_struct_to_dict:
            with patch.object(Arm, 'get_resource_name') as mock_arm_get_resource_name:
                with patch.object(Camera, 'get_resource_name') as mock_camera_get_resource_name:
                    with patch.object(PoseTracker, 'get_resource_name') as mock_pose_tracker_get_resource_name:
                        mock_struct_to_dict.return_value = test_config
                        mock_arm_get_resource_name.return_value = ResourceName(
                            namespace='rdk', type='component', subtype='arm', name='test_arm'
                        )
                        mock_camera_get_resource_name.return_value = ResourceName(
                            namespace='rdk', type='component', subtype='camera', name='test_camera'
                        )
                        mock_pose_tracker_get_resource_name.return_value = ResourceName(
                            namespace='rdk', type='component', subtype='pose_tracker', name='test_pose_tracker'
                        )
                        
                        hand_eye_calibration.reconfigure(config, dependencies)

                        assert hand_eye_calibration.arm == mock_arm
                        assert hand_eye_calibration.camera == None
                        assert hand_eye_calibration.pose_tracker == mock_pose_tracker

    def test_missing_pose_tracker_dependency(self, test_config):
        """Test that missing pose_tracker dependency raises exception."""
        hand_eye_calibration = HandEyeCalibration('test_hand_eye_calibration')

        config = Mock(spec=ComponentConfig)
        config.attributes = Mock()  
        
        mock_arm = Mock(spec=Arm)
        mock_camera = Mock(spec=Camera)
        mock_arm.name = 'test_arm'
        mock_camera.name = 'test_camera'
        
        dependencies = {
            ResourceName(namespace='rdk', type='component', subtype='arm', name='test_arm'): mock_arm,
            ResourceName(namespace='rdk', type='component', subtype='camera', name='test_camera'): mock_camera,
        }

        with patch('src.models.hand_eye_calibration.struct_to_dict') as mock_struct_to_dict:
            with patch.object(Arm, 'get_resource_name') as mock_arm_get_resource_name:
                with patch.object(Camera, 'get_resource_name') as mock_camera_get_resource_name:
                    with patch.object(PoseTracker, 'get_resource_name') as mock_pose_tracker_get_resource_name:
                        mock_struct_to_dict.return_value = test_config
                        mock_arm_get_resource_name.return_value = ResourceName(
                            namespace='rdk', type='component', subtype='arm', name='test_arm'
                        )
                        mock_camera_get_resource_name.return_value = ResourceName(
                            namespace='rdk', type='component', subtype='camera', name='test_camera'
                        )
                        mock_pose_tracker_get_resource_name.return_value = ResourceName(
                            namespace='rdk', type='component', subtype='pose_tracker', name='test_pose_tracker'
                        )
                        
                        hand_eye_calibration.reconfigure(config, dependencies)

                        assert hand_eye_calibration.arm == mock_arm
                        assert hand_eye_calibration.camera == mock_camera
                        assert hand_eye_calibration.pose_tracker == None


class TestHandEyeCalibrationGetCalibrationValues:
    """Test get_calibration_values"""
    @pytest.mark.asyncio
    async def test_get_calibration_values_success(self, mock_pose):
        hand_eye_calibration = HandEyeCalibration('test_hand_eye_calibration')

        mock_arm = AsyncMock(spec=Arm)
        mock_pose_tracker = AsyncMock(spec=PoseTracker)
        hand_eye_calibration.arm = mock_arm
        hand_eye_calibration.pose_tracker = mock_pose_tracker

        mock_arm.get_end_position.return_value = mock_pose

        mock_pif = PoseInFrame(
            reference_frame="test",
            pose=mock_pose
        )
        mock_poses = {"pose": mock_pif}
        mock_pose_tracker.get_poses.return_value = mock_poses

        expected_R = np.array([
            [-0.71919021, 0.69480796, -0.00270914],
            [ 0.68608837, 0.7107698, 0.15520646],
            [ 0.10976426, 0.10976426, -0.98787834]
        ])
        expected_t = np.array([[0.1], [0.2], [0.3]])

        with patch('src.models.hand_eye_calibration.call_go_ov2mat') as mock_call_go_ov2mat:
            mock_call_go_ov2mat.return_value = expected_R
            
            R_base2gripper, t_base2gripper, R_cam2target, t_cam2target = await hand_eye_calibration.get_calibration_values()
            
            np.testing.assert_array_almost_equal(R_base2gripper, expected_R)
            np.testing.assert_array_almost_equal(t_base2gripper, expected_t)
            np.testing.assert_array_almost_equal(R_cam2target, expected_R)
            np.testing.assert_array_almost_equal(t_cam2target, expected_t)
            mock_call_go_ov2mat.assert_called_with(mock_pose.o_x, mock_pose.o_y, mock_pose.o_z, mock_pose.theta)
    def test_get_calibration_values_incorrect_ov2mat(self):
        pass
    def test_get_calibration_values_no_tracked_bodies(self):
        pass
    def test_get_calibration_values_multiple_tracked_bodies(self):
        pass

class TestHandEyeCalibrationDoCommand:
    def test_do_command_run_calibration_success(self):
        pass
    def test_do_command_run_calibration_not_enough_valid_measurements(self):
        pass
    def test_do_command_run_calibration_successful_calibrate_hand_eye(self):
        pass
    def test_do_command_run_calibration_incorrect_calibrate_hand_eye(self):
        pass
    def test_do_command_run_calibration_correct_response(self):
        pass
    def test_do_command_check_tags_success(self):
        pass
    def test_do_command_check_tags_failure(self):
        pass
    def test_do_command_save_calibration_position_success(self):
        pass
    def test_do_command_save_calibration_position_failure(self):
        pass
    def test_do_command_move_arm_to_position_success(self):
        pass
    def test_do_command_move_arm_to_position_failure(self):
        pass
    def test_do_command_delete_calibration_position_success(self):
        pass
    def test_do_command_delete_calibration_position_success(self):
        pass
    def test_do_command_clear_calibration_positions_success(self):
        pass
    def test_do_command_unsupported_key(self):
        pass
