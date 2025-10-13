"""Unit tests for models.chessboard module."""

import os
from PIL import Image
import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.components.camera import Camera
from viam.media.video import CameraMimeType

from src.models.chessboard import Chessboard


class TestValidateChessboard:
    @pytest.fixture
    def mock_camera(self):
        """Create a realistic Camera mock"""
        camera = AsyncMock(spec=Camera)
        camera.name = 'test_camera'

        camera.get_images.return_value = [
            [Mock( 
                mime_type=CameraMimeType.JPEG,
                data=b'fake_jpeg_data'
            )]
        ]

        camera.do_command.return_value = {
            'Color': {
                'intrinsics': {'fx': 800, 'fy': 800, 'cx': 320, 'cy': 240},
                'distortion': {'k1': 0.1, 'k2': 0.05, 'p1': 0.001, 'p2': 0.002, 'k3': 0.01}
            }
        }
        return camera

    @pytest.fixture 
    def mock_component_config(self):
        """Mock Viam ComponentConfig"""
        config = Mock(spec=ComponentConfig)
        with patch('src.models.chessboard.struct_to_dict') as mock_struct:
            mock_struct.return_value = {
                'camera_name': 'test_camera',
                'pattern_size': [9, 6],
                'square_size_mm': 25.0
            }
            return config
        
    def test_validate_config_success(self, mock_component_config):
        """Test that valid config passes validation"""
        with patch('src.models.chessboard.struct_to_dict') as mock_struct:
            mock_struct.return_value = {
                'camera_name': 'test_camera',
                'pattern_size': [9, 6],
                'square_size_mm': 25.0
            }
            required, optional = Chessboard.validate_config(mock_struct)
            assert 'test_camera' in required
            assert optional == []

    def test_validate_config_missing_camera(self):
        """Test that missing camera raises exception"""
        with patch('src.models.chessboard.struct_to_dict') as mock_struct:
            mock_struct.return_value = {
                'pattern_size': [9, 6],
                'square_size_mm': 25.0
            }

            with pytest.raises(Exception, match="Missing required camera_name attribute."):
                Chessboard.validate_config(mock_struct)

    def test_validate_config_missing_pattern(self):
        """Test that missing pattern size raises exception"""
        with patch('src.models.chessboard.struct_to_dict') as mock_struct:
            mock_struct.return_value = {
                'camera_name': "camera",
                'square_size_mm': 25.0
            }

            with pytest.raises(Exception, match="Missing required pattern_size attribute."):
                Chessboard.validate_config(mock_struct)

    def test_validate_config_missing_square_size(self):
        """Test that missing square size mm raises exception"""
        with patch('src.models.chessboard.struct_to_dict') as mock_struct:
            mock_struct.return_value = {
                'camera_name': "camera",
                'pattern_size': [9, 6]
            }

            with pytest.raises(Exception, match="Missing required square_size_mm attribute."):
                Chessboard.validate_config(mock_struct)


class TestChessboardReconfigure:
    """Test the reconfigure method."""
    
    def test_successful_reconfigure(self):
        """Test successful reconfiguration with valid dependencies."""
        chessboard = Chessboard('test_chessboard')
        
        config = Mock(spec=ComponentConfig)
        config.attributes = Mock()  
        
        mock_camera = Mock(spec=Camera)
        mock_camera.name = 'test_camera'
        
        dependencies = {
            ResourceName(namespace='rdk', type='component', subtype='camera', name='test_camera'): mock_camera
        }
        
        with patch('src.models.chessboard.struct_to_dict') as mock_struct_to_dict:
            with patch.object(Camera, 'get_resource_name') as mock_get_resource_name:
                mock_struct_to_dict.return_value = {
                    'camera_name': 'test_camera',
                    'pattern_size': [9, 6],
                    'square_size_mm': 25.0
                }
                mock_get_resource_name.return_value = ResourceName(
                    namespace='rdk', type='component', subtype='camera', name='test_camera'
                )
                
                chessboard.reconfigure(config, dependencies)
                
                assert chessboard.camera == mock_camera
                assert chessboard.pattern_size == [9, 6]
                assert chessboard.square_size == 25.0
    
    def test_missing_camera_dependency(self):
        """Test that missing camera dependency raises exception."""
        chessboard = Chessboard('test_chessboard')
        
        config = Mock(spec=ComponentConfig)
        config.attributes = Mock() 
        dependencies = {}  
        
        with patch('src.models.chessboard.struct_to_dict') as mock_struct_to_dict:
            with patch.object(Camera, 'get_resource_name') as mock_get_resource_name:
                mock_struct_to_dict.return_value = {
                    'camera_name': 'test_camera',
                    'pattern_size': [9, 6],
                    'square_size_mm': 25.0
                }
                mock_get_resource_name.return_value = ResourceName(
                    namespace='rdk', type='component', subtype='camera', name='test_camera'  
                )
                
                with pytest.raises(Exception, match="Could not find camera resource"):
                    chessboard.reconfigure(config, dependencies)


class TestChessboardGetCameraIntrinsics:
    """Test the get_camera_intrinsics async method."""
    
    @pytest.mark.asyncio
    async def test_successful_intrinsics_retrieval(self):
        """Test successful retrieval of camera intrinsics."""
        chessboard = Chessboard('test_chessboard')
        mock_camera = AsyncMock(spec=Camera)
        chessboard.camera = mock_camera
        
        mock_camera.do_command.return_value = {
            'Color': {
                'intrinsics': {
                    'fx': 800.0,
                    'fy': 800.0, 
                    'cx': 320.0,
                    'cy': 240.0
                },
                'distortion': {
                    'k1': 0.1,
                    'k2': 0.05,
                    'p1': 0.001,
                    'p2': 0.002,
                    'k3': 0.01
                }
            }
        }
        
        K, dist = await chessboard.get_camera_intrinsics()
        
        assert K.shape == (3, 3)
        assert K.dtype == np.float32
        assert K[0, 0] == 800.0 
        assert K[1, 1] == 800.0 
        assert K[0, 2] == 320.0 
        assert K[1, 2] == 240.0 
        
        assert dist.shape == (5,)
        assert dist.dtype == np.float32
        expected_dist = np.array([0.1, 0.05, 0.001, 0.002, 0.01], dtype=np.float32)
        np.testing.assert_array_equal(dist, expected_dist)
        
        mock_camera.do_command.assert_called_once_with({"get_camera_params": None})


class TestChessboardGetPoses:
    """Test the get_poses async method - the main functionality."""
    
    @pytest.mark.asyncio
    async def test_successful_chessboard_detection(self):
        """Test successful chessboard detection and pose estimation using real chessboard image."""
        chessboard = Chessboard('test_chessboard')
        chessboard.pattern_size = [8, 11]
        chessboard.square_size = 25.0
        
        mock_camera = AsyncMock(spec=Camera)
        chessboard.camera = mock_camera
        mock_camera.name = 'test_camera'
        
        test_image_path = os.path.join(os.path.dirname(__file__), '..', 'chessboard_1.jpeg')
        real_chessboard_image = Image.open(test_image_path)
        
        mock_camera_image = Mock()
        mock_camera_image.mime_type = CameraMimeType.JPEG
        mock_camera_image.data = b'fake_jpeg_data'
        mock_camera.get_images.return_value = [[mock_camera_image]]
        
        with patch.object(chessboard, 'get_camera_intrinsics') as mock_get_intrinsics:
            with patch('src.models.chessboard.call_go_mat2ov') as mock_go_call:
                with patch('src.models.chessboard.viam_to_pil_image') as mock_viam_to_pil:
                    mock_viam_to_pil.return_value = real_chessboard_image
                    mock_get_intrinsics.return_value = (
                        np.eye(3, dtype=np.float32) * 800,
                        np.zeros(5, dtype=np.float32)    
                    )
                    mock_go_call.return_value = (0.0, 0.0, 1.0, 90.0)
                    
                    result = await chessboard.get_poses(['pose'])
                    
                    assert 'pose' in result
                    pose_in_frame = result['pose']
                    assert pose_in_frame.reference_frame == 'test_camera'
                    assert pose_in_frame.pose is not None
                    
                    assert pose_in_frame.pose.o_x == 0.0
                    assert pose_in_frame.pose.o_y == 0.0
                    assert pose_in_frame.pose.o_z == 1.0
                    assert pose_in_frame.pose.theta == 90.0
                    
                    mock_viam_to_pil.assert_called_once_with(mock_camera_image)
                    mock_get_intrinsics.assert_called_once()
                    mock_go_call.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_no_chessboard_found(self):
        """Test handling when no chessboard is found in image due to incorrect pattern size."""
        chessboard = Chessboard('test_chessboard')
        chessboard.pattern_size = [10, 11] # incorrect pattern size
        chessboard.square_size = 25.0
        
        mock_camera = AsyncMock(spec=Camera)
        chessboard.camera = mock_camera
        mock_camera.name = 'test_camera'
        
        test_image_path = os.path.join(os.path.dirname(__file__), '..', 'chessboard_1.jpeg')
        real_chessboard_image = Image.open(test_image_path)
        
        mock_camera_image = Mock()
        mock_camera_image.mime_type = CameraMimeType.JPEG
        mock_camera_image.data = b'fake_jpeg_data'
        mock_camera.get_images.return_value = [[mock_camera_image]]
        
        with patch.object(chessboard, 'get_camera_intrinsics') as mock_get_intrinsics:
            with patch('src.models.chessboard.call_go_mat2ov') as mock_go_call:
                with patch('src.models.chessboard.viam_to_pil_image') as mock_viam_to_pil:
                    mock_viam_to_pil.return_value = real_chessboard_image
                    mock_get_intrinsics.return_value = (
                        np.eye(3, dtype=np.float32) * 800,
                        np.zeros(5, dtype=np.float32)    
                    )
                    mock_go_call.return_value = (0.0, 0.0, 1.0, 90.0)
                    
                    with pytest.raises(Exception, match="Could not find chessboard pattern in image"):
                        await chessboard.get_poses(['pose'])
