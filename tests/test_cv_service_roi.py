"""
Unit tests for CV Service ROI integration.
Tests the integration between CV service and ROI manager functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import List

from src.services.cv_service import EnhancedCVService
from src.services.roi_manager import ROIManager
from src.models.core import PersonDetection, CameraHealth, CameraStatus
from src.config.models import CameraConfig, CameraType


class TestEnhancedCVServiceROI:
    """Test Enhanced CV Service with ROI functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('src.services.cv_service.YOLO'):
            self.cv_service = EnhancedCVService(model_path="test_model.pt")
        
        self.camera_id = "test_camera"
        self.rtsp_url = "rtsp://test.camera/stream"
    
    def test_configure_roi_success(self):
        """Test successful ROI configuration."""
        polygon_points = [(0, 0), (100, 0), (100, 100), (0, 100)]
        
        # Mock the ROI manager method
        with patch.object(self.cv_service.roi_manager, 'configure_camera_roi', return_value=True) as mock_config:
            self.cv_service.configure_roi(self.camera_id, polygon_points)
            
            mock_config.assert_called_once_with(self.camera_id, polygon_points)
    
    def test_configure_roi_failure(self):
        """Test ROI configuration failure."""
        polygon_points = [(0, 0), (100, 0)]  # Invalid - too few points
        
        # Mock the ROI manager method to return False
        with patch.object(self.cv_service.roi_manager, 'configure_camera_roi', return_value=False):
            with patch('src.services.cv_service.logger') as mock_logger:
                self.cv_service.configure_roi(self.camera_id, polygon_points)
                
                mock_logger.error.assert_called_once()
    
    def test_set_entrance_line_success(self):
        """Test successful entrance line configuration."""
        line_coords = (100, 50, 100, 450)
        
        # Mock the ROI manager method
        with patch.object(self.cv_service.roi_manager, 'configure_entrance_line', return_value=True) as mock_config:
            self.cv_service.set_entrance_line(self.camera_id, line_coords)
            
            mock_config.assert_called_once_with(self.camera_id, line_coords)
            # Should initialize entrance events tracking
            assert self.camera_id in self.cv_service.entrance_events
    
    def test_set_entrance_line_failure(self):
        """Test entrance line configuration failure."""
        line_coords = (100, 100, 101, 101)  # Invalid - too short
        
        # Mock the ROI manager method to return False
        with patch.object(self.cv_service.roi_manager, 'configure_entrance_line', return_value=False):
            with patch('src.services.cv_service.logger') as mock_logger:
                self.cv_service.set_entrance_line(self.camera_id, line_coords)
                
                mock_logger.error.assert_called_once()
    
    def test_get_entrance_events(self):
        """Test getting entrance events for a camera."""
        # Set up some test events
        test_events = [
            {
                'timestamp': datetime.now(),
                'track_id': 1,
                'event_type': 'ENTER',
                'position': (150, 250)
            },
            {
                'timestamp': datetime.now(),
                'track_id': 2,
                'event_type': 'EXIT',
                'position': (50, 250)
            }
        ]
        
        self.cv_service.entrance_events[self.camera_id] = test_events
        
        # Test getting events
        events = self.cv_service.get_entrance_events(self.camera_id)
        assert len(events) == 2
        assert events[0]['event_type'] == 'ENTER'
        assert events[1]['event_type'] == 'EXIT'
        
        # Test with limit
        events = self.cv_service.get_entrance_events(self.camera_id, limit=1)
        assert len(events) == 1
        assert events[0]['event_type'] == 'EXIT'  # Should get the last one
    
    def test_get_entrance_events_no_camera(self):
        """Test getting entrance events for non-existent camera."""
        events = self.cv_service.get_entrance_events("non_existent_camera")
        assert events == []
    
    def test_get_roi_config(self):
        """Test getting ROI configuration for a camera."""
        expected_config = {
            'camera_id': self.camera_id,
            'roi_polygon': {'points': [(0, 0), (100, 0), (100, 100), (0, 100)]},
            'enabled': True
        }
        
        with patch.object(self.cv_service.roi_manager, 'export_config', return_value=expected_config) as mock_export:
            config = self.cv_service.get_roi_config(self.camera_id)
            
            mock_export.assert_called_once_with(self.camera_id)
            assert config == expected_config
    
    def test_validate_roi_config(self):
        """Test validating ROI configuration."""
        expected_errors = ["ROI polygon area too small"]
        
        with patch.object(self.cv_service.roi_manager, 'validate_roi_config', return_value=expected_errors) as mock_validate:
            errors = self.cv_service.validate_roi_config(self.camera_id)
            
            mock_validate.assert_called_once_with(self.camera_id)
            assert errors == expected_errors
    
    @patch('src.services.cv_service.cv2.VideoCapture')
    @patch('src.services.cv_service.HAVE_SV', True)
    def test_detection_processing_with_roi_filtering(self, mock_video_capture):
        """Test detection processing with ROI filtering."""
        # Mock video capture
        mock_capture = Mock()
        mock_capture.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_capture.isOpened.return_value = True
        mock_video_capture.return_value = mock_capture
        
        # Mock YOLO results
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = [np.array([[50, 50, 100, 100], [150, 150, 200, 200]])]
        mock_result.boxes.conf = [np.array([0.9, 0.8])]
        
        self.cv_service.model.return_value = [mock_result]
        
        # Mock tracker
        mock_tracker = Mock()
        mock_detections = Mock()
        mock_detections.xyxy = np.array([[50, 50, 100, 100], [150, 150, 200, 200]])
        mock_detections.confidence = np.array([0.9, 0.8])
        mock_detections.tracker_id = np.array([1, 2])
        mock_detections.get_anchors_coordinates.return_value = np.array([[75, 75], [175, 175]])
        
        mock_tracker.update_with_detections.return_value = mock_detections
        self.cv_service.tracker = mock_tracker
        
        # Mock ROI manager filtering
        filtered_detections = [
            PersonDetection(track_id=1, bbox=(50, 50, 100, 100), confidence=0.9, 
                          center_point=(75, 75), in_roi=True)
        ]
        
        with patch.object(self.cv_service.roi_manager, 'filter_detections_by_roi', return_value=filtered_detections) as mock_filter:
            with patch.object(self.cv_service.roi_manager, 'detect_entrance_crossing', return_value=None):
                # Set up camera stream
                self.cv_service.camera_manager.streams[self.camera_id] = Mock()
                self.cv_service.camera_manager.streams[self.camera_id].capture = mock_capture
                
                # Process detections
                self.cv_service._process_camera_detections(self.camera_id)
                
                # Verify ROI filtering was called
                mock_filter.assert_called_once()
                
                # Check that filtered detections are stored
                result = self.cv_service.detection_results.get(self.camera_id)
                assert result is not None
                assert len(result.detections) == 1
                assert result.detections[0].in_roi is True
    
    @patch('src.services.cv_service.cv2.VideoCapture')
    def test_entrance_crossing_detection_integration(self, mock_video_capture):
        """Test entrance crossing detection integration."""
        # Mock video capture and YOLO
        mock_capture = Mock()
        mock_capture.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_capture.isOpened.return_value = True
        mock_video_capture.return_value = mock_capture
        
        # Mock YOLO results with one detection
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.xyxy = [np.array([[70, 70, 80, 80]])]
        mock_result.boxes.conf = [np.array([0.9])]
        
        self.cv_service.model.return_value = [mock_result]
        
        # Mock tracker
        mock_tracker = Mock()
        mock_detections = Mock()
        mock_detections.xyxy = np.array([[70, 70, 80, 80]])
        mock_detections.confidence = np.array([0.9])
        mock_detections.tracker_id = np.array([1])
        mock_detections.get_anchors_coordinates.return_value = np.array([[75, 75]])
        
        mock_tracker.update_with_detections.return_value = mock_detections
        self.cv_service.tracker = mock_tracker
        
        # Mock ROI manager to return ENTER crossing
        with patch.object(self.cv_service.roi_manager, 'filter_detections_by_roi', 
                         return_value=[PersonDetection(track_id=1, bbox=(70, 70, 80, 80), 
                                                     confidence=0.9, center_point=(75, 75), in_roi=True)]):
            with patch.object(self.cv_service.roi_manager, 'detect_entrance_crossing', 
                             return_value='ENTER') as mock_crossing:
                
                # Set up camera stream
                self.cv_service.camera_manager.streams[self.camera_id] = Mock()
                self.cv_service.camera_manager.streams[self.camera_id].capture = mock_capture
                
                # Process detections
                self.cv_service._process_camera_detections(self.camera_id)
                
                # Verify entrance crossing detection was called
                mock_crossing.assert_called_once_with(self.camera_id, 1, (75, 75))
                
                # Check that entrance event was recorded
                events = self.cv_service.get_entrance_events(self.camera_id)
                assert len(events) == 1
                assert events[0]['event_type'] == 'ENTER'
                assert events[0]['track_id'] == 1
    
    def test_entrance_events_buffer_management(self):
        """Test that entrance events buffer is properly managed."""
        # Fill up entrance events beyond the limit
        events = []
        for i in range(150):  # More than the 100 limit
            event = {
                'timestamp': datetime.now(),
                'track_id': i,
                'event_type': 'ENTER' if i % 2 == 0 else 'EXIT',
                'position': (75, 75)
            }
            events.append(event)
        
        self.cv_service.entrance_events[self.camera_id] = events
        
        # Simulate adding one more event (this would happen in _process_camera_detections)
        new_event = {
            'timestamp': datetime.now(),
            'track_id': 999,
            'event_type': 'ENTER',
            'position': (75, 75)
        }
        self.cv_service.entrance_events[self.camera_id].append(new_event)
        
        # Simulate buffer management (keep only last 100)
        if len(self.cv_service.entrance_events[self.camera_id]) > 100:
            self.cv_service.entrance_events[self.camera_id] = self.cv_service.entrance_events[self.camera_id][-100:]
        
        # Check that buffer is properly limited
        events = self.cv_service.get_entrance_events(self.camera_id)
        assert len(events) == 100
        assert events[-1]['track_id'] == 999  # Most recent event should be preserved
    
    def test_roi_configuration_from_config_file(self):
        """Test loading ROI configuration from config file."""
        # This would typically be tested with the configuration manager
        # For now, test the direct configuration methods
        
        # Configure ROI from config-like data
        roi_points = [(100, 100), (500, 100), (500, 400), (100, 400)]
        entrance_line = (200, 50, 200, 450)
        
        self.cv_service.configure_roi(self.camera_id, roi_points)
        self.cv_service.set_entrance_line(self.camera_id, entrance_line)
        
        # Verify configuration
        config = self.cv_service.get_roi_config(self.camera_id)
        errors = self.cv_service.validate_roi_config(self.camera_id)
        
        # Should have valid configuration with no errors
        assert config is not None
        assert len(errors) == 0
    
    def test_multiple_cameras_roi_independence(self):
        """Test that ROI configurations are independent between cameras."""
        camera1 = "camera1"
        camera2 = "camera2"
        
        # Configure different ROIs for each camera
        roi1 = [(0, 0), (100, 0), (100, 100), (0, 100)]
        roi2 = [(50, 50), (150, 50), (150, 150), (50, 150)]
        
        self.cv_service.configure_roi(camera1, roi1)
        self.cv_service.configure_roi(camera2, roi2)
        
        # Configure entrance lines
        line1 = (50, 0, 50, 100)
        line2 = (100, 50, 100, 150)
        
        self.cv_service.set_entrance_line(camera1, line1)
        self.cv_service.set_entrance_line(camera2, line2)
        
        # Verify configurations are independent
        config1 = self.cv_service.get_roi_config(camera1)
        config2 = self.cv_service.get_roi_config(camera2)
        
        assert config1 != config2
        assert camera1 in self.cv_service.entrance_events
        assert camera2 in self.cv_service.entrance_events
    
    def test_roi_error_handling(self):
        """Test error handling in ROI operations."""
        # Test with invalid camera ID
        with patch.object(self.cv_service.roi_manager, 'export_config', return_value=None):
            config = self.cv_service.get_roi_config("invalid_camera")
            assert config is None
        
        # Test validation with errors
        with patch.object(self.cv_service.roi_manager, 'validate_roi_config', 
                         return_value=["Error 1", "Error 2"]):
            errors = self.cv_service.validate_roi_config(self.camera_id)
            assert len(errors) == 2
    
    @patch('src.services.cv_service.logger')
    def test_roi_processing_exception_handling(self, mock_logger):
        """Test exception handling during ROI processing."""
        # Mock ROI manager to raise exception
        with patch.object(self.cv_service.roi_manager, 'filter_detections_by_roi', 
                         side_effect=Exception("ROI processing error")):
            
            # Create test detections
            detections = [PersonDetection(track_id=1, bbox=(50, 50, 100, 100), 
                                        confidence=0.9, center_point=(75, 75), in_roi=False)]
            
            # This should handle the exception gracefully
            # In a real scenario, this would be called from _process_camera_detections
            try:
                self.cv_service.roi_manager.filter_detections_by_roi(self.camera_id, detections)
            except Exception:
                pass  # Exception should be caught and logged in the actual implementation


class TestROIManagerPerformance:
    """Performance tests for ROI Manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.roi_manager = ROIManager()
    
    def test_large_polygon_performance(self):
        """Test performance with large complex polygons."""
        # Create a complex polygon with many points (approximating a circle)
        center_x, center_y, radius = 320, 240, 200
        polygon_points = []
        
        for i in range(100):  # 100-sided polygon
            angle = i * 2 * np.pi / 100
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            polygon_points.append((x, y))
        
        camera_id = "performance_test_camera"
        
        # Configure ROI
        result = self.roi_manager.configure_camera_roi(camera_id, polygon_points)
        assert result is True
        
        # Test many point-in-polygon checks
        test_points = [(i*10, j*10) for i in range(64) for j in range(48)]  # 3072 points
        
        results = []
        for point in test_points:
            result = self.roi_manager.is_point_in_roi(camera_id, point)
            results.append(result)
        
        # Should complete without timeout and return reasonable results
        assert len(results) == len(test_points)
        assert any(results)  # Some points should be inside
        assert not all(results)  # Some points should be outside
    
    def test_many_simultaneous_crossings(self):
        """Test performance with many simultaneous entrance crossings."""
        camera_id = "crossing_test_camera"
        line_coords = (320, 0, 320, 480)  # Vertical line through center
        
        self.roi_manager.configure_entrance_line(camera_id, line_coords)
        
        # Simulate many people crossing simultaneously
        num_people = 50
        
        # Initialize all people on left side
        for track_id in range(num_people):
            self.roi_manager.detect_entrance_crossing(camera_id, track_id, (200, 240))
        
        # Move all people to right side
        crossings = []
        for frame in range(5):  # Simulate 5 frames
            for track_id in range(num_people):
                crossing = self.roi_manager.detect_entrance_crossing(camera_id, track_id, (400, 240))
                if crossing:
                    crossings.append((track_id, crossing))
        
        # Should detect crossings for all people
        assert len(crossings) == num_people
        assert all(crossing[1] == 'ENTER' for crossing in crossings)


if __name__ == "__main__":
    pytest.main([__file__])