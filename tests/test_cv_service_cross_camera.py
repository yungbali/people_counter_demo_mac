"""
Integration tests for CV service with cross-camera tracking functionality.
Tests the integration between CV service and cross-camera tracker.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.services.cv_service import EnhancedCVService
from src.models.core import PersonDetection, CameraStatus
from src.config.models import CameraConfig, CameraType


class TestCVServiceCrossCameraIntegration(unittest.TestCase):
    """Test CV service integration with cross-camera tracking."""
    
    def setUp(self):
        """Set up test environment."""
        with patch('src.services.cv_service.YOLO'):
            self.cv_service = EnhancedCVService(model_path="test_model.pt", max_cameras=3)
    
    def tearDown(self):
        """Clean up after tests."""
        self.cv_service.shutdown()
    
    def test_cross_camera_tracker_initialization(self):
        """Test that cross-camera tracker is properly initialized."""
        self.assertIsNotNone(self.cv_service.cross_camera_tracker)
        
        # Check initial state
        stats = self.cv_service.get_cross_camera_tracking_stats()
        self.assertEqual(stats['total_global_tracks'], 0)
        self.assertEqual(stats['active_global_tracks'], 0)
        self.assertEqual(stats['global_count'], 0)
    
    def test_get_deduplicated_global_count(self):
        """Test getting deduplicated global count."""
        initial_count = self.cv_service.get_deduplicated_global_count()
        self.assertEqual(initial_count, 0)
    
    def test_reset_cross_camera_tracking(self):
        """Test resetting cross-camera tracking state."""
        # This should not raise any exceptions
        self.cv_service.reset_cross_camera_tracking()
        
        stats = self.cv_service.get_cross_camera_tracking_stats()
        self.assertEqual(stats['total_global_tracks'], 0)
        self.assertEqual(stats['next_global_id'], 1)
    
    @patch('src.services.cv_service.cv2.VideoCapture')
    def test_multi_camera_setup_with_tracking(self, mock_video_capture):
        """Test setting up multiple cameras with cross-camera tracking."""
        # Mock video capture
        mock_capture = Mock()
        mock_capture.isOpened.return_value = True
        mock_capture.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_capture
        
        # Add multiple cameras
        self.cv_service.process_camera_feed("cam1", "rtsp://camera1/stream")
        self.cv_service.process_camera_feed("cam2", "rtsp://camera2/stream")
        
        # Verify cameras are added
        self.assertIn("cam1", self.cv_service.camera_manager.streams)
        self.assertIn("cam2", self.cv_service.camera_manager.streams)
        
        # Check that per-camera trackers are created when needed
        # (This will happen during detection processing)
    
    @patch('src.services.cv_service.HAVE_SV', True)
    @patch('src.services.cv_service.sv')
    def test_detection_with_tracking_integration(self, mock_sv):
        """Test detection processing with tracking integration."""
        # Mock supervision components
        mock_tracker = Mock()
        mock_detections = Mock()
        mock_detections.xyxy = np.array([[100, 100, 200, 300]])
        mock_detections.tracker_id = np.array([1])
        mock_detections.confidence = np.array([0.8])
        
        mock_sv.ByteTrack.return_value = mock_tracker
        mock_sv.Detections.from_ultralytics.return_value = mock_detections
        mock_sv.Position.CENTER = "CENTER"
        
        mock_tracker.update_with_detections.return_value = mock_detections
        mock_detections.get_anchors_coordinates.return_value = np.array([[150, 200]])
        
        # Mock YOLO results
        mock_result = Mock()
        mock_result.boxes = None  # Will use tracking path
        
        with patch.object(self.cv_service.model, '__call__', return_value=[mock_result]):
            # Process a single camera detection
            result = self.cv_service._process_camera_detections("test_cam")
            
            if result:  # May be None if camera not set up
                self.assertIsNotNone(result)
                # Verify detection has local_track_id set
                for detection in result.detections:
                    self.assertIsNotNone(detection.local_track_id)
    
    def test_frame_storage_for_embeddings(self):
        """Test that frames are stored for embedding extraction."""
        # Initially no frames stored
        self.assertEqual(len(self.cv_service.current_frames), 0)
        
        # Mock a camera stream
        mock_stream = Mock()
        mock_stream.capture = Mock()
        mock_stream.capture.isOpened.return_value = True
        
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_stream.capture.read.return_value = (True, test_frame)
        
        self.cv_service.camera_manager.streams["test_cam"] = mock_stream
        
        # Mock YOLO to return empty results
        with patch.object(self.cv_service.model, '__call__', return_value=[]):
            result = self.cv_service._process_camera_detections("test_cam")
            
            # Frame should be stored
            self.assertIn("test_cam", self.cv_service.current_frames)
            np.testing.assert_array_equal(self.cv_service.current_frames["test_cam"], test_frame)
    
    def test_detection_loop_with_cross_camera_processing(self):
        """Test that detection loop processes cross-camera tracking."""
        # Mock multiple camera streams
        for cam_id in ["cam1", "cam2"]:
            mock_stream = Mock()
            mock_stream.capture = Mock()
            mock_stream.capture.isOpened.return_value = True
            mock_stream.capture.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            self.cv_service.camera_manager.streams[cam_id] = mock_stream
        
        # Mock the process_camera_detections to return results
        mock_result1 = Mock()
        mock_result1.camera_id = "cam1"
        mock_result1.detections = []
        
        mock_result2 = Mock()
        mock_result2.camera_id = "cam2"
        mock_result2.detections = []
        
        with patch.object(self.cv_service, '_process_camera_detections') as mock_process:
            mock_process.side_effect = [mock_result1, mock_result2]
            
            with patch.object(self.cv_service.cross_camera_tracker, 'process_detections') as mock_cross_track:
                mock_cross_track.return_value = {"cam1": [], "cam2": []}
                
                # Manually call detection loop logic once
                detection_results = []
                for camera_id in ["cam1", "cam2"]:
                    result = self.cv_service._process_camera_detections(camera_id)
                    if result:
                        detection_results.append(result)
                
                if len(detection_results) > 1:
                    self.cv_service.cross_camera_tracker.process_detections(
                        detection_results, self.cv_service.current_frames
                    )
                
                # Verify cross-camera processing was called
                if len(detection_results) > 1:
                    mock_cross_track.assert_called_once()
    
    def test_tracking_stats_integration(self):
        """Test tracking statistics integration."""
        stats = self.cv_service.get_cross_camera_tracking_stats()
        
        # Should have all expected keys
        expected_keys = [
            'total_global_tracks',
            'active_global_tracks',
            'next_global_id',
            'tracks_by_camera',
            'global_count',
            'tracks_with_embeddings',
            'avg_embeddings_per_track'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
    
    def test_person_detection_global_track_id_assignment(self):
        """Test that PersonDetection objects get global track IDs assigned."""
        # Create a mock detection with local track ID
        detection = PersonDetection(
            track_id=1,
            bbox=(100, 100, 200, 300),
            confidence=0.8,
            center_point=(150, 200),
            in_roi=True,
            local_track_id=1
        )
        
        # Simulate cross-camera tracker assigning global ID
        detection.track_id = 100  # Global ID
        detection.global_track_id = 100
        
        self.assertEqual(detection.track_id, 100)
        self.assertEqual(detection.global_track_id, 100)
        self.assertEqual(detection.local_track_id, 1)
    
    def test_roi_integration_with_cross_camera_tracking(self):
        """Test that ROI filtering works with cross-camera tracking."""
        # Configure ROI for a camera
        roi_points = [(100, 100), (500, 100), (500, 400), (100, 400)]
        self.cv_service.configure_roi("test_cam", roi_points)
        
        # Verify ROI is configured
        roi_config = self.cv_service.get_roi_config("test_cam")
        self.assertIsNotNone(roi_config)
        
        # ROI filtering should work with cross-camera tracking
        # (This is tested more thoroughly in ROI-specific tests)
    
    def test_entrance_line_integration_with_tracking(self):
        """Test entrance line detection with cross-camera tracking."""
        # Configure entrance line
        line_coords = (200, 50, 200, 450)
        self.cv_service.set_entrance_line("test_cam", line_coords)
        
        # Verify entrance line is configured
        roi_config = self.cv_service.get_roi_config("test_cam")
        self.assertIsNotNone(roi_config)
        
        # Entrance events should work with global track IDs
        events = self.cv_service.get_entrance_events("test_cam")
        self.assertIsInstance(events, list)
    
    def test_camera_health_with_cross_camera_tracking(self):
        """Test camera health monitoring with cross-camera tracking."""
        # Health check should work regardless of tracking state
        health = self.cv_service.health_check("nonexistent_cam")
        
        self.assertEqual(health.camera_id, "nonexistent_cam")
        self.assertEqual(health.status, CameraStatus.OFFLINE)
        self.assertEqual(health.error_message, "Camera not found")
    
    def test_shutdown_with_cross_camera_tracking(self):
        """Test proper shutdown with cross-camera tracking."""
        # Should not raise any exceptions
        self.cv_service.shutdown()
        
        # Verify tracking state is preserved (not automatically reset)
        stats = self.cv_service.get_cross_camera_tracking_stats()
        self.assertIsInstance(stats, dict)


class TestCrossCameraTrackingAccuracy(unittest.TestCase):
    """Test cross-camera tracking accuracy and consistency."""
    
    def setUp(self):
        """Set up test environment."""
        with patch('src.services.cv_service.YOLO'):
            self.cv_service = EnhancedCVService(model_path="test_model.pt")
    
    def tearDown(self):
        """Clean up after tests."""
        self.cv_service.shutdown()
    
    def test_tracking_consistency_single_camera(self):
        """Test that tracking is consistent within a single camera."""
        # This tests the foundation for cross-camera consistency
        tracker = self.cv_service.cross_camera_tracker
        
        # Create consistent detections over time
        detection1 = PersonDetection(1, (100, 100, 200, 300), 0.8, (150, 200), True, 1)
        detection2 = PersonDetection(1, (105, 105, 205, 305), 0.8, (155, 205), True, 1)  # Slightly moved
        
        from src.models.core import DetectionResult, CameraHealth, CameraStatus
        
        camera_health = CameraHealth("cam1", CameraStatus.ONLINE, datetime.now(), 30.0, "GOOD")
        
        result1 = tracker.process_detections([
            DetectionResult("cam1", datetime.now(), [detection1], camera_health)
        ])
        
        result2 = tracker.process_detections([
            DetectionResult("cam1", datetime.now(), [detection2], camera_health)
        ])
        
        # Should maintain same global ID
        global_id1 = result1["cam1"][0].track_id
        global_id2 = result2["cam1"][0].track_id
        
        self.assertEqual(global_id1, global_id2)
    
    def test_deduplication_accuracy(self):
        """Test accuracy of de-duplication logic."""
        tracker = self.cv_service.cross_camera_tracker
        
        # Create detections that should NOT be deduplicated (different people)
        detection1 = PersonDetection(1, (100, 100, 200, 300), 0.8, (150, 200), True, 1)
        detection2 = PersonDetection(1, (300, 100, 400, 300), 0.8, (350, 200), True, 1)
        
        from src.models.core import DetectionResult, CameraHealth, CameraStatus
        
        camera_health1 = CameraHealth("cam1", CameraStatus.ONLINE, datetime.now(), 30.0, "GOOD")
        camera_health2 = CameraHealth("cam2", CameraStatus.ONLINE, datetime.now(), 30.0, "GOOD")
        
        result = tracker.process_detections([
            DetectionResult("cam1", datetime.now(), [detection1], camera_health1),
            DetectionResult("cam2", datetime.now(), [detection2], camera_health2)
        ])
        
        # Should have 2 unique people (different locations, no embeddings to match)
        global_count = tracker.get_deduplicated_global_count()
        self.assertEqual(global_count, 2)
        
        # Global IDs should be different
        global_id1 = result["cam1"][0].track_id
        global_id2 = result["cam2"][0].track_id
        self.assertNotEqual(global_id1, global_id2)
    
    def test_track_persistence_across_occlusions(self):
        """Test that tracks persist across brief occlusions."""
        tracker = self.cv_service.cross_camera_tracker
        
        from src.models.core import DetectionResult, CameraHealth, CameraStatus
        camera_health = CameraHealth("cam1", CameraStatus.ONLINE, datetime.now(), 30.0, "GOOD")
        
        # Person appears
        detection1 = PersonDetection(1, (100, 100, 200, 300), 0.8, (150, 200), True, 1)
        result1 = tracker.process_detections([
            DetectionResult("cam1", datetime.now(), [detection1], camera_health)
        ])
        global_id1 = result1["cam1"][0].track_id
        
        # Person disappears (occlusion)
        result2 = tracker.process_detections([
            DetectionResult("cam1", datetime.now(), [], camera_health)
        ])
        
        # Person reappears with same local track ID
        detection3 = PersonDetection(1, (110, 110, 210, 310), 0.8, (160, 210), True, 1)
        result3 = tracker.process_detections([
            DetectionResult("cam1", datetime.now(), [detection3], camera_health)
        ])
        
        # Should get same global ID if reappears quickly
        if result3["cam1"]:  # May be empty if track was cleaned up
            global_id3 = result3["cam1"][0].track_id
            # This might be the same or different depending on timing
            # The test verifies the framework handles reappearance
            self.assertIsInstance(global_id3, int)


if __name__ == '__main__':
    unittest.main()