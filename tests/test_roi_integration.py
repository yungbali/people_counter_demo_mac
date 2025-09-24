"""
Integration tests for ROI management system.
Tests the complete workflow from configuration to detection filtering.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from src.services.roi_manager import ROIManager
from src.services.cv_service import EnhancedCVService
from src.models.core import PersonDetection
from src.config.models import CameraConfig, CameraType


class TestROIIntegration:
    """Integration tests for ROI system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.roi_manager = ROIManager()
        
        # Mock YOLO to avoid loading actual model
        with patch('src.services.cv_service.YOLO'):
            self.cv_service = EnhancedCVService(model_path="test_model.pt")
    
    def test_complete_roi_workflow(self):
        """Test complete ROI configuration and filtering workflow."""
        camera_id = "entrance_camera"
        
        # Step 1: Configure ROI polygon (entrance area)
        entrance_roi = [
            (100, 100),  # Top-left
            (500, 100),  # Top-right
            (500, 400),  # Bottom-right
            (100, 400)   # Bottom-left
        ]
        
        success = self.roi_manager.configure_camera_roi(camera_id, entrance_roi, "entrance_area")
        assert success is True
        
        # Step 2: Configure entrance line (vertical line through middle)
        entrance_line = (300, 100, 300, 400)
        success = self.roi_manager.configure_entrance_line(camera_id, entrance_line)
        assert success is True
        
        # Step 3: Create test detections (some inside ROI, some outside)
        test_detections = [
            # Inside ROI
            PersonDetection(track_id=1, bbox=(250, 200, 270, 240), confidence=0.9, 
                          center_point=(260, 220), in_roi=False),
            PersonDetection(track_id=2, bbox=(350, 300, 370, 340), confidence=0.8, 
                          center_point=(360, 320), in_roi=False),
            
            # Outside ROI
            PersonDetection(track_id=3, bbox=(50, 50, 70, 90), confidence=0.7, 
                          center_point=(60, 70), in_roi=False),
            PersonDetection(track_id=4, bbox=(550, 450, 570, 490), confidence=0.6, 
                          center_point=(560, 470), in_roi=False),
            
            # On ROI boundary
            PersonDetection(track_id=5, bbox=(95, 95, 105, 105), confidence=0.8, 
                          center_point=(100, 100), in_roi=False),
        ]
        
        # Step 4: Filter detections by ROI
        filtered_detections = self.roi_manager.filter_detections_by_roi(camera_id, test_detections)
        
        # Step 5: Verify filtering results
        assert len(filtered_detections) == 3  # Should keep tracks 1, 2, and 5
        
        filtered_track_ids = {d.track_id for d in filtered_detections}
        assert filtered_track_ids == {1, 2, 5}
        
        # All filtered detections should have in_roi=True
        assert all(d.in_roi for d in filtered_detections)
        
        # Step 6: Test entrance line crossing detection
        track_id = 1
        
        # Simulate person moving from left to right (ENTER)
        positions = [
            (250, 220),  # Left side of line
            (280, 220),  # Approaching line
            (320, 220),  # Right side of line
            (350, 220),  # Moving away
        ]
        
        crossings = []
        for pos in positions:
            crossing = self.roi_manager.detect_entrance_crossing(camera_id, track_id, pos)
            if crossing:
                crossings.append(crossing)
        
        # Should detect one ENTER event
        assert len(crossings) == 1
        assert crossings[0] == 'ENTER'
    
    def test_multi_camera_roi_system(self):
        """Test ROI system with multiple cameras."""
        # Configure entrance camera
        entrance_camera = "entrance"
        entrance_roi = [(100, 100), (500, 100), (500, 400), (100, 400)]
        entrance_line = (300, 100, 300, 400)
        
        self.roi_manager.configure_camera_roi(entrance_camera, entrance_roi)
        self.roi_manager.configure_entrance_line(entrance_camera, entrance_line)
        
        # Configure lounge camera (circular ROI)
        lounge_camera = "lounge"
        center_x, center_y, radius = 320, 240, 150
        lounge_roi = []
        for i in range(8):  # Octagon approximating circle
            angle = i * 2 * np.pi / 8
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            lounge_roi.append((x, y))
        
        self.roi_manager.configure_camera_roi(lounge_camera, lounge_roi)
        
        # Test detections for entrance camera
        entrance_detections = [
            PersonDetection(track_id=1, bbox=(250, 200, 270, 240), confidence=0.9, 
                          center_point=(260, 220), in_roi=False),  # Inside
            PersonDetection(track_id=2, bbox=(50, 50, 70, 90), confidence=0.8, 
                          center_point=(60, 70), in_roi=False),    # Outside
        ]
        
        # Test detections for lounge camera
        lounge_detections = [
            PersonDetection(track_id=3, bbox=(310, 230, 330, 250), confidence=0.9, 
                          center_point=(320, 240), in_roi=False),  # Center (inside)
            PersonDetection(track_id=4, bbox=(100, 100, 120, 120), confidence=0.8, 
                          center_point=(110, 110), in_roi=False),  # Outside circle
        ]
        
        # Filter detections for each camera
        entrance_filtered = self.roi_manager.filter_detections_by_roi(entrance_camera, entrance_detections)
        lounge_filtered = self.roi_manager.filter_detections_by_roi(lounge_camera, lounge_detections)
        
        # Verify results
        assert len(entrance_filtered) == 1
        assert entrance_filtered[0].track_id == 1
        
        assert len(lounge_filtered) == 1
        assert lounge_filtered[0].track_id == 3
        
        # Test entrance crossing only works for entrance camera
        crossing = self.roi_manager.detect_entrance_crossing(entrance_camera, 1, (350, 220))
        assert crossing is None  # First position, no crossing yet
        
        crossing = self.roi_manager.detect_entrance_crossing(lounge_camera, 3, (350, 240))
        assert crossing is None  # No entrance line configured for lounge
    
    def test_roi_configuration_persistence(self):
        """Test ROI configuration export/import functionality."""
        camera_id = "test_camera"
        
        # Configure ROI and entrance line
        roi_points = [(0, 0), (100, 0), (100, 100), (0, 100)]
        line_coords = (50, 0, 50, 100)
        
        self.roi_manager.configure_camera_roi(camera_id, roi_points, "test_area")
        self.roi_manager.configure_entrance_line(camera_id, line_coords)
        
        # Export configuration
        exported_config = self.roi_manager.export_config(camera_id)
        assert exported_config is not None
        assert 'roi_polygon' in exported_config
        assert 'entrance_line' in exported_config
        
        # Clear configuration
        self.roi_manager.camera_configs.clear()
        assert self.roi_manager.get_camera_config(camera_id) is None
        
        # Import configuration
        success = self.roi_manager.import_config(camera_id, exported_config)
        assert success is True
        
        # Verify imported configuration
        config = self.roi_manager.get_camera_config(camera_id)
        assert config is not None
        assert config.roi_polygon is not None
        assert config.entrance_line is not None
        assert len(config.roi_polygon.points) == 4
        assert config.entrance_line.to_tuple() == line_coords
        
        # Test functionality still works
        inside_point = (50, 50)
        outside_point = (150, 150)
        
        assert self.roi_manager.is_point_in_roi(camera_id, inside_point) is True
        assert self.roi_manager.is_point_in_roi(camera_id, outside_point) is False
    
    def test_roi_validation_workflow(self):
        """Test ROI validation in complete workflow."""
        camera_id = "validation_test"
        
        # Test valid configuration
        valid_roi = [(0, 0), (200, 0), (200, 200), (0, 200)]
        valid_line = (100, 0, 100, 200)
        
        self.roi_manager.configure_camera_roi(camera_id, valid_roi)
        self.roi_manager.configure_entrance_line(camera_id, valid_line)
        
        errors = self.roi_manager.validate_roi_config(camera_id)
        assert len(errors) == 0
        
        # Test invalid configuration (too small area)
        invalid_roi = [(0, 0), (5, 0), (0, 5)]
        self.roi_manager.configure_camera_roi(camera_id, invalid_roi)
        
        errors = self.roi_manager.validate_roi_config(camera_id)
        assert len(errors) > 0
        assert any("too small" in error for error in errors)
    
    def test_cv_service_roi_integration(self):
        """Test CV service integration with ROI manager."""
        camera_id = "cv_integration_test"
        
        # Configure ROI through CV service
        roi_points = [(100, 100), (500, 100), (500, 400), (100, 400)]
        line_coords = (300, 100, 300, 400)
        
        self.cv_service.configure_roi(camera_id, roi_points)
        self.cv_service.set_entrance_line(camera_id, line_coords)
        
        # Verify configuration through CV service
        config = self.cv_service.get_roi_config(camera_id)
        assert config is not None
        
        errors = self.cv_service.validate_roi_config(camera_id)
        assert len(errors) == 0
        
        # Test entrance events tracking
        assert camera_id in self.cv_service.entrance_events
        
        events = self.cv_service.get_entrance_events(camera_id)
        assert events == []  # No events initially
    
    def test_error_recovery_and_robustness(self):
        """Test system robustness and error recovery."""
        camera_id = "robustness_test"
        
        # Test with invalid inputs
        invalid_roi = [(0, 0)]  # Too few points
        success = self.roi_manager.configure_camera_roi(camera_id, invalid_roi)
        assert success is False
        
        invalid_line = (100, 100, 101, 101)  # Too short
        success = self.roi_manager.configure_entrance_line(camera_id, invalid_line)
        assert success is False
        
        # Test operations on non-existent camera
        result = self.roi_manager.is_point_in_roi("non_existent", (50, 50))
        assert result is True  # Should default to accepting all points
        
        crossing = self.roi_manager.detect_entrance_crossing("non_existent", 1, (50, 50))
        assert crossing is None
        
        # Test with empty detections list
        filtered = self.roi_manager.filter_detections_by_roi(camera_id, [])
        assert filtered == []
        
        # Test with malformed detection data
        malformed_detection = PersonDetection(
            track_id=1, bbox=(0, 0, 0, 0), confidence=0.0, 
            center_point=(0, 0), in_roi=False
        )
        
        # Should handle gracefully
        filtered = self.roi_manager.filter_detections_by_roi(camera_id, [malformed_detection])
        assert len(filtered) == 1  # Should still process, just mark as not in ROI
    
    def test_performance_with_realistic_data(self):
        """Test performance with realistic detection data."""
        camera_id = "performance_test"
        
        # Configure realistic ROI (entrance area)
        roi_points = [(150, 100), (490, 100), (490, 380), (150, 380)]
        line_coords = (320, 100, 320, 380)
        
        self.roi_manager.configure_camera_roi(camera_id, roi_points)
        self.roi_manager.configure_entrance_line(camera_id, line_coords)
        
        # Generate realistic detection data (30 people)
        detections = []
        for i in range(30):
            # Distribute people across the frame
            x = 100 + (i % 10) * 50
            y = 100 + (i // 10) * 100
            
            detection = PersonDetection(
                track_id=i,
                bbox=(x, y, x+20, y+40),
                confidence=0.7 + (i % 3) * 0.1,
                center_point=(x+10, y+20),
                in_roi=False
            )
            detections.append(detection)
        
        # Filter detections
        filtered = self.roi_manager.filter_detections_by_roi(camera_id, detections)
        
        # Should complete quickly and return reasonable results
        assert len(filtered) <= len(detections)
        assert len(filtered) > 0  # Some should be in ROI
        
        # Test entrance crossing with multiple people
        crossings = []
        for track_id in range(10):
            # Simulate crossing from left to right
            for frame in range(5):
                pos_x = 250 + frame * 20  # Move from left to right
                crossing = self.roi_manager.detect_entrance_crossing(camera_id, track_id, (pos_x, 240))
                if crossing:
                    crossings.append((track_id, crossing))
        
        # Should detect crossings for multiple people
        assert len(crossings) > 0
        assert all(crossing[1] == 'ENTER' for crossing in crossings)


if __name__ == "__main__":
    pytest.main([__file__])