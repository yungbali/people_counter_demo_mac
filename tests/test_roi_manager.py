"""
Unit tests for ROI Manager functionality.
Tests point-in-polygon calculations, entrance line detection, and spatial filtering.
"""

import pytest
import numpy as np
from typing import List, Tuple
from unittest.mock import Mock, patch

from src.services.roi_manager import (
    ROIManager, 
    ROIPolygon, 
    EntranceLine, 
    CameraROIConfig,
    LinePosition
)
from src.models.core import PersonDetection


class TestROIPolygon:
    """Test ROI polygon functionality."""
    
    def test_valid_polygon_creation(self):
        """Test creating a valid polygon."""
        points = [(0, 0), (100, 0), (100, 100), (0, 100)]
        polygon = ROIPolygon(points=points, name="test_roi")
        
        assert polygon.is_valid()
        assert len(polygon.points) == 4
        assert polygon.name == "test_roi"
    
    def test_invalid_polygon_creation(self):
        """Test creating an invalid polygon with too few points."""
        points = [(0, 0), (100, 0)]
        polygon = ROIPolygon(points=points)
        
        assert not polygon.is_valid()
    
    def test_polygon_area_calculation(self):
        """Test polygon area calculation using shoelace formula."""
        # Square with side length 100
        points = [(0, 0), (100, 0), (100, 100), (0, 100)]
        polygon = ROIPolygon(points=points)
        
        expected_area = 10000  # 100 * 100
        assert abs(polygon.area() - expected_area) < 1e-6
    
    def test_polygon_area_triangle(self):
        """Test area calculation for a triangle."""
        # Right triangle with base 100, height 50
        points = [(0, 0), (100, 0), (0, 50)]
        polygon = ROIPolygon(points=points)
        
        expected_area = 2500  # 0.5 * 100 * 50
        assert abs(polygon.area() - expected_area) < 1e-6
    
    def test_polygon_to_numpy(self):
        """Test conversion to numpy array."""
        points = [(0, 0), (100, 0), (100, 100), (0, 100)]
        polygon = ROIPolygon(points=points)
        
        np_array = polygon.to_numpy()
        assert isinstance(np_array, np.ndarray)
        assert np_array.shape == (4, 2)
        assert np_array.dtype == np.int32


class TestEntranceLine:
    """Test entrance line functionality."""
    
    def test_entrance_line_creation(self):
        """Test creating an entrance line."""
        line = EntranceLine(x1=100, y1=50, x2=100, y2=450)
        
        assert line.x1 == 100
        assert line.y1 == 50
        assert line.x2 == 100
        assert line.y2 == 450
    
    def test_entrance_line_tuple_conversion(self):
        """Test conversion to and from tuple format."""
        coords = (100, 50, 100, 450)
        line = EntranceLine.from_tuple(coords)
        
        assert line.to_tuple() == coords
    
    def test_entrance_line_length_calculation(self):
        """Test calculating entrance line length."""
        line = EntranceLine(x1=0, y1=0, x2=100, y2=0)  # Horizontal line
        
        # Length calculation would be done externally
        length = np.sqrt((line.x2 - line.x1)**2 + (line.y2 - line.y1)**2)
        assert abs(length - 100) < 1e-6


class TestROIManager:
    """Test ROI Manager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.roi_manager = ROIManager()
        self.camera_id = "test_camera"
    
    def test_configure_camera_roi_valid(self):
        """Test configuring a valid ROI for a camera."""
        polygon_points = [(0, 0), (100, 0), (100, 100), (0, 100)]
        
        result = self.roi_manager.configure_camera_roi(self.camera_id, polygon_points, "test_roi")
        
        assert result is True
        assert self.camera_id in self.roi_manager.camera_configs
        
        config = self.roi_manager.camera_configs[self.camera_id]
        assert config.roi_polygon is not None
        assert len(config.roi_polygon.points) == 4
        assert config.roi_polygon.name == "test_roi"
    
    def test_configure_camera_roi_invalid(self):
        """Test configuring an invalid ROI (too few points)."""
        polygon_points = [(0, 0), (100, 0)]  # Only 2 points
        
        result = self.roi_manager.configure_camera_roi(self.camera_id, polygon_points)
        
        assert result is False
        # Should not create config for invalid ROI
        config = self.roi_manager.camera_configs.get(self.camera_id)
        assert config is None or config.roi_polygon is None
    
    def test_configure_entrance_line_valid(self):
        """Test configuring a valid entrance line."""
        line_coords = (100, 50, 100, 450)
        
        result = self.roi_manager.configure_entrance_line(self.camera_id, line_coords)
        
        assert result is True
        assert self.camera_id in self.roi_manager.camera_configs
        
        config = self.roi_manager.camera_configs[self.camera_id]
        assert config.entrance_line is not None
        assert config.entrance_line.to_tuple() == line_coords
    
    def test_configure_entrance_line_invalid(self):
        """Test configuring an invalid entrance line (too short)."""
        line_coords = (100, 100, 101, 101)  # Very short line
        
        result = self.roi_manager.configure_entrance_line(self.camera_id, line_coords)
        
        assert result is False
    
    def test_point_in_roi_inside(self):
        """Test point inside ROI polygon."""
        # Square ROI
        polygon_points = [(0, 0), (100, 0), (100, 100), (0, 100)]
        self.roi_manager.configure_camera_roi(self.camera_id, polygon_points)
        
        # Point inside the square
        result = self.roi_manager.is_point_in_roi(self.camera_id, (50, 50))
        assert result is True
    
    def test_point_in_roi_outside(self):
        """Test point outside ROI polygon."""
        # Square ROI
        polygon_points = [(0, 0), (100, 0), (100, 100), (0, 100)]
        self.roi_manager.configure_camera_roi(self.camera_id, polygon_points)
        
        # Point outside the square
        result = self.roi_manager.is_point_in_roi(self.camera_id, (150, 150))
        assert result is False
    
    def test_point_in_roi_on_edge(self):
        """Test point on ROI polygon edge."""
        # Square ROI
        polygon_points = [(0, 0), (100, 0), (100, 100), (0, 100)]
        self.roi_manager.configure_camera_roi(self.camera_id, polygon_points)
        
        # Point on the edge
        result = self.roi_manager.is_point_in_roi(self.camera_id, (50, 0))
        assert result is True  # OpenCV considers edge points as inside
    
    def test_point_in_roi_no_config(self):
        """Test point check when no ROI is configured."""
        # No ROI configured - should return True (accept all points)
        result = self.roi_manager.is_point_in_roi(self.camera_id, (50, 50))
        assert result is True
    
    def test_filter_detections_by_roi(self):
        """Test filtering detections by ROI."""
        # Configure square ROI
        polygon_points = [(0, 0), (100, 0), (100, 100), (0, 100)]
        self.roi_manager.configure_camera_roi(self.camera_id, polygon_points)
        
        # Create test detections
        detections = [
            PersonDetection(track_id=1, bbox=(40, 40, 60, 60), confidence=0.9, 
                          center_point=(50, 50), in_roi=False),  # Inside ROI
            PersonDetection(track_id=2, bbox=(140, 140, 160, 160), confidence=0.8, 
                          center_point=(150, 150), in_roi=False),  # Outside ROI
            PersonDetection(track_id=3, bbox=(90, 90, 110, 110), confidence=0.7, 
                          center_point=(100, 100), in_roi=False),  # On edge
        ]
        
        filtered = self.roi_manager.filter_detections_by_roi(self.camera_id, detections)
        
        # Should keep detections 1 and 3 (inside and on edge)
        assert len(filtered) == 2
        assert filtered[0].track_id == 1
        assert filtered[1].track_id == 3
        
        # Check in_roi flags are set correctly
        assert filtered[0].in_roi is True
        assert filtered[1].in_roi is True
    
    def test_filter_detections_no_roi(self):
        """Test filtering detections when no ROI is configured."""
        detections = [
            PersonDetection(track_id=1, bbox=(40, 40, 60, 60), confidence=0.9, 
                          center_point=(50, 50), in_roi=False),
        ]
        
        filtered = self.roi_manager.filter_detections_by_roi(self.camera_id, detections)
        
        # Should keep all detections and set in_roi to True
        assert len(filtered) == 1
        assert filtered[0].in_roi is True
    
    def test_get_line_position_left(self):
        """Test determining position left of entrance line."""
        # Vertical line from (100, 50) to (100, 450)
        line_coords = (100, 50, 100, 450)
        self.roi_manager.configure_entrance_line(self.camera_id, line_coords)
        
        # Point to the left of the line
        position = self.roi_manager.get_line_position(self.camera_id, (50, 250))
        assert position == LinePosition.LEFT
    
    def test_get_line_position_right(self):
        """Test determining position right of entrance line."""
        # Vertical line from (100, 50) to (100, 450)
        line_coords = (100, 50, 100, 450)
        self.roi_manager.configure_entrance_line(self.camera_id, line_coords)
        
        # Point to the right of the line
        position = self.roi_manager.get_line_position(self.camera_id, (150, 250))
        assert position == LinePosition.RIGHT
    
    def test_get_line_position_on_line(self):
        """Test determining position on entrance line."""
        # Vertical line from (100, 50) to (100, 450)
        line_coords = (100, 50, 100, 450)
        self.roi_manager.configure_entrance_line(self.camera_id, line_coords)
        
        # Point on the line
        position = self.roi_manager.get_line_position(self.camera_id, (100, 250))
        assert position == LinePosition.ON_LINE
    
    def test_detect_entrance_crossing_enter(self):
        """Test detecting entrance crossing (LEFT to RIGHT = ENTER)."""
        # Vertical line from (100, 50) to (100, 450)
        line_coords = (100, 50, 100, 450)
        self.roi_manager.configure_entrance_line(self.camera_id, line_coords)
        
        track_id = 1
        
        # First position: left side
        crossing = self.roi_manager.detect_entrance_crossing(self.camera_id, track_id, (50, 250))
        assert crossing is None  # First position, no crossing yet
        
        # Move to right side (simulate multiple frames for stability)
        for _ in range(3):
            crossing = self.roi_manager.detect_entrance_crossing(self.camera_id, track_id, (150, 250))
        
        assert crossing == 'ENTER'
    
    def test_detect_entrance_crossing_exit(self):
        """Test detecting entrance crossing (RIGHT to LEFT = EXIT)."""
        # Vertical line from (100, 50) to (100, 450)
        line_coords = (100, 50, 100, 450)
        self.roi_manager.configure_entrance_line(self.camera_id, line_coords)
        
        track_id = 2
        
        # First position: right side
        crossing = self.roi_manager.detect_entrance_crossing(self.camera_id, track_id, (150, 250))
        assert crossing is None  # First position, no crossing yet
        
        # Move to left side (simulate multiple frames for stability)
        for _ in range(3):
            crossing = self.roi_manager.detect_entrance_crossing(self.camera_id, track_id, (50, 250))
        
        assert crossing == 'EXIT'
    
    def test_detect_entrance_crossing_no_line(self):
        """Test entrance crossing detection when no line is configured."""
        track_id = 1
        crossing = self.roi_manager.detect_entrance_crossing(self.camera_id, track_id, (50, 250))
        assert crossing is None
    
    def test_cleanup_person_tracking(self):
        """Test cleaning up person tracking data."""
        # Set up tracking data
        track_id = 1
        self.roi_manager.person_positions[track_id] = {
            'last_position': (50, 250),
            'last_side': LinePosition.LEFT,
            'crossing_buffer': [LinePosition.LEFT]
        }
        
        # Clean up
        self.roi_manager.cleanup_person_tracking(track_id)
        
        assert track_id not in self.roi_manager.person_positions
    
    def test_validate_roi_config_valid(self):
        """Test validating a valid ROI configuration."""
        polygon_points = [(0, 0), (100, 0), (100, 100), (0, 100)]
        self.roi_manager.configure_camera_roi(self.camera_id, polygon_points)
        
        errors = self.roi_manager.validate_roi_config(self.camera_id)
        assert len(errors) == 0
    
    def test_validate_roi_config_invalid_area(self):
        """Test validating ROI with too small area."""
        # Very small triangle
        polygon_points = [(0, 0), (5, 0), (0, 5)]
        self.roi_manager.configure_camera_roi(self.camera_id, polygon_points)
        
        errors = self.roi_manager.validate_roi_config(self.camera_id)
        assert len(errors) > 0
        assert any("too small" in error for error in errors)
    
    def test_validate_roi_config_no_config(self):
        """Test validating when no configuration exists."""
        errors = self.roi_manager.validate_roi_config(self.camera_id)
        assert len(errors) > 0
        assert any("No ROI configuration found" in error for error in errors)
    
    def test_export_import_config(self):
        """Test exporting and importing ROI configuration."""
        # Configure ROI and entrance line
        polygon_points = [(0, 0), (100, 0), (100, 100), (0, 100)]
        line_coords = (100, 50, 100, 450)
        
        self.roi_manager.configure_camera_roi(self.camera_id, polygon_points, "test_roi")
        self.roi_manager.configure_entrance_line(self.camera_id, line_coords)
        
        # Export configuration
        exported = self.roi_manager.export_config(self.camera_id)
        assert exported is not None
        assert 'roi_polygon' in exported
        assert 'entrance_line' in exported
        
        # Clear and import
        self.roi_manager.camera_configs.clear()
        result = self.roi_manager.import_config(self.camera_id, exported)
        
        assert result is True
        assert self.camera_id in self.roi_manager.camera_configs
        
        config = self.roi_manager.camera_configs[self.camera_id]
        assert config.roi_polygon is not None
        assert config.entrance_line is not None
        assert len(config.roi_polygon.points) == 4
        assert config.entrance_line.to_tuple() == line_coords
    
    def test_enable_disable_camera_roi(self):
        """Test enabling and disabling ROI for a camera."""
        polygon_points = [(0, 0), (100, 0), (100, 100), (0, 100)]
        self.roi_manager.configure_camera_roi(self.camera_id, polygon_points)
        
        # Disable ROI
        result = self.roi_manager.enable_camera_roi(self.camera_id, False)
        assert result is True
        assert not self.roi_manager.camera_configs[self.camera_id].enabled
        
        # Enable ROI
        result = self.roi_manager.enable_camera_roi(self.camera_id, True)
        assert result is True
        assert self.roi_manager.camera_configs[self.camera_id].enabled
    
    def test_complex_polygon_shapes(self):
        """Test with more complex polygon shapes."""
        # L-shaped polygon
        polygon_points = [
            (0, 0), (100, 0), (100, 50), (50, 50), (50, 100), (0, 100)
        ]
        self.roi_manager.configure_camera_roi(self.camera_id, polygon_points)
        
        # Test points
        assert self.roi_manager.is_point_in_roi(self.camera_id, (25, 25)) is True  # Inside
        assert self.roi_manager.is_point_in_roi(self.camera_id, (75, 75)) is False  # In cutout
        assert self.roi_manager.is_point_in_roi(self.camera_id, (25, 75)) is True  # Inside
    
    def test_diagonal_entrance_line(self):
        """Test entrance line detection with diagonal line."""
        # Diagonal line from (50, 50) to (150, 150)
        line_coords = (50, 50, 150, 150)
        self.roi_manager.configure_entrance_line(self.camera_id, line_coords)
        
        track_id = 1
        
        # Start on one side of diagonal
        self.roi_manager.detect_entrance_crossing(self.camera_id, track_id, (25, 75))
        
        # Move to other side
        for _ in range(3):
            crossing = self.roi_manager.detect_entrance_crossing(self.camera_id, track_id, (125, 125))
        
        # Should detect crossing
        assert crossing is not None


class TestROIManagerIntegration:
    """Integration tests for ROI Manager with realistic scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.roi_manager = ROIManager()
    
    def test_multi_camera_configuration(self):
        """Test configuring multiple cameras with different ROIs."""
        # Configure entrance camera with rectangular ROI and entrance line
        entrance_roi = [(100, 100), (500, 100), (500, 400), (100, 400)]
        entrance_line = (200, 50, 200, 450)
        
        self.roi_manager.configure_camera_roi("entrance", entrance_roi)
        self.roi_manager.configure_entrance_line("entrance", entrance_line)
        
        # Configure lounge camera with circular-approximated ROI
        lounge_roi = []
        center_x, center_y, radius = 320, 240, 150
        for i in range(8):  # 8-sided polygon approximating circle
            angle = i * 2 * np.pi / 8
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            lounge_roi.append((x, y))
        
        self.roi_manager.configure_camera_roi("lounge", lounge_roi)
        
        # Test configurations
        assert len(self.roi_manager.camera_configs) == 2
        
        # Test entrance camera
        entrance_config = self.roi_manager.get_camera_config("entrance")
        assert entrance_config.roi_polygon is not None
        assert entrance_config.entrance_line is not None
        
        # Test lounge camera
        lounge_config = self.roi_manager.get_camera_config("lounge")
        assert lounge_config.roi_polygon is not None
        assert lounge_config.entrance_line is None
        
        # Test point filtering for each camera
        assert self.roi_manager.is_point_in_roi("entrance", (300, 250)) is True
        assert self.roi_manager.is_point_in_roi("entrance", (50, 50)) is False
        
        assert self.roi_manager.is_point_in_roi("lounge", (320, 240)) is True  # Center
        assert self.roi_manager.is_point_in_roi("lounge", (100, 100)) is False  # Outside circle
    
    def test_realistic_entrance_scenario(self):
        """Test realistic entrance crossing scenario."""
        # Configure entrance with vertical line
        line_coords = (200, 50, 200, 450)
        self.roi_manager.configure_entrance_line("entrance", line_coords)
        
        track_id = 1
        
        # Simulate person approaching entrance from left
        positions = [
            (150, 250),  # Approaching
            (170, 250),  # Getting closer
            (190, 250),  # Almost at line
            (210, 250),  # Crossed to right
            (220, 250),  # Moving away
            (230, 250),  # Further away
        ]
        
        crossings = []
        for pos in positions:
            crossing = self.roi_manager.detect_entrance_crossing("entrance", track_id, pos)
            if crossing:
                crossings.append(crossing)
        
        # Should detect one ENTER event
        assert len(crossings) == 1
        assert crossings[0] == 'ENTER'
    
    def test_performance_with_many_detections(self):
        """Test performance with many simultaneous detections."""
        # Configure large ROI
        roi_points = [(0, 0), (640, 0), (640, 480), (0, 480)]
        self.roi_manager.configure_camera_roi("test_camera", roi_points)
        
        # Create many detections
        detections = []
        for i in range(100):
            detection = PersonDetection(
                track_id=i,
                bbox=(i*5, i*3, i*5+20, i*3+30),
                confidence=0.8,
                center_point=(i*5+10, i*3+15),
                in_roi=False
            )
            detections.append(detection)
        
        # Filter detections
        filtered = self.roi_manager.filter_detections_by_roi("test_camera", detections)
        
        # All should be within the large ROI
        assert len(filtered) == len(detections)
        assert all(d.in_roi for d in filtered)


if __name__ == "__main__":
    pytest.main([__file__])