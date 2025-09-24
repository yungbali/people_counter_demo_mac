"""
ROI (Region of Interest) Management System for multi-camera detection.
Handles polygon-based ROI definitions, entrance line detection, and spatial filtering.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from src.models.core import PersonDetection


logger = logging.getLogger(__name__)


class LinePosition(Enum):
    """Position relative to entrance line."""
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    ON_LINE = "ON_LINE"


@dataclass
class EntranceLine:
    """Entrance line configuration for in/out counting."""
    x1: int
    y1: int
    x2: int
    y2: int
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to tuple format."""
        return (self.x1, self.y1, self.x2, self.y2)
    
    @classmethod
    def from_tuple(cls, coords: Tuple[int, int, int, int]) -> 'EntranceLine':
        """Create from tuple format."""
        return cls(coords[0], coords[1], coords[2], coords[3])


@dataclass
class ROIPolygon:
    """ROI polygon configuration."""
    points: List[Tuple[int, int]]
    name: Optional[str] = None
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array for OpenCV operations."""
        return np.array(self.points, dtype=np.int32)
    
    def is_valid(self) -> bool:
        """Check if polygon is valid (at least 3 points)."""
        return len(self.points) >= 3
    
    def area(self) -> float:
        """Calculate polygon area using shoelace formula."""
        if not self.is_valid():
            return 0.0
        
        n = len(self.points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.points[i][0] * self.points[j][1]
            area -= self.points[j][0] * self.points[i][1]
        return abs(area) / 2.0


@dataclass
class CameraROIConfig:
    """Complete ROI configuration for a camera."""
    camera_id: str
    roi_polygon: Optional[ROIPolygon] = None
    entrance_line: Optional[EntranceLine] = None
    enabled: bool = True


class ROIManager:
    """Manages ROI configurations and spatial filtering for multiple cameras."""
    
    def __init__(self):
        self.camera_configs: Dict[str, CameraROIConfig] = {}
        self.person_positions: Dict[int, Dict[str, Any]] = {}  # track_id -> position history
        
    def configure_camera_roi(self, camera_id: str, polygon_points: List[Tuple[int, int]], 
                           name: Optional[str] = None) -> bool:
        """Configure ROI polygon for a camera."""
        try:
            roi_polygon = ROIPolygon(points=polygon_points, name=name)
            
            if not roi_polygon.is_valid():
                logger.error(f"Invalid ROI polygon for camera {camera_id}: need at least 3 points")
                return False
            
            # Get or create camera config
            if camera_id not in self.camera_configs:
                self.camera_configs[camera_id] = CameraROIConfig(camera_id=camera_id)
            
            self.camera_configs[camera_id].roi_polygon = roi_polygon
            
            logger.info(f"Configured ROI for camera {camera_id} with {len(polygon_points)} points, area: {roi_polygon.area():.1f}")
            return True
            
        except Exception as e:
            logger.error(f"Error configuring ROI for camera {camera_id}: {e}")
            return False
    
    def configure_entrance_line(self, camera_id: str, line_coords: Tuple[int, int, int, int]) -> bool:
        """Configure entrance line for a camera."""
        try:
            entrance_line = EntranceLine.from_tuple(line_coords)
            
            # Validate line (should have non-zero length)
            line_length = np.sqrt((entrance_line.x2 - entrance_line.x1)**2 + 
                                (entrance_line.y2 - entrance_line.y1)**2)
            if line_length < 10:  # Minimum 10 pixels
                logger.error(f"Entrance line too short for camera {camera_id}: {line_length:.1f} pixels")
                return False
            
            # Get or create camera config
            if camera_id not in self.camera_configs:
                self.camera_configs[camera_id] = CameraROIConfig(camera_id=camera_id)
            
            self.camera_configs[camera_id].entrance_line = entrance_line
            
            logger.info(f"Configured entrance line for camera {camera_id}: {line_coords}")
            return True
            
        except Exception as e:
            logger.error(f"Error configuring entrance line for camera {camera_id}: {e}")
            return False
    
    def is_point_in_roi(self, camera_id: str, point: Tuple[float, float]) -> bool:
        """Check if a point is within the camera's ROI."""
        config = self.camera_configs.get(camera_id)
        if not config or not config.enabled or not config.roi_polygon:
            return True  # No ROI configured, consider all points valid
        
        try:
            roi_np = config.roi_polygon.to_numpy()
            result = cv2.pointPolygonTest(roi_np, (float(point[0]), float(point[1])), False)
            return result >= 0
        except Exception as e:
            logger.error(f"Error checking point in ROI for camera {camera_id}: {e}")
            return True  # Default to accepting point on error
    
    def filter_detections_by_roi(self, camera_id: str, detections: List[PersonDetection]) -> List[PersonDetection]:
        """Filter detections to only include those within ROI."""
        if not detections:
            return detections
        
        config = self.camera_configs.get(camera_id)
        if not config or not config.enabled or not config.roi_polygon:
            # No ROI filtering, but still update in_roi flag
            for detection in detections:
                detection.in_roi = True
            return detections
        
        filtered_detections = []
        for detection in detections:
            in_roi = self.is_point_in_roi(camera_id, detection.center_point)
            detection.in_roi = in_roi
            
            if in_roi:
                filtered_detections.append(detection)
        
        logger.debug(f"Camera {camera_id}: filtered {len(detections)} -> {len(filtered_detections)} detections")
        return filtered_detections
    
    def get_line_position(self, camera_id: str, point: Tuple[float, float]) -> LinePosition:
        """Determine which side of the entrance line a point is on."""
        config = self.camera_configs.get(camera_id)
        if not config or not config.entrance_line:
            return LinePosition.ON_LINE
        
        line = config.entrance_line
        
        # Calculate cross product to determine side
        # Vector from line start to point
        px, py = point[0] - line.x1, point[1] - line.y1
        # Line direction vector
        lx, ly = line.x2 - line.x1, line.y2 - line.y1
        
        # Cross product
        cross = px * ly - py * lx
        
        if abs(cross) < 5:  # Tolerance for "on line"
            return LinePosition.ON_LINE
        elif cross > 0:
            return LinePosition.LEFT
        else:
            return LinePosition.RIGHT
    
    def detect_entrance_crossing(self, camera_id: str, track_id: int, 
                               current_position: Tuple[float, float]) -> Optional[str]:
        """
        Detect if a person has crossed the entrance line.
        Returns 'ENTER' or 'EXIT' if crossing detected, None otherwise.
        """
        config = self.camera_configs.get(camera_id)
        if not config or not config.entrance_line:
            return None
        
        current_side = self.get_line_position(camera_id, current_position)
        
        # Get previous position
        if track_id not in self.person_positions:
            self.person_positions[track_id] = {
                'last_position': current_position,
                'last_side': current_side,
                'crossing_buffer': []
            }
            return None
        
        person_data = self.person_positions[track_id]
        previous_side = person_data['last_side']
        
        # Update position history
        person_data['last_position'] = current_position
        person_data['last_side'] = current_side
        
        # Add to crossing buffer for stability
        person_data['crossing_buffer'].append(current_side)
        if len(person_data['crossing_buffer']) > 5:  # Keep last 5 positions
            person_data['crossing_buffer'].pop(0)
        
        # Detect crossing (need stable transition)
        if len(person_data['crossing_buffer']) >= 3:
            # Check for consistent crossing pattern
            recent_sides = person_data['crossing_buffer'][-3:]
            
            if (previous_side == LinePosition.LEFT and 
                all(side == LinePosition.RIGHT for side in recent_sides)):
                logger.info(f"Camera {camera_id}: Person {track_id} crossed LEFT->RIGHT (ENTER)")
                return 'ENTER'
            elif (previous_side == LinePosition.RIGHT and 
                  all(side == LinePosition.LEFT for side in recent_sides)):
                logger.info(f"Camera {camera_id}: Person {track_id} crossed RIGHT->LEFT (EXIT)")
                return 'EXIT'
        
        return None
    
    def cleanup_person_tracking(self, track_id: int) -> None:
        """Clean up tracking data for a person who is no longer detected."""
        if track_id in self.person_positions:
            del self.person_positions[track_id]
    
    def get_camera_config(self, camera_id: str) -> Optional[CameraROIConfig]:
        """Get ROI configuration for a camera."""
        return self.camera_configs.get(camera_id)
    
    def get_all_configs(self) -> Dict[str, CameraROIConfig]:
        """Get all camera ROI configurations."""
        return self.camera_configs.copy()
    
    def enable_camera_roi(self, camera_id: str, enabled: bool = True) -> bool:
        """Enable or disable ROI filtering for a camera."""
        if camera_id not in self.camera_configs:
            return False
        
        self.camera_configs[camera_id].enabled = enabled
        logger.info(f"Camera {camera_id} ROI filtering {'enabled' if enabled else 'disabled'}")
        return True
    
    def validate_roi_config(self, camera_id: str) -> List[str]:
        """Validate ROI configuration for a camera and return any errors."""
        errors = []
        config = self.camera_configs.get(camera_id)
        
        if not config:
            errors.append(f"No ROI configuration found for camera {camera_id}")
            return errors
        
        # Validate ROI polygon
        if config.roi_polygon:
            if not config.roi_polygon.is_valid():
                errors.append(f"ROI polygon for camera {camera_id} has fewer than 3 points")
            
            area = config.roi_polygon.area()
            if area < 100:  # Minimum area threshold
                errors.append(f"ROI polygon for camera {camera_id} is too small (area: {area:.1f})")
        
        # Validate entrance line
        if config.entrance_line:
            line = config.entrance_line
            line_length = np.sqrt((line.x2 - line.x1)**2 + (line.y2 - line.y1)**2)
            if line_length < 10:
                errors.append(f"Entrance line for camera {camera_id} is too short ({line_length:.1f} pixels)")
        
        return errors
    
    def export_config(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Export ROI configuration for a camera to dictionary format."""
        config = self.camera_configs.get(camera_id)
        if not config:
            return None
        
        result = {
            'camera_id': config.camera_id,
            'enabled': config.enabled
        }
        
        if config.roi_polygon:
            result['roi_polygon'] = {
                'points': config.roi_polygon.points,
                'name': config.roi_polygon.name,
                'area': config.roi_polygon.area()
            }
        
        if config.entrance_line:
            result['entrance_line'] = {
                'coordinates': config.entrance_line.to_tuple(),
                'length': np.sqrt((config.entrance_line.x2 - config.entrance_line.x1)**2 + 
                                (config.entrance_line.y2 - config.entrance_line.y1)**2)
            }
        
        return result
    
    def import_config(self, camera_id: str, config_data: Dict[str, Any]) -> bool:
        """Import ROI configuration for a camera from dictionary format."""
        try:
            # Create camera config
            camera_config = CameraROIConfig(
                camera_id=camera_id,
                enabled=config_data.get('enabled', True)
            )
            
            # Import ROI polygon
            if 'roi_polygon' in config_data:
                roi_data = config_data['roi_polygon']
                camera_config.roi_polygon = ROIPolygon(
                    points=roi_data['points'],
                    name=roi_data.get('name')
                )
            
            # Import entrance line
            if 'entrance_line' in config_data:
                line_data = config_data['entrance_line']
                camera_config.entrance_line = EntranceLine.from_tuple(
                    tuple(line_data['coordinates'])
                )
            
            self.camera_configs[camera_id] = camera_config
            logger.info(f"Imported ROI configuration for camera {camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing ROI configuration for camera {camera_id}: {e}")
            return False