"""
Camera Configuration UI Components for ROI drawing and entrance line setup.
Provides interactive tools for configuring camera regions of interest and entrance lines.
"""

import json
import logging
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ConfigMode(Enum):
    """Configuration mode enumeration."""
    ROI_POLYGON = "roi_polygon"
    ENTRANCE_LINE = "entrance_line"
    VIEW_ONLY = "view_only"


@dataclass
class UIPoint:
    """UI point with screen coordinates."""
    x: int
    y: int
    
    def to_tuple(self) -> Tuple[int, int]:
        """Convert to tuple format."""
        return (self.x, self.y)


@dataclass
class UIConfig:
    """UI configuration settings."""
    canvas_width: int = 640
    canvas_height: int = 480
    point_radius: int = 5
    line_width: int = 2
    roi_color: str = "#00ff00"  # Green
    entrance_line_color: str = "#ff0000"  # Red
    point_color: str = "#ffff00"  # Yellow
    grid_enabled: bool = True
    grid_size: int = 20
    snap_to_grid: bool = False


class CameraConfigUI:
    """Interactive camera configuration UI for ROI and entrance line setup."""
    
    def __init__(self, ui_config: Optional[UIConfig] = None):
        self.ui_config = ui_config or UIConfig()
        self.current_mode = ConfigMode.VIEW_ONLY
        self.roi_points: List[UIPoint] = []
        self.entrance_line_points: List[UIPoint] = []
        self.is_drawing = False
        self.selected_point_index: Optional[int] = None
        
        # Callbacks
        self.on_roi_changed: Optional[Callable[[List[Tuple[int, int]]], None]] = None
        self.on_entrance_line_changed: Optional[Callable[[Tuple[int, int, int, int]], None]] = None
        self.on_config_validated: Optional[Callable[[List[str]], None]] = None
    
    def set_mode(self, mode: ConfigMode) -> None:
        """Set the current configuration mode."""
        self.current_mode = mode
        self.is_drawing = False
        self.selected_point_index = None
        logger.info(f"Configuration mode set to: {mode.value}")
    
    def start_roi_drawing(self) -> None:
        """Start drawing ROI polygon."""
        self.set_mode(ConfigMode.ROI_POLYGON)
        self.roi_points.clear()
        self.is_drawing = True
        logger.info("Started ROI polygon drawing")
    
    def start_entrance_line_drawing(self) -> None:
        """Start drawing entrance line."""
        self.set_mode(ConfigMode.ENTRANCE_LINE)
        self.entrance_line_points.clear()
        self.is_drawing = True
        logger.info("Started entrance line drawing")
    
    def add_point(self, x: int, y: int) -> bool:
        """Add a point based on current mode."""
        if self.ui_config.snap_to_grid:
            x, y = self._snap_to_grid(x, y)
        
        point = UIPoint(x, y)
        
        if self.current_mode == ConfigMode.ROI_POLYGON:
            return self._add_roi_point(point)
        elif self.current_mode == ConfigMode.ENTRANCE_LINE:
            return self._add_entrance_line_point(point)
        
        return False
    
    def _add_roi_point(self, point: UIPoint) -> bool:
        """Add point to ROI polygon."""
        if not self.is_drawing:
            return False
        
        # Check for polygon closure (click near first point)
        if len(self.roi_points) >= 3:
            first_point = self.roi_points[0]
            distance = ((point.x - first_point.x)**2 + (point.y - first_point.y)**2)**0.5
            if distance < self.ui_config.point_radius * 2:
                # Close polygon
                self._finish_roi_drawing()
                return True
        
        self.roi_points.append(point)
        logger.debug(f"Added ROI point: ({point.x}, {point.y})")
        return True
    
    def _add_entrance_line_point(self, point: UIPoint) -> bool:
        """Add point to entrance line."""
        if not self.is_drawing:
            return False
        
        self.entrance_line_points.append(point)
        
        # Entrance line needs exactly 2 points
        if len(self.entrance_line_points) >= 2:
            self._finish_entrance_line_drawing()
        
        logger.debug(f"Added entrance line point: ({point.x}, {point.y})")
        return True
    
    def _finish_roi_drawing(self) -> None:
        """Finish ROI polygon drawing."""
        if len(self.roi_points) < 3:
            logger.warning("ROI polygon needs at least 3 points")
            return
        
        self.is_drawing = False
        roi_tuples = [point.to_tuple() for point in self.roi_points]
        
        # Validate polygon
        errors = self._validate_roi_polygon(roi_tuples)
        if errors:
            logger.error(f"ROI polygon validation failed: {errors}")
            if self.on_config_validated:
                self.on_config_validated(errors)
            return
        
        # Notify callback
        if self.on_roi_changed:
            self.on_roi_changed(roi_tuples)
        
        logger.info(f"Finished ROI polygon with {len(self.roi_points)} points")
    
    def _finish_entrance_line_drawing(self) -> None:
        """Finish entrance line drawing."""
        if len(self.entrance_line_points) != 2:
            logger.warning("Entrance line needs exactly 2 points")
            return
        
        self.is_drawing = False
        p1, p2 = self.entrance_line_points
        line_coords = (p1.x, p1.y, p2.x, p2.y)
        
        # Validate line
        errors = self._validate_entrance_line(line_coords)
        if errors:
            logger.error(f"Entrance line validation failed: {errors}")
            if self.on_config_validated:
                self.on_config_validated(errors)
            return
        
        # Notify callback
        if self.on_entrance_line_changed:
            self.on_entrance_line_changed(line_coords)
        
        logger.info(f"Finished entrance line: {line_coords}")
    
    def move_point(self, point_index: int, new_x: int, new_y: int) -> bool:
        """Move an existing point."""
        if self.ui_config.snap_to_grid:
            new_x, new_y = self._snap_to_grid(new_x, new_y)
        
        if self.current_mode == ConfigMode.ROI_POLYGON:
            if 0 <= point_index < len(self.roi_points):
                self.roi_points[point_index] = UIPoint(new_x, new_y)
                roi_tuples = [point.to_tuple() for point in self.roi_points]
                if self.on_roi_changed:
                    self.on_roi_changed(roi_tuples)
                return True
        
        elif self.current_mode == ConfigMode.ENTRANCE_LINE:
            if 0 <= point_index < len(self.entrance_line_points):
                self.entrance_line_points[point_index] = UIPoint(new_x, new_y)
                if len(self.entrance_line_points) == 2:
                    p1, p2 = self.entrance_line_points
                    line_coords = (p1.x, p1.y, p2.x, p2.y)
                    if self.on_entrance_line_changed:
                        self.on_entrance_line_changed(line_coords)
                return True
        
        return False
    
    def select_point(self, x: int, y: int) -> Optional[int]:
        """Select a point near the given coordinates."""
        min_distance = float('inf')
        selected_index = None
        
        points = []
        if self.current_mode == ConfigMode.ROI_POLYGON:
            points = self.roi_points
        elif self.current_mode == ConfigMode.ENTRANCE_LINE:
            points = self.entrance_line_points
        
        for i, point in enumerate(points):
            distance = ((x - point.x)**2 + (y - point.y)**2)**0.5
            if distance < self.ui_config.point_radius * 2 and distance < min_distance:
                min_distance = distance
                selected_index = i
        
        self.selected_point_index = selected_index
        return selected_index
    
    def delete_selected_point(self) -> bool:
        """Delete the currently selected point."""
        if self.selected_point_index is None:
            return False
        
        if self.current_mode == ConfigMode.ROI_POLYGON:
            if 0 <= self.selected_point_index < len(self.roi_points):
                del self.roi_points[self.selected_point_index]
                roi_tuples = [point.to_tuple() for point in self.roi_points]
                if self.on_roi_changed:
                    self.on_roi_changed(roi_tuples)
                self.selected_point_index = None
                return True
        
        elif self.current_mode == ConfigMode.ENTRANCE_LINE:
            if 0 <= self.selected_point_index < len(self.entrance_line_points):
                del self.entrance_line_points[self.selected_point_index]
                if len(self.entrance_line_points) == 2:
                    p1, p2 = self.entrance_line_points
                    line_coords = (p1.x, p1.y, p2.x, p2.y)
                    if self.on_entrance_line_changed:
                        self.on_entrance_line_changed(line_coords)
                self.selected_point_index = None
                return True
        
        return False
    
    def clear_current_config(self) -> None:
        """Clear the current configuration."""
        if self.current_mode == ConfigMode.ROI_POLYGON:
            self.roi_points.clear()
            if self.on_roi_changed:
                self.on_roi_changed([])
        elif self.current_mode == ConfigMode.ENTRANCE_LINE:
            self.entrance_line_points.clear()
            if self.on_entrance_line_changed:
                self.on_entrance_line_changed((0, 0, 0, 0))
        
        self.is_drawing = False
        self.selected_point_index = None
        logger.info(f"Cleared {self.current_mode.value} configuration")
    
    def load_roi_config(self, roi_points: List[Tuple[int, int]]) -> None:
        """Load existing ROI configuration."""
        self.roi_points = [UIPoint(x, y) for x, y in roi_points]
        logger.info(f"Loaded ROI configuration with {len(roi_points)} points")
    
    def load_entrance_line_config(self, line_coords: Tuple[int, int, int, int]) -> None:
        """Load existing entrance line configuration."""
        x1, y1, x2, y2 = line_coords
        self.entrance_line_points = [UIPoint(x1, y1), UIPoint(x2, y2)]
        logger.info(f"Loaded entrance line configuration: {line_coords}")
    
    def get_render_data(self) -> Dict[str, Any]:
        """Get data for rendering the UI."""
        return {
            'mode': self.current_mode.value,
            'is_drawing': self.is_drawing,
            'roi_points': [point.to_tuple() for point in self.roi_points],
            'entrance_line_points': [point.to_tuple() for point in self.entrance_line_points],
            'selected_point_index': self.selected_point_index,
            'ui_config': {
                'canvas_width': self.ui_config.canvas_width,
                'canvas_height': self.ui_config.canvas_height,
                'point_radius': self.ui_config.point_radius,
                'line_width': self.ui_config.line_width,
                'roi_color': self.ui_config.roi_color,
                'entrance_line_color': self.ui_config.entrance_line_color,
                'point_color': self.ui_config.point_color,
                'grid_enabled': self.ui_config.grid_enabled,
                'grid_size': self.ui_config.grid_size
            }
        }
    
    def export_config(self) -> Dict[str, Any]:
        """Export current configuration to dictionary."""
        config = {}
        
        if self.roi_points and len(self.roi_points) >= 3:
            config['roi_polygon'] = [point.to_tuple() for point in self.roi_points]
        
        if self.entrance_line_points and len(self.entrance_line_points) == 2:
            p1, p2 = self.entrance_line_points
            config['entrance_line'] = (p1.x, p1.y, p2.x, p2.y)
        
        return config
    
    def import_config(self, config: Dict[str, Any]) -> List[str]:
        """Import configuration from dictionary."""
        errors = []
        
        try:
            if 'roi_polygon' in config:
                roi_points = config['roi_polygon']
                validation_errors = self._validate_roi_polygon(roi_points)
                if validation_errors:
                    errors.extend(validation_errors)
                else:
                    self.load_roi_config(roi_points)
            
            if 'entrance_line' in config:
                line_coords = tuple(config['entrance_line'])
                validation_errors = self._validate_entrance_line(line_coords)
                if validation_errors:
                    errors.extend(validation_errors)
                else:
                    self.load_entrance_line_config(line_coords)
        
        except Exception as e:
            errors.append(f"Error importing configuration: {str(e)}")
        
        return errors
    
    def _snap_to_grid(self, x: int, y: int) -> Tuple[int, int]:
        """Snap coordinates to grid."""
        grid_size = self.ui_config.grid_size
        snapped_x = round(x / grid_size) * grid_size
        snapped_y = round(y / grid_size) * grid_size
        return (snapped_x, snapped_y)
    
    def _validate_roi_polygon(self, points: List[Tuple[int, int]]) -> List[str]:
        """Validate ROI polygon configuration."""
        errors = []
        
        if len(points) < 3:
            errors.append("ROI polygon must have at least 3 points")
            return errors
        
        # Check for valid coordinates
        for i, (x, y) in enumerate(points):
            if not (0 <= x <= self.ui_config.canvas_width and 0 <= y <= self.ui_config.canvas_height):
                errors.append(f"Point {i+1} is outside canvas bounds: ({x}, {y})")
        
        # Check for self-intersection (basic check)
        if len(points) > 3:
            # Simple check: no three consecutive points should be collinear
            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]
                p3 = points[(i + 2) % len(points)]
                
                # Calculate cross product
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                if abs(cross) < 1:  # Nearly collinear
                    errors.append(f"Points {i+1}, {i+2}, {i+3} are nearly collinear")
        
        # Check minimum area
        if len(points) >= 3:
            area = self._calculate_polygon_area(points)
            if area < 100:  # Minimum 100 square pixels
                errors.append(f"ROI polygon area too small: {area:.1f} square pixels")
        
        return errors
    
    def _validate_entrance_line(self, line_coords: Tuple[int, int, int, int]) -> List[str]:
        """Validate entrance line configuration."""
        errors = []
        x1, y1, x2, y2 = line_coords
        
        # Check coordinates are within bounds
        for i, coord in enumerate([x1, y1, x2, y2]):
            bound = self.ui_config.canvas_width if i % 2 == 0 else self.ui_config.canvas_height
            if not (0 <= coord <= bound):
                errors.append(f"Entrance line coordinate {i+1} is outside bounds: {coord}")
        
        # Check line length
        length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        if length < 10:
            errors.append(f"Entrance line too short: {length:.1f} pixels")
        
        return errors
    
    def _calculate_polygon_area(self, points: List[Tuple[int, int]]) -> float:
        """Calculate polygon area using shoelace formula."""
        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        return abs(area) / 2.0


class CameraConfigManager:
    """Manages camera configurations and UI interactions."""
    
    def __init__(self):
        self.camera_uis: Dict[str, CameraConfigUI] = {}
        self.active_camera: Optional[str] = None
    
    def create_camera_ui(self, camera_id: str, ui_config: Optional[UIConfig] = None) -> CameraConfigUI:
        """Create UI for a camera."""
        ui = CameraConfigUI(ui_config)
        self.camera_uis[camera_id] = ui
        
        # Set up callbacks
        ui.on_roi_changed = lambda points: self._on_roi_changed(camera_id, points)
        ui.on_entrance_line_changed = lambda coords: self._on_entrance_line_changed(camera_id, coords)
        ui.on_config_validated = lambda errors: self._on_config_validated(camera_id, errors)
        
        logger.info(f"Created camera UI for {camera_id}")
        return ui
    
    def get_camera_ui(self, camera_id: str) -> Optional[CameraConfigUI]:
        """Get UI for a camera."""
        return self.camera_uis.get(camera_id)
    
    def set_active_camera(self, camera_id: str) -> bool:
        """Set the active camera for configuration."""
        if camera_id not in self.camera_uis:
            return False
        
        self.active_camera = camera_id
        logger.info(f"Set active camera to {camera_id}")
        return True
    
    def get_active_ui(self) -> Optional[CameraConfigUI]:
        """Get the active camera UI."""
        if self.active_camera:
            return self.camera_uis.get(self.active_camera)
        return None
    
    def export_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Export all camera configurations."""
        configs = {}
        for camera_id, ui in self.camera_uis.items():
            config = ui.export_config()
            if config:  # Only include cameras with configuration
                configs[camera_id] = config
        return configs
    
    def import_all_configs(self, configs: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Import configurations for all cameras."""
        all_errors = {}
        
        for camera_id, config in configs.items():
            if camera_id not in self.camera_uis:
                self.create_camera_ui(camera_id)
            
            ui = self.camera_uis[camera_id]
            errors = ui.import_config(config)
            if errors:
                all_errors[camera_id] = errors
        
        return all_errors
    
    def _on_roi_changed(self, camera_id: str, points: List[Tuple[int, int]]) -> None:
        """Handle ROI configuration change."""
        logger.info(f"ROI changed for camera {camera_id}: {len(points)} points")
        # This would typically trigger saving to configuration system
    
    def _on_entrance_line_changed(self, camera_id: str, coords: Tuple[int, int, int, int]) -> None:
        """Handle entrance line configuration change."""
        logger.info(f"Entrance line changed for camera {camera_id}: {coords}")
        # This would typically trigger saving to configuration system
    
    def _on_config_validated(self, camera_id: str, errors: List[str]) -> None:
        """Handle configuration validation results."""
        if errors:
            logger.warning(f"Configuration validation errors for camera {camera_id}: {errors}")
        else:
            logger.info(f"Configuration validated successfully for camera {camera_id}")