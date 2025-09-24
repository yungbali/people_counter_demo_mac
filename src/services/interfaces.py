"""
Service interfaces for the occupancy security system.
Defines the contracts that all service implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from src.models.core import (
    DetectionResult, 
    OccupancyState, 
    ZoneOccupancy, 
    Alert, 
    AlertResult,
    CameraHealth
)


class CVService(ABC):
    """Computer Vision Service interface for person detection and tracking."""
    
    @abstractmethod
    def process_camera_feed(self, camera_id: str, rtsp_url: str) -> None:
        """Start processing a camera feed."""
        pass
    
    @abstractmethod
    def configure_roi(self, camera_id: str, polygon_points: List[tuple]) -> None:
        """Configure region of interest for a camera."""
        pass
    
    @abstractmethod
    def set_entrance_line(self, camera_id: str, line_coords: tuple) -> None:
        """Set entrance line coordinates for in/out counting."""
        pass
    
    @abstractmethod
    def get_detection_results(self, camera_id: str) -> Optional[DetectionResult]:
        """Get latest detection results for a camera."""
        pass
    
    @abstractmethod
    def health_check(self, camera_id: str) -> CameraHealth:
        """Get health status for a camera."""
        pass
    
    @abstractmethod
    def stop_camera_feed(self, camera_id: str) -> None:
        """Stop processing a camera feed."""
        pass
    
    @abstractmethod
    def get_entrance_events(self, camera_id: str, limit: int = 50) -> List[dict]:
        """Get recent entrance/exit events for a camera."""
        pass
    
    @abstractmethod
    def get_roi_config(self, camera_id: str) -> Optional[dict]:
        """Get ROI configuration for a camera."""
        pass
    
    @abstractmethod
    def validate_roi_config(self, camera_id: str) -> List[str]:
        """Validate ROI configuration and return any errors."""
        pass
    
    @abstractmethod
    def get_cross_camera_tracking_stats(self) -> dict:
        """Get cross-camera tracking statistics."""
        pass
    
    @abstractmethod
    def get_deduplicated_global_count(self) -> int:
        """Get total count of unique people across all cameras."""
        pass
    
    @abstractmethod
    def reset_cross_camera_tracking(self) -> None:
        """Reset cross-camera tracking state."""
        pass


class AggregationService(ABC):
    """Aggregation Service interface for multi-camera data fusion."""
    
    @abstractmethod
    def process_detection_batch(self, detections: List[DetectionResult]) -> None:
        """Process a batch of detection results from multiple cameras."""
        pass
    
    @abstractmethod
    def get_global_occupancy(self) -> OccupancyState:
        """Get current global occupancy state."""
        pass
    
    @abstractmethod
    def get_zone_occupancy(self, zone_id: str) -> Optional[ZoneOccupancy]:
        """Get occupancy for a specific zone."""
        pass
    
    @abstractmethod
    def handle_camera_offline(self, camera_id: str) -> None:
        """Handle camera going offline."""
        pass
    
    @abstractmethod
    def reset_zone_count(self, zone_id: str) -> None:
        """Reset count for a specific zone."""
        pass


class AlertService(ABC):
    """Alert Service interface for threshold monitoring and notifications."""
    
    @abstractmethod
    def check_thresholds(self, occupancy_state: OccupancyState) -> List[Alert]:
        """Check occupancy against configured thresholds."""
        pass
    
    @abstractmethod
    def send_alert(self, alert: Alert, channels: List[str]) -> AlertResult:
        """Send alert via specified channels."""
        pass
    
    @abstractmethod
    def acknowledge_alert(self, alert_id: str, operator_id: str, notes: str) -> None:
        """Acknowledge an alert."""
        pass
    
    @abstractmethod
    def escalate_alert(self, alert_id: str, escalation_level: int) -> None:
        """Escalate an alert to higher severity."""
        pass
    
    @abstractmethod
    def get_active_alerts(self) -> List[Alert]:
        """Get all currently active alerts."""
        pass