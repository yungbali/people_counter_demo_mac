"""
Core data models for the occupancy security system.
Defines the main data structures used throughout the application.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum


class CameraStatus(Enum):
    """Camera connection status enumeration."""
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    DEGRADED = "DEGRADED"


class OccupancyStatus(Enum):
    """Occupancy status enumeration."""
    OK = "OK"
    OVER = "OVER"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    """Alert type enumeration."""
    THRESHOLD_EXCEEDED = "THRESHOLD_EXCEEDED"
    CAMERA_OFFLINE = "CAMERA_OFFLINE"
    WATCHLIST_MATCH = "WATCHLIST_MATCH"


class AlertSeverity(Enum):
    """Alert severity enumeration."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class PersonDetection:
    """Individual person detection data."""
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    center_point: Tuple[int, int]
    in_roi: bool
    global_track_id: Optional[int] = None  # Cross-camera global track ID
    local_track_id: Optional[int] = None   # Original camera-specific track ID


@dataclass
class CameraHealth:
    """Camera health and status information."""
    camera_id: str
    status: CameraStatus
    last_frame_time: datetime
    fps: float
    connection_quality: str
    error_message: Optional[str] = None


@dataclass
class DetectionResult:
    """Result from computer vision processing."""
    camera_id: str
    timestamp: datetime
    detections: List[PersonDetection]
    camera_health: CameraHealth


@dataclass
class ZoneOccupancy:
    """Occupancy information for a specific zone."""
    zone_id: str
    count: int
    max_capacity: int
    status: OccupancyStatus
    last_updated: datetime


@dataclass
class OccupancyState:
    """Global occupancy state across all zones."""
    site_id: str
    timestamp: datetime
    zones: Dict[str, ZoneOccupancy]
    global_count: int
    status: OccupancyStatus


@dataclass
class Alert:
    """Alert information and metadata."""
    alert_id: str
    type: AlertType
    severity: AlertSeverity
    message: str
    zone_id: Optional[str]
    camera_id: Optional[str]
    timestamp: datetime
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    notes: Optional[str] = None


@dataclass
class AlertResult:
    """Result of alert delivery attempt."""
    alert_id: str
    success: bool
    channels_attempted: List[str]
    channels_successful: List[str]
    error_messages: Dict[str, str]
    timestamp: datetime