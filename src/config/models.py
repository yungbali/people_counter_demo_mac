"""
Configuration data models for the occupancy security system.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class CameraType(Enum):
    """Camera type enumeration."""
    ENTRANCE = "entrance"
    ZONE = "zone"


class NotificationChannel(Enum):
    """Notification channel enumeration."""
    WHATSAPP = "whatsapp"
    SMS = "sms"
    EMAIL = "email"


@dataclass
class CameraConfig:
    """Configuration for a single camera."""
    camera_id: str
    rtsp_url: str
    camera_type: CameraType
    zone_id: str
    roi_polygon: Optional[List[Tuple[int, int]]] = None
    entrance_line: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2
    enabled: bool = True


@dataclass
class ZoneConfig:
    """Configuration for a monitoring zone."""
    zone_id: str
    name: str
    max_capacity: int
    alert_threshold: int
    zone_type: str  # 'entrance' | 'lounge' | 'general'
    enabled: bool = True


@dataclass
class AlertRuleConfig:
    """Configuration for alert rules."""
    rule_id: str
    zone_id: str
    rule_type: str
    threshold_value: int
    channels: List[NotificationChannel]
    hold_duration: int = 3  # seconds
    rate_limit: int = 10  # alerts per hour
    enabled: bool = True


@dataclass
class NotificationChannelConfig:
    """Configuration for notification channels."""
    channel: NotificationChannel
    api_key: str
    recipients: List[str]
    rate_limit: int
    enabled: bool = True


@dataclass
class SiteConfig:
    """Main site configuration."""
    site_id: str
    name: str
    cameras: Dict[str, CameraConfig]
    zones: Dict[str, ZoneConfig]
    alert_rules: Dict[str, AlertRuleConfig]
    notification_channels: Dict[str, NotificationChannelConfig]
    timezone: str = "UTC"


@dataclass
class MQTTConfig:
    """MQTT broker configuration."""
    host: str = "localhost"
    port: int = 1883
    username: Optional[str] = None
    password: Optional[str] = None
    topic_prefix: str = "site"
    keepalive: int = 60


@dataclass
class DatabaseConfig:
    """Database configuration."""
    # Time series database (for occupancy data)
    timeseries_url: str = "postgresql://localhost:5432/occupancy_ts"
    
    # Metadata database (for configuration and events)
    metadata_url: str = "postgresql://localhost:5432/occupancy_meta"
    
    # Connection pool settings
    pool_size: int = 10
    max_overflow: int = 20


@dataclass
class SystemConfig:
    """Complete system configuration."""
    site: SiteConfig
    mqtt: MQTTConfig
    database: DatabaseConfig
    
    # Processing settings
    detection_interval: float = 0.1  # seconds
    aggregation_interval: float = 1.0  # seconds
    alert_check_interval: float = 5.0  # seconds
    
    # Performance settings
    max_concurrent_cameras: int = 10
    frame_buffer_size: int = 30
    
    # Optional features
    biometric_enabled: bool = False
    aws_telemetry_enabled: bool = False