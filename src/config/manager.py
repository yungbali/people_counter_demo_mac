"""
Configuration management system for the occupancy security system.
Handles loading, validation, and management of system configuration.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

from .models import (
    SystemConfig, 
    SiteConfig, 
    CameraConfig, 
    ZoneConfig, 
    AlertRuleConfig,
    NotificationChannelConfig,
    MQTTConfig,
    DatabaseConfig,
    CameraType,
    NotificationChannel
)


class ConfigurationManager:
    """Manages system configuration loading and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_path = config_path or "config/system.yaml"
        self._config: Optional[SystemConfig] = None
    
    def load_config(self) -> SystemConfig:
        """Load configuration from file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)
        
        self._config = self._parse_config(config_data)
        return self._config
    
    def get_config(self) -> SystemConfig:
        """Get current configuration, loading if necessary."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def save_config(self, config: SystemConfig) -> None:
        """Save configuration to file."""
        config_data = asdict(config)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            else:
                json.dump(config_data, f, indent=2, default=str)
        
        self._config = config
    
    def _parse_config(self, config_data: Dict[str, Any]) -> SystemConfig:
        """Parse configuration data into SystemConfig object."""
        # Parse site configuration
        site_data = config_data.get('site', {})
        
        # Parse cameras
        cameras = {}
        for cam_id, cam_data in site_data.get('cameras', {}).items():
            cameras[cam_id] = CameraConfig(
                camera_id=cam_id,
                rtsp_url=cam_data['rtsp_url'],
                camera_type=CameraType(cam_data['camera_type']),
                zone_id=cam_data['zone_id'],
                roi_polygon=cam_data.get('roi_polygon'),
                entrance_line=cam_data.get('entrance_line'),
                enabled=cam_data.get('enabled', True)
            )
        
        # Parse zones
        zones = {}
        for zone_id, zone_data in site_data.get('zones', {}).items():
            zones[zone_id] = ZoneConfig(
                zone_id=zone_id,
                name=zone_data['name'],
                max_capacity=zone_data['max_capacity'],
                alert_threshold=zone_data['alert_threshold'],
                zone_type=zone_data['zone_type'],
                enabled=zone_data.get('enabled', True)
            )
        
        # Parse alert rules
        alert_rules = {}
        for rule_id, rule_data in site_data.get('alert_rules', {}).items():
            channels = [NotificationChannel(ch) for ch in rule_data['channels']]
            alert_rules[rule_id] = AlertRuleConfig(
                rule_id=rule_id,
                zone_id=rule_data['zone_id'],
                rule_type=rule_data['rule_type'],
                threshold_value=rule_data['threshold_value'],
                hold_duration=rule_data.get('hold_duration', 3),
                channels=channels,
                rate_limit=rule_data.get('rate_limit', 10),
                enabled=rule_data.get('enabled', True)
            )
        
        # Parse notification channels
        notification_channels = {}
        for ch_name, ch_data in site_data.get('notification_channels', {}).items():
            notification_channels[ch_name] = NotificationChannelConfig(
                channel=NotificationChannel(ch_name),
                api_key=ch_data['api_key'],
                recipients=ch_data['recipients'],
                rate_limit=ch_data.get('rate_limit', 10),
                enabled=ch_data.get('enabled', True)
            )
        
        site_config = SiteConfig(
            site_id=site_data['site_id'],
            name=site_data['name'],
            timezone=site_data.get('timezone', 'UTC'),
            cameras=cameras,
            zones=zones,
            alert_rules=alert_rules,
            notification_channels=notification_channels
        )
        
        # Parse MQTT configuration
        mqtt_data = config_data.get('mqtt', {})
        mqtt_config = MQTTConfig(
            host=mqtt_data.get('host', 'localhost'),
            port=mqtt_data.get('port', 1883),
            username=mqtt_data.get('username'),
            password=mqtt_data.get('password'),
            topic_prefix=mqtt_data.get('topic_prefix', 'site'),
            keepalive=mqtt_data.get('keepalive', 60)
        )
        
        # Parse database configuration
        db_data = config_data.get('database', {})
        db_config = DatabaseConfig(
            timeseries_url=db_data.get('timeseries_url', 'postgresql://localhost:5432/occupancy_ts'),
            metadata_url=db_data.get('metadata_url', 'postgresql://localhost:5432/occupancy_meta'),
            pool_size=db_data.get('pool_size', 10),
            max_overflow=db_data.get('max_overflow', 20)
        )
        
        # Create system configuration
        return SystemConfig(
            site=site_config,
            mqtt=mqtt_config,
            database=db_config,
            detection_interval=config_data.get('detection_interval', 0.1),
            aggregation_interval=config_data.get('aggregation_interval', 1.0),
            alert_check_interval=config_data.get('alert_check_interval', 5.0),
            max_concurrent_cameras=config_data.get('max_concurrent_cameras', 10),
            frame_buffer_size=config_data.get('frame_buffer_size', 30),
            biometric_enabled=config_data.get('biometric_enabled', False),
            aws_telemetry_enabled=config_data.get('aws_telemetry_enabled', False)
        )
    
    def get_camera_config(self, camera_id: str) -> Optional[CameraConfig]:
        """Get configuration for a specific camera."""
        config = self.get_config()
        return config.site.cameras.get(camera_id)
    
    def get_zone_config(self, zone_id: str) -> Optional[ZoneConfig]:
        """Get configuration for a specific zone."""
        config = self.get_config()
        return config.site.zones.get(zone_id)
    
    def get_alert_rules_for_zone(self, zone_id: str) -> Dict[str, AlertRuleConfig]:
        """Get all alert rules for a specific zone."""
        config = self.get_config()
        return {
            rule_id: rule for rule_id, rule in config.site.alert_rules.items()
            if rule.zone_id == zone_id and rule.enabled
        }
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        config = self.get_config()
        
        # Validate cameras reference valid zones
        for camera_id, camera in config.site.cameras.items():
            if camera.zone_id not in config.site.zones:
                errors.append(f"Camera {camera_id} references non-existent zone {camera.zone_id}")
        
        # Validate alert rules reference valid zones
        for rule_id, rule in config.site.alert_rules.items():
            if rule.zone_id not in config.site.zones:
                errors.append(f"Alert rule {rule_id} references non-existent zone {rule.zone_id}")
        
        # Validate notification channels are configured
        for rule_id, rule in config.site.alert_rules.items():
            for channel in rule.channels:
                if channel.value not in config.site.notification_channels:
                    errors.append(f"Alert rule {rule_id} uses unconfigured channel {channel.value}")
        
        return errors


# Global configuration manager instance
config_manager = ConfigurationManager()