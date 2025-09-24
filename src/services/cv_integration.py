"""
Integration module that connects the Enhanced CV Service with MQTT publishing.
Extends the original demo functionality to work with multiple RTSP cameras.
"""

import json
import logging
import threading
import time
from datetime import datetime
from typing import Dict, Optional

import paho.mqtt.client as mqtt

from src.services.cv_service import EnhancedCVService
from src.models.core import DetectionResult, CameraHealth
from src.config.models import CameraConfig, CameraType

logger = logging.getLogger(__name__)


class MQTTPublisher:
    """MQTT publisher for occupancy and camera health data."""
    
    def __init__(self, host: str = "localhost", port: int = 1883, 
                 username: Optional[str] = None, password: Optional[str] = None,
                 topic_prefix: str = "site/demo"):
        self.topic_prefix = topic_prefix.rstrip("/")
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        
        if username and password:
            self.client.username_pw_set(username, password)
        
        try:
            self.client.connect(host, port, 60)
            self.client.loop_start()
            self.connected = True
            logger.info(f"Connected to MQTT broker at {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            self.connected = False
    
    def publish_occupancy_state(self, zone_id: str, count: int, max_capacity: int, status: str):
        """Publish occupancy state for a zone."""
        if not self.connected:
            return
        
        topic = f"{self.topic_prefix}/occupancy/{zone_id}/state"
        payload = {
            "timestamp": int(time.time()),
            "zone_id": zone_id,
            "count": count,
            "max_capacity": max_capacity,
            "status": status
        }
        
        self.client.publish(topic, json.dumps(payload), qos=0, retain=True)
    
    def publish_camera_health(self, camera_id: str, health: CameraHealth):
        """Publish camera health status."""
        if not self.connected:
            return
        
        topic = f"{self.topic_prefix}/cameras/{camera_id}/health"
        payload = {
            "timestamp": int(time.time()),
            "camera_id": camera_id,
            "status": health.status.value,
            "fps": health.fps,
            "connection_quality": health.connection_quality,
            "last_frame_time": int(health.last_frame_time.timestamp()),
            "error_message": health.error_message
        }
        
        self.client.publish(topic, json.dumps(payload), qos=0, retain=False)
    
    def publish_detection_event(self, camera_id: str, zone_id: str, detection_count: int):
        """Publish detection event."""
        if not self.connected:
            return
        
        topic = f"{self.topic_prefix}/occupancy/{zone_id}/events"
        payload = {
            "timestamp": int(time.time()),
            "camera_id": camera_id,
            "zone_id": zone_id,
            "detection_count": detection_count,
            "event_type": "detection_update"
        }
        
        self.client.publish(topic, json.dumps(payload), qos=1, retain=False)
    
    def disconnect(self):
        """Disconnect from MQTT broker."""
        if self.connected:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False


class CVServiceIntegration:
    """Integration service that manages CV processing and MQTT publishing."""
    
    def __init__(self, mqtt_host: str = "localhost", mqtt_port: int = 1883,
                 mqtt_username: Optional[str] = None, mqtt_password: Optional[str] = None):
        
        # Initialize services
        self.cv_service = EnhancedCVService()
        self.mqtt_publisher = MQTTPublisher(
            host=mqtt_host,
            port=mqtt_port,
            username=mqtt_username,
            password=mqtt_password
        )
        
        # Camera configurations
        self.camera_configs: Dict[str, CameraConfig] = {}
        self.zone_occupancy: Dict[str, int] = {}
        self.zone_max_capacity: Dict[str, int] = {}
        
        # Processing control
        self.running = False
        self.processing_thread: Optional[threading.Thread] = None
        
        logger.info("CV Service Integration initialized")
    
    def add_camera(self, camera_config: CameraConfig):
        """Add a camera to the system."""
        camera_id = camera_config.camera_id
        
        # Store configuration
        self.camera_configs[camera_id] = camera_config
        
        # Initialize zone occupancy tracking
        if camera_config.zone_id not in self.zone_occupancy:
            self.zone_occupancy[camera_config.zone_id] = 0
            self.zone_max_capacity[camera_config.zone_id] = 40  # Default capacity
        
        # Start camera processing
        self.cv_service.process_camera_feed(camera_id, camera_config.rtsp_url)
        
        # Configure ROI if specified
        if camera_config.roi_polygon:
            self.cv_service.configure_roi(camera_id, camera_config.roi_polygon)
        
        # Configure entrance line if specified
        if camera_config.entrance_line:
            self.cv_service.set_entrance_line(camera_id, camera_config.entrance_line)
        
        logger.info(f"Added camera {camera_id} for zone {camera_config.zone_id}")
    
    def set_zone_capacity(self, zone_id: str, max_capacity: int):
        """Set maximum capacity for a zone."""
        self.zone_max_capacity[zone_id] = max_capacity
        logger.info(f"Set max capacity for zone {zone_id}: {max_capacity}")
    
    def start_processing(self):
        """Start the main processing loop."""
        if self.running:
            return
        
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info("Started CV processing and MQTT publishing")
    
    def stop_processing(self):
        """Stop the processing loop."""
        self.running = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        logger.info("Stopped CV processing")
    
    def _processing_loop(self):
        """Main processing loop that handles detection results and MQTT publishing."""
        last_health_publish = 0
        last_occupancy_publish = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # Process detection results from all cameras
                zone_counts = {}
                
                for camera_id, config in self.camera_configs.items():
                    # Get detection results
                    results = self.cv_service.get_detection_results(camera_id)
                    if results:
                        # Count detections in ROI
                        count = len([d for d in results.detections if d.in_roi])
                        
                        # Aggregate by zone
                        zone_id = config.zone_id
                        if zone_id not in zone_counts:
                            zone_counts[zone_id] = 0
                        zone_counts[zone_id] += count
                        
                        # Publish detection event
                        self.mqtt_publisher.publish_detection_event(camera_id, zone_id, count)
                    
                    # Publish camera health (every 10 seconds)
                    if current_time - last_health_publish >= 10:
                        health = self.cv_service.health_check(camera_id)
                        self.mqtt_publisher.publish_camera_health(camera_id, health)
                
                # Update zone occupancy and publish state (every 2 seconds)
                if current_time - last_occupancy_publish >= 2:
                    for zone_id, count in zone_counts.items():
                        self.zone_occupancy[zone_id] = count
                        max_capacity = self.zone_max_capacity.get(zone_id, 40)
                        status = "OVER" if count > max_capacity else "OK"
                        
                        self.mqtt_publisher.publish_occupancy_state(
                            zone_id, count, max_capacity, status
                        )
                    
                    last_occupancy_publish = current_time
                
                if current_time - last_health_publish >= 10:
                    last_health_publish = current_time
                
                time.sleep(1.0)  # Process every second
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(1.0)
    
    def get_zone_occupancy(self, zone_id: str) -> int:
        """Get current occupancy for a zone."""
        return self.zone_occupancy.get(zone_id, 0)
    
    def get_camera_health(self, camera_id: str) -> Optional[CameraHealth]:
        """Get health status for a camera."""
        return self.cv_service.health_check(camera_id)
    
    def shutdown(self):
        """Shutdown the integration service."""
        logger.info("Shutting down CV Service Integration")
        
        self.stop_processing()
        self.cv_service.shutdown()
        self.mqtt_publisher.disconnect()


def main():
    """Example usage of the CV Service Integration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced CV Service with MQTT Integration")
    parser.add_argument("--mqtt-host", default="localhost", help="MQTT broker host")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--mqtt-user", help="MQTT username")
    parser.add_argument("--mqtt-pass", help="MQTT password")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create integration service
    integration = CVServiceIntegration(
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
        mqtt_username=args.mqtt_user,
        mqtt_password=args.mqtt_pass
    )
    
    # Example: Add cameras from configuration
    # This would normally be loaded from config file
    entrance_camera = CameraConfig(
        camera_id="entrance",
        rtsp_url="0",  # Use webcam for demo
        camera_type=CameraType.ENTRANCE,
        zone_id="entrance_zone",
        roi_polygon=[(100, 100), (500, 100), (500, 400), (100, 400)],
        entrance_line=(200, 50, 200, 450)
    )
    
    integration.add_camera(entrance_camera)
    integration.set_zone_capacity("entrance_zone", 10)
    
    # Start processing
    integration.start_processing()
    
    try:
        print("CV Service Integration running. Press Ctrl+C to stop.")
        while True:
            # Print status every 5 seconds
            occupancy = integration.get_zone_occupancy("entrance_zone")
            health = integration.get_camera_health("entrance")
            
            print(f"Zone occupancy: {occupancy}, Camera status: {health.status.value if health else 'Unknown'}")
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        integration.shutdown()


if __name__ == "__main__":
    main()