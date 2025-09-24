#!/usr/bin/env python3
"""
Enhanced main entry point for the occupancy security system.
Supports multi-camera RTSP processing with health monitoring and automatic reconnection.
"""

import argparse
import logging
import signal
import sys
import time
from typing import Optional

from src.utils.logging import setup_logging
from src.services.cv_integration import CVServiceIntegration
from src.config.manager import ConfigManager
from src.config.models import CameraConfig, CameraType

# Global integration service for signal handling
integration_service: Optional[CVServiceIntegration] = None

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global integration_service
    print("\nReceived shutdown signal. Cleaning up...")
    
    if integration_service:
        integration_service.shutdown()
    
    sys.exit(0)

def main():
    """Main application entry point."""
    global integration_service
    
    parser = argparse.ArgumentParser(description="Enhanced Occupancy Security System")
    parser.add_argument("--config", default="config/system.yaml", help="Configuration file path")
    parser.add_argument("--mqtt-host", help="MQTT broker host (overrides config)")
    parser.add_argument("--mqtt-port", type=int, help="MQTT broker port (overrides config)")
    parser.add_argument("--mqtt-user", help="MQTT username (overrides config)")
    parser.add_argument("--mqtt-pass", help="MQTT password (overrides config)")
    parser.add_argument("--demo-mode", action="store_true", help="Run in demo mode with webcam")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, args.log_level.upper()))
    
    logger.info("Starting Enhanced Occupancy Security System")
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        system_config = config_manager.load_config(args.config)
        
        # Override MQTT settings if provided
        mqtt_config = system_config.mqtt
        if args.mqtt_host:
            mqtt_config.host = args.mqtt_host
        if args.mqtt_port:
            mqtt_config.port = args.mqtt_port
        if args.mqtt_user:
            mqtt_config.username = args.mqtt_user
        if args.mqtt_pass:
            mqtt_config.password = args.mqtt_pass
        
        # Initialize integration service
        integration_service = CVServiceIntegration(
            mqtt_host=mqtt_config.host,
            mqtt_port=mqtt_config.port,
            mqtt_username=mqtt_config.username,
            mqtt_password=mqtt_config.password
        )
        
        if args.demo_mode:
            # Demo mode: use webcam
            logger.info("Running in demo mode with webcam")
            
            demo_camera = CameraConfig(
                camera_id="demo_webcam",
                rtsp_url="0",  # Webcam index
                camera_type=CameraType.ZONE,
                zone_id="demo_zone",
                roi_polygon=[(100, 100), (500, 100), (500, 400), (100, 400)]
            )
            
            integration_service.add_camera(demo_camera)
            integration_service.set_zone_capacity("demo_zone", 10)
            
        else:
            # Production mode: use configured cameras
            logger.info("Running in production mode with configured cameras")
            
            for camera_id, camera_config in system_config.site.cameras.items():
                if camera_config.enabled:
                    integration_service.add_camera(camera_config)
                    logger.info(f"Added camera {camera_id}: {camera_config.rtsp_url}")
            
            # Set zone capacities
            for zone_id, zone_config in system_config.site.zones.items():
                if zone_config.enabled:
                    integration_service.set_zone_capacity(zone_id, zone_config.max_capacity)
        
        # Start processing
        integration_service.start_processing()
        logger.info("System started successfully")
        
        print("\n" + "="*60)
        print("Enhanced Occupancy Security System - RUNNING")
        print("="*60)
        print("Features:")
        print("  ✓ Multi-camera RTSP support")
        print("  ✓ Automatic reconnection with exponential backoff")
        print("  ✓ Camera health monitoring")
        print("  ✓ ROI (Region of Interest) configuration")
        print("  ✓ MQTT publishing for occupancy and health data")
        print("  ✓ Real-time person detection and tracking")
        print("\nPress Ctrl+C to stop the system")
        print("="*60)
        
        # Main loop - print status periodically
        while True:
            try:
                # Print system status every 10 seconds
                print(f"\nStatus Update - {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print("-" * 40)
                
                if args.demo_mode:
                    zones_to_check = ["demo_zone"]
                    cameras_to_check = ["demo_webcam"]
                else:
                    zones_to_check = list(system_config.site.zones.keys())
                    cameras_to_check = list(system_config.site.cameras.keys())
                
                # Show zone occupancy
                total_occupancy = 0
                for zone_id in zones_to_check:
                    occupancy = integration_service.get_zone_occupancy(zone_id)
                    max_capacity = integration_service.zone_max_capacity.get(zone_id, 0)
                    status = "🔴 OVER" if occupancy > max_capacity else "🟢 OK"
                    print(f"Zone {zone_id}: {occupancy}/{max_capacity} people {status}")
                    total_occupancy += occupancy
                
                print(f"Total Occupancy: {total_occupancy} people")
                
                # Show camera health
                print("\nCamera Status:")
                healthy_cameras = 0
                for camera_id in cameras_to_check:
                    health = integration_service.get_camera_health(camera_id)
                    if health:
                        status_icon = {
                            "ONLINE": "🟢",
                            "OFFLINE": "🔴", 
                            "DEGRADED": "🟡"
                        }.get(health.status.value, "❓")
                        
                        print(f"  {camera_id}: {status_icon} {health.status.value} - {health.fps:.1f} FPS - {health.connection_quality}")
                        if health.error_message:
                            print(f"    ⚠️  Error: {health.error_message}")
                        
                        if health.status.value == "ONLINE":
                            healthy_cameras += 1
                    else:
                        print(f"  {camera_id}: ❓ No health data")
                
                print(f"Healthy Cameras: {healthy_cameras}/{len(cameras_to_check)}")
                
                time.sleep(10)
                
            except KeyboardInterrupt:
                break
    
    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        return 1
    
    finally:
        if integration_service:
            integration_service.shutdown()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())