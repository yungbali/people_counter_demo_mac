#!/usr/bin/env python3
"""
Demo script for the Enhanced CV Service without requiring actual cameras.
Shows the service architecture and health monitoring capabilities.
"""

import time
import logging
from src.services.cv_integration import CVServiceIntegration
from src.config.models import CameraConfig, CameraType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def demo_cv_service():
    """Demo the enhanced CV service functionality."""
    print("="*60)
    print("Enhanced CV Service Demo")
    print("="*60)
    print("Features demonstrated:")
    print("  ✓ Multi-camera RTSP support")
    print("  ✓ Automatic reconnection with exponential backoff")
    print("  ✓ Camera health monitoring")
    print("  ✓ ROI (Region of Interest) configuration")
    print("  ✓ MQTT publishing simulation")
    print("  ✓ Real-time status monitoring")
    print("="*60)
    
    # Initialize integration service (MQTT will fail gracefully if broker not available)
    integration = CVServiceIntegration(
        mqtt_host="localhost",
        mqtt_port=1883
    )
    
    # Add demo cameras (these will show connection attempts and health monitoring)
    print("\n1. Adding demo cameras...")
    
    # Camera 1: Entrance camera
    entrance_camera = CameraConfig(
        camera_id="entrance_cam",
        rtsp_url="rtsp://demo.camera1/stream",  # Fake RTSP URL
        camera_type=CameraType.ENTRANCE,
        zone_id="entrance_zone",
        roi_polygon=[(100, 100), (500, 100), (500, 400), (100, 400)],
        entrance_line=(200, 50, 200, 450)
    )
    
    # Camera 2: Lounge camera
    lounge_camera = CameraConfig(
        camera_id="lounge_cam",
        rtsp_url="rtsp://demo.camera2/stream",  # Fake RTSP URL
        camera_type=CameraType.ZONE,
        zone_id="lounge_zone",
        roi_polygon=[(50, 50), (590, 50), (590, 430), (50, 430)]
    )
    
    integration.add_camera(entrance_camera)
    integration.add_camera(lounge_camera)
    
    # Set zone capacities
    integration.set_zone_capacity("entrance_zone", 10)
    integration.set_zone_capacity("lounge_zone", 25)
    
    print("✓ Added 2 demo cameras")
    
    # Start processing
    print("\n2. Starting CV processing...")
    integration.start_processing()
    print("✓ Processing started")
    
    # Monitor for 30 seconds
    print("\n3. Monitoring system status (30 seconds)...")
    print("   Note: Cameras will show connection attempts and health status")
    print("   This demonstrates the robust error handling and reconnection logic")
    
    for i in range(6):  # 6 iterations of 5 seconds each
        print(f"\n--- Status Update {i+1}/6 ---")
        
        # Show zone occupancy (will be 0 since no actual cameras)
        for zone_id in ["entrance_zone", "lounge_zone"]:
            occupancy = integration.get_zone_occupancy(zone_id)
            max_capacity = integration.zone_max_capacity.get(zone_id, 0)
            status = "🔴 OVER" if occupancy > max_capacity else "🟢 OK"
            print(f"Zone {zone_id}: {occupancy}/{max_capacity} people {status}")
        
        # Show camera health
        print("Camera Health:")
        for camera_id in ["entrance_cam", "lounge_cam"]:
            health = integration.get_camera_health(camera_id)
            if health:
                status_icon = {
                    "ONLINE": "🟢",
                    "OFFLINE": "🔴", 
                    "DEGRADED": "🟡"
                }.get(health.status.value, "❓")
                
                print(f"  {camera_id}: {status_icon} {health.status.value} - {health.fps:.1f} FPS - {health.connection_quality}")
                if health.error_message:
                    print(f"    ⚠️  Error: {health.error_message}")
            else:
                print(f"  {camera_id}: ❓ No health data")
        
        time.sleep(5)
    
    print("\n4. Shutting down...")
    integration.shutdown()
    print("✓ Demo completed successfully!")
    
    print("\n" + "="*60)
    print("Demo Summary:")
    print("="*60)
    print("The Enhanced CV Service demonstrated:")
    print("  ✅ Camera management with health monitoring")
    print("  ✅ Automatic reconnection attempts with exponential backoff")
    print("  ✅ ROI and entrance line configuration")
    print("  ✅ Zone-based occupancy tracking")
    print("  ✅ Graceful error handling for unavailable cameras")
    print("  ✅ Clean shutdown and resource management")
    print("\nIn a production environment with real RTSP cameras,")
    print("this system would provide robust multi-camera person")
    print("detection and occupancy monitoring.")
    print("="*60)

if __name__ == "__main__":
    demo_cv_service()