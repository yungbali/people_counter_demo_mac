#!/usr/bin/env python3
"""
Test script for the Enhanced CV Service.
Tests RTSP camera connections, health monitoring, and detection processing.
"""

import time
import logging
from src.services.cv_service import EnhancedCVService

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_cv_service():
    """Test the enhanced CV service functionality."""
    print("Testing Enhanced CV Service...")
    
    # Initialize service
    cv_service = EnhancedCVService()
    
    # Test with webcam (fallback if no RTSP cameras available)
    print("\n1. Testing webcam connection...")
    cv_service.process_camera_feed("test_webcam", "0")  # Use webcam index 0
    
    # Configure ROI for test camera
    print("\n2. Configuring ROI...")
    roi_points = [(100, 100), (500, 100), (500, 400), (100, 400)]
    cv_service.configure_roi("test_webcam", roi_points)
    
    # Set entrance line
    print("\n3. Setting entrance line...")
    cv_service.set_entrance_line("test_webcam", (200, 50, 200, 450))
    
    # Monitor for a few seconds
    print("\n4. Monitoring camera health and detections...")
    for i in range(10):
        # Check camera health
        health = cv_service.health_check("test_webcam")
        print(f"Camera health: {health.status.value}, FPS: {health.fps:.1f}, Quality: {health.connection_quality}")
        
        # Get detection results
        results = cv_service.get_detection_results("test_webcam")
        if results:
            count = len([d for d in results.detections if d.in_roi])
            print(f"Detections in ROI: {count}")
        else:
            print("No detection results yet")
        
        time.sleep(2)
    
    # Test RTSP camera (will fail gracefully if not available)
    print("\n5. Testing RTSP camera connection...")
    cv_service.process_camera_feed("test_rtsp", "rtsp://demo:demo@ipvmdemo.dyndns.org:5541/onvif-media/media.amp?profile=profile_1_h264&sessiontimeout=60&streamtype=unicast")
    
    # Monitor RTSP camera
    print("\n6. Monitoring RTSP camera...")
    for i in range(5):
        health = cv_service.health_check("test_rtsp")
        print(f"RTSP Camera health: {health.status.value}, FPS: {health.fps:.1f}")
        time.sleep(2)
    
    # Cleanup
    print("\n7. Cleaning up...")
    cv_service.stop_camera_feed("test_webcam")
    cv_service.stop_camera_feed("test_rtsp")
    cv_service.shutdown()
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_cv_service()