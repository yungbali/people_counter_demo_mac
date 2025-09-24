#!/usr/bin/env python3
"""
Demo script showcasing ROI management functionality.
Demonstrates polygon-based ROI filtering and entrance line detection.
"""

import sys
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

from src.services.roi_manager import ROIManager, ROIPolygon, EntranceLine, LinePosition
from src.models.core import PersonDetection


def demo_basic_roi_functionality():
    """Demonstrate basic ROI polygon functionality."""
    print("🔹 Basic ROI Polygon Functionality")
    print("-" * 40)
    
    roi_manager = ROIManager()
    camera_id = "demo_camera"
    
    # Configure a rectangular ROI
    roi_points = [(100, 100), (500, 100), (500, 400), (100, 400)]
    success = roi_manager.configure_camera_roi(camera_id, roi_points, "entrance_area")
    
    print(f"✅ ROI Configuration: {success}")
    
    # Test various points
    test_points = [
        (300, 250, "Center of ROI"),
        (50, 50, "Outside ROI (top-left)"),
        (600, 500, "Outside ROI (bottom-right)"),
        (100, 100, "On ROI corner"),
        (300, 100, "On ROI edge"),
    ]
    
    print("\n📍 Point-in-ROI Tests:")
    for x, y, description in test_points:
        inside = roi_manager.is_point_in_roi(camera_id, (x, y))
        status = "✅ INSIDE" if inside else "❌ OUTSIDE"
        print(f"   {description}: ({x}, {y}) -> {status}")
    
    # Get ROI configuration
    config = roi_manager.get_camera_config(camera_id)
    if config and config.roi_polygon:
        area = config.roi_polygon.area()
        print(f"\n📐 ROI Area: {area:.1f} square pixels")
    
    print()


def demo_entrance_line_detection():
    """Demonstrate entrance line crossing detection."""
    print("🔹 Entrance Line Detection")
    print("-" * 40)
    
    roi_manager = ROIManager()
    camera_id = "entrance_camera"
    
    # Configure vertical entrance line
    line_coords = (300, 100, 300, 400)
    success = roi_manager.configure_entrance_line(camera_id, line_coords)
    
    print(f"✅ Entrance Line Configuration: {success}")
    print(f"   Line coordinates: {line_coords}")
    
    # Test line position detection
    test_positions = [
        (250, 250, "Left of line"),
        (350, 250, "Right of line"),
        (300, 250, "On the line"),
    ]
    
    print("\n📍 Line Position Tests:")
    for x, y, description in test_positions:
        position = roi_manager.get_line_position(camera_id, (x, y))
        print(f"   {description}: ({x}, {y}) -> {position.value}")
    
    # Simulate person crossing entrance
    print("\n🚶 Simulating Person Crossing:")
    track_id = 1
    
    # Person approaches from left
    crossing_positions = [
        (200, 250, "Approaching from left"),
        (250, 250, "Getting closer"),
        (280, 250, "Near the line"),
        (320, 250, "Crossed to right"),
        (350, 250, "Moving away"),
        (380, 250, "Further right"),
    ]
    
    for x, y, description in crossing_positions:
        crossing = roi_manager.detect_entrance_crossing(camera_id, track_id, (x, y))
        if crossing:
            print(f"   🎯 CROSSING DETECTED: {crossing} at ({x}, {y})")
        else:
            print(f"   {description}: ({x}, {y}) -> No crossing")
    
    print()


def demo_detection_filtering():
    """Demonstrate detection filtering with ROI."""
    print("🔹 Detection Filtering with ROI")
    print("-" * 40)
    
    roi_manager = ROIManager()
    camera_id = "filter_demo"
    
    # Configure L-shaped ROI (more complex polygon)
    roi_points = [
        (100, 100), (300, 100), (300, 200),
        (200, 200), (200, 300), (100, 300)
    ]
    roi_manager.configure_camera_roi(camera_id, roi_points, "L_shaped_area")
    
    print(f"✅ L-shaped ROI configured with {len(roi_points)} points")
    
    # Create test detections
    test_detections = [
        PersonDetection(1, (150, 120, 170, 160), 0.9, (160, 140), False),  # Inside L
        PersonDetection(2, (250, 120, 270, 160), 0.8, (260, 140), False),  # Inside L
        PersonDetection(3, (250, 220, 270, 260), 0.7, (260, 240), False),  # Outside L (cutout)
        PersonDetection(4, (150, 220, 170, 260), 0.9, (160, 240), False),  # Inside L
        PersonDetection(5, (350, 350, 370, 390), 0.6, (360, 370), False),  # Outside L
        PersonDetection(6, (50, 50, 70, 90), 0.8, (60, 70), False),        # Outside L
    ]
    
    print(f"\n📊 Original detections: {len(test_detections)}")
    for detection in test_detections:
        print(f"   Track {detection.track_id}: center {detection.center_point}, conf {detection.confidence:.1f}")
    
    # Filter detections
    filtered_detections = roi_manager.filter_detections_by_roi(camera_id, test_detections)
    
    print(f"\n✅ Filtered detections: {len(filtered_detections)}")
    for detection in filtered_detections:
        print(f"   Track {detection.track_id}: center {detection.center_point}, in_roi {detection.in_roi}")
    
    print(f"\n📈 Filtering efficiency: {len(filtered_detections)}/{len(test_detections)} detections kept")
    print()


def demo_multi_camera_setup():
    """Demonstrate multi-camera ROI configuration."""
    print("🔹 Multi-Camera ROI Setup")
    print("-" * 40)
    
    roi_manager = ROIManager()
    
    # Configure entrance camera
    entrance_roi = [(100, 100), (500, 100), (500, 400), (100, 400)]
    entrance_line = (300, 100, 300, 400)
    
    roi_manager.configure_camera_roi("entrance", entrance_roi, "entrance_zone")
    roi_manager.configure_entrance_line("entrance", entrance_line)
    
    # Configure lounge camera (circular ROI approximation)
    center_x, center_y, radius = 320, 240, 150
    lounge_roi = []
    for i in range(8):  # Octagon
        angle = i * 2 * np.pi / 8
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        lounge_roi.append((x, y))
    
    roi_manager.configure_camera_roi("lounge", lounge_roi, "lounge_zone")
    
    print("✅ Configured cameras:")
    print("   - Entrance: Rectangular ROI + Entrance Line")
    print("   - Lounge: Octagonal ROI (no entrance line)")
    
    # Test configurations
    cameras = ["entrance", "lounge"]
    test_point = (320, 240)
    
    print(f"\n📍 Testing point {test_point} on all cameras:")
    for camera in cameras:
        inside = roi_manager.is_point_in_roi(camera, test_point)
        config = roi_manager.get_camera_config(camera)
        has_line = config.entrance_line is not None if config else False
        
        print(f"   {camera}: {'✅ INSIDE' if inside else '❌ OUTSIDE'} ROI, "
              f"{'✅ HAS' if has_line else '❌ NO'} entrance line")
    
    # Export configurations
    print("\n💾 Configuration Export:")
    for camera in cameras:
        config_data = roi_manager.export_config(camera)
        if config_data:
            roi_area = config_data.get('roi_polygon', {}).get('area', 0)
            line_length = config_data.get('entrance_line', {}).get('length', 0)
            print(f"   {camera}: ROI area {roi_area:.1f}, Line length {line_length:.1f}")
    
    print()


def demo_configuration_validation():
    """Demonstrate ROI configuration validation."""
    print("🔹 Configuration Validation")
    print("-" * 40)
    
    roi_manager = ROIManager()
    
    # Test valid configuration
    print("Testing valid configuration:")
    valid_roi = [(0, 0), (200, 0), (200, 200), (0, 200)]
    valid_line = (100, 0, 100, 200)
    
    roi_manager.configure_camera_roi("valid_camera", valid_roi)
    roi_manager.configure_entrance_line("valid_camera", valid_line)
    
    errors = roi_manager.validate_roi_config("valid_camera")
    if errors:
        print(f"   ❌ Validation errors: {errors}")
    else:
        print("   ✅ Configuration is valid")
    
    # Test invalid configuration
    print("\nTesting invalid configuration:")
    invalid_roi = [(0, 0), (2, 0), (0, 2)]  # Very small triangle
    invalid_line = (100, 100, 101, 101)     # Very short line
    
    roi_manager.configure_camera_roi("invalid_camera", invalid_roi)
    roi_manager.configure_entrance_line("invalid_camera", invalid_line)
    
    errors = roi_manager.validate_roi_config("invalid_camera")
    if errors:
        print(f"   ❌ Validation errors found:")
        for error in errors:
            print(f"      - {error}")
    else:
        print("   ✅ Configuration is valid")
    
    print()


def demo_performance_test():
    """Demonstrate performance with many detections."""
    print("🔹 Performance Test")
    print("-" * 40)
    
    roi_manager = ROIManager()
    camera_id = "performance_test"
    
    # Configure complex polygon (many-sided)
    center_x, center_y, radius = 320, 240, 200
    roi_points = []
    num_sides = 20  # 20-sided polygon
    
    for i in range(num_sides):
        angle = i * 2 * np.pi / num_sides
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        roi_points.append((x, y))
    
    roi_manager.configure_camera_roi(camera_id, roi_points, "complex_polygon")
    
    print(f"✅ Configured {num_sides}-sided polygon ROI")
    
    # Generate many test detections
    num_detections = 100
    detections = []
    
    for i in range(num_detections):
        x = 50 + (i % 20) * 30
        y = 50 + (i // 20) * 50
        
        detection = PersonDetection(
            track_id=i,
            bbox=(x, y, x+20, y+40),
            confidence=0.7 + (i % 3) * 0.1,
            center_point=(x+10, y+20),
            in_roi=False
        )
        detections.append(detection)
    
    print(f"📊 Generated {num_detections} test detections")
    
    # Time the filtering operation
    import time
    start_time = time.time()
    
    filtered_detections = roi_manager.filter_detections_by_roi(camera_id, detections)
    
    end_time = time.time()
    processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    print(f"⚡ Filtering completed in {processing_time:.2f}ms")
    print(f"📈 Kept {len(filtered_detections)}/{num_detections} detections")
    print(f"🔄 Processing rate: {num_detections/processing_time*1000:.0f} detections/second")
    
    print()


def main():
    """Run all ROI functionality demos."""
    print("🎯 ROI Management System Demo")
    print("=" * 50)
    print()
    
    try:
        demo_basic_roi_functionality()
        demo_entrance_line_detection()
        demo_detection_filtering()
        demo_multi_camera_setup()
        demo_configuration_validation()
        demo_performance_test()
        
        print("🎉 All demos completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ Polygon-based ROI configuration")
        print("✅ Point-in-polygon detection")
        print("✅ Entrance line crossing detection")
        print("✅ Detection filtering by ROI")
        print("✅ Multi-camera support")
        print("✅ Configuration validation")
        print("✅ Performance optimization")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)