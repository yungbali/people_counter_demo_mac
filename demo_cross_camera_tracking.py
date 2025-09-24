#!/usr/bin/env python3
"""
Demo script for cross-camera person tracking and de-duplication.
Shows how the cross-camera tracker works with multiple detection streams.
"""

import numpy as np
import time
from datetime import datetime
from src.services.cross_camera_tracker import CrossCameraTracker
from src.models.core import PersonDetection, DetectionResult, CameraHealth, CameraStatus


def create_mock_detection_result(camera_id: str, detections: list) -> DetectionResult:
    """Create a mock detection result for testing."""
    camera_health = CameraHealth(
        camera_id=camera_id,
        status=CameraStatus.ONLINE,
        last_frame_time=datetime.now(),
        fps=30.0,
        connection_quality="GOOD"
    )
    
    return DetectionResult(
        camera_id=camera_id,
        timestamp=datetime.now(),
        detections=detections,
        camera_health=camera_health
    )


def create_person_detection(track_id: int, bbox: tuple, confidence: float = 0.8) -> PersonDetection:
    """Create a person detection for testing."""
    center_x = (bbox[0] + bbox[2]) // 2
    center_y = (bbox[1] + bbox[3]) // 2
    
    return PersonDetection(
        track_id=track_id,
        bbox=bbox,
        confidence=confidence,
        center_point=(center_x, center_y),
        in_roi=True,
        local_track_id=track_id
    )


def demo_single_camera_tracking():
    """Demo single camera tracking consistency."""
    print("=== Single Camera Tracking Demo ===")
    
    tracker = CrossCameraTracker()
    
    # Person appears in camera 1
    detections1 = [create_person_detection(1, (100, 100, 200, 300))]
    result1 = tracker.process_detections([create_mock_detection_result("cam1", detections1)])
    
    global_id1 = result1["cam1"][0].track_id
    print(f"Frame 1: Person with local ID 1 gets global ID {global_id1}")
    
    # Same person moves slightly
    detections2 = [create_person_detection(1, (110, 110, 210, 310))]
    result2 = tracker.process_detections([create_mock_detection_result("cam1", detections2)])
    
    global_id2 = result2["cam1"][0].track_id
    print(f"Frame 2: Same person (local ID 1) maintains global ID {global_id2}")
    
    print(f"Tracking consistency: {'✓' if global_id1 == global_id2 else '✗'}")
    print()


def demo_multi_camera_no_overlap():
    """Demo multi-camera tracking with no person overlap."""
    print("=== Multi-Camera No Overlap Demo ===")
    
    tracker = CrossCameraTracker()
    
    # Different people in different cameras
    detections_cam1 = [create_person_detection(1, (100, 100, 200, 300))]
    detections_cam2 = [create_person_detection(1, (300, 100, 400, 300))]
    
    results = tracker.process_detections([
        create_mock_detection_result("cam1", detections_cam1),
        create_mock_detection_result("cam2", detections_cam2)
    ])
    
    global_id1 = results["cam1"][0].track_id
    global_id2 = results["cam2"][0].track_id
    
    print(f"Camera 1: Person gets global ID {global_id1}")
    print(f"Camera 2: Person gets global ID {global_id2}")
    print(f"Different global IDs: {'✓' if global_id1 != global_id2 else '✗'}")
    print(f"Total unique people: {tracker.get_deduplicated_global_count()}")
    print()


def demo_cross_camera_matching():
    """Demo cross-camera person matching with embeddings."""
    print("=== Cross-Camera Matching Demo ===")
    
    tracker = CrossCameraTracker(similarity_threshold=0.6)
    
    # Create similar frame data for embedding extraction
    frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame2 = frame1.copy()  # Identical frames for testing
    
    # Person appears in cam1
    detections_cam1 = [create_person_detection(1, (100, 100, 200, 300), confidence=0.9)]
    results1 = tracker.process_detections(
        [create_mock_detection_result("cam1", detections_cam1)],
        {"cam1": frame1}
    )
    
    global_id1 = results1["cam1"][0].track_id
    print(f"Frame 1: Person appears in cam1 with global ID {global_id1}")
    
    # Same person appears in cam2 (simulating movement)
    detections_cam2 = [create_person_detection(1, (100, 100, 200, 300), confidence=0.9)]
    results2 = tracker.process_detections([
        create_mock_detection_result("cam1", []),  # Person left cam1
        create_mock_detection_result("cam2", detections_cam2)
    ], {"cam1": frame1, "cam2": frame2})
    
    if results2["cam2"]:
        global_id2 = results2["cam2"][0].track_id
        print(f"Frame 2: Person appears in cam2 with global ID {global_id2}")
        
        # Note: Without sophisticated embeddings, matching may not occur
        # This demonstrates the framework is working
        print(f"Potential match: {'✓' if global_id1 == global_id2 else '✗ (expected with simple embeddings)'}")
    
    print(f"Total unique people: {tracker.get_deduplicated_global_count()}")
    print()


def demo_tracking_persistence():
    """Demo tracking persistence across occlusions."""
    print("=== Tracking Persistence Demo ===")
    
    tracker = CrossCameraTracker()
    
    # Person appears
    detections1 = [create_person_detection(1, (100, 100, 200, 300))]
    result1 = tracker.process_detections([create_mock_detection_result("cam1", detections1)])
    global_id1 = result1["cam1"][0].track_id
    print(f"Frame 1: Person appears with global ID {global_id1}")
    
    # Person disappears (occlusion)
    result2 = tracker.process_detections([create_mock_detection_result("cam1", [])])
    print("Frame 2: Person disappears (occlusion)")
    
    # Person reappears quickly
    detections3 = [create_person_detection(1, (110, 110, 210, 310))]
    result3 = tracker.process_detections([create_mock_detection_result("cam1", detections3)])
    
    if result3["cam1"]:
        global_id3 = result3["cam1"][0].track_id
        print(f"Frame 3: Person reappears with global ID {global_id3}")
        print(f"ID persistence: {'✓' if global_id1 == global_id3 else '✗'}")
    else:
        print("Frame 3: Person not detected (track may have been cleaned up)")
    
    print()


def demo_tracking_statistics():
    """Demo tracking statistics and monitoring."""
    print("=== Tracking Statistics Demo ===")
    
    tracker = CrossCameraTracker()
    
    # Add multiple people across cameras
    detections_cam1 = [
        create_person_detection(1, (100, 100, 200, 300)),
        create_person_detection(2, (300, 100, 400, 300))
    ]
    detections_cam2 = [
        create_person_detection(1, (100, 100, 200, 300))
    ]
    
    tracker.process_detections([
        create_mock_detection_result("cam1", detections_cam1),
        create_mock_detection_result("cam2", detections_cam2)
    ])
    
    stats = tracker.get_tracking_stats()
    
    print("Tracking Statistics:")
    print(f"  Total global tracks: {stats['total_global_tracks']}")
    print(f"  Active global tracks: {stats['active_global_tracks']}")
    print(f"  Global count: {stats['global_count']}")
    print(f"  Tracks by camera: {stats['tracks_by_camera']}")
    print(f"  Next global ID: {stats['next_global_id']}")
    print()


def main():
    """Run all demos."""
    print("Cross-Camera Person Tracking Demo")
    print("=" * 50)
    print()
    
    demo_single_camera_tracking()
    demo_multi_camera_no_overlap()
    demo_cross_camera_matching()
    demo_tracking_persistence()
    demo_tracking_statistics()
    
    print("Demo completed! ✓")
    print()
    print("Key Features Demonstrated:")
    print("- Single camera tracking consistency")
    print("- Multi-camera unique person identification")
    print("- Cross-camera person matching framework")
    print("- Tracking persistence across occlusions")
    print("- Comprehensive tracking statistics")


if __name__ == "__main__":
    main()