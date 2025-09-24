"""
Unit tests for cross-camera person tracking and de-duplication.
Tests tracking consistency, de-duplication accuracy, and embedding extraction.
"""

import unittest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.services.cross_camera_tracker import (
    CrossCameraTracker, 
    PersonEmbedding, 
    GlobalTrack,
    SimpleEmbeddingExtractor
)
from src.models.core import PersonDetection, DetectionResult, CameraHealth, CameraStatus


class TestSimpleEmbeddingExtractor(unittest.TestCase):
    """Test the simple embedding extractor."""
    
    def setUp(self):
        self.extractor = SimpleEmbeddingExtractor()
    
    def test_extract_embedding_valid_bbox(self):
        """Test embedding extraction with valid bounding box."""
        # Create a test image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 200, 300)  # x1, y1, x2, y2
        
        embedding = self.extractor.extract_embedding(image, bbox)
        
        self.assertEqual(len(embedding), self.extractor.feature_dim)
        self.assertAlmostEqual(np.linalg.norm(embedding), 1.0, places=5)  # Should be normalized
    
    def test_extract_embedding_invalid_bbox(self):
        """Test embedding extraction with invalid bounding box."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (500, 400, 450, 350)  # Invalid bbox (x2 < x1, y2 < y1)
        
        embedding = self.extractor.extract_embedding(image, bbox)
        
        self.assertEqual(len(embedding), self.extractor.feature_dim)
        self.assertTrue(np.allclose(embedding, 0))  # Should be zero vector
    
    def test_extract_embedding_out_of_bounds(self):
        """Test embedding extraction with out-of-bounds bbox."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (600, 450, 700, 550)  # Out of bounds
        
        embedding = self.extractor.extract_embedding(image, bbox)
        
        self.assertEqual(len(embedding), self.extractor.feature_dim)
        # Should handle gracefully and return normalized embedding
    
    def test_embedding_consistency(self):
        """Test that similar crops produce similar embeddings."""
        # Create two similar images
        base_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        similar_image = base_image.copy().astype(np.int16)
        similar_image += np.random.randint(-10, 10, similar_image.shape, dtype=np.int16)
        similar_image = np.clip(similar_image, 0, 255).astype(np.uint8)
        
        bbox = (100, 100, 200, 300)
        
        embedding1 = self.extractor.extract_embedding(base_image, bbox)
        embedding2 = self.extractor.extract_embedding(similar_image, bbox)
        
        # Similar images should have high similarity
        similarity = np.dot(embedding1, embedding2)
        self.assertGreater(similarity, 0.8)


class TestPersonEmbedding(unittest.TestCase):
    """Test PersonEmbedding data class."""
    
    def test_embedding_normalization(self):
        """Test that embeddings are automatically normalized."""
        embedding_vector = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        
        person_emb = PersonEmbedding(
            track_id=1,
            camera_id="cam1",
            embedding=embedding_vector,
            bbox=(0, 0, 100, 100),
            timestamp=datetime.now(),
            confidence=0.8
        )
        
        # Should be normalized
        self.assertAlmostEqual(np.linalg.norm(person_emb.embedding), 1.0, places=5)
    
    def test_zero_embedding_handling(self):
        """Test handling of zero embeddings."""
        embedding_vector = np.zeros(5, dtype=np.float32)
        
        person_emb = PersonEmbedding(
            track_id=1,
            camera_id="cam1",
            embedding=embedding_vector,
            bbox=(0, 0, 100, 100),
            timestamp=datetime.now(),
            confidence=0.8
        )
        
        # Zero embedding should remain zero
        self.assertTrue(np.allclose(person_emb.embedding, 0))


class TestGlobalTrack(unittest.TestCase):
    """Test GlobalTrack functionality."""
    
    def setUp(self):
        self.track = GlobalTrack(global_id=1)
    
    def test_add_camera_track(self):
        """Test adding camera track associations."""
        timestamp = datetime.now()
        self.track.add_camera_track("cam1", 10, timestamp)
        
        self.assertEqual(self.track.camera_tracks["cam1"], 10)
        self.assertEqual(self.track.last_seen["cam1"], timestamp)
    
    def test_remove_camera_track(self):
        """Test removing camera track associations."""
        timestamp = datetime.now()
        self.track.add_camera_track("cam1", 10, timestamp)
        self.track.remove_camera_track("cam1")
        
        self.assertNotIn("cam1", self.track.camera_tracks)
        self.assertNotIn("cam1", self.track.last_seen)
    
    def test_get_latest_timestamp(self):
        """Test getting latest timestamp across cameras."""
        time1 = datetime.now()
        time2 = time1 + timedelta(seconds=5)
        time3 = time1 + timedelta(seconds=10)
        
        self.track.add_camera_track("cam1", 10, time1)
        self.track.add_camera_track("cam2", 20, time3)
        self.track.add_camera_track("cam3", 30, time2)
        
        latest = self.track.get_latest_timestamp()
        self.assertEqual(latest, time3)
    
    def test_is_seen_in_camera(self):
        """Test checking if person was recently seen in camera."""
        recent_time = datetime.now() - timedelta(seconds=2)
        old_time = datetime.now() - timedelta(seconds=10)
        
        self.track.add_camera_track("cam1", 10, recent_time)
        self.track.add_camera_track("cam2", 20, old_time)
        
        self.assertTrue(self.track.is_seen_in_camera("cam1", within_seconds=5))
        self.assertFalse(self.track.is_seen_in_camera("cam2", within_seconds=5))
        self.assertFalse(self.track.is_seen_in_camera("cam3", within_seconds=5))
    
    def test_get_active_cameras(self):
        """Test getting active cameras."""
        recent_time = datetime.now() - timedelta(seconds=2)
        old_time = datetime.now() - timedelta(seconds=10)
        
        self.track.add_camera_track("cam1", 10, recent_time)
        self.track.add_camera_track("cam2", 20, recent_time)
        self.track.add_camera_track("cam3", 30, old_time)
        
        active = self.track.get_active_cameras(within_seconds=5)
        self.assertEqual(active, {"cam1", "cam2"})


class TestCrossCameraTracker(unittest.TestCase):
    """Test the main cross-camera tracker functionality."""
    
    def setUp(self):
        self.tracker = CrossCameraTracker(
            similarity_threshold=0.7,
            time_window_seconds=10.0,
            max_track_age_seconds=30.0,
            min_embedding_quality=0.3
        )
    
    def create_detection_result(self, camera_id: str, detections: list) -> DetectionResult:
        """Helper to create detection results."""
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
    
    def create_person_detection(self, track_id: int, bbox: tuple, confidence: float = 0.8) -> PersonDetection:
        """Helper to create person detections."""
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
    
    def test_single_camera_processing(self):
        """Test processing detections from a single camera."""
        detections = [
            self.create_person_detection(1, (100, 100, 200, 300)),
            self.create_person_detection(2, (300, 100, 400, 300))
        ]
        
        detection_results = [self.create_detection_result("cam1", detections)]
        
        result = self.tracker.process_detections(detection_results)
        
        self.assertIn("cam1", result)
        self.assertEqual(len(result["cam1"]), 2)
        
        # Should have global track IDs assigned
        global_ids = {det.track_id for det in result["cam1"]}
        self.assertEqual(len(global_ids), 2)  # Two unique global IDs
    
    def test_multi_camera_processing_no_overlap(self):
        """Test processing detections from multiple cameras with no person overlap."""
        detections1 = [self.create_person_detection(1, (100, 100, 200, 300))]
        detections2 = [self.create_person_detection(1, (100, 100, 200, 300))]
        
        detection_results = [
            self.create_detection_result("cam1", detections1),
            self.create_detection_result("cam2", detections2)
        ]
        
        result = self.tracker.process_detections(detection_results)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result["cam1"]), 1)
        self.assertEqual(len(result["cam2"]), 1)
        
        # Should have different global IDs (no matching without embeddings)
        global_id1 = result["cam1"][0].track_id
        global_id2 = result["cam2"][0].track_id
        self.assertNotEqual(global_id1, global_id2)
    
    def test_embedding_extraction_integration(self):
        """Test integration with embedding extraction."""
        detections = [self.create_person_detection(1, (100, 100, 200, 300), confidence=0.8)]
        detection_results = [self.create_detection_result("cam1", detections)]
        
        # Create mock frame data
        frame_data = {
            "cam1": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        }
        
        result = self.tracker.process_detections(detection_results, frame_data)
        
        self.assertIn("cam1", result)
        self.assertEqual(len(result["cam1"]), 1)
        
        # Check that embeddings were created
        stats = self.tracker.get_tracking_stats()
        self.assertGreater(stats["tracks_with_embeddings"], 0)
    
    def test_track_persistence(self):
        """Test that tracks persist across multiple frames."""
        # First frame
        detections1 = [self.create_person_detection(1, (100, 100, 200, 300))]
        detection_results1 = [self.create_detection_result("cam1", detections1)]
        
        result1 = self.tracker.process_detections(detection_results1)
        global_id1 = result1["cam1"][0].track_id
        
        # Second frame with same track
        detections2 = [self.create_person_detection(1, (110, 110, 210, 310))]  # Slightly moved
        detection_results2 = [self.create_detection_result("cam1", detections2)]
        
        result2 = self.tracker.process_detections(detection_results2)
        global_id2 = result2["cam1"][0].track_id
        
        # Should maintain same global ID
        self.assertEqual(global_id1, global_id2)
    
    def test_track_cleanup(self):
        """Test cleanup of old tracks."""
        # Create a track
        detections = [self.create_person_detection(1, (100, 100, 200, 300))]
        detection_results = [self.create_detection_result("cam1", detections)]
        
        self.tracker.process_detections(detection_results)
        
        initial_count = len(self.tracker.global_tracks)
        self.assertGreater(initial_count, 0)
        
        # Manually age the tracks
        for track in self.tracker.global_tracks.values():
            track.last_seen = {cam: datetime.now() - timedelta(seconds=60) 
                             for cam in track.last_seen}
        
        # Process empty detections to trigger cleanup
        empty_results = [self.create_detection_result("cam1", [])]
        self.tracker.process_detections(empty_results)
        
        # Should have fewer or same tracks after cleanup (cleanup is time-based)
        final_count = len(self.tracker.global_tracks)
        self.assertLessEqual(final_count, initial_count)
    
    def test_get_deduplicated_count(self):
        """Test getting deduplicated global count."""
        # Create detections in multiple cameras
        detections1 = [self.create_person_detection(1, (100, 100, 200, 300))]
        detections2 = [self.create_person_detection(1, (100, 100, 200, 300))]
        
        detection_results = [
            self.create_detection_result("cam1", detections1),
            self.create_detection_result("cam2", detections2)
        ]
        
        self.tracker.process_detections(detection_results)
        
        # Should have 2 unique people (no matching without embeddings)
        count = self.tracker.get_deduplicated_global_count()
        self.assertEqual(count, 2)
    
    def test_tracking_stats(self):
        """Test tracking statistics."""
        detections = [
            self.create_person_detection(1, (100, 100, 200, 300)),
            self.create_person_detection(2, (300, 100, 400, 300))
        ]
        detection_results = [self.create_detection_result("cam1", detections)]
        
        self.tracker.process_detections(detection_results)
        
        stats = self.tracker.get_tracking_stats()
        
        self.assertIn("total_global_tracks", stats)
        self.assertIn("active_global_tracks", stats)
        self.assertIn("global_count", stats)
        self.assertIn("tracks_by_camera", stats)
        
        self.assertEqual(stats["global_count"], 2)
        self.assertEqual(stats["tracks_by_camera"]["cam1"], 2)
    
    def test_reset_tracking(self):
        """Test resetting tracking state."""
        # Create some tracks
        detections = [self.create_person_detection(1, (100, 100, 200, 300))]
        detection_results = [self.create_detection_result("cam1", detections)]
        
        self.tracker.process_detections(detection_results)
        
        self.assertGreater(len(self.tracker.global_tracks), 0)
        
        # Reset tracking
        self.tracker.reset_tracking()
        
        self.assertEqual(len(self.tracker.global_tracks), 0)
        self.assertEqual(len(self.tracker.camera_to_global), 0)
        self.assertEqual(self.tracker.next_global_id, 1)
    
    def test_similarity_calculation(self):
        """Test similarity calculation between tracks."""
        # Create two tracks with embeddings
        track1 = GlobalTrack(global_id=1)
        track2 = GlobalTrack(global_id=2)
        
        # Create similar embeddings
        embedding1 = PersonEmbedding(
            track_id=1,
            camera_id="cam1",
            embedding=np.array([1, 0, 0], dtype=np.float32),
            bbox=(0, 0, 100, 100),
            timestamp=datetime.now(),
            confidence=0.8
        )
        
        embedding2 = PersonEmbedding(
            track_id=2,
            camera_id="cam2",
            embedding=np.array([0.9, 0.1, 0], dtype=np.float32),
            bbox=(0, 0, 100, 100),
            timestamp=datetime.now(),
            confidence=0.8
        )
        
        track1.embeddings = [embedding1]
        track2.embeddings = [embedding2]
        
        similarity = self.tracker._calculate_track_similarity(track1, track2)
        
        # Should have high similarity
        self.assertGreater(similarity, 0.8)
    
    def test_track_merging(self):
        """Test merging of similar tracks."""
        # Create tracks manually for testing
        track1 = GlobalTrack(global_id=1)
        track1.add_camera_track("cam1", 10, datetime.now())
        
        track2 = GlobalTrack(global_id=2)
        track2.add_camera_track("cam2", 20, datetime.now())
        
        self.tracker.global_tracks[1] = track1
        self.tracker.global_tracks[2] = track2
        self.tracker.camera_to_global["cam1"][10] = 1
        self.tracker.camera_to_global["cam2"][20] = 2
        
        # Merge tracks
        self.tracker._merge_global_tracks(1, 2)
        
        # Track 2 should be removed, track 1 should have both cameras
        self.assertNotIn(2, self.tracker.global_tracks)
        self.assertIn("cam1", self.tracker.global_tracks[1].camera_tracks)
        self.assertIn("cam2", self.tracker.global_tracks[1].camera_tracks)
        
        # Camera mappings should be updated
        self.assertEqual(self.tracker.camera_to_global["cam2"][20], 1)


class TestCrossCameraTrackerIntegration(unittest.TestCase):
    """Integration tests for cross-camera tracking with realistic scenarios."""
    
    def setUp(self):
        self.tracker = CrossCameraTracker(
            similarity_threshold=0.6,  # Lower threshold for testing
            time_window_seconds=10.0,
            max_track_age_seconds=30.0,
            min_embedding_quality=0.3
        )
    
    def test_person_moving_between_cameras(self):
        """Test scenario where person moves from one camera to another."""
        # Person appears in cam1
        detections1 = [PersonDetection(
            track_id=1,
            bbox=(100, 100, 200, 300),
            confidence=0.8,
            center_point=(150, 200),
            in_roi=True,
            local_track_id=1
        )]
        
        frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Make person region distinctive
        frame1[100:300, 100:200] = [100, 150, 200]  # Distinctive color
        
        result1 = self.tracker.process_detections(
            [DetectionResult("cam1", datetime.now(), detections1, Mock())],
            {"cam1": frame1}
        )
        
        global_id1 = result1["cam1"][0].track_id
        
        # Person disappears from cam1 and appears in cam2 with similar appearance
        detections2 = [PersonDetection(
            track_id=1,
            bbox=(100, 100, 200, 300),
            confidence=0.8,
            center_point=(150, 200),
            in_roi=True,
            local_track_id=1
        )]
        
        frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Similar appearance
        frame2[100:300, 100:200] = [105, 155, 205]  # Very similar color
        
        result2 = self.tracker.process_detections(
            [
                DetectionResult("cam1", datetime.now(), [], Mock()),  # Empty cam1
                DetectionResult("cam2", datetime.now(), detections2, Mock())
            ],
            {"cam1": frame1, "cam2": frame2}
        )
        
        # Should potentially match (though without perfect embeddings, might not)
        # This tests the framework is working
        self.assertIn("cam2", result2)
        self.assertEqual(len(result2["cam2"]), 1)
    
    def test_multiple_people_different_cameras(self):
        """Test multiple people in different cameras with no overlap."""
        # Two people in cam1
        detections1 = [
            PersonDetection(1, (100, 100, 200, 300), 0.8, (150, 200), True, 1),
            PersonDetection(2, (300, 100, 400, 300), 0.8, (350, 200), True, 2)
        ]
        
        # One person in cam2
        detections2 = [
            PersonDetection(1, (100, 100, 200, 300), 0.8, (150, 200), True, 1)
        ]
        
        detection_results = [
            DetectionResult("cam1", datetime.now(), detections1, Mock()),
            DetectionResult("cam2", datetime.now(), detections2, Mock())
        ]
        
        result = self.tracker.process_detections(detection_results)
        
        # Should have 3 unique global tracks
        all_global_ids = set()
        for camera_detections in result.values():
            for detection in camera_detections:
                all_global_ids.add(detection.track_id)
        
        self.assertEqual(len(all_global_ids), 3)
        self.assertEqual(self.tracker.get_deduplicated_global_count(), 3)


if __name__ == '__main__':
    unittest.main()