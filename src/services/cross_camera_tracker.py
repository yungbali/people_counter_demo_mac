"""
Cross-Camera Person Tracking and De-duplication Service.
Implements multi-camera person re-identification and tracking consistency.
"""

import cv2
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

from src.models.core import PersonDetection, DetectionResult


logger = logging.getLogger(__name__)


@dataclass
class PersonEmbedding:
    """Person appearance embedding for re-identification."""
    track_id: int
    camera_id: str
    embedding: np.ndarray
    bbox: Tuple[int, int, int, int]
    timestamp: datetime
    confidence: float
    
    def __post_init__(self):
        """Ensure embedding is normalized."""
        if self.embedding is not None and len(self.embedding) > 0:
            norm = np.linalg.norm(self.embedding)
            if norm > 0:
                self.embedding = self.embedding / norm


@dataclass
class GlobalTrack:
    """Global person track across multiple cameras."""
    global_id: int
    camera_tracks: Dict[str, int] = field(default_factory=dict)  # camera_id -> local_track_id
    last_seen: Dict[str, datetime] = field(default_factory=dict)  # camera_id -> timestamp
    embeddings: List[PersonEmbedding] = field(default_factory=list)
    creation_time: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    
    def add_camera_track(self, camera_id: str, local_track_id: int, timestamp: datetime):
        """Add or update camera track association."""
        self.camera_tracks[camera_id] = local_track_id
        self.last_seen[camera_id] = timestamp
    
    def remove_camera_track(self, camera_id: str):
        """Remove camera track association."""
        if camera_id in self.camera_tracks:
            del self.camera_tracks[camera_id]
        if camera_id in self.last_seen:
            del self.last_seen[camera_id]
    
    def get_latest_timestamp(self) -> datetime:
        """Get the most recent timestamp across all cameras."""
        if not self.last_seen:
            return self.creation_time
        return max(self.last_seen.values())
    
    def is_seen_in_camera(self, camera_id: str, within_seconds: float = 5.0) -> bool:
        """Check if person was recently seen in a specific camera."""
        if camera_id not in self.last_seen:
            return False
        
        time_diff = (datetime.now() - self.last_seen[camera_id]).total_seconds()
        return time_diff <= within_seconds
    
    def get_active_cameras(self, within_seconds: float = 5.0) -> Set[str]:
        """Get cameras where person was recently seen."""
        active_cameras = set()
        current_time = datetime.now()
        
        for camera_id, last_time in self.last_seen.items():
            if (current_time - last_time).total_seconds() <= within_seconds:
                active_cameras.add(camera_id)
        
        return active_cameras


class SimpleEmbeddingExtractor:
    """Simple appearance embedding extractor using color histograms and basic features."""
    
    def __init__(self):
        self.hist_bins = 32  # Bins for color histogram
        self.feature_dim = 96  # 32*3 for RGB histograms
    
    def extract_embedding(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract appearance embedding from person crop."""
        try:
            x1, y1, x2, y2 = bbox
            
            # Ensure bbox is within image bounds
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return np.zeros(self.feature_dim)
            
            # Extract person crop
            person_crop = image[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                return np.zeros(self.feature_dim)
            
            # Resize to standard size for consistency
            person_crop = cv2.resize(person_crop, (64, 128))
            
            # Extract color histograms for each channel
            features = []
            
            # RGB histograms
            for channel in range(3):
                hist = cv2.calcHist([person_crop], [channel], None, [self.hist_bins], [0, 256])
                hist = hist.flatten()
                # Normalize histogram
                if hist.sum() > 0:
                    hist = hist / hist.sum()
                features.extend(hist)
            
            embedding = np.array(features, dtype=np.float32)
            
            # Normalize the final embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return np.zeros(self.feature_dim)


class CrossCameraTracker:
    """Cross-camera person tracking and de-duplication system."""
    
    def __init__(self, 
                 similarity_threshold: float = 0.7,
                 time_window_seconds: float = 10.0,
                 max_track_age_seconds: float = 30.0,
                 min_embedding_quality: float = 0.3):
        """
        Initialize cross-camera tracker.
        
        Args:
            similarity_threshold: Minimum similarity for person matching
            time_window_seconds: Time window for cross-camera matching
            max_track_age_seconds: Maximum age before track is considered inactive
            min_embedding_quality: Minimum detection confidence for embedding extraction
        """
        self.similarity_threshold = similarity_threshold
        self.time_window_seconds = time_window_seconds
        self.max_track_age_seconds = max_track_age_seconds
        self.min_embedding_quality = min_embedding_quality
        
        # Tracking state
        self.global_tracks: Dict[int, GlobalTrack] = {}
        self.next_global_id = 1
        self.camera_to_global: Dict[str, Dict[int, int]] = defaultdict(dict)  # camera_id -> {local_id: global_id}
        
        # Embedding extractor
        self.embedding_extractor = SimpleEmbeddingExtractor()
        
        # Performance tracking
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 10.0  # seconds
        
        logger.info("Cross-camera tracker initialized")
    
    def process_detections(self, detection_results: List[DetectionResult], 
                         frame_data: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, List[PersonDetection]]:
        """
        Process detection results from multiple cameras and perform cross-camera tracking.
        
        Args:
            detection_results: List of detection results from different cameras
            frame_data: Optional dictionary of camera_id -> frame image for embedding extraction
        
        Returns:
            Dictionary of camera_id -> deduplicated detections with global track IDs
        """
        current_time = datetime.now()
        
        # Extract embeddings for high-quality detections
        embeddings_by_camera = {}
        if frame_data:
            embeddings_by_camera = self._extract_embeddings(detection_results, frame_data)
        
        # Process each camera's detections
        deduplicated_results = {}
        
        for result in detection_results:
            camera_id = result.camera_id
            detections = result.detections
            
            # Get embeddings for this camera
            camera_embeddings = embeddings_by_camera.get(camera_id, {})
            
            # Update tracking for this camera
            deduplicated_detections = self._update_camera_tracking(
                camera_id, detections, camera_embeddings, current_time
            )
            
            deduplicated_results[camera_id] = deduplicated_detections
        
        # Perform cross-camera matching
        self._perform_cross_camera_matching(current_time)
        
        # Update detection results with global track IDs
        for camera_id, detections in deduplicated_results.items():
            for detection in detections:
                global_id = self._get_global_id(camera_id, detection.track_id)
                if global_id is not None:
                    # Store global ID in a custom field (we'll modify PersonDetection later)
                    detection.track_id = global_id
        
        # Cleanup old tracks periodically
        if time.time() - self.last_cleanup_time > self.cleanup_interval:
            self._cleanup_old_tracks(current_time)
            self.last_cleanup_time = time.time()
        
        return deduplicated_results
    
    def _extract_embeddings(self, detection_results: List[DetectionResult], 
                          frame_data: Dict[str, np.ndarray]) -> Dict[str, Dict[int, PersonEmbedding]]:
        """Extract embeddings for detections with sufficient quality."""
        embeddings_by_camera = {}
        
        for result in detection_results:
            camera_id = result.camera_id
            
            if camera_id not in frame_data:
                continue
            
            frame = frame_data[camera_id]
            camera_embeddings = {}
            
            for detection in result.detections:
                # Only extract embeddings for high-confidence detections
                if detection.confidence < self.min_embedding_quality:
                    continue
                
                embedding_vector = self.embedding_extractor.extract_embedding(frame, detection.bbox)
                
                if embedding_vector is not None and np.any(embedding_vector):
                    embedding = PersonEmbedding(
                        track_id=detection.track_id,
                        camera_id=camera_id,
                        embedding=embedding_vector,
                        bbox=detection.bbox,
                        timestamp=result.timestamp,
                        confidence=detection.confidence
                    )
                    camera_embeddings[detection.track_id] = embedding
            
            embeddings_by_camera[camera_id] = camera_embeddings
        
        return embeddings_by_camera
    
    def _update_camera_tracking(self, camera_id: str, detections: List[PersonDetection],
                              embeddings: Dict[int, PersonEmbedding], 
                              current_time: datetime) -> List[PersonDetection]:
        """Update tracking state for a single camera."""
        # Get current local track IDs for this camera
        current_local_ids = {det.track_id for det in detections}
        previous_local_ids = set(self.camera_to_global[camera_id].keys())
        
        # Handle disappeared tracks
        disappeared_ids = previous_local_ids - current_local_ids
        for local_id in disappeared_ids:
            global_id = self.camera_to_global[camera_id].get(local_id)
            if global_id and global_id in self.global_tracks:
                self.global_tracks[global_id].remove_camera_track(camera_id)
            
            # Remove from camera mapping
            if local_id in self.camera_to_global[camera_id]:
                del self.camera_to_global[camera_id][local_id]
        
        # Handle new and existing tracks
        for detection in detections:
            local_id = detection.track_id
            
            # Check if this local track already has a global ID
            if local_id in self.camera_to_global[camera_id]:
                global_id = self.camera_to_global[camera_id][local_id]
                if global_id in self.global_tracks:
                    # Update existing global track
                    self.global_tracks[global_id].add_camera_track(camera_id, local_id, current_time)
                    
                    # Add embedding if available
                    if local_id in embeddings:
                        self.global_tracks[global_id].embeddings.append(embeddings[local_id])
                        # Keep only recent embeddings
                        self._trim_embeddings(self.global_tracks[global_id])
            else:
                # New local track - create new global track for now
                # Cross-camera matching will handle merging later
                global_id = self._create_new_global_track(camera_id, local_id, current_time)
                
                if local_id in embeddings:
                    self.global_tracks[global_id].embeddings.append(embeddings[local_id])
        
        return detections
    
    def _create_new_global_track(self, camera_id: str, local_id: int, timestamp: datetime) -> int:
        """Create a new global track."""
        global_id = self.next_global_id
        self.next_global_id += 1
        
        global_track = GlobalTrack(global_id=global_id)
        global_track.add_camera_track(camera_id, local_id, timestamp)
        
        self.global_tracks[global_id] = global_track
        self.camera_to_global[camera_id][local_id] = global_id
        
        return global_id
    
    def _perform_cross_camera_matching(self, current_time: datetime):
        """Perform cross-camera person matching and merge tracks."""
        active_tracks = [track for track in self.global_tracks.values() 
                        if track.is_active and track.embeddings]
        
        # Find potential matches between tracks
        matches_to_merge = []
        
        for i, track1 in enumerate(active_tracks):
            for j, track2 in enumerate(active_tracks[i+1:], i+1):
                # Skip if tracks are from the same camera set (already matched)
                if track1.get_active_cameras() & track2.get_active_cameras():
                    continue
                
                # Check temporal overlap (should be within time window)
                time_diff = abs((track1.get_latest_timestamp() - track2.get_latest_timestamp()).total_seconds())
                if time_diff > self.time_window_seconds:
                    continue
                
                # Calculate embedding similarity
                similarity = self._calculate_track_similarity(track1, track2)
                
                if similarity > self.similarity_threshold:
                    matches_to_merge.append((track1.global_id, track2.global_id, similarity))
        
        # Sort matches by similarity (highest first)
        matches_to_merge.sort(key=lambda x: x[2], reverse=True)
        
        # Merge tracks (avoid conflicts)
        merged_tracks = set()
        for track1_id, track2_id, similarity in matches_to_merge:
            if track1_id in merged_tracks or track2_id in merged_tracks:
                continue
            
            if track1_id in self.global_tracks and track2_id in self.global_tracks:
                self._merge_global_tracks(track1_id, track2_id)
                merged_tracks.add(track2_id)
                
                logger.info(f"Merged global tracks {track1_id} and {track2_id} (similarity: {similarity:.3f})")
    
    def _calculate_track_similarity(self, track1: GlobalTrack, track2: GlobalTrack) -> float:
        """Calculate similarity between two global tracks based on embeddings."""
        if not track1.embeddings or not track2.embeddings:
            return 0.0
        
        # Get recent embeddings from each track
        recent_embeddings1 = [emb for emb in track1.embeddings 
                            if (datetime.now() - emb.timestamp).total_seconds() <= self.time_window_seconds]
        recent_embeddings2 = [emb for emb in track2.embeddings 
                            if (datetime.now() - emb.timestamp).total_seconds() <= self.time_window_seconds]
        
        if not recent_embeddings1 or not recent_embeddings2:
            return 0.0
        
        # Calculate pairwise similarities and take the maximum
        max_similarity = 0.0
        
        for emb1 in recent_embeddings1:
            for emb2 in recent_embeddings2:
                # Cosine similarity
                similarity = np.dot(emb1.embedding, emb2.embedding)
                max_similarity = max(max_similarity, similarity)
        
        return max_similarity
    
    def _merge_global_tracks(self, keep_id: int, merge_id: int):
        """Merge two global tracks."""
        if keep_id not in self.global_tracks or merge_id not in self.global_tracks:
            return
        
        keep_track = self.global_tracks[keep_id]
        merge_track = self.global_tracks[merge_id]
        
        # Merge camera tracks
        for camera_id, local_id in merge_track.camera_tracks.items():
            keep_track.add_camera_track(camera_id, local_id, merge_track.last_seen[camera_id])
            # Update camera mapping
            self.camera_to_global[camera_id][local_id] = keep_id
        
        # Merge embeddings
        keep_track.embeddings.extend(merge_track.embeddings)
        self._trim_embeddings(keep_track)
        
        # Remove merged track
        del self.global_tracks[merge_id]
    
    def _trim_embeddings(self, track: GlobalTrack, max_embeddings: int = 10):
        """Keep only the most recent embeddings for a track."""
        if len(track.embeddings) > max_embeddings:
            # Sort by timestamp and keep the most recent
            track.embeddings.sort(key=lambda x: x.timestamp, reverse=True)
            track.embeddings = track.embeddings[:max_embeddings]
    
    def _cleanup_old_tracks(self, current_time: datetime):
        """Remove old inactive tracks."""
        tracks_to_remove = []
        
        for global_id, track in self.global_tracks.items():
            # Check if track is too old
            age = (current_time - track.get_latest_timestamp()).total_seconds()
            
            if age > self.max_track_age_seconds:
                tracks_to_remove.append(global_id)
        
        # Remove old tracks
        for global_id in tracks_to_remove:
            track = self.global_tracks[global_id]
            
            # Remove from camera mappings
            for camera_id, local_id in track.camera_tracks.items():
                if camera_id in self.camera_to_global and local_id in self.camera_to_global[camera_id]:
                    del self.camera_to_global[camera_id][local_id]
            
            # Remove global track
            del self.global_tracks[global_id]
            
            logger.debug(f"Removed old global track {global_id}")
    
    def _get_global_id(self, camera_id: str, local_id: int) -> Optional[int]:
        """Get global ID for a local track ID."""
        return self.camera_to_global[camera_id].get(local_id)
    
    def get_active_global_tracks(self) -> List[GlobalTrack]:
        """Get all currently active global tracks."""
        current_time = datetime.now()
        active_tracks = []
        
        for track in self.global_tracks.values():
            if track.is_active:
                # Check if track has recent activity
                age = (current_time - track.get_latest_timestamp()).total_seconds()
                if age <= self.max_track_age_seconds:
                    active_tracks.append(track)
        
        return active_tracks
    
    def get_track_count_by_camera(self) -> Dict[str, int]:
        """Get count of active tracks per camera."""
        counts = defaultdict(int)
        
        for track in self.get_active_global_tracks():
            active_cameras = track.get_active_cameras()
            for camera_id in active_cameras:
                counts[camera_id] += 1
        
        return dict(counts)
    
    def get_deduplicated_global_count(self) -> int:
        """Get total count of unique people across all cameras."""
        return len(self.get_active_global_tracks())
    
    def reset_tracking(self):
        """Reset all tracking state."""
        self.global_tracks.clear()
        self.camera_to_global.clear()
        self.next_global_id = 1
        logger.info("Cross-camera tracking state reset")
    
    def get_tracking_stats(self) -> Dict[str, any]:
        """Get tracking statistics for monitoring."""
        active_tracks = self.get_active_global_tracks()
        
        stats = {
            'total_global_tracks': len(self.global_tracks),
            'active_global_tracks': len(active_tracks),
            'next_global_id': self.next_global_id,
            'tracks_by_camera': self.get_track_count_by_camera(),
            'global_count': self.get_deduplicated_global_count(),
            'tracks_with_embeddings': sum(1 for track in active_tracks if track.embeddings),
            'avg_embeddings_per_track': np.mean([len(track.embeddings) for track in active_tracks]) if active_tracks else 0
        }
        
        return stats