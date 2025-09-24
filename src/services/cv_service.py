"""
Enhanced Computer Vision Service for multi-camera RTSP processing.
Extends the existing webcam functionality to support multiple concurrent RTSP streams
with health monitoring and automatic reconnection.
"""

import asyncio
import cv2
import json
import logging
import numpy as np
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from queue import Queue, Empty

# Ultralytics YOLO
from ultralytics import YOLO

# Optional: tracking for more stable counts
try:
    import supervision as sv
    HAVE_SV = True
except ImportError:
    HAVE_SV = False

from src.models.core import (
    DetectionResult, 
    PersonDetection, 
    CameraHealth, 
    CameraStatus
)
from src.services.interfaces import CVService
from src.services.roi_manager import ROIManager
from src.services.cross_camera_tracker import CrossCameraTracker
from src.config.models import CameraConfig


logger = logging.getLogger(__name__)


@dataclass
class CameraStream:
    """Internal camera stream management data."""
    camera_id: str
    config: CameraConfig
    capture: Optional[cv2.VideoCapture] = None
    thread: Optional[threading.Thread] = None
    running: bool = False
    last_frame_time: Optional[datetime] = None
    fps_counter: int = 0
    fps_start_time: float = 0
    current_fps: float = 0
    error_count: int = 0
    last_error: Optional[str] = None
    reconnect_attempts: int = 0
    next_reconnect_time: float = 0
    detection_queue: Optional[Queue] = None


class CameraManager:
    """Manages multiple concurrent RTSP camera streams with health monitoring."""
    
    def __init__(self, max_cameras: int = 10):
        self.max_cameras = max_cameras
        self.streams: Dict[str, CameraStream] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_cameras)
        self._shutdown = False
        
        # Start health monitoring thread
        self.health_thread = threading.Thread(target=self._health_monitor_loop, daemon=True)
        self.health_thread.start()
    
    def add_camera(self, camera_id: str, config: CameraConfig) -> bool:
        """Add a new camera stream."""
        if len(self.streams) >= self.max_cameras:
            logger.error(f"Maximum camera limit ({self.max_cameras}) reached")
            return False
        
        if camera_id in self.streams:
            logger.warning(f"Camera {camera_id} already exists, stopping existing stream")
            self.remove_camera(camera_id)
        
        stream = CameraStream(
            camera_id=camera_id,
            config=config,
            detection_queue=Queue(maxsize=30)
        )
        
        self.streams[camera_id] = stream
        self._start_camera_stream(stream)
        
        logger.info(f"Added camera {camera_id} with RTSP URL: {config.rtsp_url}")
        return True
    
    def remove_camera(self, camera_id: str) -> bool:
        """Remove a camera stream."""
        if camera_id not in self.streams:
            return False
        
        stream = self.streams[camera_id]
        self._stop_camera_stream(stream)
        del self.streams[camera_id]
        
        logger.info(f"Removed camera {camera_id}")
        return True
    
    def get_camera_health(self, camera_id: str) -> Optional[CameraHealth]:
        """Get health status for a camera."""
        if camera_id not in self.streams:
            return None
        
        stream = self.streams[camera_id]
        
        # Determine status
        if not stream.running:
            status = CameraStatus.OFFLINE
        elif stream.error_count > 5:
            status = CameraStatus.DEGRADED
        elif stream.last_frame_time and (datetime.now() - stream.last_frame_time).total_seconds() > 10:
            status = CameraStatus.OFFLINE
        else:
            status = CameraStatus.ONLINE
        
        # Determine connection quality
        if stream.current_fps > 15:
            quality = "EXCELLENT"
        elif stream.current_fps > 10:
            quality = "GOOD"
        elif stream.current_fps > 5:
            quality = "FAIR"
        else:
            quality = "POOR"
        
        return CameraHealth(
            camera_id=camera_id,
            status=status,
            last_frame_time=stream.last_frame_time or datetime.now(),
            fps=stream.current_fps,
            connection_quality=quality,
            error_message=stream.last_error
        )
    
    def get_latest_detections(self, camera_id: str) -> Optional[List[PersonDetection]]:
        """Get latest detection results from camera queue."""
        if camera_id not in self.streams:
            return None
        
        stream = self.streams[camera_id]
        if not stream.detection_queue:
            return None
        
        try:
            # Get the most recent detection result
            detections = None
            while not stream.detection_queue.empty():
                detections = stream.detection_queue.get_nowait()
            return detections
        except Empty:
            return None
    
    def _start_camera_stream(self, stream: CameraStream):
        """Start processing a camera stream."""
        stream.running = True
        stream.thread = threading.Thread(
            target=self._camera_processing_loop,
            args=(stream,),
            daemon=True
        )
        stream.thread.start()
    
    def _stop_camera_stream(self, stream: CameraStream):
        """Stop processing a camera stream."""
        stream.running = False
        
        if stream.thread and stream.thread.is_alive():
            stream.thread.join(timeout=5.0)
        
        if stream.capture:
            stream.capture.release()
            stream.capture = None
    
    def _camera_processing_loop(self, stream: CameraStream):
        """Main processing loop for a single camera stream."""
        logger.info(f"Starting camera processing loop for {stream.camera_id}")
        
        while stream.running and not self._shutdown:
            try:
                # Check if we need to reconnect
                if not stream.capture or not stream.capture.isOpened():
                    if not self._connect_camera(stream):
                        time.sleep(1.0)
                        continue
                
                # Read frame
                ret, frame = stream.capture.read()
                if not ret:
                    logger.warning(f"Failed to read frame from {stream.camera_id}")
                    stream.error_count += 1
                    stream.last_error = "Failed to read frame"
                    
                    # Trigger reconnection
                    if stream.capture:
                        stream.capture.release()
                        stream.capture = None
                    continue
                
                # Update frame timing
                stream.last_frame_time = datetime.now()
                stream.fps_counter += 1
                
                # Calculate FPS every second
                current_time = time.time()
                if current_time - stream.fps_start_time >= 1.0:
                    stream.current_fps = stream.fps_counter / (current_time - stream.fps_start_time)
                    stream.fps_counter = 0
                    stream.fps_start_time = current_time
                
                # Reset error count on successful frame
                stream.error_count = 0
                stream.last_error = None
                
                # Process frame (this will be handled by the main CV service)
                # For now, just put a placeholder in the queue
                if stream.detection_queue and not stream.detection_queue.full():
                    # This will be replaced with actual detection results
                    stream.detection_queue.put([])
                
            except Exception as e:
                logger.error(f"Error in camera processing loop for {stream.camera_id}: {e}")
                stream.error_count += 1
                stream.last_error = str(e)
                time.sleep(1.0)
        
        logger.info(f"Camera processing loop ended for {stream.camera_id}")
    
    def _connect_camera(self, stream: CameraStream) -> bool:
        """Connect to camera with exponential backoff."""
        current_time = time.time()
        
        # Check if we should attempt reconnection
        if current_time < stream.next_reconnect_time:
            return False
        
        logger.info(f"Attempting to connect to camera {stream.camera_id}")
        
        try:
            # Create new capture
            capture = cv2.VideoCapture(stream.config.rtsp_url)
            
            # Set buffer size to reduce latency
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test if we can read a frame
            ret, frame = capture.read()
            if not ret:
                capture.release()
                raise Exception("Cannot read test frame")
            
            stream.capture = capture
            stream.reconnect_attempts = 0
            stream.fps_start_time = time.time()
            
            logger.info(f"Successfully connected to camera {stream.camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to camera {stream.camera_id}: {e}")
            stream.reconnect_attempts += 1
            stream.last_error = f"Connection failed: {str(e)}"
            
            # Exponential backoff: 2^attempts seconds, max 300 seconds (5 minutes)
            backoff_time = min(2 ** stream.reconnect_attempts, 300)
            stream.next_reconnect_time = current_time + backoff_time
            
            logger.info(f"Next reconnection attempt for {stream.camera_id} in {backoff_time} seconds")
            return False
    
    def _health_monitor_loop(self):
        """Monitor camera health and trigger reconnections."""
        while not self._shutdown:
            try:
                for stream in list(self.streams.values()):
                    if not stream.running:
                        continue
                    
                    # Check if camera has been silent for too long
                    if stream.last_frame_time:
                        silence_duration = (datetime.now() - stream.last_frame_time).total_seconds()
                        if silence_duration > 30:  # 30 seconds without frames
                            logger.warning(f"Camera {stream.camera_id} silent for {silence_duration:.1f}s")
                            stream.error_count += 1
                            
                            # Force reconnection
                            if stream.capture:
                                stream.capture.release()
                                stream.capture = None
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                time.sleep(10)
    
    def shutdown(self):
        """Shutdown camera manager and all streams."""
        logger.info("Shutting down camera manager")
        self._shutdown = True
        
        # Stop all camera streams
        for stream in list(self.streams.values()):
            self._stop_camera_stream(stream)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)


class EnhancedCVService(CVService):
    """Enhanced Computer Vision Service with multi-camera RTSP support."""
    
    def __init__(self, model_path: str = "yolov8n.pt", max_cameras: int = 10):
        self.model = YOLO(model_path)
        self.camera_manager = CameraManager(max_cameras)
        self.roi_manager = ROIManager()
        
        # Initialize cross-camera tracker
        self.cross_camera_tracker = CrossCameraTracker(
            similarity_threshold=0.7,
            time_window_seconds=10.0,
            max_track_age_seconds=30.0,
            min_embedding_quality=0.3
        )
        
        # Initialize per-camera tracker if available
        self.camera_trackers: Dict[str, any] = {}
        if HAVE_SV:
            # We'll create trackers per camera for better local tracking
            pass
        
        # Detection processing
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_results: Dict[str, DetectionResult] = {}
        self.entrance_events: Dict[str, List[Dict[str, Any]]] = {}  # Track entrance/exit events
        self.current_frames: Dict[str, np.ndarray] = {}  # Store current frames for embedding extraction
        self.detection_thread.start()
        
        logger.info("Enhanced CV Service with cross-camera tracking initialized")
    
    def process_camera_feed(self, camera_id: str, rtsp_url: str) -> None:
        """Start processing a camera feed."""
        # Create camera config
        from src.config.models import CameraConfig, CameraType
        
        config = CameraConfig(
            camera_id=camera_id,
            rtsp_url=rtsp_url,
            camera_type=CameraType.ZONE,  # Default to zone
            zone_id=f"{camera_id}_zone"
        )
        
        self.camera_manager.add_camera(camera_id, config)
    
    def configure_roi(self, camera_id: str, polygon_points: List[tuple]) -> None:
        """Configure region of interest for a camera."""
        success = self.roi_manager.configure_camera_roi(camera_id, polygon_points)
        if not success:
            logger.error(f"Failed to configure ROI for camera {camera_id}")
    
    def set_entrance_line(self, camera_id: str, line_coords: tuple) -> None:
        """Set entrance line coordinates for in/out counting."""
        success = self.roi_manager.configure_entrance_line(camera_id, line_coords)
        if not success:
            logger.error(f"Failed to configure entrance line for camera {camera_id}")
        else:
            # Initialize entrance events tracking for this camera
            if camera_id not in self.entrance_events:
                self.entrance_events[camera_id] = []
    
    def get_detection_results(self, camera_id: str) -> Optional[DetectionResult]:
        """Get latest detection results for a camera."""
        return self.detection_results.get(camera_id)
    
    def health_check(self, camera_id: str) -> CameraHealth:
        """Get health status for a camera."""
        health = self.camera_manager.get_camera_health(camera_id)
        if health:
            return health
        
        # Return offline status if camera not found
        return CameraHealth(
            camera_id=camera_id,
            status=CameraStatus.OFFLINE,
            last_frame_time=datetime.now(),
            fps=0.0,
            connection_quality="OFFLINE",
            error_message="Camera not found"
        )
    
    def stop_camera_feed(self, camera_id: str) -> None:
        """Stop processing a camera feed."""
        self.camera_manager.remove_camera(camera_id)
        if camera_id in self.detection_results:
            del self.detection_results[camera_id]
    
    def _detection_loop(self):
        """Main detection processing loop with cross-camera tracking."""
        logger.info("Starting detection processing loop with cross-camera tracking")
        
        while True:
            try:
                # Collect detection results from all cameras
                detection_results = []
                
                for camera_id in list(self.camera_manager.streams.keys()):
                    result = self._process_camera_detections(camera_id)
                    if result:
                        detection_results.append(result)
                
                # Perform cross-camera tracking if we have multiple cameras
                if len(detection_results) > 1:
                    deduplicated_results = self.cross_camera_tracker.process_detections(
                        detection_results, self.current_frames
                    )
                    
                    # Update stored results with deduplicated data
                    for camera_id, detections in deduplicated_results.items():
                        if camera_id in self.detection_results:
                            # Update detections with global track IDs
                            self.detection_results[camera_id].detections = detections
                
                time.sleep(0.1)  # 10 FPS detection processing
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                time.sleep(1.0)
    
    def _process_camera_detections(self, camera_id: str) -> Optional[DetectionResult]:
        """Process detections for a single camera."""
        stream = self.camera_manager.streams.get(camera_id)
        if not stream or not stream.capture or not stream.capture.isOpened():
            return None
        
        try:
            # Read latest frame
            ret, frame = stream.capture.read()
            if not ret:
                return None
            
            # Store frame for embedding extraction
            self.current_frames[camera_id] = frame.copy()
            
            # Run YOLO detection
            results = self.model(frame, conf=0.35, classes=[0], verbose=False)  # Only detect persons
            
            detections = []
            
            if results and len(results) > 0:
                result = results[0]
                
                # Get or create camera-specific tracker
                if HAVE_SV:
                    if camera_id not in self.camera_trackers:
                        self.camera_trackers[camera_id] = sv.ByteTrack(
                            track_activation_threshold=0.25,
                            lost_track_buffer=30,
                            minimum_matching_threshold=0.8,
                            frame_rate=30
                        )
                    
                    tracker = self.camera_trackers[camera_id]
                    
                    # Use tracking for consistent IDs
                    det = sv.Detections.from_ultralytics(result)
                    tracks = tracker.update_with_detections(det)
                    
                    if len(tracks) > 0:
                        centers = tracks.get_anchors_coordinates(sv.Position.CENTER)
                        
                        for i, center in enumerate(centers):
                            x1, y1, x2, y2 = tracks.xyxy[i]
                            track_id = tracks.tracker_id[i] if tracks.tracker_id is not None else i
                            confidence = tracks.confidence[i] if tracks.confidence is not None else 0.5
                            
                            detection = PersonDetection(
                                track_id=int(track_id),
                                bbox=(int(x1), int(y1), int(x2), int(y2)),
                                confidence=float(confidence),
                                center_point=(int(center[0]), int(center[1])),
                                in_roi=False,  # Will be set by ROI manager
                                local_track_id=int(track_id)  # Store original local ID
                            )
                            detections.append(detection)
                            
                            # Check for entrance line crossing
                            crossing = self.roi_manager.detect_entrance_crossing(
                                camera_id, int(track_id), center
                            )
                            if crossing:
                                event = {
                                    'timestamp': datetime.now(),
                                    'track_id': int(track_id),
                                    'event_type': crossing,
                                    'position': center
                                }
                                if camera_id not in self.entrance_events:
                                    self.entrance_events[camera_id] = []
                                self.entrance_events[camera_id].append(event)
                                
                                # Keep only recent events (last 100)
                                if len(self.entrance_events[camera_id]) > 100:
                                    self.entrance_events[camera_id] = self.entrance_events[camera_id][-100:]
                
                else:
                    # Fallback without tracking
                    if result.boxes is not None:
                        for i, box in enumerate(result.boxes):
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            confidence = float(box.conf[0].item())
                            center = ((x1 + x2) / 2, (y1 + y2) / 2)
                            
                            detection = PersonDetection(
                                track_id=i,
                                bbox=(int(x1), int(y1), int(x2), int(y2)),
                                confidence=confidence,
                                center_point=(int(center[0]), int(center[1])),
                                in_roi=False,  # Will be set by ROI manager
                                local_track_id=i
                            )
                            detections.append(detection)
            
            # Filter detections by ROI
            filtered_detections = self.roi_manager.filter_detections_by_roi(camera_id, detections)
            
            # Get camera health
            camera_health = self.camera_manager.get_camera_health(camera_id)
            if not camera_health:
                return None
            
            # Create detection result
            detection_result = DetectionResult(
                camera_id=camera_id,
                timestamp=datetime.now(),
                detections=filtered_detections,
                camera_health=camera_health
            )
            
            # Store result
            self.detection_results[camera_id] = detection_result
            
            return detection_result
            
        except Exception as e:
            logger.error(f"Error processing detections for camera {camera_id}: {e}")
            return None
    
    def get_entrance_events(self, camera_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent entrance/exit events for a camera."""
        events = self.entrance_events.get(camera_id, [])
        return events[-limit:] if events else []
    
    def get_roi_config(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get ROI configuration for a camera."""
        return self.roi_manager.export_config(camera_id)
    
    def validate_roi_config(self, camera_id: str) -> List[str]:
        """Validate ROI configuration and return any errors."""
        return self.roi_manager.validate_roi_config(camera_id)
    
    def get_cross_camera_tracking_stats(self) -> dict:
        """Get cross-camera tracking statistics."""
        return self.cross_camera_tracker.get_tracking_stats()
    
    def get_deduplicated_global_count(self) -> int:
        """Get total count of unique people across all cameras."""
        return self.cross_camera_tracker.get_deduplicated_global_count()
    
    def reset_cross_camera_tracking(self) -> None:
        """Reset cross-camera tracking state."""
        self.cross_camera_tracker.reset_tracking()
    
    def shutdown(self):
        """Shutdown the CV service."""
        logger.info("Shutting down Enhanced CV Service")
        self.camera_manager.shutdown()