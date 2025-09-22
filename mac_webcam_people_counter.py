#!/usr/bin/env python3
"""
Mac Webcam People Counting Demo
- Uses your Mac webcam via OpenCV.
- Runs YOLOv8n (Ultralytics) to detect persons.
- Optional ROI polygon you can draw with the mouse.
- Shows live count; triggers a simple Mac voice alert when over capacity.

Controls:
  q  = quit
  z  = start ROI drawing (click around area)
  c  = commit ROI polygon
  x  = clear ROI
"""
import os
import time
import argparse
import cv2
import numpy as np
import json
import paho.mqtt.client as mqtt

# Ultralytics YOLO
from ultralytics import YOLO

# Optional: tracking for more stable counts
try:
    import supervision as sv
    HAVE_SV = True
except Exception:
    HAVE_SV = False


class Publisher:
    def __init__(self, host, port, user=None, pw=None, base="site/demo/occupancy/lounge"):
        self.base = base.rstrip("/")
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if user and pw:
            self.client.username_pw_set(user, pw)
        try:
            self.client.connect(host, port, 60)
            self.client.loop_start()
            self.ok = True
        except Exception as e:
            print(f"[MQTT] Connect failed: {e}")
            self.ok = False

    def publish_state(self, payload: dict):
        if not self.ok: return
        topic = f"{self.base}/state"
        self.client.publish(topic, json.dumps(payload), qos=0, retain=True)

    def publish_event(self, payload: dict):
        if not self.ok: return
        topic = f"{self.base}/events"
        self.client.publish(topic, json.dumps(payload), qos=1, retain=False)

    def publish_health(self, payload: dict):
        if not self.ok: return
        topic = f"{self.base}/health"
        self.client.publish(topic, json.dumps(payload), qos=0, retain=False)


def mac_say(msg: str):
    try:
        # Simple voice alert on macOS; non-blocking
        os.system(f'say "{msg}" &')
    except Exception:
        pass

def point_in_polygon(point, polygon_np):
    # point: (x, y), polygon_np: Nx2 np.array
    # returns True if point inside polygon (or no polygon defined)
    if polygon_np is None or len(polygon_np) < 3:
        return True
    return cv2.pointPolygonTest(polygon_np, (float(point[0]), float(point[1])), False) >= 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=40, help="Max occupancy threshold")
    ap.add_argument("--hold", type=float, default=3.0, help="Seconds over threshold before alert")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    ap.add_argument("--imgsz", type=int, default=960, help="YOLO input size")
    ap.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    ap.add_argument("--mqtt-host", type=str, default="localhost", help="MQTT broker host")
    ap.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    ap.add_argument("--mqtt-base", type=str, default="site/demo/occupancy/lounge", help="Base topic for state/events")
    ap.add_argument("--mqtt-user", type=str, default=None, help="MQTT username (optional)")
    ap.add_argument("--mqtt-pass", type=str, default=None, help="MQTT password (optional)")
    args = ap.parse_args()

    publisher = Publisher(
        host=args.mqtt_host, port=args.mqtt_port,
        user=args.mqtt_user, pw=args.mqtt_pass,
        base=args.mqtt_base,
    )
    print(f"[MQTT] Publishing to {args.mqtt_host}:{args.mqtt_port} base={args.mqtt_base}")

    # Load YOLOv8n (auto-downloads weights on first run)
    model = YOLO("yolov8n.pt")

    # Try to use Apple Silicon MPS if available
    device = None
    try:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    except Exception:
        device = "cpu"

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Could not open webcam. Try a different --camera index.")
        print("On macOS, you may need to grant camera permissions to Terminal/Python.")
        print("Go to System Preferences > Security & Privacy > Camera and enable access.")
        return
    
    # Test if we can actually read from the camera
    ret, test_frame = cap.read()
    if not ret:
        print("Camera opened but cannot read frames. Check camera permissions.")
        cap.release()
        return

    roi_points = []
    roi_polygon = None
    drawing = False

    def on_mouse(event, x, y, flags, param):
        nonlocal roi_points, drawing
        if drawing and event == cv2.EVENT_LBUTTONDOWN:
            roi_points.append((x, y))

    cv2.namedWindow("People Counter")
    cv2.setMouseCallback("People Counter", on_mouse)

    over_since = None
    last_alert = 0.0
    last_state_pub = 0.0
    ALERT_COOLDOWN = 120.0  # seconds

    tracker = None
    if HAVE_SV:
        tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30, minimum_matching_threshold=0.8, frame_rate=30)

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        # Inference
        # results list; take first
        res = model(frame, imgsz=args.imgsz, conf=args.conf, classes=[0], device=device)[0]

        count = 0
        boxes_to_draw = []  # (x1,y1,x2,y2, id_or_score)

        if HAVE_SV and tracker is not None:
            det = sv.Detections.from_ultralytics(res)
            tracks = tracker.update_with_detections(det)
            # compute centers and check ROI
            if len(tracks) > 0:
                centers = tracks.get_anchors_coordinates(sv.Position.CENTER)
                # Prepare polygon
                poly_np = None
                if roi_polygon is not None:
                    poly_np = np.array(roi_polygon, dtype=np.int32)

                keep_idx = []
                for i, c in enumerate(centers):
                    inside = point_in_polygon((c[0], c[1]), poly_np)
                    if inside:
                        keep_idx.append(i)

                count = len(keep_idx)
                # Collect boxes to draw only for kept indices
                for i in keep_idx:
                    x1, y1, x2, y2 = tracks.xyxy[i]
                    tid = tracks.tracker_id[i] if tracks.tracker_id is not None else -1
                    boxes_to_draw.append((int(x1), int(y1), int(x2), int(y2), int(tid)))
        else:
            # Fallback: no tracking, just count detections in ROI
            poly_np = None
            if roi_polygon is not None:
                poly_np = np.array(roi_polygon, dtype=np.int32)

            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    cls = int(b.cls[0].item()) if hasattr(b, "cls") else 0
                    if cls != 0:
                        continue
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    if point_in_polygon((cx, cy), poly_np):
                        count += 1
                        score = float(b.conf[0].item()) if hasattr(b, "conf") else 0.0
                        boxes_to_draw.append((int(x1), int(y1), int(x2), int(y2), score))

        # ALERT LOGIC
        now = time.time()
        status = "OVER" if count > args.max else "OK"

        # Publish state every ~2s
        if time.time() - last_state_pub >= 2.0:
            state = {
                "ts": int(time.time()),
                "site": "demo",
                "zone": "lounge",
                "camera": f"webcam-{args.camera}",
                "count": int(count),
                "max": int(args.max),
                "status": status
            }
            publisher.publish_state(state)
            last_state_pub = time.time()

        if status == "OVER":
            over_since = over_since or now
            if (now - over_since) >= args.hold and (now - last_alert) >= ALERT_COOLDOWN:
                mac_say(f"Over capacity. Count {count} over {args.max}.")
                print(f"[ALERT] Threshold exceeded: {count} > {args.max}")
                last_alert = now
                over_since = None
        else:
            over_since = None

        # Draw overlay
        overlay = frame.copy()

        # ROI polygon
        if roi_polygon and len(roi_polygon) >= 3:
            pts = np.array(roi_polygon, dtype=np.int32)
            cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

        # If in drawing mode, show current poly and points
        if drawing:
            if len(roi_points) > 0:
                pts = np.array(roi_points, dtype=np.int32)
                cv2.polylines(overlay, [pts], isClosed=False, color=(255, 200, 0), thickness=2)
                for p in roi_points:
                    cv2.circle(overlay, p, 3, (255, 200, 0), -1)

        # Boxes
        for (x1, y1, x2, y2, ident) in boxes_to_draw:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(overlay, f"{ident}", (x1, y1 - 6), font, 0.5, (0, 200, 0), 1, cv2.LINE_AA)

        # Header
        h, w = overlay.shape[:2]
        bar_h = 36
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
        color = (0, 200, 0) if status == "OK" else (0, 0, 255)
        cv2.putText(overlay, f"Count: {count}  Status: {status}  Max: {args.max}", (10, 24), font, 0.7, color, 2, cv2.LINE_AA)

        cv2.imshow("People Counter", overlay)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('z'):
            # start ROI drawing
            drawing = True
            roi_points = []
            roi_polygon = None
            print("ROI drawing: click points, then press 'c' to commit, 'x' to clear.")
        elif key == ord('c'):
            # commit ROI
            if len(roi_points) >= 3:
                roi_polygon = roi_points.copy()
                print(f"ROI committed with {len(roi_polygon)} points.")
            else:
                print("Need at least 3 points for ROI.")
            drawing = False
        elif key == ord('x'):
            # clear ROI
            roi_points = []
            roi_polygon = None
            drawing = False
            print("ROI cleared.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
