# Mac Webcam People Counting Demo

A tiny demo to count people using your Mac's webcam. It uses YOLOv8n and optional tracking to stabilize counts. You can draw a polygon ROI and set a max occupancy threshold. When the count stays above the threshold for a few seconds, your Mac will speak an alert.

## Prereqs
- macOS with Python 3.10+
- A working webcam (built-in is fine).

## Setup (fresh venv recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip wheel
pip install -r requirements.txt
```

> First run will auto-download `yolov8n.pt` (about ~6–7 MB).

## Run
```bash
python mac_webcam_people_counter.py --max 40 --hold 3 --conf 0.35 --imgsz 960
```

- Press **z** to start drawing an ROI (click around the area).
- Press **c** to commit the polygon.
- Press **x** to clear the ROI.
- Press **q** to quit.

## Notes
- On Apple Silicon, Ultralytics will try to use **MPS** if available; otherwise CPU.
- For best speed, keep `imgsz` at 640–960 and good lighting.
- This is a demo: tracking helps, but crowd occlusion can still affect results.

## Next steps
- Replace webcam with RTSP camera input.
- Publish counts over MQTT and wire alerts with n8n/WhatsApp.
- Save ROI polygons to `json` and load per camera.


## MQTT + Mini Web Dashboard

### 1) Start a local MQTT broker (with WebSockets)
Requires Docker Desktop.
```bash
docker compose up -d
# Broker on mqtt://localhost:1883 and ws://localhost:9001
```

### 2) Run the counter with MQTT publishing
```bash
source .venv/bin/activate
python mac_webcam_people_counter.py \
  --max 40 --hold 3 --conf 0.35 --imgsz 960 \  --mqtt-host localhost --mqtt-port 1883 --mqtt-base site/demo/occupancy/lounge
```

### 3) Open the dashboard
Serve the static files (recommended):
```bash
python3 -m http.server 8080 -d dashboard
# then visit http://localhost:8080
```
Or open `dashboard/index.html` directly. It connects to `ws://localhost:9001`.

### Topics
- State:  `site/demo/occupancy/lounge/state`  (retained, ~every 2s)
- Events: `site/demo/occupancy/lounge/events`
- Health: `site/demo/occupancy/lounge/health`

### Payloads
State:
```json
{"ts": 1727000000, "site": "demo", "zone": "lounge", "camera": "webcam-0", "count": 3, "max": 40, "status": "OK"}
```
Event:
```json
{"ts": 1727000123, "site": "demo", "zone": "lounge", "camera": "webcam-0", "event": "THRESHOLD_EXCEEDED", "count": 45, "max": 40, "hold_seconds": 3}
```
