# Design Document: 3-Camera Occupancy & Security System

## Overview

The 3-Camera Occupancy & Security System is a comprehensive real-time monitoring solution that combines computer vision, IoT messaging, and web-based dashboards to provide accurate people counting, security monitoring, and incident management. The system is designed with a microservices architecture that prioritizes on-premises deployment while supporting optional cloud telemetry.

The system extends the existing single-camera demo implementation to support multiple RTSP cameras, advanced security workflows, multi-zone monitoring, and optional biometric authentication. It maintains the proven MQTT-based messaging architecture while adding enterprise-grade features for security operations.

## Architecture

### High-Level Architecture

```mermaid
graph TB
    subgraph "Camera Layer"
        C1[Entrance Camera<br/>RTSP]
        C2[Lounge Zone 1<br/>RTSP]
        C3[Lounge Zone 2<br/>RTSP]
    end
    
    subgraph "Processing Layer"
        CV[Computer Vision<br/>Service]
        AGG[Aggregation<br/>Service]
        BIO[Biometric Service<br/>(Optional)]
    end
    
    subgraph "Messaging Layer"
        MQTT[MQTT Broker<br/>Eclipse Mosquitto]
    end
    
    subgraph "Application Layer"
        ALERT[Alert Service]
        API[REST API Service]
        DASH[Security Dashboard]
        CONFIG[Configuration UI]
    end
    
    subgraph "Storage Layer"
        TS[(Time Series DB)]
        META[(Metadata DB)]
        LOGS[(Event Logs)]
    end
    
    subgraph "External Integrations"
        WA[WhatsApp API]
        SMS[SMS Gateway]
        EMAIL[Email Service]
        AWS[AWS IoT<br/>(Optional)]
    end
    
    C1 --> CV
    C2 --> CV
    C3 --> CV
    CV --> MQTT
    CV --> BIO
    MQTT --> AGG
    MQTT --> ALERT
    MQTT --> API
    AGG --> MQTT
    ALERT --> WA
    ALERT --> SMS
    ALERT --> EMAIL
    API --> TS
    API --> META
    API --> LOGS
    DASH --> API
    CONFIG --> API
    MQTT --> AWS
```

### Deployment Architecture

The system supports three deployment modes:

1. **100% On-Premises**: All components run locally with no external dependencies
2. **Light AWS Telemetry**: Local processing with optional cloud metrics
3. **Hybrid with Biometrics**: On-premises with cloud-based biometric processing

### Service Communication

- **MQTT Topics Structure**:
  - `site/{site_id}/occupancy/{zone}/state` - Real-time occupancy state
  - `site/{site_id}/occupancy/{zone}/events` - Security events
  - `site/{site_id}/cameras/{camera_id}/health` - Camera health status
  - `site/{site_id}/alerts/{alert_type}` - Alert notifications
  - `site/{site_id}/biometric/{event_type}` - Biometric events (optional)

## Components and Interfaces

### 1. Computer Vision Service

**Purpose**: Real-time person detection and tracking across multiple RTSP camera feeds

**Key Features**:
- Multi-camera RTSP stream processing
- YOLOv8-based person detection
- ByteTrack multi-object tracking for ID consistency
- ROI (Region of Interest) configuration per camera
- Entrance line detection for accurate in/out counting
- Cross-camera person re-identification for de-duplication

**Interfaces**:
```python
class CVService:
    def process_camera_feed(self, camera_id: str, rtsp_url: str) -> None
    def configure_roi(self, camera_id: str, polygon_points: List[Tuple[int, int]]) -> None
    def set_entrance_line(self, camera_id: str, line_coords: Tuple[int, int, int, int]) -> None
    def get_detection_results(self, camera_id: str) -> DetectionResult
    def health_check(self, camera_id: str) -> CameraHealth
```

**Configuration**:
```yaml
cameras:
  entrance:
    rtsp_url: "rtsp://camera1.local/stream"
    type: "entrance"
    roi_polygon: [[100,100], [500,100], [500,400], [100,400]]
    entrance_line: [200, 50, 200, 450]
  lounge_1:
    rtsp_url: "rtsp://camera2.local/stream"
    type: "zone"
    roi_polygon: [[0,0], [640,0], [640,480], [0,480]]
  lounge_2:
    rtsp_url: "rtsp://camera3.local/stream"
    type: "zone"
    roi_polygon: [[50,50], [590,50], [590,430], [50,430]]
```

### 2. Aggregation Service

**Purpose**: Combine data from multiple cameras, handle de-duplication, and maintain global occupancy state

**Key Features**:
- Multi-camera data fusion
- Short-window person re-identification
- Global occupancy calculation
- Entrance/exit flow tracking
- Data validation and anomaly detection

**Interfaces**:
```python
class AggregationService:
    def process_detection_batch(self, detections: List[DetectionResult]) -> None
    def get_global_occupancy(self) -> OccupancyState
    def get_zone_occupancy(self, zone_id: str) -> ZoneOccupancy
    def handle_camera_offline(self, camera_id: str) -> None
```

### 3. Alert Service

**Purpose**: Monitor thresholds, manage alert logic, and deliver notifications via multiple channels

**Key Features**:
- Configurable threshold monitoring
- Rate limiting and alert cooldown
- Multi-channel notification delivery
- Alert acknowledgment and escalation
- Retry logic with fallback channels

**Interfaces**:
```python
class AlertService:
    def check_thresholds(self, occupancy_state: OccupancyState) -> List[Alert]
    def send_alert(self, alert: Alert, channels: List[str]) -> AlertResult
    def acknowledge_alert(self, alert_id: str, operator_id: str, notes: str) -> None
    def escalate_alert(self, alert_id: str, escalation_level: int) -> None
```

**Alert Channels Configuration**:
```yaml
alert_channels:
  whatsapp:
    api_key: "${WHATSAPP_API_KEY}"
    recipients: ["+234XXXXXXXXX"]
    rate_limit: 10  # per hour
  sms:
    provider: "twilio"
    api_key: "${SMS_API_KEY}"
    recipients: ["+234XXXXXXXXX"]
    rate_limit: 20  # per hour
  email:
    smtp_server: "smtp.company.com"
    recipients: ["security@company.com"]
    rate_limit: 50  # per hour
```

### 4. Security Dashboard

**Purpose**: Real-time monitoring interface for security operators

**Key Features**:
- Multi-zone occupancy display
- Live camera health monitoring
- Alert management interface
- Historical trend visualization
- Event logging and export
- Operator workflow management

**Interface Components**:
- Real-time occupancy cards per zone
- Camera health status indicators
- Alert notification panel with acknowledge/escalate buttons
- Historical trend charts (Chart.js)
- Event log table with filtering and export
- Configuration panels for thresholds and ROIs

### 5. REST API Service

**Purpose**: Backend API for dashboard, configuration, and data access

**Key Endpoints**:
```python
# Occupancy Data
GET /api/occupancy/current
GET /api/occupancy/history?zone={zone}&from={timestamp}&to={timestamp}

# Camera Management
GET /api/cameras
POST /api/cameras/{camera_id}/configure
GET /api/cameras/{camera_id}/health

# Alert Management
GET /api/alerts?status={status}&from={timestamp}
POST /api/alerts/{alert_id}/acknowledge
POST /api/alerts/{alert_id}/escalate

# Configuration
GET /api/config/thresholds
PUT /api/config/thresholds
GET /api/config/zones
PUT /api/config/zones/{zone_id}

# Events and Reporting
GET /api/events?type={type}&from={timestamp}&to={timestamp}
GET /api/reports/occupancy?format=csv&from={timestamp}&to={timestamp}
```

### 6. Biometric Service (Optional Module)

**Purpose**: Staff attendance tracking and watchlist monitoring at entrance

**Key Features**:
- Face detection and recognition
- Encrypted biometric embedding storage
- Liveness detection
- Staff enrollment workflow
- Watchlist matching with human review
- Audit logging for compliance

**Interfaces**:
```python
class BiometricService:
    def enroll_staff(self, staff_id: str, images: List[bytes]) -> EnrollmentResult
    def identify_person(self, image: bytes) -> IdentificationResult
    def check_watchlist(self, embedding: bytes) -> WatchlistResult
    def log_biometric_event(self, event: BiometricEvent) -> None
```

## Data Models

### Core Data Structures

```python
@dataclass
class DetectionResult:
    camera_id: str
    timestamp: datetime
    detections: List[PersonDetection]
    camera_health: CameraHealth

@dataclass
class PersonDetection:
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    center_point: Tuple[int, int]
    in_roi: bool

@dataclass
class OccupancyState:
    site_id: str
    timestamp: datetime
    zones: Dict[str, ZoneOccupancy]
    global_count: int
    status: str  # "OK" | "OVER" | "CRITICAL"

@dataclass
class ZoneOccupancy:
    zone_id: str
    count: int
    max_capacity: int
    status: str
    last_updated: datetime

@dataclass
class Alert:
    alert_id: str
    type: str  # "THRESHOLD_EXCEEDED" | "CAMERA_OFFLINE" | "WATCHLIST_MATCH"
    severity: str  # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    message: str
    zone_id: Optional[str]
    camera_id: Optional[str]
    timestamp: datetime
    acknowledged: bool
    acknowledged_by: Optional[str]
    acknowledged_at: Optional[datetime]
    notes: Optional[str]

@dataclass
class CameraHealth:
    camera_id: str
    status: str  # "ONLINE" | "OFFLINE" | "DEGRADED"
    last_frame_time: datetime
    fps: float
    connection_quality: str
    error_message: Optional[str]
```

### Database Schema

**Time Series Data (InfluxDB/TimescaleDB)**:
```sql
-- Occupancy measurements
CREATE TABLE occupancy_measurements (
    time TIMESTAMPTZ NOT NULL,
    site_id TEXT NOT NULL,
    zone_id TEXT NOT NULL,
    count INTEGER NOT NULL,
    max_capacity INTEGER NOT NULL,
    status TEXT NOT NULL
);

-- Camera health metrics
CREATE TABLE camera_health (
    time TIMESTAMPTZ NOT NULL,
    camera_id TEXT NOT NULL,
    status TEXT NOT NULL,
    fps REAL,
    connection_quality TEXT,
    error_count INTEGER DEFAULT 0
);
```

**Metadata Database (PostgreSQL)**:
```sql
-- Site configuration
CREATE TABLE sites (
    site_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    timezone TEXT DEFAULT 'UTC',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Zone configuration
CREATE TABLE zones (
    zone_id TEXT PRIMARY KEY,
    site_id TEXT REFERENCES sites(site_id),
    name TEXT NOT NULL,
    max_capacity INTEGER NOT NULL,
    alert_threshold INTEGER NOT NULL,
    zone_type TEXT NOT NULL -- 'entrance' | 'lounge' | 'general'
);

-- Camera configuration
CREATE TABLE cameras (
    camera_id TEXT PRIMARY KEY,
    site_id TEXT REFERENCES sites(site_id),
    zone_id TEXT REFERENCES zones(zone_id),
    rtsp_url TEXT NOT NULL,
    roi_polygon JSONB,
    entrance_line JSONB,
    status TEXT DEFAULT 'ACTIVE'
);

-- Alert configuration
CREATE TABLE alert_rules (
    rule_id TEXT PRIMARY KEY,
    site_id TEXT REFERENCES sites(site_id),
    zone_id TEXT REFERENCES zones(zone_id),
    rule_type TEXT NOT NULL,
    threshold_value INTEGER,
    hold_duration INTEGER DEFAULT 3,
    channels JSONB NOT NULL -- ['whatsapp', 'sms', 'email']
);
```

**Event Logging (PostgreSQL)**:
```sql
-- Security events
CREATE TABLE security_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    site_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    zone_id TEXT,
    camera_id TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Alert history
CREATE TABLE alert_history (
    alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_id TEXT REFERENCES alert_rules(rule_id),
    triggered_at TIMESTAMPTZ NOT NULL,
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by TEXT,
    escalated_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,
    notes TEXT,
    metadata JSONB
);
```

## Error Handling

### Camera Connection Failures

1. **Detection**: Monitor frame timestamps and connection status
2. **Recovery**: Implement exponential backoff reconnection strategy
3. **Fallback**: Continue operation with remaining cameras
4. **Alerting**: Immediate notification to operators
5. **Logging**: Detailed error logs for troubleshooting

### MQTT Broker Failures

1. **Detection**: Connection heartbeat monitoring
2. **Recovery**: Automatic reconnection with message queuing
3. **Fallback**: Local buffering of critical messages
4. **Alerting**: System health alerts via alternative channels

### Processing Overload

1. **Detection**: Monitor processing latency and queue depths
2. **Recovery**: Dynamic frame rate adjustment and load balancing
3. **Fallback**: Graceful degradation with reduced accuracy
4. **Alerting**: Performance alerts to administrators

### Alert Delivery Failures

1. **Detection**: Delivery confirmation tracking
2. **Recovery**: Retry logic with exponential backoff
3. **Fallback**: Alternative channel routing (SMS → Email → Local)
4. **Logging**: Failed delivery tracking for audit

## Testing Strategy

### Unit Testing

- **Computer Vision Service**: Mock camera feeds, test detection accuracy
- **Aggregation Service**: Test de-duplication logic with synthetic data
- **Alert Service**: Test threshold logic and rate limiting
- **API Service**: Test all endpoints with various data scenarios

### Integration Testing

- **Camera Integration**: Test with real RTSP streams
- **MQTT Integration**: Test message flow and reliability
- **Database Integration**: Test data persistence and retrieval
- **Alert Integration**: Test end-to-end notification delivery

### Performance Testing

- **Load Testing**: Simulate multiple concurrent camera feeds
- **Stress Testing**: Test system behavior under high occupancy
- **Latency Testing**: Measure end-to-end detection to alert timing
- **Memory Testing**: Monitor memory usage under extended operation

### Security Testing

- **Authentication Testing**: Test API security and access controls
- **Data Privacy Testing**: Verify no video data leaves premises
- **Biometric Security**: Test encryption and secure storage
- **Network Security**: Test MQTT security and SSL/TLS

### Acceptance Testing

- **Accuracy Testing**: Validate ±2 people accuracy for <30 occupancy
- **Timing Testing**: Verify <2s detection latency and <10s alert delivery
- **Recovery Testing**: Verify <60s auto-recovery from failures
- **Usability Testing**: Test operator workflows and dashboard usability

### Test Data and Scenarios

1. **Normal Operation**: Standard occupancy patterns
2. **Peak Load**: High occupancy with rapid changes
3. **Edge Cases**: Partial occlusion, lighting changes, camera angles
4. **Failure Scenarios**: Camera disconnections, network issues
5. **Security Scenarios**: Watchlist matches, unauthorized access
6. **Multi-Zone Scenarios**: People moving between zones

The testing strategy ensures the system meets all acceptance criteria while maintaining reliability and security standards required for production deployment.