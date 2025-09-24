# Implementation Plan

- [x] 1. Set up project structure and core interfaces
  - Create directory structure for services, models, database, and API components
  - Define core data model interfaces and types for DetectionResult, OccupancyState, Alert, CameraHealth
  - Create base service interfaces for CVService, AggregationService, AlertService
  - Set up configuration management system for cameras, zones, and alert rules
  - _Requirements: 1.1, 4.1, 4.2_

- [x] 2. Implement enhanced computer vision service foundation
  - Extend existing mac_webcam_people_counter.py to support RTSP camera connections
  - Create CameraManager class to handle multiple concurrent RTSP streams
  - Implement camera health monitoring with connection status and FPS tracking
  - Add automatic reconnection logic with exponential backoff for failed camera connections
  - _Requirements: 1.1, 1.5, 5.1, 5.2_

- [x] 3. Implement multi-camera detection and ROI management
  - Create ROI configuration system that supports per-camera polygon definitions
  - Implement entrance line detection for accurate in/out counting at entrance camera
  - Add detection filtering logic to only count people within configured ROIs
  - Create camera configuration UI components for ROI drawing and entrance line setup
  - Write unit tests for ROI point-in-polygon calculations and entrance line detection
  - _Requirements: 4.2, 4.3, 4.4_

- [x] 4. Implement cross-camera person tracking and de-duplication
  - Extend ByteTrack implementation to support multi-camera person re-identification
  - Create person embedding extraction for short-window cross-camera matching
  - Implement de-duplication logic to prevent counting same person across multiple cameras
  - Add tracking persistence to maintain person IDs across brief occlusions
  - Write unit tests for tracking consistency and de-duplication accuracy
  - _Requirements: 1.2, 1.3_

- [ ] 5. Create aggregation service for multi-zone occupancy management
  - Implement AggregationService class that consumes detection results from multiple cameras
  - Create zone-based occupancy calculation with entrance/exit flow tracking
  - Add global occupancy state management that combines all zones
  - Implement data validation and anomaly detection for unrealistic occupancy changes
  - Write unit tests for occupancy calculations and zone management logic
  - _Requirements: 1.1, 1.2, 4.3_

- [ ] 6. Implement enhanced MQTT messaging architecture
  - Extend existing MQTT publisher to support structured topic hierarchy for multiple zones
  - Create MQTT message schemas for occupancy state, events, camera health, and alerts
  - Implement message persistence and retry logic for critical notifications
  - Add MQTT broker health monitoring and automatic reconnection
  - Write integration tests for MQTT message flow and reliability
  - _Requirements: 1.1, 5.2, 5.3_

- [ ] 7. Create alert service with threshold monitoring
  - Implement AlertService class with configurable threshold monitoring logic
  - Add support for hold duration (3+ seconds over threshold) before triggering alerts
  - Create rate limiting system to prevent alert spam with configurable cooldown periods
  - Implement alert acknowledgment and escalation workflows
  - Write unit tests for threshold detection, rate limiting, and alert state management
  - _Requirements: 2.1, 2.2, 2.3, 3.4, 3.5_

- [ ] 8. Implement multi-channel notification system
  - Create notification channel interfaces for WhatsApp, SMS, and email delivery
  - Implement WhatsApp Business API integration with message templates and rate limiting
  - Add SMS gateway integration (Twilio) with delivery confirmation tracking
  - Create email notification service with SMTP configuration and HTML templates
  - Implement fallback channel routing when primary channels fail
  - Write integration tests for each notification channel and fallback logic
  - _Requirements: 2.2, 2.4, 2.5_

- [ ] 9. Create database layer and data persistence
  - Set up PostgreSQL database with tables for sites, zones, cameras, alert rules, and events
  - Implement time series database (TimescaleDB) for occupancy measurements and camera health
  - Create database connection management with connection pooling and retry logic
  - Implement data access layer with repositories for each entity type
  - Add database migration system for schema versioning
  - Write unit tests for all database operations and data integrity
  - _Requirements: 3.3, 3.6, 5.4_

- [ ] 10. Implement REST API service for dashboard backend
  - Create FastAPI application with endpoints for occupancy data, camera management, and alerts
  - Implement real-time occupancy endpoints with current state and historical data queries
  - Add camera management endpoints for configuration, health status, and ROI updates
  - Create alert management endpoints for listing, acknowledging, and escalating alerts
  - Implement event logging endpoints with filtering and CSV export functionality
  - Write API integration tests for all endpoints with various data scenarios
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [ ] 11. Create enhanced security dashboard frontend
  - Extend existing dashboard/index.html to support multiple zones with individual occupancy cards
  - Implement real-time camera health status indicators with connection quality display
  - Create alert notification panel with acknowledge and escalate buttons for operator workflows
  - Add historical trend visualization using Chart.js for occupancy patterns over time
  - Implement event log table with filtering, search, and CSV export functionality
  - Write frontend unit tests for dashboard components and user interactions
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [ ] 12. Implement configuration management UI
  - Create configuration interface for camera RTSP URLs, ROI polygons, and entrance lines
  - Add zone configuration UI for setting occupancy thresholds and alert parameters
  - Implement alert rule configuration with channel selection and rate limiting settings
  - Create recipient management interface for WhatsApp, SMS, and email contacts
  - Add system settings panel for MQTT broker configuration and database connections
  - Write UI tests for configuration workflows and validation
  - _Requirements: 4.1, 4.2, 4.4, 4.5_

- [ ] 13. Implement system reliability and monitoring features
  - Create watchdog processes for automatic service restart on failures
  - Implement system health monitoring with CPU, memory, and disk usage tracking
  - Add comprehensive logging system with structured logs and log rotation
  - Create system status dashboard showing service health and performance metrics
  - Implement backup and recovery procedures for configuration and historical data
  - Write reliability tests for failure scenarios and auto-recovery mechanisms
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 14. Create biometric authentication module (optional add-on)
  - Implement face detection and recognition using deep learning models
  - Create staff enrollment workflow with multiple image capture and embedding generation
  - Add encrypted biometric embedding storage with secure key management
  - Implement liveness detection to prevent spoofing attacks
  - Create watchlist matching system with human review workflow for security alerts
  - Add comprehensive audit logging for all biometric events and access attempts
  - Write security tests for biometric data protection and access controls
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [ ] 15. Implement performance optimization and testing
  - Add performance monitoring for detection latency, processing throughput, and memory usage
  - Implement dynamic frame rate adjustment based on system load and processing capacity
  - Create load balancing for multiple camera streams across available CPU cores
  - Add caching layers for frequently accessed configuration and historical data
  - Implement database query optimization and indexing for fast data retrieval
  - Write performance tests to validate <2s detection latency and <10s alert delivery requirements
  - _Requirements: 1.1, 2.2, 8.1, 8.2, 8.3, 8.4_

- [ ] 16. Create deployment and infrastructure automation
  - Create Docker containers for all services with optimized images and health checks
  - Implement Docker Compose configuration for local development and testing environments
  - Add Kubernetes deployment manifests for production scalability and high availability
  - Create infrastructure as code (Terraform) for AWS resources if cloud telemetry is enabled
  - Implement automated backup procedures for databases and configuration files
  - Write deployment tests to validate system functionality after deployment
  - _Requirements: 7.1, 7.2, 7.4, 7.5_

- [ ] 17. Implement comprehensive testing and validation
  - Create end-to-end integration tests that validate complete workflows from detection to alerts
  - Implement accuracy testing with synthetic and real video data to validate ±2 people accuracy
  - Add stress testing for high occupancy scenarios and rapid occupancy changes
  - Create security testing for API authentication, data encryption, and access controls
  - Implement user acceptance testing scenarios for operator workflows and dashboard usability
  - Write automated test suite that can be run in CI/CD pipeline for continuous validation
  - _Requirements: 1.3, 1.4, 2.2, 5.5, 8.1, 8.2_

- [ ] 18. Create documentation and operator training materials
  - Write comprehensive API documentation with OpenAPI specifications and examples
  - Create operator manual with step-by-step workflows for alert management and system monitoring
  - Implement system administration guide with installation, configuration, and troubleshooting procedures
  - Add inline help system in dashboard with contextual guidance for operators
  - Create training videos and materials for security staff on system operation
  - Write runbook for common maintenance tasks and emergency procedures
  - _Requirements: 3.4, 3.5, 3.6_

- [ ] 19. Implement optional AWS cloud telemetry integration
  - Create AWS IoT Core integration for sending occupancy metrics and system health data
  - Implement AWS Timestream integration for cloud-based historical data storage
  - Add AWS CloudWatch integration for system monitoring and alerting
  - Create optional Grafana dashboard for cloud-based data visualization
  - Implement secure data transmission with encryption and authentication
  - Write cloud integration tests to validate data flow and security
  - _Requirements: 7.2, 7.3, 7.5_

- [ ] 20. Final system integration and production readiness
  - Integrate all services into complete system with proper service orchestration
  - Implement final end-to-end testing with real camera hardware and network conditions
  - Add production monitoring and alerting for system health and performance
  - Create disaster recovery procedures and backup/restore functionality
  - Implement security hardening with proper authentication, authorization, and encryption
  - Conduct final user acceptance testing with actual security operators
  - Prepare production deployment checklist and go-live procedures
  - _Requirements: All requirements validation and system acceptance criteria_