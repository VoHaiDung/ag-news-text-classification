"""
Monitoring Services Module for AG News Text Classification
================================================================================
This module provides comprehensive monitoring capabilities for system health,
performance metrics, alerting, and logging across all components.

The monitoring services enable:
- Real-time metrics collection and aggregation
- Health status monitoring and auto-recovery
- Alert management with multi-channel notifications
- Centralized logging with pattern detection

References:
    - Google SRE Book: Monitoring Distributed Systems
    - Prometheus: Systems and Service Monitoring
    - The Art of Monitoring (James Turnbull)

Author: Võ Hải Dũng
License: MIT
"""

from src.services.monitoring.metrics_service import (
    MetricsService,
    MetricType,
    AggregationType,
    Metric,
    MetricWindow
)
from src.services.monitoring.health_service import (
    HealthService,
    HealthStatus,
    ComponentHealth,
    HealthCheck
)
from src.services.monitoring.alerting_service import (
    AlertingService,
    AlertSeverity,
    AlertStatus,
    AlertRule,
    Alert,
    AlertChannel
)
from src.services.monitoring.logging_service import (
    LoggingService,
    LogLevel,
    LogEntry,
    LogPattern
)

__all__ = [
    # Metrics Service
    "MetricsService",
    "MetricType",
    "AggregationType",
    "Metric",
    "MetricWindow",
    
    # Health Service
    "HealthService",
    "HealthStatus",
    "ComponentHealth",
    "HealthCheck",
    
    # Alerting Service
    "AlertingService",
    "AlertSeverity",
    "AlertStatus",
    "AlertRule",
    "Alert",
    "AlertChannel",
    
    # Logging Service
    "LoggingService",
    "LogLevel",
    "LogEntry",
    "LogPattern"
]

# Module version
__version__ = "1.0.0"
