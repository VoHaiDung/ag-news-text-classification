"""
gRPC API Module
================================================================================
This module provides gRPC-based API interface for high-performance communication,
implementing Protocol Buffers for efficient serialization and bidirectional streaming.

The gRPC implementation offers:
- High-performance binary protocol
- Bidirectional streaming support
- Strong typing through protobuf
- Built-in load balancing and health checking

References:
    - gRPC Documentation (https://grpc.io)
    - Protocol Buffers Language Guide
    - Hohpe, G., & Woolf, B. (2003). Enterprise Integration Patterns

Author: Võ Hải Dũng
License: MIT
"""

from typing import Dict, Any

# Version information
__version__ = "1.0.0"
__grpc_version__ = "1.54.0"

# gRPC configuration defaults
DEFAULT_CONFIG = {
    "host": "0.0.0.0",
    "port": 50051,
    "max_workers": 10,
    "max_message_length": 4 * 1024 * 1024,  # 4MB
    "keepalive_time_ms": 10000,
    "keepalive_timeout_ms": 5000,
    "keepalive_permit_without_calls": True,
    "http2_max_pings_without_data": 2,
    "options": [
        ('grpc.max_send_message_length', 4 * 1024 * 1024),
        ('grpc.max_receive_message_length', 4 * 1024 * 1024),
        ('grpc.enable_retries', 1),
        ('grpc.keepalive_time_ms', 10000),
        ('grpc.keepalive_timeout_ms', 5000)
    ]
}

# Service names
SERVICES = {
    "classification": "ClassificationService",
    "training": "TrainingService",
    "model": "ModelManagementService",
    "data": "DataService",
    "health": "HealthService",
    "monitoring": "MonitoringService"
}

# Export configuration
__all__ = [
    "__version__",
    "__grpc_version__",
    "DEFAULT_CONFIG",
    "SERVICES"
]
