"""
gRPC Compiled Protocol Buffers Module
================================================================================
This module contains compiled Protocol Buffer definitions for gRPC services.
The compilation is automated through the build process using protoc compiler.

Protocol Buffer compilation follows Google's Protocol Buffer style guide and
gRPC best practices for service definition and message structuring.

References:
    - Protocol Buffers Documentation: https://developers.google.com/protocol-buffers
    - gRPC Python Documentation: https://grpc.io/docs/languages/python/
    - Google API Design Guide: https://cloud.google.com/apis/design

Author: Project Team
License: MIT
"""

import sys
from pathlib import Path

# Add proto path to system path for proper imports
PROTO_PATH = Path(__file__).parent
sys.path.insert(0, str(PROTO_PATH))

# Version information for compiled protos
PROTO_VERSION = "3.19.0"
GRPC_VERSION = "1.50.0"

# Export all compiled proto modules
__all__ = [
    "classification_pb2",
    "classification_pb2_grpc",
    "model_management_pb2",
    "model_management_pb2_grpc",
    "training_pb2",
    "training_pb2_grpc",
    "data_service_pb2",
    "data_service_pb2_grpc",
    "health_pb2",
    "health_pb2_grpc",
    "monitoring_pb2",
    "monitoring_pb2_grpc",
]
