"""
Common Protocol Buffer Types Module
================================================================================
This module contains shared Protocol Buffer message types and enums used across
multiple gRPC services in the AG News Text Classification system.

These common types ensure consistency in data structures and reduce duplication
across service definitions.

References:
    - Protocol Buffers Style Guide: https://developers.google.com/protocol-buffers/docs/style
    - gRPC Common Patterns: https://grpc.io/docs/guides/

Author: Võ Hải Dũng
License: MIT
"""

from typing import List

# Import common protocol buffer types
from .types_pb2 import (
    Timestamp,
    Metadata,
    Pagination,
    SortOrder,
    FieldMask,
    ErrorDetail,
    RequestOptions,
    ResponseMetadata
)

from .status_pb2 import (
    Status,
    StatusCode,
    ErrorInfo,
    RetryInfo,
    DebugInfo,
    QuotaViolation,
    PreconditionFailure,
    BadRequest
)

# Version information
COMMON_PROTO_VERSION = "1.0.0"

# Export all common types
__all__: List[str] = [
    # From types_pb2
    "Timestamp",
    "Metadata",
    "Pagination",
    "SortOrder",
    "FieldMask",
    "ErrorDetail",
    "RequestOptions",
    "ResponseMetadata",
    
    # From status_pb2
    "Status",
    "StatusCode",
    "ErrorInfo",
    "RetryInfo",
    "DebugInfo",
    "QuotaViolation",
    "PreconditionFailure",
    "BadRequest",
    
    # Version
    "COMMON_PROTO_VERSION"
]
