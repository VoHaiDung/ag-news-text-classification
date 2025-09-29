"""
API Metrics Collection and Monitoring
================================================================================
This module implements comprehensive metrics collection for API endpoints,
tracking performance, usage patterns, and error rates across different API protocols.

The implementation supports REST, gRPC, and GraphQL APIs with protocol-specific
metrics and unified reporting.

Author: Võ Hải Dũng
License: MIT
"""

import time
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import asyncio
import functools

import numpy as np
from prometheus_client import Counter, Histogram, Gauge, Summary

from monitoring.metrics.custom_metrics import CustomMetrics, MetricDefinition, MetricType

logger = logging.getLogger(__name__)


@dataclass
class APIMetricConfig:
    """Configuration for API metrics collection."""
    
    track_request_body: bool = False
    track_response_body: bool = False
    track_headers: bool = True
    track_user_agent: bool = True
    
    # Latency buckets in seconds
    latency_buckets: List[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
    )
    
    # Request size buckets in bytes
    size_buckets: List[float] = field(
        default_factory=lambda: [100, 1000, 10000, 100000, 1000000, 10000000]
    )
    
    # Rate limiting
    rate_limit_window: int = 60  # seconds
    rate_limit_threshold: int = 1000  # requests per window


class APIMetrics:
    """
    Comprehensive API metrics collection and monitoring.
    
    This class provides:
    - Request/response metrics
    - Latency tracking
    - Error rate monitoring
    - Rate limiting metrics
    - Protocol-specific metrics
    """
    
    def __init__(
        self,
        metrics_manager: CustomMetrics,
        config: Optional[APIMetricConfig] = None
    ):
        """
        Initialize API metrics.
        
        Args:
            metrics_manager: Custom metrics manager
            config: API metrics configuration
        """
        self.metrics_manager = metrics_manager
        self.config = config or APIMetricConfig()
        
        # Request tracking
        self.active_requests = defaultdict(int)
        self.request_history = defaultdict(list)
        
        # Initialize API metrics
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize API-specific metrics."""
        
        # Request metrics
        self.metrics_manager.register_metric(MetricDefinition(
            name="api_requests_total",
            description="Total number of API requests",
            metric_type=MetricType.COUNTER,
            labels=["method", "endpoint", "protocol", "status"]
        ))
        
        self.metrics_manager.register_metric(MetricDefinition(
            name="api_request_duration_seconds",
            description="API request duration in seconds",
            metric_type=MetricType.HISTOGRAM,
            labels=["method", "endpoint", "protocol"],
            buckets=self.config.latency_buckets
        ))
        
        self.metrics_manager.register_metric(MetricDefinition(
            name="api_request_size_bytes",
            description="API request size in bytes",
            metric_type=MetricType.HISTOGRAM,
            labels=["method", "endpoint", "protocol"],
            buckets=self.config.size_buckets
        ))
        
        self.metrics_manager.register_metric(MetricDefinition(
            name="api_response_size_bytes",
            description="API response size in bytes",
            metric_type=MetricType.HISTOGRAM,
            labels=["method", "endpoint", "protocol"],
            buckets=self.config.size_buckets
        ))
        
        # Concurrent requests
        self.metrics_manager.register_metric(MetricDefinition(
            name="api_concurrent_requests",
            description="Number of concurrent API requests",
            metric_type=MetricType.GAUGE,
            labels=["endpoint", "protocol"]
        ))
        
        # Error metrics
        self.metrics_manager.register_metric(MetricDefinition(
            name="api_errors_total",
            description="Total number of API errors",
            metric_type=MetricType.COUNTER,
            labels=["endpoint", "protocol", "error_type", "status_code"]
        ))
        
        # Rate limiting metrics
        self.metrics_manager.register_metric(MetricDefinition(
            name="api_rate_limit_exceeded",
            description="Number of rate limit exceeded events",
            metric_type=MetricType.COUNTER,
            labels=["endpoint", "client_id"]
        ))
        
        # Authentication metrics
        self.metrics_manager.register_metric(MetricDefinition(
            name="api_auth_attempts",
            description="Number of authentication attempts",
            metric_type=MetricType.COUNTER,
            labels=["auth_type", "status"]
        ))
        
        # Protocol-specific metrics
        self._initialize_protocol_metrics()
    
    def _initialize_protocol_metrics(self):
        """Initialize protocol-specific metrics."""
        
        # REST-specific
        self.metrics_manager.register_metric(MetricDefinition(
            name="rest_api_cache_hits",
            description="Number of REST API cache hits",
            metric_type=MetricType.COUNTER,
            labels=["endpoint", "cache_type"]
        ))
        
        # gRPC-specific
        self.metrics_manager.register_metric(MetricDefinition(
            name="grpc_stream_duration_seconds",
            description="gRPC stream duration in seconds",
            metric_type=MetricType.HISTOGRAM,
            labels=["service", "method"],
            buckets=[1, 5, 10, 30, 60, 120, 300]
        ))
        
        self.metrics_manager.register_metric(MetricDefinition(
            name="grpc_message_received_total",
            description="Total gRPC messages received",
            metric_type=MetricType.COUNTER,
            labels=["service", "method", "type"]
        ))
        
        # GraphQL-specific
        self.metrics_manager.register_metric(MetricDefinition(
            name="graphql_query_complexity",
            description="GraphQL query complexity score",
            metric_type=MetricType.HISTOGRAM,
            labels=["operation_type", "operation_name"],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500]
        ))
        
        self.metrics_manager.register_metric(MetricDefinition(
            name="graphql_field_resolution_time",
            description="GraphQL field resolution time",
            metric_type=MetricType.HISTOGRAM,
            labels=["type_name", "field_name"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        ))
    
    def track_request(
        self,
        method: str,
        endpoint: str,
        protocol: str = "REST",
        request_size: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Track API request start.
        
        Args:
            method: HTTP method or RPC method
            endpoint: API endpoint or service name
            protocol: API protocol (REST, gRPC, GraphQL)
            request_size: Request body size in bytes
            metadata: Additional request metadata
            
        Returns:
            Request tracking ID
        """
        request_id = f"{protocol}_{endpoint}_{time.time()}"
        
        # Track concurrent requests
        key = f"{protocol}:{endpoint}"
        self.active_requests[key] += 1
        
        self.metrics_manager.update_metric(
            "api_concurrent_requests",
            self.active_requests[key],
            labels={"endpoint": endpoint, "protocol": protocol}
        )
        
        # Track request size
        if request_size:
            self.metrics_manager.update_metric(
                "api_request_size_bytes",
                request_size,
                labels={"method": method, "endpoint": endpoint, "protocol": protocol},
                operation="observe"
            )
        
        # Store request start time
        self.request_history[request_id] = {
            "start_time": time.time(),
            "method": method,
            "endpoint": endpoint,
            "protocol": protocol,
            "metadata": metadata or {}
        }
        
        return request_id
    
    def track_response(
        self,
        request_id: str,
        status_code: int,
        response_size: Optional[int] = None,
        error: Optional[str] = None
    ):
        """
        Track API response.
        
        Args:
            request_id: Request tracking ID
            status_code: HTTP status code or gRPC status
            response_size: Response body size in bytes
            error: Error message if any
        """
        if request_id not in self.request_history:
            logger.warning(f"Unknown request ID: {request_id}")
            return
        
        request_info = self.request_history[request_id]
        duration = time.time() - request_info["start_time"]
        
        # Update request counter
        self.metrics_manager.update_metric(
            "api_requests_total",
            1,
            labels={
                "method": request_info["method"],
                "endpoint": request_info["endpoint"],
                "protocol": request_info["protocol"],
                "status": str(status_code)
            },
            operation="inc"
        )
        
        # Track request duration
        self.metrics_manager.update_metric(
            "api_request_duration_seconds",
            duration,
            labels={
                "method": request_info["method"],
                "endpoint": request_info["endpoint"],
                "protocol": request_info["protocol"]
            },
            operation="observe"
        )
        
        # Track response size
        if response_size:
            self.metrics_manager.update_metric(
                "api_response_size_bytes",
                response_size,
                labels={
                    "method": request_info["method"],
                    "endpoint": request_info["endpoint"],
                    "protocol": request_info["protocol"]
                },
                operation="observe"
            )
        
        # Track errors
        if error or status_code >= 400:
            self.metrics_manager.update_metric(
                "api_errors_total",
                1,
                labels={
                    "endpoint": request_info["endpoint"],
                    "protocol": request_info["protocol"],
                    "error_type": error or "unknown",
                    "status_code": str(status_code)
                },
                operation="inc"
            )
        
        # Update concurrent requests
        key = f"{request_info['protocol']}:{request_info['endpoint']}"
        self.active_requests[key] = max(0, self.active_requests[key] - 1)
        
        self.metrics_manager.update_metric(
            "api_concurrent_requests",
            self.active_requests[key],
            labels={
                "endpoint": request_info["endpoint"],
                "protocol": request_info["protocol"]
            }
        )
        
        # Clean up request history
        del self.request_history[request_id]
    
    def track_grpc_stream(
        self,
        service: str,
        method: str,
        message_type: str,
        message_count: int = 1
    ):
        """
        Track gRPC streaming metrics.
        
        Args:
            service: gRPC service name
            method: RPC method name
            message_type: Type of message (request/response)
            message_count: Number of messages
        """
        self.metrics_manager.update_metric(
            "grpc_message_received_total",
            message_count,
            labels={
                "service": service,
                "method": method,
                "type": message_type
            },
            operation="inc"
        )
    
    def track_graphql_query(
        self,
        operation_type: str,
        operation_name: str,
        complexity: int,
        field_resolutions: Dict[str, float]
    ):
        """
        Track GraphQL query metrics.
        
        Args:
            operation_type: Query, Mutation, or Subscription
            operation_name: Name of the operation
            complexity: Query complexity score
            field_resolutions: Field resolution times
        """
        # Track query complexity
        self.metrics_manager.update_metric(
            "graphql_query_complexity",
            complexity,
            labels={
                "operation_type": operation_type,
                "operation_name": operation_name
            },
            operation="observe"
        )
        
        # Track field resolution times
        for field_name, resolution_time in field_resolutions.items():
            type_name, field = field_name.rsplit(".", 1)
            self.metrics_manager.update_metric(
                "graphql_field_resolution_time",
                resolution_time,
                labels={
                    "type_name": type_name,
                    "field_name": field
                },
                operation="observe"
            )
    
    def track_authentication(
        self,
        auth_type: str,
        success: bool,
        user_id: Optional[str] = None
    ):
        """
        Track authentication attempts.
        
        Args:
            auth_type: Type of authentication (JWT, OAuth, API Key)
            success: Whether authentication succeeded
            user_id: User identifier if available
        """
        self.metrics_manager.update_metric(
            "api_auth_attempts",
            1,
            labels={
                "auth_type": auth_type,
                "status": "success" if success else "failure"
            },
            operation="inc"
        )
    
    def check_rate_limit(
        self,
        client_id: str,
        endpoint: str
    ) -> bool:
        """
        Check if client has exceeded rate limit.
        
        Args:
            client_id: Client identifier
            endpoint: API endpoint
            
        Returns:
            True if rate limit exceeded
        """
        current_time = time.time()
        window_start = current_time - self.config.rate_limit_window
        
        # Clean old requests
        key = f"{client_id}:{endpoint}"
        self.request_history[key] = [
            t for t in self.request_history.get(key, [])
            if t > window_start
        ]
        
        # Check rate limit
        request_count = len(self.request_history[key])
        
        if request_count >= self.config.rate_limit_threshold:
            self.metrics_manager.update_metric(
                "api_rate_limit_exceeded",
                1,
                labels={
                    "endpoint": endpoint,
                    "client_id": client_id
                },
                operation="inc"
            )
            return True
        
        # Add current request
        self.request_history[key].append(current_time)
        return False
    
    def get_endpoint_statistics(
        self,
        endpoint: str,
        time_window: int = 3600
    ) -> Dict[str, Any]:
        """
        Get statistics for a specific endpoint.
        
        Args:
            endpoint: API endpoint
            time_window: Time window in seconds
            
        Returns:
            Endpoint statistics
        """
        # This would query the metrics backend
        # Placeholder implementation
        return {
            "endpoint": endpoint,
            "request_count": 1000,
            "error_rate": 0.02,
            "avg_latency": 0.150,
            "p95_latency": 0.450,
            "p99_latency": 0.850
        }


def api_metrics_decorator(
    metrics: APIMetrics,
    endpoint: str,
    protocol: str = "REST"
):
    """
    Decorator for automatic API metrics collection.
    
    Args:
        metrics: APIMetrics instance
        endpoint: API endpoint name
        protocol: API protocol
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Track request
            request_id = metrics.track_request(
                method=func.__name__,
                endpoint=endpoint,
                protocol=protocol
            )
            
            try:
                # Execute function
                result = await func(*args, **kwargs)
                
                # Track successful response
                metrics.track_response(
                    request_id=request_id,
                    status_code=200
                )
                
                return result
                
            except Exception as e:
                # Track error response
                metrics.track_response(
                    request_id=request_id,
                    status_code=500,
                    error=str(e)
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Track request
            request_id = metrics.track_request(
                method=func.__name__,
                endpoint=endpoint,
                protocol=protocol
            )
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Track successful response
                metrics.track_response(
                    request_id=request_id,
                    status_code=200
                )
                
                return result
                
            except Exception as e:
                # Track error response
                metrics.track_response(
                    request_id=request_id,
                    status_code=500,
                    error=str(e)
                )
                raise
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
