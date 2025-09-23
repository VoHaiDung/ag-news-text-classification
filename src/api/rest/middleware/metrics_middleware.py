"""
Metrics Collection Middleware for REST API
================================================================================
Implements metrics collection for API performance monitoring including
request counts, latency, error rates, and resource utilization.

This middleware integrates with monitoring systems like Prometheus
for comprehensive observability.

References:
    - Google SRE Book: Chapter 6 - Monitoring Distributed Systems
    - Prometheus Best Practices
    - OpenTelemetry Specification

Author: Võ Hải Dũng
License: MIT
"""

import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """
    Metrics collector for aggregating API metrics.
    
    Implements various metric types including counters, gauges,
    histograms, and summaries following Prometheus conventions.
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        # Counters
        self.request_count = defaultdict(int)
        self.error_count = defaultdict(int)
        
        # Histograms (bucketed latencies)
        self.latency_buckets = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        self.latency_histogram = defaultdict(lambda: defaultdict(int))
        
        # Current values (gauges)
        self.active_requests = 0
        self.last_request_time = None
        
        # Summaries
        self.latency_sum = defaultdict(float)
        self.latency_count = defaultdict(int)
        
        # Error tracking
        self.error_types = defaultdict(lambda: defaultdict(int))
        
        # Resource metrics
        self.request_sizes = []
        self.response_sizes = []
    
    def record_request(
        self,
        method: str,
        path: str,
        status_code: int,
        latency: float,
        request_size: Optional[int] = None,
        response_size: Optional[int] = None,
        error_type: Optional[str] = None
    ):
        """
        Record metrics for a request.
        
        Args:
            method: HTTP method
            path: Request path
            status_code: Response status code
            latency: Request latency in seconds
            request_size: Request body size in bytes
            response_size: Response body size in bytes
            error_type: Type of error if occurred
        """
        # Create metric labels
        labels = f"{method}:{path}:{status_code}"
        
        # Update counters
        self.request_count[labels] += 1
        
        if status_code >= 400:
            self.error_count[labels] += 1
            if error_type:
                self.error_types[path][error_type] += 1
        
        # Update latency histogram
        for bucket in self.latency_buckets:
            if latency <= bucket:
                self.latency_histogram[labels][bucket] += 1
        
        # Update summaries
        self.latency_sum[labels] += latency
        self.latency_count[labels] += 1
        
        # Update sizes
        if request_size is not None:
            self.request_sizes.append(request_size)
            # Keep only last 1000 to prevent memory issues
            if len(self.request_sizes) > 1000:
                self.request_sizes = self.request_sizes[-1000:]
        
        if response_size is not None:
            self.response_sizes.append(response_size)
            if len(self.response_sizes) > 1000:
                self.response_sizes = self.response_sizes[-1000:]
        
        self.last_request_time = datetime.now(timezone.utc)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics snapshot.
        
        Returns:
            Dictionary of current metrics
        """
        metrics = {
            "request_total": sum(self.request_count.values()),
            "error_total": sum(self.error_count.values()),
            "active_requests": self.active_requests,
            "last_request_time": self.last_request_time.isoformat() if self.last_request_time else None
        }
        
        # Calculate error rate
        if metrics["request_total"] > 0:
            metrics["error_rate"] = metrics["error_total"] / metrics["request_total"]
        else:
            metrics["error_rate"] = 0.0
        
        # Calculate average latencies
        latency_averages = {}
        for labels in self.latency_count:
            if self.latency_count[labels] > 0:
                latency_averages[labels] = (
                    self.latency_sum[labels] / self.latency_count[labels]
                )
        metrics["latency_averages"] = latency_averages
        
        # Add percentiles
        metrics["latency_percentiles"] = self._calculate_percentiles()
        
        # Add size metrics
        if self.request_sizes:
            metrics["avg_request_size"] = sum(self.request_sizes) / len(self.request_sizes)
        if self.response_sizes:
            metrics["avg_response_size"] = sum(self.response_sizes) / len(self.response_sizes)
        
        # Add error breakdown
        metrics["error_types"] = dict(self.error_types)
        
        return metrics
    
    def _calculate_percentiles(self) -> Dict[str, float]:
        """
        Calculate latency percentiles.
        
        Returns:
            Dictionary of percentiles (p50, p90, p95, p99)
        """
        all_latencies = []
        
        for labels, histogram in self.latency_histogram.items():
            for bucket, count in histogram.items():
                all_latencies.extend([bucket] * count)
        
        if not all_latencies:
            return {"p50": 0, "p90": 0, "p95": 0, "p99": 0}
        
        all_latencies.sort()
        n = len(all_latencies)
        
        return {
            "p50": all_latencies[int(n * 0.5)],
            "p90": all_latencies[int(n * 0.9)],
            "p95": all_latencies[int(n * 0.95)],
            "p99": all_latencies[int(n * 0.99)]
        }
    
    def reset_metrics(self):
        """Reset all metrics to initial state."""
        self.request_count.clear()
        self.error_count.clear()
        self.latency_histogram.clear()
        self.latency_sum.clear()
        self.latency_count.clear()
        self.error_types.clear()
        self.request_sizes.clear()
        self.response_sizes.clear()
        self.active_requests = 0


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting API metrics.
    
    Captures performance metrics, error rates, and resource utilization
    for monitoring and alerting purposes.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        collector: Optional[MetricsCollector] = None,
        detailed_metrics: bool = True,
        exclude_paths: Optional[list] = None
    ):
        """
        Initialize metrics middleware.
        
        Args:
            app: ASGI application
            collector: Metrics collector instance
            detailed_metrics: Whether to collect detailed metrics
            exclude_paths: Paths to exclude from metrics
        """
        super().__init__(app)
        self.collector = collector or MetricsCollector()
        self.detailed_metrics = detailed_metrics
        self.exclude_paths = exclude_paths or ["/metrics", "/health"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and collect metrics.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response from the application
        """
        # Skip metrics for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Start timing and increment active requests
        start_time = time.time()
        self.collector.active_requests += 1
        
        # Get request size
        request_size = None
        if self.detailed_metrics:
            request_size = int(request.headers.get("content-length", 0))
        
        response = None
        error_type = None
        
        try:
            # Process request
            response = await call_next(request)
            
            # Get response size
            response_size = None
            if self.detailed_metrics and response:
                response_size = int(response.headers.get("content-length", 0))
            
            return response
            
        except Exception as e:
            error_type = type(e).__name__
            raise
            
        finally:
            # Calculate latency
            latency = time.time() - start_time
            
            # Decrement active requests
            self.collector.active_requests -= 1
            
            # Record metrics
            status_code = response.status_code if response else 500
            
            # Normalize path for metrics (remove path parameters)
            metric_path = self._normalize_path(request.url.path)
            
            self.collector.record_request(
                method=request.method,
                path=metric_path,
                status_code=status_code,
                latency=latency,
                request_size=request_size,
                response_size=response_size if response else None,
                error_type=error_type
            )
            
            # Add metrics to response headers if detailed metrics enabled
            if self.detailed_metrics and response:
                response.headers["X-Response-Time"] = f"{latency:.3f}"
                response.headers["X-Request-ID"] = getattr(
                    request.state,
                    "request_id",
                    "unknown"
                )
    
    def _normalize_path(self, path: str) -> str:
        """
        Normalize path for metrics aggregation.
        
        Replaces path parameters with placeholders to avoid
        high cardinality metrics.
        
        Args:
            path: Request path
            
        Returns:
            Normalized path
        """
        # Common patterns to normalize
        import re
        
        # UUID pattern
        path = re.sub(
            r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '/{id}',
            path
        )
        
        # Numeric IDs
        path = re.sub(r'/\d+', '/{id}', path)
        
        # Remove trailing slashes
        path = path.rstrip('/')
        
        return path or '/'
