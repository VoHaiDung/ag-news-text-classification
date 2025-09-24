"""
Metrics Interceptor
================================================================================
This module implements metrics collection interceptor for gRPC services,
providing comprehensive performance monitoring and observability.

Collects metrics including:
- Request count and rate
- Response time distribution
- Error rates by type
- Method-level statistics

References:
    - Prometheus Best Practices
    - Google SRE Book: Chapter 6 - Monitoring Distributed Systems
    - OpenTelemetry Specification

Author: Võ Hải Dũng
License: MIT
"""

import time
import logging
from typing import Callable, Any, Dict, List
from collections import defaultdict
import grpc
import numpy as np
from datetime import datetime, timedelta

from . import BaseInterceptor

logger = logging.getLogger(__name__)

class MetricsInterceptor(BaseInterceptor):
    """
    Metrics collection interceptor for monitoring and observability.
    
    Collects detailed metrics for all RPC calls including latency,
    error rates, and throughput for SLO monitoring.
    """
    
    def __init__(self):
        """Initialize metrics interceptor."""
        super().__init__("MetricsInterceptor")
        
        # Method-level metrics
        self.method_metrics = defaultdict(lambda: {
            'count': 0,
            'errors': 0,
            'latencies': [],
            'status_codes': defaultdict(int)
        })
        
        # Time-based metrics
        self.time_buckets = defaultdict(lambda: {
            'requests': 0,
            'errors': 0,
            'total_latency': 0
        })
        
        # Global metrics
        self.global_metrics = {
            'total_requests': 0,
            'total_errors': 0,
            'active_requests': 0,
            'start_time': time.time()
        }
        
    def intercept_unary_unary(
        self,
        request: Any,
        context: grpc.ServicerContext,
        method: Callable,
        handler_call_details: grpc.HandlerCallDetails
    ) -> Any:
        """
        Intercept and collect metrics for unary-unary RPC.
        
        Args:
            request: Request message
            context: gRPC context
            method: Original method
            handler_call_details: Call details
            
        Returns:
            Any: Response message
        """
        # Start timing
        start_time = time.time()
        method_name = handler_call_details.method
        
        # Increment active requests
        self.global_metrics['active_requests'] += 1
        
        try:
            # Call original method
            response = method(request, context)
            
            # Record success metrics
            self._record_success(method_name, start_time)
            
            return response
            
        except grpc.RpcError as e:
            # Record error metrics
            self._record_error(method_name, start_time, e.code())
            raise
            
        except Exception as e:
            # Record unexpected error
            self._record_error(method_name, start_time, grpc.StatusCode.INTERNAL)
            raise
            
        finally:
            # Decrement active requests
            self.global_metrics['active_requests'] -= 1
    
    def _record_success(self, method: str, start_time: float) -> None:
        """
        Record successful request metrics.
        
        Args:
            method: Method name
            start_time: Request start time
        """
        latency = time.time() - start_time
        
        # Update method metrics
        metrics = self.method_metrics[method]
        metrics['count'] += 1
        metrics['latencies'].append(latency)
        metrics['status_codes'][grpc.StatusCode.OK] += 1
        
        # Keep only recent latencies for percentile calculation
        if len(metrics['latencies']) > 1000:
            metrics['latencies'] = metrics['latencies'][-1000:]
        
        # Update time bucket
        bucket = self._get_time_bucket()
        self.time_buckets[bucket]['requests'] += 1
        self.time_buckets[bucket]['total_latency'] += latency
        
        # Update global metrics
        self.global_metrics['total_requests'] += 1
        self.metrics["requests_intercepted"] += 1
    
    def _record_error(
        self,
        method: str,
        start_time: float,
        status_code: grpc.StatusCode
    ) -> None:
        """
        Record error metrics.
        
        Args:
            method: Method name
            start_time: Request start time
            status_code: gRPC status code
        """
        latency = time.time() - start_time
        
        # Update method metrics
        metrics = self.method_metrics[method]
        metrics['count'] += 1
        metrics['errors'] += 1
        metrics['latencies'].append(latency)
        metrics['status_codes'][status_code] += 1
        
        # Update time bucket
        bucket = self._get_time_bucket()
        self.time_buckets[bucket]['requests'] += 1
        self.time_buckets[bucket]['errors'] += 1
        self.time_buckets[bucket]['total_latency'] += latency
        
        # Update global metrics
        self.global_metrics['total_requests'] += 1
        self.global_metrics['total_errors'] += 1
        self.metrics["errors_handled"] += 1
    
    def _get_time_bucket(self) -> str:
        """
        Get current time bucket for time-series metrics.
        
        Returns:
            str: Time bucket identifier
        """
        # 1-minute buckets
        now = datetime.utcnow()
        return now.strftime("%Y-%m-%d %H:%M:00")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary.
        
        Returns:
            Dict[str, Any]: Metrics summary
        """
        uptime = time.time() - self.global_metrics['start_time']
        
        # Calculate aggregate metrics
        metrics_summary = {
            'global': {
                'total_requests': self.global_metrics['total_requests'],
                'total_errors': self.global_metrics['total_errors'],
                'error_rate': self._calculate_error_rate(),
                'active_requests': self.global_metrics['active_requests'],
                'uptime_seconds': uptime,
                'requests_per_second': self.global_metrics['total_requests'] / uptime if uptime > 0 else 0
            },
            'methods': {},
            'recent_performance': self._get_recent_performance()
        }
        
        # Add method-level metrics
        for method, metrics in self.method_metrics.items():
            method_summary = {
                'count': metrics['count'],
                'errors': metrics['errors'],
                'error_rate': metrics['errors'] / metrics['count'] if metrics['count'] > 0 else 0,
                'latency_stats': self._calculate_latency_stats(metrics['latencies']),
                'status_distribution': dict(metrics['status_codes'])
            }
            
            # Simplify method name for readability
            method_name = method.split('/')[-1] if '/' in method else method
            metrics_summary['methods'][method_name] = method_summary
        
        return metrics_summary
    
    def _calculate_error_rate(self) -> float:
        """
        Calculate overall error rate.
        
        Returns:
            float: Error rate (0-1)
        """
        total = self.global_metrics['total_requests']
        if total == 0:
            return 0.0
        
        return self.global_metrics['total_errors'] / total
    
    def _calculate_latency_stats(self, latencies: List[float]) -> Dict[str, float]:
        """
        Calculate latency statistics.
        
        Args:
            latencies: List of latency values
            
        Returns:
            Dict[str, float]: Latency statistics
        """
        if not latencies:
            return {
                'min': 0,
                'max': 0,
                'mean': 0,
                'p50': 0,
                'p95': 0,
                'p99': 0
            }
        
        latencies_ms = [l * 1000 for l in latencies]  # Convert to milliseconds
        
        return {
            'min': np.min(latencies_ms),
            'max': np.max(latencies_ms),
            'mean': np.mean(latencies_ms),
            'p50': np.percentile(latencies_ms, 50),
            'p95': np.percentile(latencies_ms, 95),
            'p99': np.percentile(latencies_ms, 99)
        }
    
    def _get_recent_performance(self, minutes: int = 5) -> Dict[str, Any]:
        """
        Get recent performance metrics.
        
        Args:
            minutes: Number of recent minutes to include
            
        Returns:
            Dict[str, Any]: Recent performance metrics
        """
        now = datetime.utcnow()
        recent_buckets = []
        
        # Collect recent buckets
        for i in range(minutes):
            bucket_time = now - timedelta(minutes=i)
            bucket = bucket_time.strftime("%Y-%m-%d %H:%M:00")
            if bucket in self.time_buckets:
                recent_buckets.append(self.time_buckets[bucket])
        
        if not recent_buckets:
            return {
                'requests': 0,
                'errors': 0,
                'error_rate': 0,
                'avg_latency_ms': 0
            }
        
        # Aggregate recent metrics
        total_requests = sum(b['requests'] for b in recent_buckets)
        total_errors = sum(b['errors'] for b in recent_buckets)
        total_latency = sum(b['total_latency'] for b in recent_buckets)
        
        return {
            'requests': total_requests,
            'errors': total_errors,
            'error_rate': total_errors / total_requests if total_requests > 0 else 0,
            'avg_latency_ms': (total_latency / total_requests * 1000) if total_requests > 0 else 0
        }
