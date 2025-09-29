"""
Service Metrics Collection and Monitoring
================================================================================
This module implements comprehensive metrics collection for microservices,
tracking service health, performance, and inter-service communication patterns.

The implementation follows distributed tracing principles and provides
service mesh observability capabilities.

References:
    - Distributed Tracing in Practice (Austin Parker et al., 2020)
    - Site Reliability Engineering (Google, 2016)

Author: Võ Hải Dũng
License: MIT
"""

import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import threading
import uuid

import numpy as np
from prometheus_client import Counter, Histogram, Gauge, Summary

from monitoring.metrics.custom_metrics import CustomMetrics, MetricDefinition, MetricType
from src.services.base_service import BaseService

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service health status enumeration."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceDependency:
    """Service dependency definition."""
    
    service_name: str
    required: bool = True
    timeout: float = 30.0
    retry_policy: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceMetricConfig:
    """Configuration for service metrics collection."""
    
    # Health check settings
    health_check_interval: int = 30  # seconds
    health_check_timeout: int = 10  # seconds
    unhealthy_threshold: int = 3  # consecutive failures
    
    # Circuit breaker settings
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: float = 0.5  # error rate
    circuit_breaker_window: int = 60  # seconds
    circuit_breaker_cooldown: int = 30  # seconds
    
    # Tracing settings
    enable_distributed_tracing: bool = True
    trace_sample_rate: float = 0.1  # 10% sampling
    
    # SLA monitoring
    sla_targets: Dict[str, float] = field(default_factory=lambda: {
        "availability": 0.999,  # 99.9%
        "latency_p99": 1.0,  # 1 second
        "error_rate": 0.01  # 1%
    })


class ServiceMetrics:
    """
    Comprehensive service metrics collection and monitoring.
    
    This class provides:
    - Service health monitoring
    - Inter-service communication tracking
    - Circuit breaker implementation
    - Distributed tracing support
    - SLA monitoring and alerting
    """
    
    def __init__(
        self,
        service_name: str,
        metrics_manager: CustomMetrics,
        config: Optional[ServiceMetricConfig] = None
    ):
        """
        Initialize service metrics.
        
        Args:
            service_name: Name of the service
            metrics_manager: Custom metrics manager
            config: Service metrics configuration
        """
        self.service_name = service_name
        self.metrics_manager = metrics_manager
        self.config = config or ServiceMetricConfig()
        
        # Service state
        self.status = ServiceStatus.UNKNOWN
        self.health_check_failures = 0
        self.dependencies: Dict[str, ServiceDependency] = {}
        
        # Circuit breaker state
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Tracing
        self.active_traces: Dict[str, TraceContext] = {}
        self.trace_buffer = deque(maxlen=10000)
        
        # SLA tracking
        self.sla_violations = defaultdict(list)
        
        # Initialize metrics
        self._initialize_metrics()
        
        # Start monitoring
        self._start_monitoring()
    
    def _initialize_metrics(self):
        """Initialize service-specific metrics."""
        
        # Service health metrics
        self.metrics_manager.register_metric(MetricDefinition(
            name="service_health_status",
            description="Service health status (1=healthy, 0=unhealthy)",
            metric_type=MetricType.GAUGE,
            labels=["service_name", "instance_id"]
        ))
        
        self.metrics_manager.register_metric(MetricDefinition(
            name="service_uptime_seconds",
            description="Service uptime in seconds",
            metric_type=MetricType.GAUGE,
            labels=["service_name"]
        ))
        
        # Request metrics
        self.metrics_manager.register_metric(MetricDefinition(
            name="service_request_total",
            description="Total service requests",
            metric_type=MetricType.COUNTER,
            labels=["service_name", "method", "status"]
        ))
        
        self.metrics_manager.register_metric(MetricDefinition(
            name="service_request_duration_seconds",
            description="Service request duration",
            metric_type=MetricType.HISTOGRAM,
            labels=["service_name", "method"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        ))
        
        # Inter-service communication
        self.metrics_manager.register_metric(MetricDefinition(
            name="service_dependency_calls",
            description="Calls to service dependencies",
            metric_type=MetricType.COUNTER,
            labels=["source_service", "target_service", "status"]
        ))
        
        self.metrics_manager.register_metric(MetricDefinition(
            name="service_dependency_latency",
            description="Dependency call latency",
            metric_type=MetricType.HISTOGRAM,
            labels=["source_service", "target_service"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        ))
        
        # Circuit breaker metrics
        self.metrics_manager.register_metric(MetricDefinition(
            name="circuit_breaker_state",
            description="Circuit breaker state (0=closed, 1=open, 2=half-open)",
            metric_type=MetricType.GAUGE,
            labels=["service_name", "target_service"]
        ))
        
        self.metrics_manager.register_metric(MetricDefinition(
            name="circuit_breaker_trips",
            description="Circuit breaker trip count",
            metric_type=MetricType.COUNTER,
            labels=["service_name", "target_service"]
        ))
        
        # Queue metrics
        self.metrics_manager.register_metric(MetricDefinition(
            name="service_queue_size",
            description="Service queue size",
            metric_type=MetricType.GAUGE,
            labels=["service_name", "queue_name"]
        ))
        
        self.metrics_manager.register_metric(MetricDefinition(
            name="service_queue_latency",
            description="Queue processing latency",
            metric_type=MetricType.HISTOGRAM,
            labels=["service_name", "queue_name"],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]
        ))
        
        # Resource metrics
        self.metrics_manager.register_metric(MetricDefinition(
            name="service_thread_pool_size",
            description="Service thread pool size",
            metric_type=MetricType.GAUGE,
            labels=["service_name", "pool_name"]
        ))
        
        self.metrics_manager.register_metric(MetricDefinition(
            name="service_connection_pool_size",
            description="Service connection pool size",
            metric_type=MetricType.GAUGE,
            labels=["service_name", "pool_type"]
        ))
        
        # SLA metrics
        self.metrics_manager.register_metric(MetricDefinition(
            name="service_sla_compliance",
            description="SLA compliance rate",
            metric_type=MetricType.GAUGE,
            labels=["service_name", "sla_metric"]
        ))
        
        self.metrics_manager.register_metric(MetricDefinition(
            name="service_sla_violations",
            description="SLA violation count",
            metric_type=MetricType.COUNTER,
            labels=["service_name", "sla_metric", "severity"]
        ))
    
    def _start_monitoring(self):
        """Start background monitoring tasks."""
        # Health check thread
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self.health_check_thread.start()
        
        # SLA monitoring thread
        self.sla_monitor_thread = threading.Thread(
            target=self._sla_monitor_loop,
            daemon=True
        )
        self.sla_monitor_thread.start()
    
    def _health_check_loop(self):
        """Background health check loop."""
        while True:
            try:
                # Perform health check
                is_healthy = self._perform_health_check()
                
                # Update status
                if is_healthy:
                    self.status = ServiceStatus.HEALTHY
                    self.health_check_failures = 0
                else:
                    self.health_check_failures += 1
                    
                    if self.health_check_failures >= self.config.unhealthy_threshold:
                        self.status = ServiceStatus.UNHEALTHY
                    else:
                        self.status = ServiceStatus.DEGRADED
                
                # Update metrics
                self.metrics_manager.update_metric(
                    "service_health_status",
                    1 if self.status == ServiceStatus.HEALTHY else 0,
                    labels={"service_name": self.service_name, "instance_id": self._get_instance_id()}
                )
                
                # Sleep until next check
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                time.sleep(10)
    
    def _perform_health_check(self) -> bool:
        """
        Perform service health check.
        
        Returns:
            True if service is healthy
        """
        try:
            # Check service responsiveness
            # This would be implemented based on service type
            
            # Check dependencies
            for dep_name, dependency in self.dependencies.items():
                if dependency.required and not self._check_dependency_health(dep_name):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def _check_dependency_health(self, dependency_name: str) -> bool:
        """Check health of a service dependency."""
        # Placeholder implementation
        return True
    
    def _sla_monitor_loop(self):
        """Background SLA monitoring loop."""
        while True:
            try:
                # Calculate SLA metrics
                sla_metrics = self._calculate_sla_metrics()
                
                # Check for violations
                for metric_name, current_value in sla_metrics.items():
                    target_value = self.config.sla_targets.get(metric_name)
                    
                    if target_value:
                        compliance = self._check_sla_compliance(
                            metric_name,
                            current_value,
                            target_value
                        )
                        
                        # Update metrics
                        self.metrics_manager.update_metric(
                            "service_sla_compliance",
                            compliance,
                            labels={
                                "service_name": self.service_name,
                                "sla_metric": metric_name
                            }
                        )
                        
                        # Track violations
                        if compliance < 1.0:
                            self._record_sla_violation(metric_name, current_value, target_value)
                
                # Sleep for monitoring interval
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in SLA monitor loop: {e}")
                time.sleep(10)
    
    def _calculate_sla_metrics(self) -> Dict[str, float]:
        """Calculate current SLA metrics."""
        # Placeholder implementation
        return {
            "availability": 0.998,
            "latency_p99": 0.850,
            "error_rate": 0.012
        }
    
    def _check_sla_compliance(
        self,
        metric_name: str,
        current_value: float,
        target_value: float
    ) -> float:
        """Check SLA compliance for a metric."""
        if metric_name == "error_rate":
            # Lower is better
            return 1.0 if current_value <= target_value else target_value / current_value
        else:
            # Higher is better
            return min(1.0, current_value / target_value)
    
    def _record_sla_violation(
        self,
        metric_name: str,
        current_value: float,
        target_value: float
    ):
        """Record an SLA violation."""
        violation = {
            "timestamp": datetime.now(),
            "metric": metric_name,
            "current_value": current_value,
            "target_value": target_value,
            "severity": self._calculate_violation_severity(current_value, target_value)
        }
        
        self.sla_violations[metric_name].append(violation)
        
        # Update metrics
        self.metrics_manager.update_metric(
            "service_sla_violations",
            1,
            labels={
                "service_name": self.service_name,
                "sla_metric": metric_name,
                "severity": violation["severity"]
            },
            operation="inc"
        )
        
        logger.warning(f"SLA violation: {violation}")
    
    def _calculate_violation_severity(
        self,
        current_value: float,
        target_value: float
    ) -> str:
        """Calculate severity of SLA violation."""
        deviation = abs(current_value - target_value) / target_value
        
        if deviation < 0.1:
            return "low"
        elif deviation < 0.25:
            return "medium"
        else:
            return "high"
    
    def track_request(
        self,
        method: str,
        duration: float,
        status: str,
        trace_id: Optional[str] = None
    ):
        """
        Track a service request.
        
        Args:
            method: Request method/operation
            duration: Request duration in seconds
            status: Request status (success/failure)
            trace_id: Distributed trace ID
        """
        # Update request metrics
        self.metrics_manager.update_metric(
            "service_request_total",
            1,
            labels={
                "service_name": self.service_name,
                "method": method,
                "status": status
            },
            operation="inc"
        )
        
        self.metrics_manager.update_metric(
            "service_request_duration_seconds",
            duration,
            labels={
                "service_name": self.service_name,
                "method": method
            },
            operation="observe"
        )
        
        # Update trace if provided
        if trace_id and trace_id in self.active_traces:
            self.active_traces[trace_id].add_span(
                service=self.service_name,
                operation=method,
                duration=duration,
                status=status
            )
    
    def track_dependency_call(
        self,
        target_service: str,
        duration: float,
        status: str
    ):
        """
        Track a call to a service dependency.
        
        Args:
            target_service: Target service name
            duration: Call duration in seconds
            status: Call status
        """
        # Update dependency metrics
        self.metrics_manager.update_metric(
            "service_dependency_calls",
            1,
            labels={
                "source_service": self.service_name,
                "target_service": target_service,
                "status": status
            },
            operation="inc"
        )
        
        self.metrics_manager.update_metric(
            "service_dependency_latency",
            duration,
            labels={
                "source_service": self.service_name,
                "target_service": target_service
            },
            operation="observe"
        )
        
        # Update circuit breaker
        if target_service in self.circuit_breakers:
            self.circuit_breakers[target_service].record_result(
                success=(status == "success")
            )
    
    def get_circuit_breaker(self, target_service: str) -> "CircuitBreaker":
        """
        Get or create circuit breaker for target service.
        
        Args:
            target_service: Target service name
            
        Returns:
            Circuit breaker instance
        """
        if target_service not in self.circuit_breakers:
            self.circuit_breakers[target_service] = CircuitBreaker(
                service_name=self.service_name,
                target_service=target_service,
                metrics_manager=self.metrics_manager,
                config=self.config
            )
        
        return self.circuit_breakers[target_service]
    
    def start_trace(self, operation: str) -> str:
        """
        Start a new distributed trace.
        
        Args:
            operation: Initial operation name
            
        Returns:
            Trace ID
        """
        if not self.config.enable_distributed_tracing:
            return ""
        
        # Sample based on configured rate
        if np.random.random() > self.config.trace_sample_rate:
            return ""
        
        trace_id = str(uuid.uuid4())
        self.active_traces[trace_id] = TraceContext(
            trace_id=trace_id,
            service=self.service_name,
            operation=operation,
            start_time=time.time()
        )
        
        return trace_id
    
    def end_trace(self, trace_id: str):
        """
        End a distributed trace.
        
        Args:
            trace_id: Trace ID
        """
        if trace_id in self.active_traces:
            trace = self.active_traces.pop(trace_id)
            trace.end_time = time.time()
            
            # Store in buffer for export
            self.trace_buffer.append(trace)
    
    def _get_instance_id(self) -> str:
        """Get service instance ID."""
        # This could be from environment variable or generated
        return f"{self.service_name}-001"


@dataclass
class TraceContext:
    """Distributed trace context."""
    
    trace_id: str
    service: str
    operation: str
    start_time: float
    end_time: Optional[float] = None
    spans: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_span(
        self,
        service: str,
        operation: str,
        duration: float,
        status: str
    ):
        """Add a span to the trace."""
        self.spans.append({
            "service": service,
            "operation": operation,
            "duration": duration,
            "status": status,
            "timestamp": time.time()
        })


class CircuitBreaker:
    """
    Circuit breaker implementation for service protection.
    
    Implements the circuit breaker pattern to prevent cascading failures
    in distributed systems.
    """
    
    def __init__(
        self,
        service_name: str,
        target_service: str,
        metrics_manager: CustomMetrics,
        config: ServiceMetricConfig
    ):
        """
        Initialize circuit breaker.
        
        Args:
            service_name: Source service name
            target_service: Target service name
            metrics_manager: Metrics manager
            config: Service metrics configuration
        """
        self.service_name = service_name
        self.target_service = target_service
        self.metrics_manager = metrics_manager
        self.config = config
        
        # State
        self.state = "closed"  # closed, open, half_open
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.window_start = time.time()
        
        # Metrics window
        self.request_window = deque(maxlen=100)
        self.lock = threading.Lock()
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        with self.lock:
            self._update_state()
            return self.state == "open"
    
    def record_result(self, success: bool):
        """
        Record request result.
        
        Args:
            success: Whether request succeeded
        """
        with self.lock:
            self.request_window.append({
                "timestamp": time.time(),
                "success": success
            })
            
            if success:
                self.success_count += 1
                
                # Reset on success in half-open state
                if self.state == "half_open":
                    self._close()
            else:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                # Check if should trip
                if self._should_trip():
                    self._trip()
            
            self._update_metrics()
    
    def _update_state(self):
        """Update circuit breaker state."""
        if self.state == "open":
            # Check if cooldown period has passed
            if (self.last_failure_time and 
                time.time() - self.last_failure_time > self.config.circuit_breaker_cooldown):
                self.state = "half_open"
                logger.info(f"Circuit breaker half-open: {self.service_name} -> {self.target_service}")
    
    def _should_trip(self) -> bool:
        """Check if circuit breaker should trip."""
        if not self.config.circuit_breaker_enabled:
            return False
        
        # Calculate error rate in window
        current_time = time.time()
        window_requests = [
            r for r in self.request_window
            if current_time - r["timestamp"] < self.config.circuit_breaker_window
        ]
        
        if len(window_requests) < 10:  # Minimum requests
            return False
        
        error_rate = sum(1 for r in window_requests if not r["success"]) / len(window_requests)
        
        return error_rate > self.config.circuit_breaker_threshold
    
    def _trip(self):
        """Trip the circuit breaker."""
        self.state = "open"
        logger.warning(f"Circuit breaker tripped: {self.service_name} -> {self.target_service}")
        
        # Update metrics
        self.metrics_manager.update_metric(
            "circuit_breaker_trips",
            1,
            labels={
                "service_name": self.service_name,
                "target_service": self.target_service
            },
            operation="inc"
        )
    
    def _close(self):
        """Close the circuit breaker."""
        self.state = "closed"
        self.failure_count = 0
        logger.info(f"Circuit breaker closed: {self.service_name} -> {self.target_service}")
    
    def _update_metrics(self):
        """Update circuit breaker metrics."""
        state_value = {"closed": 0, "open": 1, "half_open": 2}[self.state]
        
        self.metrics_manager.update_metric(
            "circuit_breaker_state",
            state_value,
            labels={
                "service_name": self.service_name,
                "target_service": self.target_service
            }
        )
