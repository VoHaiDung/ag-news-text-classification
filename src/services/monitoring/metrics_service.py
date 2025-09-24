"""
Metrics Service Implementation for AG News Text Classification
================================================================================
This module implements comprehensive metrics collection and monitoring for
system performance, model accuracy, and resource utilization.

The metrics service provides:
- Real-time metrics collection
- Aggregation and computation
- Time-series storage
- Alert triggering

References:
    - Google SRE Book: Monitoring Distributed Systems
    - Prometheus Best Practices

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import statistics

from src.services.base_service import BaseService, ServiceConfig
from src.core.exceptions import ServiceException
from src.utils.logging_config import get_logger


class MetricType(Enum):
    """
    Types of metrics supported.
    
    Types:
        COUNTER: Monotonically increasing value
        GAUGE: Value that can go up or down
        HISTOGRAM: Distribution of values
        SUMMARY: Statistical summary of values
    """
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AggregationType(Enum):
    """
    Aggregation methods for metrics.
    
    Types:
        SUM: Sum of values
        AVG: Average of values
        MIN: Minimum value
        MAX: Maximum value
        P50: 50th percentile
        P90: 90th percentile
        P99: 99th percentile
    """
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    P50 = "p50"
    P90 = "p90"
    P99 = "p99"


@dataclass
class Metric:
    """
    Individual metric definition.
    
    Attributes:
        name: Metric name
        type: Metric type
        value: Current value
        unit: Unit of measurement
        labels: Metric labels for grouping
        timestamp: Timestamp of last update
        description: Metric description
    """
    name: str
    type: MetricType
    value: Union[float, int] = 0
    unit: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "name": self.name,
            "type": self.type.value,
            "value": self.value,
            "unit": self.unit,
            "labels": self.labels,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description
        }


@dataclass
class MetricWindow:
    """
    Time window for metric aggregation.
    
    Attributes:
        duration: Window duration
        values: Values within window
        max_size: Maximum number of values to store
    """
    duration: timedelta
    values: deque = field(default_factory=lambda: deque(maxlen=10000))
    max_size: int = 10000
    
    def add_value(self, value: float, timestamp: datetime) -> None:
        """Add value to window."""
        self.values.append((timestamp, value))
        self._cleanup_old_values()
    
    def _cleanup_old_values(self) -> None:
        """Remove values outside window."""
        cutoff = datetime.now() - self.duration
        while self.values and self.values[0][0] < cutoff:
            self.values.popleft()
    
    def get_values(self) -> List[float]:
        """Get all values in window."""
        self._cleanup_old_values()
        return [v for _, v in self.values]


class MetricsService(BaseService):
    """
    Service for collecting and managing system metrics.
    
    This service provides comprehensive metrics collection with support for
    various metric types, aggregations, and time-series analysis.
    """
    
    def __init__(
        self,
        config: Optional[ServiceConfig] = None,
        flush_interval_seconds: int = 60,
        retention_hours: int = 24
    ):
        """
        Initialize metrics service.
        
        Args:
            config: Service configuration
            flush_interval_seconds: Interval for flushing metrics
            retention_hours: Hours to retain metrics
        """
        if config is None:
            config = ServiceConfig(name="metrics_service")
        super().__init__(config)
        
        # Metrics storage
        self._metrics: Dict[str, Metric] = {}
        self._time_series: Dict[str, MetricWindow] = {}
        
        # Configuration
        self.flush_interval = timedelta(seconds=flush_interval_seconds)
        self.retention_period = timedelta(hours=retention_hours)
        
        # Background tasks
        self._flush_task: Optional[asyncio.Task] = None
        
        # Alert thresholds
        self._thresholds: Dict[str, Dict[str, float]] = {}
        self._alert_callbacks: List[Callable] = []
        
        self.logger = get_logger("service.metrics")
    
    async def _initialize(self) -> None:
        """Initialize metrics service."""
        self.logger.info("Initializing metrics service")
        
        # Initialize default metrics
        self._initialize_default_metrics()
        
        # Start background flush task
        self._flush_task = asyncio.create_task(self._flush_loop())
    
    async def _start(self) -> None:
        """Start metrics service."""
        self.logger.info("Metrics service started")
    
    async def _stop(self) -> None:
        """Stop metrics service."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        await self._flush_metrics()
        
        self.logger.info("Metrics service stopped")
    
    async def _cleanup(self) -> None:
        """Cleanup metrics resources."""
        self._metrics.clear()
        self._time_series.clear()
    
    async def _check_health(self) -> bool:
        """Check metrics service health."""
        return len(self._metrics) < 10000  # Max metrics limit
    
    def _initialize_default_metrics(self) -> None:
        """Initialize default system metrics."""
        # Request metrics
        self.register_metric(
            "request_count",
            MetricType.COUNTER,
            "Total number of requests",
            unit="requests"
        )
        
        self.register_metric(
            "request_latency",
            MetricType.HISTOGRAM,
            "Request latency distribution",
            unit="milliseconds"
        )
        
        # Model metrics
        self.register_metric(
            "model_accuracy",
            MetricType.GAUGE,
            "Model accuracy score",
            unit="percentage"
        )
        
        self.register_metric(
            "prediction_count",
            MetricType.COUNTER,
            "Total predictions made",
            unit="predictions"
        )
        
        # Resource metrics
        self.register_metric(
            "memory_usage",
            MetricType.GAUGE,
            "Memory usage",
            unit="bytes"
        )
        
        self.register_metric(
            "cpu_usage",
            MetricType.GAUGE,
            "CPU usage percentage",
            unit="percentage"
        )
    
    def register_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str = "",
        unit: str = "",
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Register a new metric.
        
        Args:
            name: Metric name
            metric_type: Type of metric
            description: Metric description
            unit: Unit of measurement
            labels: Metric labels
        """
        metric = Metric(
            name=name,
            type=metric_type,
            description=description,
            unit=unit,
            labels=labels or {}
        )
        
        self._metrics[name] = metric
        
        # Initialize time series window
        self._time_series[name] = MetricWindow(
            duration=self.retention_period
        )
        
        self.logger.debug(f"Registered metric: {name} ({metric_type.value})")
    
    def record(
        self,
        name: str,
        value: Union[float, int],
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            labels: Additional labels
            
        Raises:
            ValueError: If metric not registered
        """
        if name not in self._metrics:
            raise ValueError(f"Metric not registered: {name}")
        
        metric = self._metrics[name]
        timestamp = datetime.now()
        
        # Update based on metric type
        if metric.type == MetricType.COUNTER:
            metric.value += value
        elif metric.type == MetricType.GAUGE:
            metric.value = value
        elif metric.type in [MetricType.HISTOGRAM, MetricType.SUMMARY]:
            # Store in time series for aggregation
            self._time_series[name].add_value(value, timestamp)
            metric.value = value  # Store latest value
        
        metric.timestamp = timestamp
        
        # Merge labels
        if labels:
            metric.labels.update(labels)
        
        # Check thresholds
        self._check_thresholds(name, value)
        
        self.logger.debug(f"Recorded metric: {name}={value}")
    
    def increment(
        self,
        name: str,
        value: Union[float, int] = 1,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Increment a counter metric.
        
        Args:
            name: Metric name
            value: Increment value
            labels: Additional labels
        """
        if name not in self._metrics:
            self.register_metric(name, MetricType.COUNTER)
        
        if self._metrics[name].type != MetricType.COUNTER:
            raise ValueError(f"Metric {name} is not a counter")
        
        self.record(name, value, labels)
    
    def set_gauge(
        self,
        name: str,
        value: Union[float, int],
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Set a gauge metric value.
        
        Args:
            name: Metric name
            value: Gauge value
            labels: Additional labels
        """
        if name not in self._metrics:
            self.register_metric(name, MetricType.GAUGE)
        
        if self._metrics[name].type != MetricType.GAUGE:
            raise ValueError(f"Metric {name} is not a gauge")
        
        self.record(name, value, labels)
    
    def observe(
        self,
        name: str,
        value: Union[float, int],
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Observe a value for histogram/summary metrics.
        
        Args:
            name: Metric name
            value: Observed value
            labels: Additional labels
        """
        if name not in self._metrics:
            self.register_metric(name, MetricType.HISTOGRAM)
        
        if self._metrics[name].type not in [MetricType.HISTOGRAM, MetricType.SUMMARY]:
            raise ValueError(f"Metric {name} is not a histogram or summary")
        
        self.record(name, value, labels)
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """
        Get metric by name.
        
        Args:
            name: Metric name
            
        Returns:
            Metric: Metric instance or None
        """
        return self._metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, Metric]:
        """
        Get all registered metrics.
        
        Returns:
            Dict[str, Metric]: All metrics
        """
        return self._metrics.copy()
    
    def aggregate(
        self,
        name: str,
        aggregation: AggregationType,
        window_minutes: Optional[int] = None
    ) -> Optional[float]:
        """
        Aggregate metric values.
        
        Args:
            name: Metric name
            aggregation: Aggregation type
            window_minutes: Time window in minutes
            
        Returns:
            float: Aggregated value or None
        """
        if name not in self._time_series:
            return None
        
        values = self._time_series[name].get_values()
        
        if not values:
            return None
        
        # Apply aggregation
        if aggregation == AggregationType.SUM:
            return sum(values)
        elif aggregation == AggregationType.AVG:
            return statistics.mean(values)
        elif aggregation == AggregationType.MIN:
            return min(values)
        elif aggregation == AggregationType.MAX:
            return max(values)
        elif aggregation == AggregationType.P50:
            return statistics.median(values)
        elif aggregation == AggregationType.P90:
            return statistics.quantiles(values, n=10)[8]
        elif aggregation == AggregationType.P99:
            return statistics.quantiles(values, n=100)[98]
        
        return None
    
    def set_threshold(
        self,
        name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> None:
        """
        Set alert threshold for metric.
        
        Args:
            name: Metric name
            min_value: Minimum threshold
            max_value: Maximum threshold
        """
        self._thresholds[name] = {
            "min": min_value,
            "max": max_value
        }
        
        self.logger.info(f"Set threshold for {name}: min={min_value}, max={max_value}")
    
    def _check_thresholds(self, name: str, value: float) -> None:
        """
        Check if metric value exceeds thresholds.
        
        Args:
            name: Metric name
            value: Metric value
        """
        if name not in self._thresholds:
            return
        
        threshold = self._thresholds[name]
        
        if threshold["min"] is not None and value < threshold["min"]:
            self._trigger_alert(name, value, "below_minimum", threshold["min"])
        
        if threshold["max"] is not None and value > threshold["max"]:
            self._trigger_alert(name, value, "above_maximum", threshold["max"])
    
    def _trigger_alert(
        self,
        metric_name: str,
        value: float,
        alert_type: str,
        threshold: float
    ) -> None:
        """
        Trigger alert for threshold violation.
        
        Args:
            metric_name: Metric name
            value: Current value
            alert_type: Type of threshold violation
            threshold: Threshold value
        """
        alert_data = {
            "metric": metric_name,
            "value": value,
            "type": alert_type,
            "threshold": threshold,
            "timestamp": datetime.now()
        }
        
        self.logger.warning(
            f"Alert: {metric_name} {alert_type} "
            f"(value={value}, threshold={threshold})"
        )
        
        # Trigger callbacks
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(alert_data))
                else:
                    callback(alert_data)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
    
    def add_alert_callback(self, callback: Callable) -> None:
        """
        Add alert callback function.
        
        Args:
            callback: Function to call on alert
        """
        self._alert_callbacks.append(callback)
        self.logger.debug(f"Added alert callback: {callback.__name__}")
    
    async def _flush_loop(self) -> None:
        """Background task for periodic metric flushing."""
        while True:
            try:
                await asyncio.sleep(self.flush_interval.total_seconds())
                await self._flush_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Flush loop error: {e}")
    
    async def _flush_metrics(self) -> None:
        """Flush metrics to storage/export."""
        self.logger.debug("Flushing metrics")
        
        # Export metrics (implement actual export logic)
        metrics_data = {
            name: metric.to_dict()
            for name, metric in self._metrics.items()
        }
        
        # Here you would send metrics to monitoring system
        # For now, just log summary
        self.logger.info(f"Flushed {len(metrics_data)} metrics")
