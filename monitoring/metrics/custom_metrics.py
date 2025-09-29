"""
Custom Metrics Collection for AG News Text Classification
================================================================================
This module defines custom metrics for comprehensive monitoring of the ML system,
including model performance, resource utilization, and business metrics.

The implementation follows Prometheus metric conventions and provides extensible
metric definitions for different components of the system.

References:
    - Prometheus Best Practices: https://prometheus.io/docs/practices/
    - Google SRE Book: Chapter on Monitoring Distributed Systems

Author: Võ Hải Dũng
License: MIT
"""

import time
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
from collections import defaultdict, deque

import numpy as np
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    CollectorRegistry, generate_latest,
    CONTENT_TYPE_LATEST, Info
)

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Enumeration of supported metric types."""
    
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


@dataclass
class MetricDefinition:
    """Definition of a custom metric."""
    
    name: str
    description: str
    metric_type: MetricType
    labels: List[str] = field(default_factory=list)
    unit: Optional[str] = None
    buckets: Optional[List[float]] = None
    objectives: Optional[Dict[float, float]] = None


class CustomMetrics:
    """
    Manager for custom metrics collection and export.
    
    This class provides:
    - Custom metric definition and registration
    - Metric collection and aggregation
    - Export functionality for monitoring systems
    - Thread-safe metric updates
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize custom metrics manager.
        
        Args:
            registry: Prometheus collector registry
        """
        self.registry = registry or CollectorRegistry()
        self.metrics = {}
        self.lock = threading.Lock()
        
        # Initialize standard custom metrics
        self._initialize_standard_metrics()
        
        # Metric aggregation buffers
        self.aggregation_buffers = defaultdict(lambda: deque(maxlen=1000))
        
        # Start background aggregation thread
        self.aggregation_thread = threading.Thread(
            target=self._aggregation_worker,
            daemon=True
        )
        self.aggregation_thread.start()
    
    def _initialize_standard_metrics(self):
        """Initialize standard custom metrics for ML system."""
        
        # Model performance metrics
        self.register_metric(MetricDefinition(
            name="model_prediction_accuracy",
            description="Real-time prediction accuracy",
            metric_type=MetricType.GAUGE,
            labels=["model_name", "version"]
        ))
        
        self.register_metric(MetricDefinition(
            name="model_prediction_latency",
            description="Prediction latency in seconds",
            metric_type=MetricType.HISTOGRAM,
            labels=["model_name", "endpoint"],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        ))
        
        self.register_metric(MetricDefinition(
            name="model_predictions_total",
            description="Total number of predictions",
            metric_type=MetricType.COUNTER,
            labels=["model_name", "class_label", "status"]
        ))
        
        # Data quality metrics
        self.register_metric(MetricDefinition(
            name="data_quality_score",
            description="Data quality assessment score",
            metric_type=MetricType.GAUGE,
            labels=["dataset", "quality_dimension"]
        ))
        
        self.register_metric(MetricDefinition(
            name="data_drift_score",
            description="Data drift detection score",
            metric_type=MetricType.GAUGE,
            labels=["feature", "drift_type"]
        ))
        
        # Training metrics
        self.register_metric(MetricDefinition(
            name="training_loss",
            description="Training loss value",
            metric_type=MetricType.GAUGE,
            labels=["model_name", "epoch", "phase"]
        ))
        
        self.register_metric(MetricDefinition(
            name="training_duration_seconds",
            description="Training duration in seconds",
            metric_type=MetricType.HISTOGRAM,
            labels=["model_name", "training_type"],
            buckets=[60, 300, 600, 1800, 3600, 7200, 14400]
        ))
        
        # Resource utilization metrics
        self.register_metric(MetricDefinition(
            name="gpu_memory_usage_bytes",
            description="GPU memory usage in bytes",
            metric_type=MetricType.GAUGE,
            labels=["device_id", "process"]
        ))
        
        self.register_metric(MetricDefinition(
            name="model_size_bytes",
            description="Model size in bytes",
            metric_type=MetricType.GAUGE,
            labels=["model_name", "version"]
        ))
        
        # Business metrics
        self.register_metric(MetricDefinition(
            name="classification_confidence",
            description="Classification confidence distribution",
            metric_type=MetricType.HISTOGRAM,
            labels=["model_name", "class_label"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        ))
        
        self.register_metric(MetricDefinition(
            name="false_positive_rate",
            description="False positive rate by class",
            metric_type=MetricType.GAUGE,
            labels=["model_name", "class_label"]
        ))
        
        # System health metrics
        self.register_metric(MetricDefinition(
            name="model_staleness_seconds",
            description="Time since last model update",
            metric_type=MetricType.GAUGE,
            labels=["model_name"]
        ))
        
        self.register_metric(MetricDefinition(
            name="cache_hit_ratio",
            description="Cache hit ratio for predictions",
            metric_type=MetricType.GAUGE,
            labels=["cache_type", "model_name"]
        ))
    
    def register_metric(self, definition: MetricDefinition) -> None:
        """
        Register a custom metric.
        
        Args:
            definition: Metric definition
        """
        with self.lock:
            if definition.name in self.metrics:
                logger.warning(f"Metric {definition.name} already registered")
                return
            
            # Create appropriate Prometheus metric
            if definition.metric_type == MetricType.COUNTER:
                metric = Counter(
                    definition.name,
                    definition.description,
                    definition.labels,
                    registry=self.registry
                )
            elif definition.metric_type == MetricType.GAUGE:
                metric = Gauge(
                    definition.name,
                    definition.description,
                    definition.labels,
                    registry=self.registry
                )
            elif definition.metric_type == MetricType.HISTOGRAM:
                metric = Histogram(
                    definition.name,
                    definition.description,
                    definition.labels,
                    buckets=definition.buckets or Histogram.DEFAULT_BUCKETS,
                    registry=self.registry
                )
            elif definition.metric_type == MetricType.SUMMARY:
                metric = Summary(
                    definition.name,
                    definition.description,
                    definition.labels,
                    registry=self.registry
                )
            elif definition.metric_type == MetricType.INFO:
                metric = Info(
                    definition.name,
                    definition.description,
                    registry=self.registry
                )
            else:
                raise ValueError(f"Unsupported metric type: {definition.metric_type}")
            
            self.metrics[definition.name] = {
                "definition": definition,
                "metric": metric
            }
            
            logger.info(f"Registered metric: {definition.name}")
    
    def update_metric(
        self,
        name: str,
        value: Union[float, Dict[str, Any]],
        labels: Optional[Dict[str, str]] = None,
        operation: str = "set"
    ) -> None:
        """
        Update a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            labels: Label values
            operation: Operation to perform (set, inc, dec, observe)
        """
        with self.lock:
            if name not in self.metrics:
                logger.warning(f"Metric {name} not registered")
                return
            
            metric_info = self.metrics[name]
            metric = metric_info["metric"]
            definition = metric_info["definition"]
            
            # Apply labels if provided
            if labels and definition.labels:
                metric = metric.labels(**labels)
            
            # Perform operation based on metric type
            if definition.metric_type == MetricType.COUNTER:
                if operation == "inc":
                    metric.inc(value)
                else:
                    logger.warning(f"Invalid operation {operation} for Counter")
            
            elif definition.metric_type == MetricType.GAUGE:
                if operation == "set":
                    metric.set(value)
                elif operation == "inc":
                    metric.inc(value)
                elif operation == "dec":
                    metric.dec(value)
            
            elif definition.metric_type in [MetricType.HISTOGRAM, MetricType.SUMMARY]:
                if operation == "observe":
                    metric.observe(value)
                else:
                    logger.warning(f"Invalid operation {operation} for {definition.metric_type}")
            
            elif definition.metric_type == MetricType.INFO:
                if isinstance(value, dict):
                    metric.info(value)
                else:
                    logger.warning("Info metric requires dict value")
    
    def batch_update_metrics(self, updates: List[Dict[str, Any]]) -> None:
        """
        Update multiple metrics in batch.
        
        Args:
            updates: List of metric updates
        """
        for update in updates:
            self.update_metric(
                name=update["name"],
                value=update["value"],
                labels=update.get("labels"),
                operation=update.get("operation", "set")
            )
    
    def calculate_derived_metrics(self) -> Dict[str, float]:
        """
        Calculate derived metrics from base metrics.
        
        Returns:
            Dictionary of derived metric values
        """
        derived = {}
        
        # Calculate model efficiency score
        if "model_prediction_latency" in self.metrics:
            latency_samples = self._get_metric_samples("model_prediction_latency")
            if latency_samples:
                avg_latency = np.mean(latency_samples)
                derived["model_efficiency_score"] = 1.0 / (1.0 + avg_latency)
        
        # Calculate overall system health score
        health_components = []
        
        if "model_prediction_accuracy" in self.metrics:
            accuracy = self._get_metric_value("model_prediction_accuracy")
            if accuracy is not None:
                health_components.append(accuracy)
        
        if "cache_hit_ratio" in self.metrics:
            cache_ratio = self._get_metric_value("cache_hit_ratio")
            if cache_ratio is not None:
                health_components.append(cache_ratio)
        
        if health_components:
            derived["system_health_score"] = np.mean(health_components)
        
        # Calculate data quality index
        if "data_quality_score" in self.metrics:
            quality_scores = self._get_metric_samples("data_quality_score")
            if quality_scores:
                derived["data_quality_index"] = np.mean(quality_scores)
        
        return derived
    
    def get_metric_summary(self, name: str) -> Dict[str, Any]:
        """
        Get summary statistics for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Summary statistics
        """
        if name not in self.metrics:
            return {}
        
        samples = self._get_metric_samples(name)
        if not samples:
            return {}
        
        return {
            "mean": np.mean(samples),
            "std": np.std(samples),
            "min": np.min(samples),
            "max": np.max(samples),
            "p50": np.percentile(samples, 50),
            "p95": np.percentile(samples, 95),
            "p99": np.percentile(samples, 99),
            "count": len(samples)
        }
    
    def export_metrics(self, format: str = "prometheus") -> Union[str, Dict[str, Any]]:
        """
        Export metrics in specified format.
        
        Args:
            format: Export format (prometheus, json)
            
        Returns:
            Exported metrics
        """
        if format == "prometheus":
            return generate_latest(self.registry).decode("utf-8")
        
        elif format == "json":
            metrics_data = {}
            
            for name, metric_info in self.metrics.items():
                metric = metric_info["metric"]
                definition = metric_info["definition"]
                
                metrics_data[name] = {
                    "description": definition.description,
                    "type": definition.metric_type.value,
                    "labels": definition.labels,
                    "value": self._get_metric_value(name),
                    "summary": self.get_metric_summary(name)
                }
            
            # Add derived metrics
            metrics_data["derived"] = self.calculate_derived_metrics()
            
            return metrics_data
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _get_metric_value(self, name: str) -> Optional[float]:
        """Get current value of a metric."""
        if name in self.aggregation_buffers:
            samples = list(self.aggregation_buffers[name])
            if samples:
                return samples[-1]
        return None
    
    def _get_metric_samples(self, name: str) -> List[float]:
        """Get samples for a metric from aggregation buffer."""
        if name in self.aggregation_buffers:
            return list(self.aggregation_buffers[name])
        return []
    
    def _aggregation_worker(self):
        """Background worker for metric aggregation."""
        while True:
            try:
                # Aggregate metrics every 10 seconds
                time.sleep(10)
                
                # Calculate and update derived metrics
                derived = self.calculate_derived_metrics()
                
                for name, value in derived.items():
                    logger.debug(f"Updated derived metric {name}: {value}")
                
            except Exception as e:
                logger.error(f"Error in aggregation worker: {e}")
    
    def reset_metrics(self, names: Optional[List[str]] = None) -> None:
        """
        Reset specified metrics or all metrics.
        
        Args:
            names: List of metric names to reset (None for all)
        """
        with self.lock:
            if names is None:
                names = list(self.metrics.keys())
            
            for name in names:
                if name in self.aggregation_buffers:
                    self.aggregation_buffers[name].clear()
                    logger.info(f"Reset metric: {name}")
