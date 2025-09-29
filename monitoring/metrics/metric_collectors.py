"""
Metric Collectors for System Monitoring
================================================================================
This module implements various metric collectors that gather metrics from
different components of the ML system including models, data, and infrastructure.

The collectors follow a plugin-based architecture allowing easy extension
and customization of metric collection strategies.

Author: Võ Hải Dũng
License: MIT
"""

import os
import time
import psutil
import logging
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import asyncio

import torch
import numpy as np
import pandas as pd
from prometheus_client import CollectorRegistry

from monitoring.metrics.custom_metrics import CustomMetrics, MetricDefinition, MetricType

logger = logging.getLogger(__name__)


@dataclass
class CollectorConfig:
    """Configuration for metric collector."""
    
    name: str
    enabled: bool = True
    collection_interval: int = 60  # seconds
    batch_size: int = 100
    timeout: int = 30  # seconds
    retry_attempts: int = 3
    labels: Dict[str, str] = None


class BaseMetricCollector(ABC):
    """
    Abstract base class for metric collectors.
    
    Provides common functionality for metric collection
    including scheduling, error handling, and batching.
    """
    
    def __init__(
        self,
        config: CollectorConfig,
        metrics_manager: CustomMetrics
    ):
        """
        Initialize metric collector.
        
        Args:
            config: Collector configuration
            metrics_manager: Custom metrics manager
        """
        self.config = config
        self.metrics_manager = metrics_manager
        self.is_running = False
        self.collection_thread = None
        self.last_collection_time = None
        self.collection_errors = 0
        
        # Initialize collector-specific metrics
        self._initialize_metrics()
    
    @abstractmethod
    def _initialize_metrics(self):
        """Initialize metrics specific to this collector."""
        pass
    
    @abstractmethod
    def collect(self) -> Dict[str, Any]:
        """
        Collect metrics from the source.
        
        Returns:
            Dictionary of collected metrics
        """
        pass
    
    def start(self):
        """Start metric collection."""
        if self.is_running:
            logger.warning(f"Collector {self.config.name} already running")
            return
        
        self.is_running = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True
        )
        self.collection_thread.start()
        logger.info(f"Started collector: {self.config.name}")
    
    def stop(self):
        """Stop metric collection."""
        self.is_running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info(f"Stopped collector: {self.config.name}")
    
    def _collection_loop(self):
        """Main collection loop."""
        while self.is_running:
            try:
                # Collect metrics
                start_time = time.time()
                metrics = self.collect()
                collection_duration = time.time() - start_time
                
                # Update metrics
                self._update_metrics(metrics)
                
                # Update collection metadata
                self.last_collection_time = datetime.now()
                self.collection_errors = 0
                
                # Log collection stats
                logger.debug(
                    f"Collected {len(metrics)} metrics from {self.config.name} "
                    f"in {collection_duration:.2f}s"
                )
                
                # Sleep until next collection
                time.sleep(self.config.collection_interval)
                
            except Exception as e:
                self.collection_errors += 1
                logger.error(f"Error in collector {self.config.name}: {e}")
                
                if self.collection_errors >= self.config.retry_attempts:
                    logger.error(
                        f"Collector {self.config.name} exceeded retry attempts, stopping"
                    )
                    self.is_running = False
                else:
                    time.sleep(10)  # Wait before retry
    
    def _update_metrics(self, metrics: Dict[str, Any]):
        """Update metrics in the metrics manager."""
        updates = []
        
        for name, value in metrics.items():
            update = {
                "name": name,
                "value": value,
                "labels": self.config.labels
            }
            
            # Determine operation based on metric type
            if isinstance(value, (int, float)):
                if "latency" in name or "duration" in name:
                    update["operation"] = "observe"
                elif "total" in name or "count" in name:
                    update["operation"] = "inc"
                else:
                    update["operation"] = "set"
            
            updates.append(update)
        
        self.metrics_manager.batch_update_metrics(updates)


class ModelMetricCollector(BaseMetricCollector):
    """Collector for model-related metrics."""
    
    def __init__(
        self,
        config: CollectorConfig,
        metrics_manager: CustomMetrics,
        model_registry: Optional[Any] = None
    ):
        self.model_registry = model_registry
        super().__init__(config, metrics_manager)
    
    def _initialize_metrics(self):
        """Initialize model-specific metrics."""
        # Model performance metrics
        self.metrics_manager.register_metric(MetricDefinition(
            name="model_inference_throughput",
            description="Model inference throughput (samples/sec)",
            metric_type=MetricType.GAUGE,
            labels=["model_name", "device"]
        ))
        
        self.metrics_manager.register_metric(MetricDefinition(
            name="model_batch_size",
            description="Average batch size for inference",
            metric_type=MetricType.GAUGE,
            labels=["model_name"]
        ))
        
        self.metrics_manager.register_metric(MetricDefinition(
            name="model_parameter_count",
            description="Number of model parameters",
            metric_type=MetricType.GAUGE,
            labels=["model_name", "layer_type"]
        ))
    
    def collect(self) -> Dict[str, Any]:
        """Collect model metrics."""
        metrics = {}
        
        try:
            # Collect model inference metrics
            if hasattr(self, "_inference_stats"):
                metrics["model_inference_throughput"] = self._calculate_throughput()
                metrics["model_batch_size"] = self._get_average_batch_size()
            
            # Collect model size metrics
            if self.model_registry:
                for model_name, model in self._get_active_models().items():
                    param_count = sum(p.numel() for p in model.parameters())
                    metrics[f"model_parameter_count_{model_name}"] = param_count
                    
                    # Memory usage
                    if torch.cuda.is_available():
                        memory_usage = torch.cuda.memory_allocated()
                        metrics[f"model_gpu_memory_{model_name}"] = memory_usage
            
            # Collect gradient norms if training
            if self._is_training():
                metrics["gradient_norm"] = self._get_gradient_norm()
            
        except Exception as e:
            logger.error(f"Error collecting model metrics: {e}")
        
        return metrics
    
    def _calculate_throughput(self) -> float:
        """Calculate model inference throughput."""
        # Placeholder implementation
        return 100.0
    
    def _get_average_batch_size(self) -> float:
        """Get average batch size."""
        # Placeholder implementation
        return 32.0
    
    def _get_active_models(self) -> Dict[str, Any]:
        """Get currently active models."""
        # Placeholder implementation
        return {}
    
    def _is_training(self) -> bool:
        """Check if model is currently training."""
        return False
    
    def _get_gradient_norm(self) -> float:
        """Get gradient norm during training."""
        return 0.0


class SystemMetricCollector(BaseMetricCollector):
    """Collector for system-level metrics."""
    
    def _initialize_metrics(self):
        """Initialize system metrics."""
        self.metrics_manager.register_metric(MetricDefinition(
            name="system_cpu_usage_percent",
            description="CPU usage percentage",
            metric_type=MetricType.GAUGE,
            labels=["host", "cpu_id"]
        ))
        
        self.metrics_manager.register_metric(MetricDefinition(
            name="system_memory_usage_bytes",
            description="Memory usage in bytes",
            metric_type=MetricType.GAUGE,
            labels=["host", "memory_type"]
        ))
        
        self.metrics_manager.register_metric(MetricDefinition(
            name="system_disk_usage_bytes",
            description="Disk usage in bytes",
            metric_type=MetricType.GAUGE,
            labels=["host", "mount_point"]
        ))
        
        self.metrics_manager.register_metric(MetricDefinition(
            name="system_network_bytes_sent",
            description="Network bytes sent",
            metric_type=MetricType.COUNTER,
            labels=["host", "interface"]
        ))
        
        self.metrics_manager.register_metric(MetricDefinition(
            name="system_network_bytes_recv",
            description="Network bytes received",
            metric_type=MetricType.COUNTER,
            labels=["host", "interface"]
        ))
    
    def collect(self) -> Dict[str, Any]:
        """Collect system metrics."""
        metrics = {}
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics["system_cpu_usage_percent"] = cpu_percent
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics["system_memory_usage_bytes"] = memory.used
            metrics["system_memory_available_bytes"] = memory.available
            metrics["system_memory_percent"] = memory.percent
            
            # Disk metrics
            disk = psutil.disk_usage("/")
            metrics["system_disk_usage_bytes"] = disk.used
            metrics["system_disk_free_bytes"] = disk.free
            metrics["system_disk_percent"] = disk.percent
            
            # Network metrics
            net_io = psutil.net_io_counters()
            metrics["system_network_bytes_sent"] = net_io.bytes_sent
            metrics["system_network_bytes_recv"] = net_io.bytes_recv
            
            # Process-specific metrics
            process = psutil.Process()
            metrics["process_cpu_percent"] = process.cpu_percent()
            metrics["process_memory_rss"] = process.memory_info().rss
            metrics["process_num_threads"] = process.num_threads()
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
        
        return metrics


class DataMetricCollector(BaseMetricCollector):
    """Collector for data-related metrics."""
    
    def __init__(
        self,
        config: CollectorConfig,
        metrics_manager: CustomMetrics,
        data_pipeline: Optional[Any] = None
    ):
        self.data_pipeline = data_pipeline
        super().__init__(config, metrics_manager)
    
    def _initialize_metrics(self):
        """Initialize data metrics."""
        self.metrics_manager.register_metric(MetricDefinition(
            name="data_processing_rate",
            description="Data processing rate (samples/sec)",
            metric_type=MetricType.GAUGE,
            labels=["pipeline", "stage"]
        ))
        
        self.metrics_manager.register_metric(MetricDefinition(
            name="data_queue_size",
            description="Size of data processing queue",
            metric_type=MetricType.GAUGE,
            labels=["queue_name"]
        ))
        
        self.metrics_manager.register_metric(MetricDefinition(
            name="data_validation_errors",
            description="Number of data validation errors",
            metric_type=MetricType.COUNTER,
            labels=["error_type", "dataset"]
        ))
    
    def collect(self) -> Dict[str, Any]:
        """Collect data metrics."""
        metrics = {}
        
        try:
            if self.data_pipeline:
                # Data processing metrics
                metrics["data_processing_rate"] = self._get_processing_rate()
                metrics["data_queue_size"] = self._get_queue_size()
                
                # Data quality metrics
                quality_stats = self._calculate_data_quality()
                metrics.update(quality_stats)
                
                # Data distribution metrics
                distribution_stats = self._analyze_data_distribution()
                metrics.update(distribution_stats)
            
        except Exception as e:
            logger.error(f"Error collecting data metrics: {e}")
        
        return metrics
    
    def _get_processing_rate(self) -> float:
        """Get data processing rate."""
        # Placeholder implementation
        return 1000.0
    
    def _get_queue_size(self) -> int:
        """Get data queue size."""
        # Placeholder implementation
        return 0
    
    def _calculate_data_quality(self) -> Dict[str, float]:
        """Calculate data quality metrics."""
        return {
            "data_completeness": 0.98,
            "data_consistency": 0.95,
            "data_validity": 0.99
        }
    
    def _analyze_data_distribution(self) -> Dict[str, Any]:
        """Analyze data distribution."""
        return {
            "data_class_balance": 0.85,
            "data_feature_variance": 0.72
        }


class MetricCollectorManager:
    """
    Manager for coordinating multiple metric collectors.
    
    This class manages the lifecycle of metric collectors
    and provides centralized control over metric collection.
    """
    
    def __init__(self, metrics_manager: CustomMetrics):
        """
        Initialize collector manager.
        
        Args:
            metrics_manager: Custom metrics manager
        """
        self.metrics_manager = metrics_manager
        self.collectors = {}
        self.is_running = False
    
    def register_collector(
        self,
        collector: BaseMetricCollector,
        name: Optional[str] = None
    ):
        """
        Register a metric collector.
        
        Args:
            collector: Metric collector instance
            name: Optional name for the collector
        """
        name = name or collector.config.name
        
        if name in self.collectors:
            logger.warning(f"Collector {name} already registered")
            return
        
        self.collectors[name] = collector
        logger.info(f"Registered collector: {name}")
        
        # Start collector if manager is running
        if self.is_running and collector.config.enabled:
            collector.start()
    
    def start_all(self):
        """Start all registered collectors."""
        self.is_running = True
        
        for name, collector in self.collectors.items():
            if collector.config.enabled:
                collector.start()
                logger.info(f"Started collector: {name}")
    
    def stop_all(self):
        """Stop all registered collectors."""
        self.is_running = False
        
        for name, collector in self.collectors.items():
            collector.stop()
            logger.info(f"Stopped collector: {name}")
    
    def get_collector_status(self) -> Dict[str, Any]:
        """
        Get status of all collectors.
        
        Returns:
            Dictionary of collector statuses
        """
        status = {}
        
        for name, collector in self.collectors.items():
            status[name] = {
                "enabled": collector.config.enabled,
                "running": collector.is_running,
                "last_collection": collector.last_collection_time,
                "errors": collector.collection_errors,
                "interval": collector.config.collection_interval
            }
        
        return status
    
    def update_collector_config(
        self,
        name: str,
        config_updates: Dict[str, Any]
    ):
        """
        Update collector configuration.
        
        Args:
            name: Collector name
            config_updates: Configuration updates
        """
        if name not in self.collectors:
            logger.warning(f"Collector {name} not found")
            return
        
        collector = self.collectors[name]
        
        # Update configuration
        for key, value in config_updates.items():
            if hasattr(collector.config, key):
                setattr(collector.config, key, value)
        
        # Restart collector if running
        if collector.is_running:
            collector.stop()
            collector.start()
        
        logger.info(f"Updated configuration for collector: {name}")
