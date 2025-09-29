"""
Log Aggregation and Analysis System
================================================================================
This module implements log aggregation functionality for collecting, processing,
and analyzing logs from multiple sources with support for real-time streaming
and batch processing.

The aggregator integrates with the log parser and anomaly detector to provide
comprehensive log analysis capabilities.

References:
    - Distributed Systems Observability (Cindy Sridharan, 2018)
    - The Art of Monitoring (James Turnbull, 2016)

Author: Võ Hải Dũng
License: MIT
"""

import time
import logging
import re
from typing import Dict, Any, List, Optional, Callable, Generator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque, Counter
import threading
import queue
import asyncio
import hashlib

import numpy as np
import pandas as pd

from monitoring.logs_analysis.log_parser import LogParser, ParsedLog, LogLevel, LogPattern
from monitoring.logs_analysis.anomaly_detector import AnomalyDetector, Anomaly

logger = logging.getLogger(__name__)


@dataclass
class LogSource:
    """Configuration for a log source."""
    
    name: str
    source_type: str  # file, stream, api, database, socket
    location: str  # file path, URL, connection string
    
    # Processing settings
    enabled: bool = True
    batch_size: int = 100
    poll_interval: int = 5  # seconds
    
    # Parsing
    parser_pattern: Optional[str] = None  # Specific pattern to use
    auto_detect_format: bool = True
    
    # Filtering
    filters: Dict[str, Any] = field(default_factory=dict)
    
    # Additional configuration
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregationResult:
    """Result of log aggregation."""
    
    timestamp: datetime
    time_window: timedelta
    total_logs: int
    
    # Aggregated metrics by level
    logs_by_level: Dict[str, int]
    
    # Aggregated metrics by service
    logs_by_service: Dict[str, int]
    
    # Aggregated metrics by module
    logs_by_module: Dict[str, int]
    
    # Error analysis
    error_rate: float
    critical_count: int
    warning_count: int
    
    # Top patterns
    top_messages: List[Tuple[str, int]]
    top_errors: List[Tuple[str, int]]
    top_services_with_errors: List[Tuple[str, int]]
    
    # Trace analysis
    unique_trace_ids: int
    incomplete_traces: List[str]
    
    # Anomalies
    anomalies: List[Anomaly]
    anomaly_rate: float
    
    # Performance metrics
    avg_message_length: float
    unique_services: int
    unique_modules: int
    unique_users: int
    
    # Extracted metrics
    extracted_metrics: Dict[str, Any]


class LogAggregator:
    """
    Central log aggregation and analysis system.
    
    This class provides:
    - Multi-source log collection
    - Real-time and batch processing
    - Aggregation and summarization
    - Integration with anomaly detection
    - Streaming and historical analysis
    - Trace correlation
    """
    
    def __init__(
        self,
        buffer_size: int = 10000,
        aggregation_interval: int = 60,  # seconds
        enable_anomaly_detection: bool = True
    ):
        """
        Initialize log aggregator.
        
        Args:
            buffer_size: Size of log buffer
            aggregation_interval: Interval for aggregation
            enable_anomaly_detection: Enable anomaly detection
        """
        self.buffer_size = buffer_size
        self.aggregation_interval = aggregation_interval
        self.enable_anomaly_detection = enable_anomaly_detection
        
        # Components
        self.parser = LogParser()
        self.anomaly_detector = AnomalyDetector() if enable_anomaly_detection else None
        
        # Log sources
        self.sources = {}
        self.source_threads = {}
        
        # Log buffer
        self.log_buffer = deque(maxlen=buffer_size)
        self.log_queue = queue.Queue()
        
        # Trace tracking
        self.active_traces = defaultdict(list)
        self.completed_traces = deque(maxlen=1000)
        
        # Aggregation state
        self.aggregation_cache = {}
        self.last_aggregation = datetime.now()
        
        # Metrics extraction
        self.metrics_buffer = deque(maxlen=10000)
        
        # Processing state
        self.is_running = False
        self.processing_thread = None
        self.aggregation_thread = None
        
        # Statistics
        self.stats = {
            "total_logs_processed": 0,
            "total_parse_errors": 0,
            "total_anomalies_detected": 0,
            "start_time": datetime.now()
        }
        
        # Callbacks
        self.log_callbacks = []
        self.anomaly_callbacks = []
        self.aggregation_callbacks = []
        self.trace_callbacks = []
    
    def add_source(self, source: LogSource):
        """
        Add a log source.
        
        Args:
            source: Log source configuration
        """
        if source.name in self.sources:
            logger.warning(f"Source {source.name} already exists")
            return
        
        self.sources[source.name] = source
        logger.info(f"Added log source: {source.name} (type: {source.source_type})")
        
        # Start source thread if aggregator is running
        if self.is_running and source.enabled:
            self._start_source_thread(source)
    
    def remove_source(self, source_name: str):
        """
        Remove a log source.
        
        Args:
            source_name: Name of the source to remove
        """
        if source_name in self.sources:
            # Stop source thread
            if source_name in self.source_threads:
                self.sources[source_name].enabled = False
                thread = self.source_threads[source_name]
                thread.join(timeout=5)
                del self.source_threads[source_name]
            
            del self.sources[source_name]
            logger.info(f"Removed log source: {source_name}")
    
    def start(self):
        """Start log aggregation."""
        if self.is_running:
            logger.warning("Aggregator already running")
            return
        
        self.is_running = True
        self.stats["start_time"] = datetime.now()
        
        # Start source threads
        for source in self.sources.values():
            if source.enabled:
                self._start_source_thread(source)
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        
        # Start aggregation thread
        self.aggregation_thread = threading.Thread(
            target=self._aggregation_loop,
            daemon=True
        )
        self.aggregation_thread.start()
        
        logger.info("Log aggregator started")
    
    def stop(self):
        """Stop log aggregation."""
        self.is_running = False
        
        # Stop source threads
        for source_name in list(self.source_threads.keys()):
            self.sources[source_name].enabled = False
        
        # Wait for threads to finish
        for thread in self.source_threads.values():
            thread.join(timeout=5)
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        if self.aggregation_thread:
            self.aggregation_thread.join(timeout=5)
        
        logger.info(f"Log aggregator stopped. Total logs processed: {self.stats['total_logs_processed']}")
    
    def _start_source_thread(self, source: LogSource):
        """Start thread for a log source."""
        thread = threading.Thread(
            target=self._source_reader,
            args=(source,),
            daemon=True
        )
        thread.start()
        self.source_threads[source.name] = thread
    
    def _source_reader(self, source: LogSource):
        """Read logs from a source."""
        logger.info(f"Started reader for source: {source.name}")
        
        while self.is_running and source.enabled:
            try:
                # Read logs based on source type
                if source.source_type == "file":
                    logs = self._read_file_logs(source)
                elif source.source_type == "stream":
                    logs = self._read_stream_logs(source)
                elif source.source_type == "api":
                    logs = self._read_api_logs(source)
                elif source.source_type == "socket":
                    logs = self._read_socket_logs(source)
                else:
                    logger.warning(f"Unknown source type: {source.source_type}")
                    logs = []
                
                # Add logs to queue with source information
                for log in logs:
                    self.log_queue.put({
                        "source": source.name,
                        "raw_log": log,
                        "pattern": source.parser_pattern,
                        "auto_detect": source.auto_detect_format
                    })
                
                # Sleep before next poll
                time.sleep(source.poll_interval)
                
            except Exception as e:
                logger.error(f"Error reading from source {source.name}: {e}")
                time.sleep(10)  # Wait before retry
    
    def _read_file_logs(self, source: LogSource) -> List[str]:
        """Read logs from file."""
        logs = []
        
        try:
            # Track file position
            position_key = f"file_position_{source.name}"
            position = self.aggregation_cache.get(position_key, 0)
            
            with open(source.location, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(position)
                
                # Read new lines
                for _ in range(source.batch_size):
                    line = f.readline()
                    if not line:
                        break
                    logs.append(line.strip())
                
                # Update position
                self.aggregation_cache[position_key] = f.tell()
        
        except Exception as e:
            logger.error(f"Error reading file {source.location}: {e}")
        
        return logs
    
    def _read_stream_logs(self, source: LogSource) -> List[str]:
        """Read logs from stream."""
        logs = []
        
        # Implementation would depend on stream type
        # Could be stdin, network stream, etc.
        
        return logs
    
    def _read_api_logs(self, source: LogSource) -> List[str]:
        """Read logs from API."""
        logs = []
        
        # Implementation would make HTTP requests to fetch logs
        # Could use requests library or async HTTP client
        
        return logs
    
    def _read_socket_logs(self, source: LogSource) -> List[str]:
        """Read logs from socket."""
        logs = []
        
        # Implementation would connect to socket and read data
        # Could be TCP/UDP socket
        
        return logs
    
    def _processing_loop(self):
        """Main processing loop."""
        while self.is_running:
            try:
                # Get log from queue
                log_data = self.log_queue.get(timeout=1)
                
                # Parse log
                parsed_log = self.parser.parse(
                    log_data["raw_log"],
                    pattern_name=log_data.get("pattern"),
                    auto_detect=log_data.get("auto_detect", True)
                )
                
                if parsed_log:
                    # Add source information
                    parsed_log.fields["_source"] = log_data["source"]
                    
                    # Apply filters
                    source = self.sources.get(log_data["source"])
                    if source and self._apply_filters(parsed_log, source.filters):
                        continue
                    
                    # Add to buffer
                    self.log_buffer.append(parsed_log)
                    self.stats["total_logs_processed"] += 1
                    
                    # Extract metrics
                    metrics = self.parser.extract_metrics(parsed_log)
                    if metrics:
                        self.metrics_buffer.append({
                            "timestamp": parsed_log.timestamp,
                            "service": parsed_log.service,
                            "metrics": metrics
                        })
                    
                    # Track traces
                    if parsed_log.trace_id:
                        self._track_trace(parsed_log)
                    
                    # Process callbacks
                    for callback in self.log_callbacks:
                        try:
                            callback(parsed_log)
                        except Exception as e:
                            logger.error(f"Error in log callback: {e}")
                    
                    # Check for anomalies
                    if self.anomaly_detector:
                        anomalies = self.anomaly_detector.process_log(parsed_log)
                        
                        if anomalies:
                            self.stats["total_anomalies_detected"] += len(anomalies)
                            
                            for anomaly in anomalies:
                                for callback in self.anomaly_callbacks:
                                    try:
                                        callback(anomaly)
                                    except Exception as e:
                                        logger.error(f"Error in anomaly callback: {e}")
                else:
                    self.stats["total_parse_errors"] += 1
                    if parsed_log and parsed_log.parse_errors:
                        logger.debug(f"Parse errors: {parsed_log.parse_errors}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
    
    def _track_trace(self, log: ParsedLog):
        """Track distributed trace."""
        trace_id = log.trace_id
        
        # Add to active traces
        self.active_traces[trace_id].append(log)
        
        # Check if trace is complete (simple heuristic)
        trace_logs = self.active_traces[trace_id]
        
        # If we see an end marker or timeout
        if self._is_trace_complete(trace_logs):
            # Move to completed traces
            self.completed_traces.append({
                "trace_id": trace_id,
                "logs": trace_logs,
                "duration": (trace_logs[-1].timestamp - trace_logs[0].timestamp).total_seconds(),
                "span_count": len(set(log.span_id for log in trace_logs if log.span_id))
            })
            
            # Process trace callbacks
            for callback in self.trace_callbacks:
                try:
                    callback(trace_id, trace_logs)
                except Exception as e:
                    logger.error(f"Error in trace callback: {e}")
            
            # Clean up active trace
            del self.active_traces[trace_id]
    
    def _is_trace_complete(self, trace_logs: List[ParsedLog]) -> bool:
        """Check if a trace is complete."""
        if not trace_logs:
            return False
        
        # Check for completion markers in messages
        last_log = trace_logs[-1]
        completion_markers = ["completed", "finished", "done", "end"]
        
        for marker in completion_markers:
            if marker in last_log.message.lower():
                return True
        
        # Check for timeout (trace older than 5 minutes)
        if (datetime.now() - trace_logs[0].timestamp).total_seconds() > 300:
            return True
        
        return False
    
    def _aggregation_loop(self):
        """Aggregation loop."""
        while self.is_running:
            try:
                # Wait for aggregation interval
                time.sleep(self.aggregation_interval)
                
                # Perform aggregation
                result = self.aggregate_logs()
                
                # Process callbacks
                for callback in self.aggregation_callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"Error in aggregation callback: {e}")
                
                # Update baselines in anomaly detector
                if self.anomaly_detector:
                    recent_logs = list(self.log_buffer)[-1000:]
                    if recent_logs:
                        self.anomaly_detector.update_baselines(recent_logs)
                
            except Exception as e:
                logger.error(f"Error in aggregation loop: {e}")
    
    def _apply_filters(self, log: ParsedLog, filters: Dict[str, Any]) -> bool:
        """
        Apply filters to log.
        
        Returns:
            True if log should be filtered out
        """
        if not filters:
            return False
        
        # Level filter
        if "min_level" in filters:
            min_level = LogLevel[filters["min_level"]] if isinstance(filters["min_level"], str) else filters["min_level"]
            if log.level.value < min_level.value:
                return True
        
        if "max_level" in filters:
            max_level = LogLevel[filters["max_level"]] if isinstance(filters["max_level"], str) else filters["max_level"]
            if log.level.value > max_level.value:
                return True
        
        # Service filter
        if "services" in filters:
            allowed_services = filters["services"]
            if log.service and log.service not in allowed_services:
                return True
        
        # Module filter
        if "modules" in filters:
            allowed_modules = filters["modules"]
            if log.module and log.module not in allowed_modules:
                return True
        
        # Pattern filter
        if "exclude_patterns" in filters:
            for pattern in filters["exclude_patterns"]:
                if re.search(pattern, log.message):
                    return True
        
        if "include_patterns" in filters:
            match_found = False
            for pattern in filters["include_patterns"]:
                if re.search(pattern, log.message):
                    match_found = True
                    break
            if not match_found:
                return True
        
        return False
    
    def aggregate_logs(
        self,
        time_window: Optional[timedelta] = None
    ) -> AggregationResult:
        """
        Aggregate logs in the buffer.
        
        Args:
            time_window: Time window for aggregation
            
        Returns:
            Aggregation result
        """
        if time_window is None:
            time_window = timedelta(seconds=self.aggregation_interval)
        
        current_time = datetime.now()
        cutoff_time = current_time - time_window
        
        # Filter logs within time window
        recent_logs = [
            log for log in self.log_buffer
            if log.timestamp > cutoff_time
        ]
        
        if not recent_logs:
            return self._empty_aggregation_result(current_time, time_window)
        
        # Aggregate by level
        logs_by_level = defaultdict(int)
        for log in recent_logs:
            logs_by_level[log.level.name] += 1
        
        # Aggregate by service
        logs_by_service = defaultdict(int)
        for log in recent_logs:
            service = log.service or "unknown"
            logs_by_service[service] += 1
        
        # Aggregate by module
        logs_by_module = defaultdict(int)
        for log in recent_logs:
            if log.module:
                logs_by_module[log.module] += 1
        
        # Calculate error metrics
        error_count = sum(
            1 for log in recent_logs
            if log.level.value >= LogLevel.ERROR.value
        )
        critical_count = sum(
            1 for log in recent_logs
            if log.level.value >= LogLevel.CRITICAL.value
        )
        warning_count = sum(
            1 for log in recent_logs
            if log.level == LogLevel.WARNING
        )
        error_rate = error_count / len(recent_logs) if recent_logs else 0
        
        # Top messages
        message_counts = defaultdict(int)
        error_counts = defaultdict(int)
        service_error_counts = defaultdict(int)
        
        for log in recent_logs:
            # Normalize message for grouping
            normalized = self._normalize_message(log.message)
            message_counts[normalized] += 1
            
            if log.level.value >= LogLevel.ERROR.value:
                error_counts[normalized] += 1
                if log.service:
                    service_error_counts[log.service] += 1
        
        top_messages = sorted(
            message_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        top_errors = sorted(
            error_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        top_services_with_errors = sorted(
            service_error_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Trace analysis
        unique_trace_ids = len(set(log.trace_id for log in recent_logs if log.trace_id))
        incomplete_traces = list(self.active_traces.keys())[:10]
        
        # Get recent anomalies
        recent_anomalies = []
        anomaly_rate = 0
        
        if self.anomaly_detector:
            recent_anomalies = [
                a for a in self.anomaly_detector.anomalies
                if a.timestamp > cutoff_time
            ]
            anomaly_rate = len(recent_anomalies) / len(recent_logs) if recent_logs else 0
        
        # Calculate statistics
        avg_message_length = np.mean([len(log.message) for log in recent_logs])
        unique_services = len(set(log.service for log in recent_logs if log.service))
        unique_modules = len(set(log.module for log in recent_logs if log.module))
        unique_users = len(set(log.user_id for log in recent_logs if log.user_id))
        
        # Extract aggregated metrics
        extracted_metrics = self._aggregate_extracted_metrics(cutoff_time)
        
        return AggregationResult(
            timestamp=current_time,
            time_window=time_window,
            total_logs=len(recent_logs),
            logs_by_level=dict(logs_by_level),
            logs_by_service=dict(logs_by_service),
            logs_by_module=dict(logs_by_module),
            error_rate=error_rate,
            critical_count=critical_count,
            warning_count=warning_count,
            top_messages=top_messages,
            top_errors=top_errors,
            top_services_with_errors=top_services_with_errors,
            unique_trace_ids=unique_trace_ids,
            incomplete_traces=incomplete_traces,
            anomalies=recent_anomalies,
            anomaly_rate=anomaly_rate,
            avg_message_length=avg_message_length,
            unique_services=unique_services,
            unique_modules=unique_modules,
            unique_users=unique_users,
            extracted_metrics=extracted_metrics
        )
    
    def _empty_aggregation_result(
        self,
        timestamp: datetime,
        time_window: timedelta
    ) -> AggregationResult:
        """Create empty aggregation result."""
        return AggregationResult(
            timestamp=timestamp,
            time_window=time_window,
            total_logs=0,
            logs_by_level={},
            logs_by_service={},
            logs_by_module={},
            error_rate=0.0,
            critical_count=0,
            warning_count=0,
            top_messages=[],
            top_errors=[],
            top_services_with_errors=[],
            unique_trace_ids=0,
            incomplete_traces=[],
            anomalies=[],
            anomaly_rate=0.0,
            avg_message_length=0.0,
            unique_services=0,
            unique_modules=0,
            unique_users=0,
            extracted_metrics={}
        )
    
    def _normalize_message(self, message: str) -> str:
        """Normalize log message for grouping."""
        # Remove numbers
        normalized = re.sub(r'\d+', 'N', message)
        
        # Remove UUIDs
        normalized = re.sub(
            r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}',
            'UUID',
            normalized,
            flags=re.IGNORECASE
        )
        
        # Remove IPs
        normalized = re.sub(
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
            'IP',
            normalized
        )
        
        # Remove file paths
        normalized = re.sub(
            r'(/[^/\s]+)+',
            'PATH',
            normalized
        )
        
        # Remove URLs
        normalized = re.sub(
            r'https?://[^\s]+',
            'URL',
            normalized
        )
        
        # Remove email addresses
        normalized = re.sub(
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'EMAIL',
            normalized
        )
        
        # Truncate
        return normalized[:200]
    
    def _aggregate_extracted_metrics(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Aggregate extracted metrics."""
        aggregated = defaultdict(list)
        
        for metric_data in self.metrics_buffer:
            if metric_data["timestamp"] > cutoff_time:
                for key, value in metric_data["metrics"].items():
                    aggregated[key].append(value)
        
        # Calculate statistics for each metric
        result = {}
        for key, values in aggregated.items():
            if values:
                result[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "p50": np.percentile(values, 50),
                    "p95": np.percentile(values, 95),
                    "p99": np.percentile(values, 99),
                    "count": len(values)
                }
        
        return result
    
    def register_log_callback(self, callback: Callable[[ParsedLog], None]):
        """Register callback for processed logs."""
        self.log_callbacks.append(callback)
    
    def register_anomaly_callback(self, callback: Callable[[Anomaly], None]):
        """Register callback for detected anomalies."""
        self.anomaly_callbacks.append(callback)
    
    def register_aggregation_callback(self, callback: Callable[[AggregationResult], None]):
        """Register callback for aggregation results."""
        self.aggregation_callbacks.append(callback)
    
    def register_trace_callback(self, callback: Callable[[str, List[ParsedLog]], None]):
        """Register callback for completed traces."""
        self.trace_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        uptime = (datetime.now() - self.stats["start_time"]).total_seconds()
        
        return {
            "uptime_seconds": uptime,
            "total_logs_processed": self.stats["total_logs_processed"],
            "total_parse_errors": self.stats["total_parse_errors"],
            "total_anomalies_detected": self.stats["total_anomalies_detected"],
            "logs_per_second": self.stats["total_logs_processed"] / uptime if uptime > 0 else 0,
            "buffer_size": len(self.log_buffer),
            "queue_size": self.log_queue.qsize(),
            "active_sources": sum(1 for s in self.sources.values() if s.enabled),
            "active_traces": len(self.active_traces),
            "completed_traces": len(self.completed_traces),
            "metrics_buffer_size": len(self.metrics_buffer)
        }
    
    def get_recent_logs(
        self,
        limit: int = 100,
        level: Optional[LogLevel] = None,
        service: Optional[str] = None
    ) -> List[ParsedLog]:
        """
        Get recent logs with optional filtering.
        
        Args:
            limit: Maximum number of logs to return
            level: Filter by log level
            service: Filter by service name
            
        Returns:
            List of recent logs
        """
        logs = list(self.log_buffer)
        
        # Apply filters
        if level:
            logs = [log for log in logs if log.level == level]
        
        if service:
            logs = [log for log in logs if log.service == service]
        
        # Return most recent
        return logs[-limit:]
    
    def search_logs(
        self,
        pattern: str,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        limit: int = 100
    ) -> List[ParsedLog]:
        """
        Search logs by pattern.
        
        Args:
            pattern: Regex pattern to search
            time_range: Optional time range filter
            limit: Maximum number of results
            
        Returns:
            List of matching logs
        """
        regex = re.compile(pattern, re.IGNORECASE)
        matching_logs = []
        
        for log in self.log_buffer:
            # Check time range
            if time_range:
                if log.timestamp < time_range[0] or log.timestamp > time_range[1]:
                    continue
            
            # Check pattern
            if regex.search(log.message):
                matching_logs.append(log)
                
                if len(matching_logs) >= limit:
                    break
        
        return matching_logs
