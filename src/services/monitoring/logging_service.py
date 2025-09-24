"""
Logging Service Implementation for AG News Text Classification
================================================================================
This module implements centralized logging management with structured logging,
log aggregation, and intelligent log analysis capabilities.

The logging service provides:
- Structured logging with context
- Log aggregation and routing
- Log level management
- Log analysis and pattern detection

References:
    - Chuvakin, A., et al. (2012). Logging and Log Management
    - ELK Stack Documentation

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
from pathlib import Path
import re

from src.services.base_service import BaseService, ServiceConfig
from src.utils.logging_config import get_logger


class LogLevel(Enum):
    """
    Log severity levels.
    
    Levels:
        DEBUG: Detailed diagnostic information
        INFO: General informational messages
        WARNING: Warning messages
        ERROR: Error messages
        CRITICAL: Critical system failures
    """
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LogEntry:
    """
    Structured log entry.
    
    Attributes:
        timestamp: Log timestamp
        level: Log level
        logger_name: Logger name
        message: Log message
        context: Contextual information
        traceback: Exception traceback if applicable
        metadata: Additional metadata
    """
    timestamp: datetime
    level: LogLevel
    logger_name: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    traceback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.name,
            "logger": self.logger_name,
            "message": self.message,
            "context": self.context,
            "traceback": self.traceback,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class LogPattern:
    """
    Log pattern for detection.
    
    Attributes:
        name: Pattern name
        pattern: Regular expression pattern
        severity: Severity when pattern matches
        action: Action to take on match
        threshold: Number of matches before action
    """
    name: str
    pattern: str
    severity: LogLevel = LogLevel.WARNING
    action: Optional[Callable] = None
    threshold: int = 1
    
    def __post_init__(self):
        """Compile pattern after initialization."""
        self.regex = re.compile(self.pattern)
    
    def matches(self, message: str) -> bool:
        """
        Check if message matches pattern.
        
        Args:
            message: Log message
            
        Returns:
            bool: True if matches
        """
        return bool(self.regex.search(message))


class LoggingService(BaseService):
    """
    Centralized logging service for system-wide log management.
    
    This service provides structured logging, aggregation, analysis,
    and intelligent pattern detection for system logs.
    """
    
    def __init__(
        self,
        config: Optional[ServiceConfig] = None,
        log_dir: Optional[Path] = None,
        max_memory_logs: int = 10000,
        rotation_size_mb: int = 100
    ):
        """
        Initialize logging service.
        
        Args:
            config: Service configuration
            log_dir: Directory for log files
            max_memory_logs: Maximum logs to keep in memory
            rotation_size_mb: Log file rotation size
        """
        if config is None:
            config = ServiceConfig(name="logging_service")
        super().__init__(config)
        
        self.log_dir = log_dir or Path("outputs/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_memory_logs = max_memory_logs
        self.rotation_size_mb = rotation_size_mb
        
        # Log storage
        self._log_buffer = deque(maxlen=max_memory_logs)
        self._log_stats = defaultdict(int)
        
        # Log patterns
        self._patterns: List[LogPattern] = []
        self._pattern_matches = defaultdict(int)
        
        # Log handlers
        self._handlers: Dict[str, logging.Handler] = {}
        
        # Background tasks
        self._flush_task: Optional[asyncio.Task] = None
        self._analysis_task: Optional[asyncio.Task] = None
        
        self.logger = get_logger("service.logging")
    
    async def _initialize(self) -> None:
        """Initialize logging service."""
        self.logger.info("Initializing logging service")
        
        # Setup log handlers
        self._setup_handlers()
        
        # Register default patterns
        self._register_default_patterns()
        
        # Start background tasks
        self._flush_task = asyncio.create_task(self._flush_loop())
        self._analysis_task = asyncio.create_task(self._analysis_loop())
    
    async def _start(self) -> None:
        """Start logging service."""
        self.logger.info("Logging service started")
    
    async def _stop(self) -> None:
        """Stop logging service."""
        # Cancel background tasks
        for task in [self._flush_task, self._analysis_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Flush remaining logs
        await self._flush_logs()
        
        # Close handlers
        for handler in self._handlers.values():
            handler.close()
        
        self.logger.info("Logging service stopped")
    
    async def _cleanup(self) -> None:
        """Cleanup logging resources."""
        self._log_buffer.clear()
        self._handlers.clear()
    
    async def _check_health(self) -> bool:
        """Check logging service health."""
        return len(self._log_buffer) < self.max_memory_logs
    
    def _setup_handlers(self) -> None:
        """Setup log handlers."""
        # File handler for all logs
        all_logs_path = self.log_dir / "all.log"
        file_handler = logging.handlers.RotatingFileHandler(
            all_logs_path,
            maxBytes=self.rotation_size_mb * 1024 * 1024,
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        self._handlers["file"] = file_handler
        
        # Error file handler
        error_logs_path = self.log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_logs_path,
            maxBytes=self.rotation_size_mb * 1024 * 1024,
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(exc_info)s'
            )
        )
        self._handlers["error"] = error_handler
        
        # JSON handler for structured logs
        json_logs_path = self.log_dir / "structured.jsonl"
        json_handler = logging.handlers.RotatingFileHandler(
            json_logs_path,
            maxBytes=self.rotation_size_mb * 1024 * 1024,
            backupCount=3
        )
        json_handler.setLevel(logging.INFO)
        self._handlers["json"] = json_handler
    
    def _register_default_patterns(self) -> None:
        """Register default log patterns."""
        # Out of memory pattern
        self.register_pattern(
            LogPattern(
                name="out_of_memory",
                pattern=r"(out of memory|OOM|memory error)",
                severity=LogLevel.CRITICAL,
                threshold=1
            )
        )
        
        # Database connection errors
        self.register_pattern(
            LogPattern(
                name="database_error",
                pattern=r"(connection refused|database error|lost connection)",
                severity=LogLevel.ERROR,
                threshold=3
            )
        )
        
        # Authentication failures
        self.register_pattern(
            LogPattern(
                name="auth_failure",
                pattern=r"(authentication failed|unauthorized|403|401)",
                severity=LogLevel.WARNING,
                threshold=5
            )
        )
        
        # Performance issues
        self.register_pattern(
            LogPattern(
                name="slow_query",
                pattern=r"(slow query|timeout|took \d{4,}ms)",
                severity=LogLevel.WARNING,
                threshold=10
            )
        )
    
    def log(
        self,
        level: LogLevel,
        message: str,
        logger_name: str = "system",
        context: Optional[Dict[str, Any]] = None,
        traceback: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a message.
        
        Args:
            level: Log level
            message: Log message
            logger_name: Logger name
            context: Contextual information
            traceback: Exception traceback
            metadata: Additional metadata
        """
        # Create log entry
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            logger_name=logger_name,
            message=message,
            context=context or {},
            traceback=traceback,
            metadata=metadata or {}
        )
        
        # Add to buffer
        self._log_buffer.append(entry)
        
        # Update statistics
        self._log_stats[level.name] += 1
        self._log_stats["total"] += 1
        
        # Write to handlers
        self._write_to_handlers(entry)
        
        # Check patterns
        self._check_patterns(entry)
    
    def _write_to_handlers(self, entry: LogEntry) -> None:
        """
        Write log entry to handlers.
        
        Args:
            entry: Log entry
        """
        # Create log record
        record = logging.LogRecord(
            name=entry.logger_name,
            level=entry.level.value,
            pathname="",
            lineno=0,
            msg=entry.message,
            args=(),
            exc_info=None
        )
        
        # Write to appropriate handlers
        for handler_name, handler in self._handlers.items():
            if handler_name == "json":
                # Write JSON format
                handler.stream.write(entry.to_json() + "\n")
                handler.stream.flush()
            else:
                # Write standard format
                if record.levelno >= handler.level:
                    handler.emit(record)
    
    def _check_patterns(self, entry: LogEntry) -> None:
        """
        Check log entry against patterns.
        
        Args:
            entry: Log entry
        """
        for pattern in self._patterns:
            if pattern.matches(entry.message):
                self._pattern_matches[pattern.name] += 1
                
                # Check threshold
                if self._pattern_matches[pattern.name] >= pattern.threshold:
                    self.logger.warning(
                        f"Pattern '{pattern.name}' matched {pattern.threshold} times"
                    )
                    
                    # Execute action if defined
                    if pattern.action:
                        try:
                            if asyncio.iscoroutinefunction(pattern.action):
                                asyncio.create_task(pattern.action(entry))
                            else:
                                pattern.action(entry)
                        except Exception as e:
                            self.logger.error(f"Pattern action failed: {e}")
                    
                    # Reset counter
                    self._pattern_matches[pattern.name] = 0
    
    def register_pattern(self, pattern: LogPattern) -> None:
        """
        Register a log pattern.
        
        Args:
            pattern: Log pattern to register
        """
        self._patterns.append(pattern)
        self.logger.info(f"Registered log pattern: {pattern.name}")
    
    async def _flush_loop(self) -> None:
        """Background loop for flushing logs."""
        while True:
            try:
                await asyncio.sleep(60)  # Flush every minute
                await self._flush_logs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Flush loop error: {e}")
    
    async def _flush_logs(self) -> None:
        """Flush logs to persistent storage."""
        # Ensure all handlers are flushed
        for handler in self._handlers.values():
            handler.flush()
        
        self.logger.debug("Flushed logs to storage")
    
    async def _analysis_loop(self) -> None:
        """Background loop for log analysis."""
        while True:
            try:
                await asyncio.sleep(300)  # Analyze every 5 minutes
                await self._analyze_logs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Analysis loop error: {e}")
    
    async def _analyze_logs(self) -> None:
        """Analyze recent logs for patterns and anomalies."""
        # Get recent logs
        recent_logs = list(self._log_buffer)
        
        if not recent_logs:
            return
        
        # Calculate error rate
        error_count = sum(
            1 for log in recent_logs
            if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]
        )
        error_rate = (error_count / len(recent_logs)) * 100
        
        if error_rate > 10:  # More than 10% errors
            self.logger.warning(f"High error rate detected: {error_rate:.2f}%")
        
        # Find most common errors
        error_messages = defaultdict(int)
        for log in recent_logs:
            if log.level == LogLevel.ERROR:
                # Normalize message for grouping
                normalized = re.sub(r'\d+', 'N', log.message)
                error_messages[normalized] += 1
        
        if error_messages:
            most_common = max(error_messages, key=error_messages.get)
            count = error_messages[most_common]
            if count > 10:
                self.logger.warning(
                    f"Frequent error detected ({count} occurrences): {most_common[:100]}"
                )
    
    def get_logs(
        self,
        level: Optional[LogLevel] = None,
        logger_name: Optional[str] = None,
        hours: int = 1,
        limit: int = 100
    ) -> List[LogEntry]:
        """
        Get recent logs.
        
        Args:
            level: Filter by log level
            logger_name: Filter by logger name
            hours: Hours of history
            limit: Maximum number of logs
            
        Returns:
            List[LogEntry]: Filtered logs
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        filtered_logs = []
        for log in reversed(self._log_buffer):
            if log.timestamp < cutoff:
                break
            
            if level and log.level != level:
                continue
            
            if logger_name and log.logger_name != logger_name:
                continue
            
            filtered_logs.append(log)
            
            if len(filtered_logs) >= limit:
                break
        
        return filtered_logs
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get logging statistics.
        
        Returns:
            Dict[str, Any]: Log statistics
        """
        return {
            "total_logs": self._log_stats["total"],
            "by_level": {
                level.name: self._log_stats.get(level.name, 0)
                for level in LogLevel
            },
            "buffer_usage": len(self._log_buffer) / self.max_memory_logs * 100,
            "pattern_matches": dict(self._pattern_matches),
            "log_rate": self._calculate_log_rate()
        }
    
    def _calculate_log_rate(self) -> float:
        """
        Calculate current log rate.
        
        Returns:
            float: Logs per second
        """
        if len(self._log_buffer) < 2:
            return 0.0
        
        # Get time span of logs in buffer
        oldest = self._log_buffer[0].timestamp
        newest = self._log_buffer[-1].timestamp
        time_span = (newest - oldest).total_seconds()
        
        if time_span > 0:
            return len(self._log_buffer) / time_span
        
        return 0.0
    
    def set_log_level(self, logger_name: str, level: LogLevel) -> None:
        """
        Set log level for a logger.
        
        Args:
            logger_name: Logger name
            level: New log level
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(level.value)
        
        self.logger.info(f"Set log level for {logger_name} to {level.name}")
