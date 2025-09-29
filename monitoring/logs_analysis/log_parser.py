"""
Log Parser for Structured and Unstructured Logs
================================================================================
This module implements comprehensive log parsing capabilities for various log formats,
extracting structured information for analysis and monitoring.

The parser supports multiple log formats including JSON, Apache, Nginx,
and custom application logs with configurable parsing rules.

References:
    - Practical Log Analysis (Michael W. Lucas, 2017)
    - Regular Expressions Cookbook (Goyvaerts & Levithan, 2012)

Author: Võ Hải Dũng
License: MIT
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Pattern, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import ast

import dateutil.parser
import yaml

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Standard log levels."""
    
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    FATAL = 60


@dataclass
class LogPattern:
    """Definition of a log pattern."""
    
    name: str
    pattern: str
    fields: List[str]
    time_format: Optional[str] = None
    level_mapping: Optional[Dict[str, LogLevel]] = None
    
    def __post_init__(self):
        """Compile regex pattern."""
        self.regex = re.compile(self.pattern)


@dataclass
class ParsedLog:
    """Parsed log entry."""
    
    timestamp: datetime
    level: LogLevel
    message: str
    
    # Optional fields
    service: Optional[str] = None
    module: Optional[str] = None
    thread: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Extracted fields
    fields: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    raw_log: Optional[str] = None
    parse_errors: List[str] = field(default_factory=list)


class LogParser:
    """
    Comprehensive log parser for multiple formats.
    
    This class provides:
    - Multi-format log parsing
    - Custom pattern support
    - Field extraction and normalization
    - Error recovery and partial parsing
    """
    
    def __init__(self):
        """Initialize log parser."""
        self.patterns = {}
        self.parsers = {}
        
        # Register default patterns
        self._register_default_patterns()
        
        # Register format parsers
        self._register_format_parsers()
    
    def _register_default_patterns(self):
        """Register default log patterns."""
        
        # Python logging format
        self.register_pattern(LogPattern(
            name="python",
            pattern=r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (?P<module>\S+) - (?P<level>\w+) - (?P<message>.*)",
            fields=["timestamp", "module", "level", "message"],
            time_format="%Y-%m-%d %H:%M:%S,%f",
            level_mapping={
                "DEBUG": LogLevel.DEBUG,
                "INFO": LogLevel.INFO,
                "WARNING": LogLevel.WARNING,
                "ERROR": LogLevel.ERROR,
                "CRITICAL": LogLevel.CRITICAL
            }
        ))
        
        # JSON structured logs
        self.register_pattern(LogPattern(
            name="json",
            pattern=r"^(?P<json>\{.*\})$",
            fields=["json"]
        ))
        
        # Apache combined log format
        self.register_pattern(LogPattern(
            name="apache_combined",
            pattern=r'(?P<ip>\S+) \S+ (?P<user>\S+) ```math
(?P<timestamp>[^```]+)``` "(?P<method>\w+) (?P<path>[^ ]+) (?P<protocol>[^"]+)" (?P<status>\d+) (?P<size>\S+) "(?P<referer>[^"]*)" "(?P<user_agent>[^"]*)"',
            fields=["ip", "user", "timestamp", "method", "path", "protocol", "status", "size", "referer", "user_agent"],
            time_format="%d/%b/%Y:%H:%M:%S %z"
        ))
        
        # Nginx error log
        self.register_pattern(LogPattern(
            name="nginx_error",
            pattern=r"(?P<timestamp>\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}) ```math
(?P<level>\w+)``` (?P<pid>\d+)#(?P<tid>\d+): (?P<message>.*)",
            fields=["timestamp", "level", "pid", "tid", "message"],
            time_format="%Y/%m/%d %H:%M:%S",
            level_mapping={
                "debug": LogLevel.DEBUG,
                "info": LogLevel.INFO,
                "notice": LogLevel.INFO,
                "warn": LogLevel.WARNING,
                "error": LogLevel.ERROR,
                "crit": LogLevel.CRITICAL,
                "alert": LogLevel.CRITICAL,
                "emerg": LogLevel.FATAL
            }
        ))
        
        # Docker container logs
        self.register_pattern(LogPattern(
            name="docker",
            pattern=r"(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z) (?P<stream>\w+) (?P<partial>\w) (?P<message>.*)",
            fields=["timestamp", "stream", "partial", "message"],
            time_format="%Y-%m-%dT%H:%M:%S.%fZ"
        ))
        
        # Kubernetes pod logs
        self.register_pattern(LogPattern(
            name="kubernetes",
            pattern=r"(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z) (?P<namespace>\S+) (?P<pod>\S+) (?P<container>\S+) (?P<message>.*)",
            fields=["timestamp", "namespace", "pod", "container", "message"]
        ))
        
        # Application specific patterns
        self.register_pattern(LogPattern(
            name="ml_training",
            pattern=r"```math
(?P<timestamp>[^```]+)``` ```math
(?P<epoch>\d+)/(?P<total_epochs>\d+)``` ```math
(?P<batch>\d+)/(?P<total_batches>\d+)``` Loss: (?P<loss>[\d.]+), Accuracy: (?P<accuracy>[\d.]+)",
            fields=["timestamp", "epoch", "total_epochs", "batch", "total_batches", "loss", "accuracy"]
        ))
        
        self.register_pattern(LogPattern(
            name="api_request",
            pattern=r"(?P<timestamp>[^```]+) ```math
(?P<request_id>[^```]+)``` (?P<method>\w+) (?P<endpoint>\S+) (?P<status>\d+) (?P<duration>[\d.]+)ms",
            fields=["timestamp", "request_id", "method", "endpoint", "status", "duration"]
        ))
    
    def _register_format_parsers(self):
        """Register format-specific parsers."""
        self.parsers = {
            "json": self._parse_json_log,
            "yaml": self._parse_yaml_log,
            "key_value": self._parse_key_value_log,
            "csv": self._parse_csv_log
        }
    
    def register_pattern(self, pattern: LogPattern):
        """
        Register a custom log pattern.
        
        Args:
            pattern: Log pattern definition
        """
        self.patterns[pattern.name] = pattern
        logger.debug(f"Registered pattern: {pattern.name}")
    
    def parse(
        self,
        log_line: str,
        pattern_name: Optional[str] = None,
        auto_detect: bool = True
    ) -> Optional[ParsedLog]:
        """
        Parse a log line.
        
        Args:
            log_line: Raw log line
            pattern_name: Specific pattern to use
            auto_detect: Auto-detect pattern if not specified
            
        Returns:
            Parsed log entry or None if parsing fails
        """
        if not log_line or not log_line.strip():
            return None
        
        log_line = log_line.strip()
        
        # Try specific pattern if provided
        if pattern_name:
            if pattern_name in self.patterns:
                return self._parse_with_pattern(log_line, self.patterns[pattern_name])
            elif pattern_name in self.parsers:
                return self.parsers[pattern_name](log_line)
            else:
                logger.warning(f"Unknown pattern: {pattern_name}")
        
        # Auto-detect pattern
        if auto_detect:
            # Try JSON first (most common structured format)
            if log_line.startswith("{"):
                parsed = self._parse_json_log(log_line)
                if parsed:
                    return parsed
            
            # Try registered patterns
            for pattern in self.patterns.values():
                parsed = self._parse_with_pattern(log_line, pattern)
                if parsed:
                    return parsed
            
            # Try other format parsers
            for parser in self.parsers.values():
                try:
                    parsed = parser(log_line)
                    if parsed:
                        return parsed
                except:
                    continue
        
        # Fallback to raw message
        return self._create_raw_log(log_line)
    
    def _parse_with_pattern(
        self,
        log_line: str,
        pattern: LogPattern
    ) -> Optional[ParsedLog]:
        """
        Parse log with specific pattern.
        
        Args:
            log_line: Raw log line
            pattern: Pattern to use
            
        Returns:
            Parsed log or None
        """
        match = pattern.regex.match(log_line)
        if not match:
            return None
        
        fields = match.groupdict()
        
        # Parse timestamp
        timestamp = self._parse_timestamp(
            fields.get("timestamp"),
            pattern.time_format
        )
        
        # Determine log level
        level = self._parse_level(
            fields.get("level"),
            pattern.level_mapping
        )
        
        # Extract message
        message = fields.get("message", "")
        
        # Handle special fields
        if "json" in fields:
            # Parse embedded JSON
            try:
                json_data = json.loads(fields["json"])
                fields.update(json_data)
                
                # Update from JSON fields
                timestamp = timestamp or self._parse_timestamp(json_data.get("timestamp"))
                level = level or self._parse_level(json_data.get("level"))
                message = message or json_data.get("message", "")
            except json.JSONDecodeError:
                pass
        
        # Create parsed log
        parsed = ParsedLog(
            timestamp=timestamp or datetime.now(),
            level=level or LogLevel.INFO,
            message=message,
            raw_log=log_line,
            fields=fields
        )
        
        # Extract additional fields
        parsed.service = fields.get("service")
        parsed.module = fields.get("module")
        parsed.thread = fields.get("thread")
        parsed.trace_id = fields.get("trace_id") or fields.get("traceId")
        parsed.span_id = fields.get("span_id") or fields.get("spanId")
        parsed.user_id = fields.get("user_id") or fields.get("userId")
        
        return parsed
    
    def _parse_json_log(self, log_line: str) -> Optional[ParsedLog]:
        """Parse JSON formatted log."""
        try:
            data = json.loads(log_line)
            
            return ParsedLog(
                timestamp=self._parse_timestamp(
                    data.get("timestamp") or data.get("time") or data.get("@timestamp")
                ),
                level=self._parse_level(
                    data.get("level") or data.get("severity") or data.get("log.level")
                ),
                message=data.get("message") or data.get("msg") or "",
                service=data.get("service") or data.get("app"),
                module=data.get("module") or data.get("logger"),
                thread=data.get("thread") or data.get("thread_name"),
                trace_id=data.get("trace_id") or data.get("traceId"),
                span_id=data.get("span_id") or data.get("spanId"),
                user_id=data.get("user_id") or data.get("userId"),
                fields=data,
                raw_log=log_line
            )
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse JSON log: {e}")
            return None
    
    def _parse_yaml_log(self, log_line: str) -> Optional[ParsedLog]:
        """Parse YAML formatted log."""
        try:
            data = yaml.safe_load(log_line)
            if not isinstance(data, dict):
                return None
            
            return ParsedLog(
                timestamp=self._parse_timestamp(data.get("timestamp")),
                level=self._parse_level(data.get("level")),
                message=data.get("message", ""),
                fields=data,
                raw_log=log_line
            )
        except yaml.YAMLError:
            return None
    
    def _parse_key_value_log(self, log_line: str) -> Optional[ParsedLog]:
        """Parse key=value formatted log."""
        pattern = re.compile(r'(\w+)=([^\s]+|"[^"]*")')
        matches = pattern.findall(log_line)
        
        if not matches:
            return None
        
        fields = {}
        for key, value in matches:
            # Remove quotes if present
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            fields[key] = value
        
        return ParsedLog(
            timestamp=self._parse_timestamp(fields.get("timestamp")),
            level=self._parse_level(fields.get("level")),
            message=fields.get("message", ""),
            fields=fields,
            raw_log=log_line
        )
    
    def _parse_csv_log(self, log_line: str) -> Optional[ParsedLog]:
        """Parse CSV formatted log."""
        # Simple CSV parsing (could be enhanced with csv module)
        parts = log_line.split(",")
        if len(parts) < 3:
            return None
        
        return ParsedLog(
            timestamp=self._parse_timestamp(parts[0]),
            level=self._parse_level(parts[1] if len(parts) > 1 else None),
            message=parts[2] if len(parts) > 2 else "",
            fields={"csv_fields": parts},
            raw_log=log_line
        )
    
    def _parse_timestamp(
        self,
        timestamp_str: Optional[Union[str, int, float]],
        format_str: Optional[str] = None
    ) -> Optional[datetime]:
        """Parse timestamp from various formats."""
        if not timestamp_str:
            return None
        
        # Handle numeric timestamps
        if isinstance(timestamp_str, (int, float)):
            return datetime.fromtimestamp(timestamp_str)
        
        # Try specific format first
        if format_str:
            try:
                return datetime.strptime(timestamp_str, format_str)
            except ValueError:
                pass
        
        # Try dateutil parser (handles many formats)
        try:
            return dateutil.parser.parse(timestamp_str)
        except (ValueError, TypeError):
            pass
        
        return None
    
    def _parse_level(
        self,
        level_str: Optional[str],
        level_mapping: Optional[Dict[str, LogLevel]] = None
    ) -> Optional[LogLevel]:
        """Parse log level from string."""
        if not level_str:
            return None
        
        level_str = level_str.upper()
        
        # Use custom mapping if provided
        if level_mapping:
            for key, value in level_mapping.items():
                if key.upper() == level_str:
                    return value
        
        # Try standard level names
        try:
            return LogLevel[level_str]
        except KeyError:
            pass
        
        # Try numeric level
        try:
            level_num = int(level_str)
            for level in LogLevel:
                if level.value == level_num:
                    return level
        except ValueError:
            pass
        
        return None
    
    def _create_raw_log(self, log_line: str) -> ParsedLog:
        """Create a parsed log entry for unparseable logs."""
        return ParsedLog(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message=log_line,
            raw_log=log_line,
            parse_errors=["Could not parse log format"]
        )
    
    def parse_batch(
        self,
        log_lines: List[str],
        pattern_name: Optional[str] = None
    ) -> List[ParsedLog]:
        """
        Parse multiple log lines.
        
        Args:
            log_lines: List of raw log lines
            pattern_name: Pattern to use for all lines
            
        Returns:
            List of parsed logs
        """
        parsed_logs = []
        
        for line in log_lines:
            parsed = self.parse(line, pattern_name)
            if parsed:
                parsed_logs.append(parsed)
        
        return parsed_logs
    
    def extract_metrics(self, parsed_log: ParsedLog) -> Dict[str, Any]:
        """
        Extract metrics from parsed log.
        
        Args:
            parsed_log: Parsed log entry
            
        Returns:
            Dictionary of extracted metrics
        """
        metrics = {}
        
        # Extract numeric fields
        for key, value in parsed_log.fields.items():
            if isinstance(value, (int, float)):
                metrics[key] = value
            elif isinstance(value, str):
                # Try to parse numeric values
                try:
                    if "." in value:
                        metrics[key] = float(value)
                    else:
                        metrics[key] = int(value)
                except ValueError:
                    pass
        
        # Extract specific metrics from message
        if "duration" in parsed_log.message.lower():
            duration_match = re.search(r"duration[:\s]+(\d+(?:\.\d+)?)", parsed_log.message, re.IGNORECASE)
            if duration_match:
                metrics["duration"] = float(duration_match.group(1))
        
        if "latency" in parsed_log.message.lower():
            latency_match = re.search(r"latency[:\s]+(\d+(?:\.\d+)?)", parsed_log.message, re.IGNORECASE)
            if latency_match:
                metrics["latency"] = float(latency_match.group(1))
        
        return metrics
