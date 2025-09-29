"""
Log Anomaly Detection System
================================================================================
This module implements anomaly detection algorithms for log analysis,
identifying unusual patterns, errors, and security threats in log streams.

The implementation combines statistical methods, machine learning models,
and pattern matching for comprehensive anomaly detection.

References:
    - Anomaly Detection Principles and Algorithms (Mehrotra et al., 2017)
    - Machine Learning for Security (Chio & Freeman, 2018)

Author: Võ Hải Dũng
License: MIT
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Set, Pattern
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter, deque
from enum import Enum
import pickle
import re
import json
import hashlib

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from monitoring.logs_analysis.log_parser import ParsedLog, LogLevel

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of log anomalies."""
    
    FREQUENCY = "frequency"
    PATTERN = "pattern"
    SEQUENCE = "sequence"
    VALUE = "value"
    CORRELATION = "correlation"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ERROR_BURST = "error_burst"
    TRACE_ANOMALY = "trace_anomaly"


@dataclass
class Anomaly:
    """Detected anomaly."""
    
    anomaly_type: AnomalyType
    severity: float  # 0.0 to 1.0
    description: str
    timestamp: datetime
    
    # Affected logs
    log_ids: List[str] = field(default_factory=list)
    affected_services: Set[str] = field(default_factory=set)
    affected_modules: Set[str] = field(default_factory=set)
    
    # Detection details
    detection_method: str = ""
    confidence: float = 0.0
    baseline_value: Optional[Any] = None
    observed_value: Optional[Any] = None
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Remediation
    suggested_action: Optional[str] = None
    auto_remediation: bool = False
    
    # Trace information
    trace_ids: Set[str] = field(default_factory=set)


@dataclass
class SecurityPattern:
    """Security threat pattern."""
    
    name: str
    pattern: Pattern
    severity: float
    category: str
    description: str
    affected_fields: List[str] = field(default_factory=list)


class BaseAnomalyModel:
    """Base class for anomaly detection models."""
    
    def fit(self, logs: List[ParsedLog]) -> None:
        """Train the model on normal logs."""
        raise NotImplementedError
    
    def predict(self, logs: List[ParsedLog]) -> List[Anomaly]:
        """Detect anomalies in logs."""
        raise NotImplementedError
    
    def update(self, logs: List[ParsedLog]) -> None:
        """Update model with new logs."""
        pass


class FrequencyAnomalyModel(BaseAnomalyModel):
    """Detects frequency-based anomalies."""
    
    def __init__(self, threshold_std: float = 3.0):
        self.threshold_std = threshold_std
        self.baseline_rates = {}
        self.time_windows = [60, 300, 3600]  # 1min, 5min, 1hour
    
    def fit(self, logs: List[ParsedLog]) -> None:
        """Calculate baseline frequencies."""
        for window in self.time_windows:
            rates = self._calculate_rates(logs, window)
            self.baseline_rates[window] = {
                "mean": np.mean(list(rates.values())) if rates else 0,
                "std": np.std(list(rates.values())) if rates else 1
            }
    
    def predict(self, logs: List[ParsedLog]) -> List[Anomaly]:
        """Detect frequency anomalies."""
        anomalies = []
        
        for window in self.time_windows:
            rates = self._calculate_rates(logs, window)
            baseline = self.baseline_rates.get(window, {"mean": 0, "std": 1})
            
            for identifier, rate in rates.items():
                z_score = abs((rate - baseline["mean"]) / (baseline["std"] + 1e-6))
                
                if z_score > self.threshold_std:
                    service_name = identifier if identifier else "unknown"
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.FREQUENCY,
                        severity=min(1.0, z_score / (self.threshold_std * 2)),
                        description=f"Abnormal log frequency for {service_name}",
                        timestamp=datetime.now(),
                        affected_services={service_name} if service_name != "unknown" else set(),
                        detection_method="z-score",
                        confidence=min(1.0, z_score / 10),
                        baseline_value=baseline["mean"],
                        observed_value=rate
                    ))
        
        return anomalies
    
    def _calculate_rates(self, logs: List[ParsedLog], window: int) -> Dict[str, float]:
        """Calculate log rates per service/module."""
        counts = defaultdict(int)
        
        for log in logs:
            # Use service or module as identifier
            identifier = log.service or log.module or "unknown"
            counts[identifier] += 1
        
        return {
            identifier: count / window 
            for identifier, count in counts.items()
        }


class SequenceAnomalyModel(BaseAnomalyModel):
    """Detects sequence-based anomalies."""
    
    def __init__(self, sequence_length: int = 5):
        self.sequence_length = sequence_length
        self.normal_sequences = set()
        self.sequence_counts = Counter()
    
    def fit(self, logs: List[ParsedLog]) -> None:
        """Learn normal log sequences."""
        sequences = self._extract_sequences(logs)
        
        for seq in sequences:
            self.sequence_counts[seq] += 1
        
        # Consider sequences appearing more than threshold as normal
        if self.sequence_counts:
            threshold = np.percentile(list(self.sequence_counts.values()), 10)
            self.normal_sequences = {
                seq for seq, count in self.sequence_counts.items()
                if count > threshold
            }
    
    def predict(self, logs: List[ParsedLog]) -> List[Anomaly]:
        """Detect sequence anomalies."""
        anomalies = []
        sequences = self._extract_sequences(logs)
        
        for seq in sequences:
            if seq not in self.normal_sequences:
                # Extract affected services and modules
                affected_services = set()
                affected_modules = set()
                
                for i, log_msg in enumerate(seq):
                    if i < len(logs):
                        if logs[i].service:
                            affected_services.add(logs[i].service)
                        if logs[i].module:
                            affected_modules.add(logs[i].module)
                
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.SEQUENCE,
                    severity=0.7,
                    description=f"Unusual log sequence detected: {seq[:3]}...",
                    timestamp=datetime.now(),
                    affected_services=affected_services,
                    affected_modules=affected_modules,
                    detection_method="sequence_matching",
                    confidence=0.8,
                    context={"sequence": seq}
                ))
        
        return anomalies
    
    def _extract_sequences(self, logs: List[ParsedLog]) -> List[Tuple[str, ...]]:
        """Extract log sequences."""
        sequences = []
        
        for i in range(len(logs) - self.sequence_length + 1):
            seq = tuple(
                log.message[:50] 
                for log in logs[i:i + self.sequence_length]
            )
            sequences.append(seq)
        
        return sequences


class CorrelationAnomalyModel(BaseAnomalyModel):
    """Detects correlation-based anomalies."""
    
    def __init__(self, correlation_threshold: float = 0.7):
        self.correlation_threshold = correlation_threshold
        self.normal_correlations = {}
    
    def fit(self, logs: List[ParsedLog]) -> None:
        """Learn normal correlations between services/modules."""
        entity_pairs = self._extract_entity_pairs(logs)
        
        for (entity1, entity2), counts in entity_pairs.items():
            if counts:  # Check if counts is not empty
                correlation = self._calculate_correlation(counts)
                if correlation > self.correlation_threshold:
                    self.normal_correlations[(entity1, entity2)] = correlation
    
    def predict(self, logs: List[ParsedLog]) -> List[Anomaly]:
        """Detect correlation anomalies."""
        anomalies = []
        entity_pairs = self._extract_entity_pairs(logs)
        
        # Check for broken correlations
        for (entity1, entity2), expected_corr in self.normal_correlations.items():
            if (entity1, entity2) in entity_pairs:
                counts = entity_pairs[(entity1, entity2)]
                if counts:  # Check if counts is not empty
                    actual_corr = self._calculate_correlation(counts)
                    
                    if abs(actual_corr - expected_corr) > 0.3:
                        anomalies.append(Anomaly(
                            anomaly_type=AnomalyType.CORRELATION,
                            severity=0.6,
                            description=f"Correlation anomaly between {entity1} and {entity2}",
                            timestamp=datetime.now(),
                            affected_services={entity1, entity2} if '/' not in entity1 else set(),
                            affected_modules={entity1, entity2} if '/' in entity1 else set(),
                            detection_method="correlation_analysis",
                            confidence=0.7,
                            baseline_value=expected_corr,
                            observed_value=actual_corr
                        ))
        
        return anomalies
    
    def _extract_entity_pairs(self, logs: List[ParsedLog]) -> Dict[Tuple[str, str], List[int]]:
        """Extract service/module pair occurrences."""
        pairs = defaultdict(list)
        
        # Group logs by time window
        time_windows = defaultdict(list)
        for log in logs:
            window = log.timestamp.timestamp() // 60  # 1-minute windows
            entity = log.service or log.module or "unknown"
            time_windows[window].append(entity)
        
        # Count co-occurrences
        for window, entities in time_windows.items():
            entity_counts = Counter(entities)
            for entity1 in entity_counts:
                for entity2 in entity_counts:
                    if entity1 < entity2:
                        pairs[(entity1, entity2)].append(
                            min(entity_counts[entity1], entity_counts[entity2])
                        )
        
        return pairs
    
    def _calculate_correlation(self, counts: List[int]) -> float:
        """Calculate correlation coefficient."""
        if len(counts) < 2:
            return 0.0
        
        x = np.arange(len(counts))
        correlation_matrix = np.corrcoef(x, counts)
        return correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0


class TraceAnomalyModel(BaseAnomalyModel):
    """Detects anomalies in distributed traces."""
    
    def __init__(self):
        self.normal_trace_patterns = {}
        self.trace_durations = defaultdict(list)
    
    def fit(self, logs: List[ParsedLog]) -> None:
        """Learn normal trace patterns."""
        traces = self._group_by_trace(logs)
        
        for trace_id, trace_logs in traces.items():
            if len(trace_logs) > 1:
                duration = (trace_logs[-1].timestamp - trace_logs[0].timestamp).total_seconds()
                
                # Group by service pattern
                service_pattern = tuple(log.service or "unknown" for log in trace_logs)
                self.trace_durations[service_pattern].append(duration)
    
    def predict(self, logs: List[ParsedLog]) -> List[Anomaly]:
        """Detect trace anomalies."""
        anomalies = []
        traces = self._group_by_trace(logs)
        
        for trace_id, trace_logs in traces.items():
            if len(trace_logs) > 1:
                duration = (trace_logs[-1].timestamp - trace_logs[0].timestamp).total_seconds()
                service_pattern = tuple(log.service or "unknown" for log in trace_logs)
                
                if service_pattern in self.trace_durations:
                    durations = self.trace_durations[service_pattern]
                    if durations:
                        mean_duration = np.mean(durations)
                        std_duration = np.std(durations) if len(durations) > 1 else mean_duration * 0.1
                        
                        if abs(duration - mean_duration) > 3 * std_duration:
                            anomalies.append(Anomaly(
                                anomaly_type=AnomalyType.TRACE_ANOMALY,
                                severity=0.7,
                                description=f"Abnormal trace duration: {duration:.2f}s",
                                timestamp=trace_logs[-1].timestamp,
                                trace_ids={trace_id},
                                affected_services=set(log.service for log in trace_logs if log.service),
                                detection_method="trace_analysis",
                                confidence=0.75,
                                baseline_value=mean_duration,
                                observed_value=duration,
                                context={"service_pattern": service_pattern}
                            ))
        
        return anomalies
    
    def _group_by_trace(self, logs: List[ParsedLog]) -> Dict[str, List[ParsedLog]]:
        """Group logs by trace ID."""
        traces = defaultdict(list)
        
        for log in logs:
            if log.trace_id:
                traces[log.trace_id].append(log)
        
        # Sort logs within each trace by timestamp
        for trace_id in traces:
            traces[trace_id].sort(key=lambda x: x.timestamp)
        
        return traces


class AnomalyDetector:
    """
    Comprehensive log anomaly detection system.
    
    This class provides:
    - Statistical anomaly detection
    - Pattern-based detection
    - Machine learning models
    - Security threat detection
    - Real-time and batch processing
    - Trace anomaly detection
    """
    
    def __init__(
        self,
        window_size: int = 3600,  # 1 hour
        sensitivity: float = 0.5,
        enable_ml: bool = True,
        enable_trace_analysis: bool = True
    ):
        """
        Initialize anomaly detector.
        
        Args:
            window_size: Time window for analysis (seconds)
            sensitivity: Detection sensitivity (0.0 to 1.0)
            enable_ml: Enable machine learning models
            enable_trace_analysis: Enable trace anomaly detection
        """
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.enable_ml = enable_ml
        self.enable_trace_analysis = enable_trace_analysis
        
        # Log history
        self.log_buffer = deque(maxlen=10000)
        self.log_stats = defaultdict(lambda: {
            "count": 0,
            "error_count": 0,
            "patterns": Counter(),
            "values": [],
            "modules": Counter()
        })
        
        # Baselines
        self.baselines = {}
        self.pattern_models = {}
        
        # ML models
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.vectorizer = TfidfVectorizer(max_features=100)
        
        # Security patterns
        self.security_patterns = self._load_security_patterns()
        
        # Detected anomalies
        self.anomalies = []
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models for anomaly detection."""
        # Anomaly models
        self.pattern_models = {
            "frequency": FrequencyAnomalyModel(threshold_std=3.0 - self.sensitivity),
            "sequence": SequenceAnomalyModel(),
            "correlation": CorrelationAnomalyModel()
        }
        
        if self.enable_trace_analysis:
            self.pattern_models["trace"] = TraceAnomalyModel()
        
        if self.enable_ml:
            # Isolation Forest for multivariate anomaly detection
            self.isolation_forest = IsolationForest(
                contamination=0.1 * self.sensitivity,
                random_state=42,
                n_estimators=100
            )
    
    def _load_security_patterns(self) -> List[SecurityPattern]:
        """Load security threat patterns."""
        patterns = []
        
        # SQL injection patterns
        patterns.append(SecurityPattern(
            name="sql_injection",
            pattern=re.compile(
                r"(\bSELECT\b.*\bFROM\b|\bUNION\b.*\bSELECT\b|\bDROP\b.*\bTABLE\b|\bOR\b.*=.*\bOR\b)",
                re.IGNORECASE
            ),
            severity=0.9,
            category="injection",
            description="Potential SQL injection attempt",
            affected_fields=["message", "fields.query", "fields.sql"]
        ))
        
        # Command injection patterns
        patterns.append(SecurityPattern(
            name="command_injection",
            pattern=re.compile(
                r"(;\s*rm\s+-rf|&&\s*wget|`.*`|\$KATEX_INLINE_OPEN.*KATEX_INLINE_CLOSE|\|\s*nc\s+)",
                re.IGNORECASE
            ),
            severity=0.9,
            category="injection",
            description="Potential command injection attempt",
            affected_fields=["message", "fields.command", "fields.cmd"]
        ))
        
        # Path traversal patterns
        patterns.append(SecurityPattern(
            name="path_traversal",
            pattern=re.compile(r"(\.\./|\.\.\KATEX_INLINE_CLOSE{2,}"),
            severity=0.8,
            category="traversal",
            description="Potential path traversal attempt",
            affected_fields=["message", "fields.path", "fields.file"]
        ))
        
        # Authentication failure patterns
        patterns.append(SecurityPattern(
            name="auth_failure",
            pattern=re.compile(
                r"(authentication failed|invalid password|login failed|unauthorized access)",
                re.IGNORECASE
            ),
            severity=0.5,
            category="authentication",
            description="Authentication failure detected",
            affected_fields=["message"]
        ))
        
        # Rate limiting patterns
        patterns.append(SecurityPattern(
            name="rate_limit",
            pattern=re.compile(
                r"(rate limit exceeded|too many requests|throttled|429 Too Many Requests)",
                re.IGNORECASE
            ),
            severity=0.4,
            category="rate_limiting",
            description="Rate limiting triggered",
            affected_fields=["message", "fields.status"]
        ))
        
        # XSS patterns
        patterns.append(SecurityPattern(
            name="xss_attempt",
            pattern=re.compile(
                r"(<script[^>]*>.*</script>|javascript:|onerror=|onclick=)",
                re.IGNORECASE
            ),
            severity=0.8,
            category="xss",
            description="Potential XSS attempt",
            affected_fields=["message", "fields.user_input", "fields.payload"]
        ))
        
        return patterns
    
    def process_log(self, log: ParsedLog) -> List[Anomaly]:
        """
        Process a single log entry for anomalies.
        
        Args:
            log: Parsed log entry
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Add to buffer
        self.log_buffer.append(log)
        
        # Update statistics
        self._update_statistics(log)
        
        # Real-time checks
        anomalies.extend(self._check_security_patterns(log))
        anomalies.extend(self._check_error_burst(log))
        anomalies.extend(self._check_performance_anomaly(log))
        
        # Check for parse errors
        if log.parse_errors:
            anomalies.append(Anomaly(
                anomaly_type=AnomalyType.PATTERN,
                severity=0.3,
                description=f"Log parsing errors: {', '.join(log.parse_errors)}",
                timestamp=log.timestamp,
                affected_services={log.service} if log.service else set(),
                affected_modules={log.module} if log.module else set(),
                detection_method="parse_error",
                confidence=1.0,
                context={"parse_errors": log.parse_errors}
            ))
        
        # Batch analysis if buffer is full
        if len(self.log_buffer) >= 100:
            anomalies.extend(self.analyze_batch(list(self.log_buffer)[-100:]))
        
        # Store detected anomalies
        self.anomalies.extend(anomalies)
        
        return anomalies
    
    def _check_security_patterns(self, log: ParsedLog) -> List[Anomaly]:
        """Check log against security patterns."""
        anomalies = []
        
        for pattern in self.security_patterns:
            # Check message
            if pattern.pattern.search(log.message):
                anomalies.append(self._create_security_anomaly(log, pattern, "message"))
            
            # Check specific fields
            if log.fields:
                for field_name in pattern.affected_fields:
                    if "." in field_name:
                        # Nested field
                        parts = field_name.split(".")
                        value = log.fields
                        for part in parts[1:]:  # Skip 'fields' prefix
                            if isinstance(value, dict) and part in value:
                                value = value[part]
                            else:
                                value = None
                                break
                        
                        if value and isinstance(value, str) and pattern.pattern.search(value):
                            anomalies.append(self._create_security_anomaly(log, pattern, field_name))
        
        return anomalies
    
    def _create_security_anomaly(self, log: ParsedLog, pattern: SecurityPattern, field: str) -> Anomaly:
        """Create security anomaly."""
        return Anomaly(
            anomaly_type=AnomalyType.SECURITY,
            severity=pattern.severity,
            description=pattern.description,
            timestamp=log.timestamp,
            affected_services={log.service} if log.service else set(),
            affected_modules={log.module} if log.module else set(),
            trace_ids={log.trace_id} if log.trace_id else set(),
            detection_method="pattern_matching",
            confidence=0.9,
            context={
                "pattern": pattern.name,
                "category": pattern.category,
                "field": field,
                "user_id": log.user_id
            },
            suggested_action=self._get_security_action(pattern.category)
        )
    
    def _check_error_burst(self, log: ParsedLog) -> List[Anomaly]:
        """Check for error bursts."""
        anomalies = []
        
        if log.level.value >= LogLevel.ERROR.value:
            # Count recent errors
            recent_errors = sum(
                1 for l in self.log_buffer
                if l.level.value >= LogLevel.ERROR.value
                and (log.timestamp - l.timestamp).total_seconds() < 60
            )
            
            if recent_errors > 10:
                affected_services = set()
                affected_modules = set()
                
                for l in self.log_buffer:
                    if l.level.value >= LogLevel.ERROR.value and (log.timestamp - l.timestamp).total_seconds() < 60:
                        if l.service:
                            affected_services.add(l.service)
                        if l.module:
                            affected_modules.add(l.module)
                
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.ERROR_BURST,
                    severity=min(1.0, recent_errors / 20),
                    description=f"Error burst detected: {recent_errors} errors in 60s",
                    timestamp=log.timestamp,
                    affected_services=affected_services,
                    affected_modules=affected_modules,
                    detection_method="threshold",
                    confidence=0.8,
                    observed_value=recent_errors,
                    suggested_action="Check service health and recent deployments"
                ))
        
        return anomalies
    
    def _check_performance_anomaly(self, log: ParsedLog) -> List[Anomaly]:
        """Check for performance anomalies."""
        anomalies = []
        
        # Check extracted metrics
        if log.fields:
            # Check for duration/latency in fields
            for key, value in log.fields.items():
                if any(perf_key in key.lower() for perf_key in ["duration", "latency", "response_time", "elapsed"]):
                    try:
                        numeric_value = float(value) if not isinstance(value, (int, float)) else value
                        
                        # Get baseline for this metric
                        metric_key = f"{log.service or 'unknown'}_{key}"
                        
                        if metric_key in self.baselines:
                            baseline = self.baselines[metric_key]
                            mean = baseline["mean"]
                            std = baseline["std"]
                            
                            if numeric_value > mean + 3 * std:
                                anomalies.append(Anomaly(
                                    anomaly_type=AnomalyType.PERFORMANCE,
                                    severity=min(1.0, (numeric_value - mean) / (mean + 1e-6)),
                                    description=f"High {key}: {numeric_value:.2f}",
                                    timestamp=log.timestamp,
                                    affected_services={log.service} if log.service else set(),
                                    affected_modules={log.module} if log.module else set(),
                                    trace_ids={log.trace_id} if log.trace_id else set(),
                                    detection_method="statistical",
                                    confidence=0.7,
                                    baseline_value=mean,
                                    observed_value=numeric_value,
                                    context={"metric": key},
                                    suggested_action="Check service load and resource utilization"
                                ))
                        
                        # Update baseline
                        if metric_key not in self.baselines:
                            self.baselines[metric_key] = {
                                "mean": numeric_value,
                                "std": numeric_value * 0.1,
                                "samples": [numeric_value]
                            }
                        else:
                            self.baselines[metric_key]["samples"].append(numeric_value)
                            if len(self.baselines[metric_key]["samples"]) > 100:
                                self.baselines[metric_key]["samples"] = self.baselines[metric_key]["samples"][-100:]
                            self.baselines[metric_key]["mean"] = np.mean(self.baselines[metric_key]["samples"])
                            self.baselines[metric_key]["std"] = np.std(self.baselines[metric_key]["samples"])
                    
                    except (ValueError, TypeError):
                        pass
        
        return anomalies
    
    def _update_statistics(self, log: ParsedLog):
        """Update running statistics."""
        identifier = log.service or log.module or "unknown"
        stats = self.log_stats[identifier]
        stats["count"] += 1
        
        if log.level.value >= LogLevel.ERROR.value:
            stats["error_count"] += 1
        
        # Update pattern counts
        pattern_key = self._extract_pattern(log.message)
        stats["patterns"][pattern_key] += 1
        
        # Track modules
        if log.module:
            stats["modules"][log.module] += 1
    
    def _extract_pattern(self, message: str) -> str:
        """Extract pattern from log message."""
        # Remove numbers and specific values
        pattern = re.sub(r'\d+', 'NUM', message)
        pattern = re.sub(r'"[^"]*"', 'STR', pattern)
        pattern = re.sub(r'\b[a-f0-9]{32}\b', 'HASH', pattern)
        pattern = re.sub(r'\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b', 'UUID', pattern)
        pattern = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', 'IP', pattern)
        
        return pattern[:100]  # Limit length
    
    def _get_security_action(self, category: str) -> str:
        """Get suggested action for security anomaly."""
        actions = {
            "injection": "Block request and investigate source IP",
            "traversal": "Validate input and check file access logs",
            "authentication": "Monitor for brute force attempts",
            "rate_limiting": "Check for DDoS or automated attacks",
            "xss": "Sanitize user input and review output encoding"
        }
        
        return actions.get(category, "Investigate security logs")
    
    def analyze_batch(self, logs: List[ParsedLog]) -> List[Anomaly]:
        """
        Analyze a batch of logs for anomalies.
        
        Args:
            logs: List of parsed logs
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Pattern-based detection
        for model_name, model in self.pattern_models.items():
            try:
                detected = model.predict(logs)
                anomalies.extend(detected)
            except Exception as e:
                logger.error(f"Error in {model_name} model: {e}")
        
        # ML-based detection
        if self.enable_ml and len(logs) > 10:
            anomalies.extend(self._ml_detection(logs))
        
        # Statistical detection
        anomalies.extend(self._statistical_detection(logs))
        
        return anomalies
    
    def _ml_detection(self, logs: List[ParsedLog]) -> List[Anomaly]:
        """ML-based anomaly detection."""
        anomalies = []
        
        try:
            # Feature extraction
            features = self._extract_features(logs)
            
            if features.shape[0] > 0:
                # Fit or predict with Isolation Forest
                if not hasattr(self.isolation_forest, "offset_"):
                    # First time - fit the model
                    self.isolation_forest.fit(features)
                    predictions = self.isolation_forest.predict(features)
                else:
                    predictions = self.isolation_forest.predict(features)
                
                # Process predictions
                for i, pred in enumerate(predictions):
                    if pred == -1:  # Anomaly
                        anomalies.append(Anomaly(
                            anomaly_type=AnomalyType.PATTERN,
                            severity=0.6,
                            description="ML model detected unusual pattern",
                            timestamp=logs[i].timestamp,
                            affected_services={logs[i].service} if logs[i].service else set(),
                            affected_modules={logs[i].module} if logs[i].module else set(),
                            trace_ids={logs[i].trace_id} if logs[i].trace_id else set(),
                            detection_method="isolation_forest",
                            confidence=0.6,
                            context={"feature_importance": self._get_feature_importance(features[i])}
                        ))
        
        except Exception as e:
            logger.error(f"Error in ML detection: {e}")
        
        return anomalies
    
    def _extract_features(self, logs: List[ParsedLog]) -> np.ndarray:
        """Extract features from logs for ML models."""
        features = []
        
        for log in logs:
            feature_vector = [
                # Time features
                log.timestamp.hour,
                log.timestamp.minute,
                log.timestamp.weekday(),
                
                # Level features
                log.level.value,
                
                # Message features
                len(log.message),
                log.message.count(" "),
                
                # Entity features
                hash(log.service or "none") % 100,
                hash(log.module or "none") % 100,
                
                # Trace features
                1 if log.trace_id else 0,
                1 if log.user_id else 0,
                
                # Error indicators
                1 if "error" in log.message.lower() else 0,
                1 if "failed" in log.message.lower() else 0,
                1 if "exception" in log.message.lower() else 0,
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _get_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """Get feature importance for anomaly explanation."""
        feature_names = [
            "hour", "minute", "weekday",
            "log_level",
            "message_length", "word_count",
            "service_hash", "module_hash",
            "has_trace", "has_user",
            "has_error", "has_failed", "has_exception"
        ]
        
        # Simple importance based on deviation from mean
        importance = {}
        for i, name in enumerate(feature_names):
            importance[name] = abs(features[i])
        
        return importance
    
    def _statistical_detection(self, logs: List[ParsedLog]) -> List[Anomaly]:
        """Statistical anomaly detection."""
        anomalies = []
        
        # Group logs by service/module
        entity_logs = defaultdict(list)
        for log in logs:
            entity = log.service or log.module or "unknown"
            entity_logs[entity].append(log)
        
        for entity, e_logs in entity_logs.items():
            # Check log rate
            rate = len(e_logs) / (self.window_size / 3600)  # logs per hour
            
            if entity in self.baselines:
                baseline = self.baselines[entity]
                z_score = abs((rate - baseline["mean"]) / (baseline["std"] + 1e-6))
                
                if z_score > 3:
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.FREQUENCY,
                        severity=min(1.0, z_score / 5),
                        description=f"Unusual log rate for {entity}",
                        timestamp=datetime.now(),
                        affected_services={entity} if not entity.startswith("/") else set(),
                        affected_modules={entity} if entity.startswith("/") else set(),
                        detection_method="z_score",
                        confidence=min(1.0, z_score / 10),
                        baseline_value=baseline["mean"],
                        observed_value=rate
                    ))
        
        return anomalies
    
    def update_baselines(self, logs: List[ParsedLog]):
        """Update baseline statistics."""
        # Group by service/module
        entity_logs = defaultdict(list)
        for log in logs:
            entity = log.service or log.module or "unknown"
            entity_logs[entity].append(log)
        
        for entity, e_logs in entity_logs.items():
            rate = len(e_logs) / (self.window_size / 3600)
            
            if entity not in self.baselines:
                self.baselines[entity] = {"mean": rate, "std": 1.0, "samples": [rate]}
            else:
                self.baselines[entity]["samples"].append(rate)
                
                # Keep only recent samples
                if len(self.baselines[entity]["samples"]) > 100:
                    self.baselines[entity]["samples"] = self.baselines[entity]["samples"][-100:]
                
                # Update statistics
                self.baselines[entity]["mean"] = np.mean(self.baselines[entity]["samples"])
                self.baselines[entity]["std"] = np.std(self.baselines[entity]["samples"])
        
        # Train pattern models
        if len(logs) > 100:
            for model in self.pattern_models.values():
                try:
                    model.fit(logs)
                except Exception as e:
                    logger.error(f"Error training model: {e}")
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of detected anomalies."""
        if not self.anomalies:
            return {"total": 0, "by_type": {}, "by_severity": {}}
        
        # Group by type
        by_type = Counter(a.anomaly_type.value for a in self.anomalies)
        
        # Group by severity
        by_severity = {
            "critical": sum(1 for a in self.anomalies if a.severity > 0.8),
            "high": sum(1 for a in self.anomalies if 0.6 < a.severity <= 0.8),
            "medium": sum(1 for a in self.anomalies if 0.4 < a.severity <= 0.6),
            "low": sum(1 for a in self.anomalies if a.severity <= 0.4)
        }
        
        # Recent anomalies
        recent = [
            a for a in self.anomalies
            if (datetime.now() - a.timestamp).total_seconds() < 3600
        ]
        
        # Affected services and modules
        affected_services = set()
        affected_modules = set()
        for a in self.anomalies:
            affected_services.update(a.affected_services)
            affected_modules.update(a.affected_modules)
        
        return {
            "total": len(self.anomalies),
            "recent_count": len(recent),
            "by_type": dict(by_type),
            "by_severity": by_severity,
            "top_services": Counter(
                service 
                for a in self.anomalies 
                for service in a.affected_services
            ).most_common(5),
            "top_modules": Counter(
                module
                for a in self.anomalies
                for module in a.affected_modules
            ).most_common(5),
            "unique_traces_affected": len(set(
                trace_id
                for a in self.anomalies
                for trace_id in a.trace_ids
            ))
        }
