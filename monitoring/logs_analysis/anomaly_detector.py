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


@dataclass
class SecurityPattern:
    """Security threat pattern."""
    
    name: str
    pattern: Pattern
    severity: float
    category: str
    description: str


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
                "mean": np.mean(list(rates.values())),
                "std": np.std(list(rates.values()))
            }
    
    def predict(self, logs: List[ParsedLog]) -> List[Anomaly]:
        """Detect frequency anomalies."""
        anomalies = []
        
        for window in self.time_windows:
            rates = self._calculate_rates(logs, window)
            baseline = self.baseline_rates.get(window, {"mean": 0, "std": 1})
            
            for service, rate in rates.items():
                z_score = abs((rate - baseline["mean"]) / (baseline["std"] + 1e-6))
                
                if z_score > self.threshold_std:
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.FREQUENCY,
                        severity=min(1.0, z_score / (self.threshold_std * 2)),
                        description=f"Abnormal log frequency for {service}",
                        timestamp=datetime.now(),
                        affected_services={service},
                        detection_method="z-score",
                        confidence=min(1.0, z_score / 10),
                        baseline_value=baseline["mean"],
                        observed_value=rate
                    ))
        
        return anomalies
    
    def _calculate_rates(self, logs: List[ParsedLog], window: int) -> Dict[str, float]:
        """Calculate log rates per service."""
        counts = defaultdict(int)
        
        for log in logs:
            counts[log.service] += 1
        
        return {
            service: count / window 
            for service, count in counts.items()
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
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.SEQUENCE,
                    severity=0.7,
                    description=f"Unusual log sequence detected: {seq[:3]}...",
                    timestamp=datetime.now(),
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
        """Learn normal correlations between services."""
        service_pairs = self._extract_service_pairs(logs)
        
        for (service1, service2), counts in service_pairs.items():
            correlation = self._calculate_correlation(counts)
            if correlation > self.correlation_threshold:
                self.normal_correlations[(service1, service2)] = correlation
    
    def predict(self, logs: List[ParsedLog]) -> List[Anomaly]:
        """Detect correlation anomalies."""
        anomalies = []
        service_pairs = self._extract_service_pairs(logs)
        
        # Check for broken correlations
        for (service1, service2), expected_corr in self.normal_correlations.items():
            if (service1, service2) in service_pairs:
                counts = service_pairs[(service1, service2)]
                actual_corr = self._calculate_correlation(counts)
                
                if abs(actual_corr - expected_corr) > 0.3:
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.CORRELATION,
                        severity=0.6,
                        description=f"Correlation anomaly between {service1} and {service2}",
                        timestamp=datetime.now(),
                        affected_services={service1, service2},
                        detection_method="correlation_analysis",
                        confidence=0.7,
                        baseline_value=expected_corr,
                        observed_value=actual_corr
                    ))
        
        return anomalies
    
    def _extract_service_pairs(self, logs: List[ParsedLog]) -> Dict[Tuple[str, str], List[int]]:
        """Extract service pair occurrences."""
        pairs = defaultdict(list)
        
        # Group logs by time window
        time_windows = defaultdict(list)
        for log in logs:
            window = log.timestamp.timestamp() // 60  # 1-minute windows
            time_windows[window].append(log.service)
        
        # Count co-occurrences
        for window, services in time_windows.items():
            service_counts = Counter(services)
            for service1 in service_counts:
                for service2 in service_counts:
                    if service1 < service2:
                        pairs[(service1, service2)].append(
                            min(service_counts[service1], service_counts[service2])
                        )
        
        return pairs
    
    def _calculate_correlation(self, counts: List[int]) -> float:
        """Calculate correlation coefficient."""
        if len(counts) < 2:
            return 0.0
        
        x = np.arange(len(counts))
        return np.corrcoef(x, counts)[0, 1]


class AnomalyDetector:
    """
    Comprehensive log anomaly detection system.
    
    This class provides:
    - Statistical anomaly detection
    - Pattern-based detection
    - Machine learning models
    - Security threat detection
    - Real-time and batch processing
    """
    
    def __init__(
        self,
        window_size: int = 3600,  # 1 hour
        sensitivity: float = 0.5,
        enable_ml: bool = True
    ):
        """
        Initialize anomaly detector.
        
        Args:
            window_size: Time window for analysis (seconds)
            sensitivity: Detection sensitivity (0.0 to 1.0)
            enable_ml: Enable machine learning models
        """
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.enable_ml = enable_ml
        
        # Log history
        self.log_buffer = deque(maxlen=10000)
        self.log_stats = defaultdict(lambda: {
            "count": 0,
            "error_count": 0,
            "patterns": Counter(),
            "values": []
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
                r"(\bSELECT\b.*\bFROM\b|\bUNION\b.*\bSELECT\b|\bDROP\b.*\bTABLE\b)",
                re.IGNORECASE
            ),
            severity=0.9,
            category="injection",
            description="Potential SQL injection attempt"
        ))
        
        # Command injection patterns
        patterns.append(SecurityPattern(
            name="command_injection",
            pattern=re.compile(
                r"(;\s*rm\s+-rf|&&\s*wget|`.*`|\$KATEX_INLINE_OPEN.*KATEX_INLINE_CLOSE)",
                re.IGNORECASE
            ),
            severity=0.9,
            category="injection",
            description="Potential command injection attempt"
        ))
        
        # Path traversal patterns
        patterns.append(SecurityPattern(
            name="path_traversal",
            pattern=re.compile(r"(\.\./|\.\.\KATEX_INLINE_CLOSE{2,}"),
            severity=0.8,
            category="traversal",
            description="Potential path traversal attempt"
        ))
        
        # Authentication failure patterns
        patterns.append(SecurityPattern(
            name="auth_failure",
            pattern=re.compile(
                r"(authentication failed|invalid password|login failed)",
                re.IGNORECASE
            ),
            severity=0.5,
            category="authentication",
            description="Authentication failure detected"
        ))
        
        # Rate limiting patterns
        patterns.append(SecurityPattern(
            name="rate_limit",
            pattern=re.compile(
                r"(rate limit exceeded|too many requests|throttled)",
                re.IGNORECASE
            ),
            severity=0.4,
            category="rate_limiting",
            description="Rate limiting triggered"
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
        
        # Batch analysis if buffer is full
        if len(self.log_buffer) >= 100:
            anomalies.extend(self.analyze_batch(list(self.log_buffer)[-100:]))
        
        # Store detected anomalies
        self.anomalies.extend(anomalies)
        
        return anomalies
    
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
    
    def _check_security_patterns(self, log: ParsedLog) -> List[Anomaly]:
        """Check log against security patterns."""
        anomalies = []
        
        for pattern in self.security_patterns:
            if pattern.pattern.search(log.message):
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.SECURITY,
                    severity=pattern.severity,
                    description=pattern.description,
                    timestamp=log.timestamp,
                    log_ids=[log.log_id],
                    affected_services={log.service},
                    detection_method="pattern_matching",
                    confidence=0.9,
                    context={
                        "pattern": pattern.name,
                        "category": pattern.category
                    },
                    suggested_action=self._get_security_action(pattern.category)
                ))
        
        return anomalies
    
    def _check_error_burst(self, log: ParsedLog) -> List[Anomaly]:
        """Check for error bursts."""
        anomalies = []
        
        if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            # Count recent errors
            recent_errors = sum(
                1 for l in self.log_buffer
                if l.level in [LogLevel.ERROR, LogLevel.CRITICAL]
                and (log.timestamp - l.timestamp).total_seconds() < 60
            )
            
            if recent_errors > 10:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.ERROR_BURST,
                    severity=min(1.0, recent_errors / 20),
                    description=f"Error burst detected: {recent_errors} errors in 60s",
                    timestamp=log.timestamp,
                    affected_services={log.service},
                    detection_method="threshold",
                    confidence=0.8,
                    observed_value=recent_errors,
                    suggested_action="Check service health and recent deployments"
                ))
        
        return anomalies
    
    def _check_performance_anomaly(self, log: ParsedLog) -> List[Anomaly]:
        """Check for performance anomalies."""
        anomalies = []
        
        # Extract response time if present
        response_time = self._extract_response_time(log.message)
        
        if response_time:
            service_stats = self.log_stats[log.service]
            
            if service_stats["values"]:
                mean = np.mean(service_stats["values"])
                std = np.std(service_stats["values"])
                
                if response_time > mean + 3 * std:
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.PERFORMANCE,
                        severity=min(1.0, (response_time - mean) / (mean + 1e-6)),
                        description=f"High response time: {response_time:.2f}ms",
                        timestamp=log.timestamp,
                        affected_services={log.service},
                        detection_method="statistical",
                        confidence=0.7,
                        baseline_value=mean,
                        observed_value=response_time,
                        suggested_action="Check service load and resource utilization"
                    ))
            
            # Update statistics
            service_stats["values"].append(response_time)
            if len(service_stats["values"]) > 1000:
                service_stats["values"] = service_stats["values"][-1000:]
        
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
                            log_ids=[logs[i].log_id],
                            detection_method="isolation_forest",
                            confidence=0.6,
                            context={"feature_importance": self._get_feature_importance(features[i])}
                        ))
        
        except Exception as e:
            logger.error(f"Error in ML detection: {e}")
        
        return anomalies
    
    def _statistical_detection(self, logs: List[ParsedLog]) -> List[Anomaly]:
        """Statistical anomaly detection."""
        anomalies = []
        
        # Group logs by service
        service_logs = defaultdict(list)
        for log in logs:
            service_logs[log.service].append(log)
        
        for service, s_logs in service_logs.items():
            # Check log rate
            rate = len(s_logs) / (self.window_size / 3600)  # logs per hour
            
            if service in self.baselines:
                baseline = self.baselines[service]
                z_score = abs((rate - baseline["mean"]) / (baseline["std"] + 1e-6))
                
                if z_score > 3:
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.FREQUENCY,
                        severity=min(1.0, z_score / 5),
                        description=f"Unusual log rate for {service}",
                        timestamp=datetime.now(),
                        affected_services={service},
                        detection_method="z_score",
                        confidence=min(1.0, z_score / 10),
                        baseline_value=baseline["mean"],
                        observed_value=rate
                    ))
        
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
                1 if log.level == LogLevel.ERROR else 0,
                1 if log.level == LogLevel.WARNING else 0,
                
                # Message features
                len(log.message),
                log.message.count(" "),
                
                # Service features
                hash(log.service) % 100,
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _extract_response_time(self, message: str) -> Optional[float]:
        """Extract response time from log message."""
        patterns = [
            r"response[_\s]time[:\s]+(\d+\.?\d*)\s*ms",
            r"latency[:\s]+(\d+\.?\d*)\s*ms",
            r"took[:\s]+(\d+\.?\d*)\s*ms"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        return None
    
    def _update_statistics(self, log: ParsedLog):
        """Update running statistics."""
        stats = self.log_stats[log.service]
        stats["count"] += 1
        
        if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            stats["error_count"] += 1
        
        # Update pattern counts
        pattern_key = self._extract_pattern(log.message)
        stats["patterns"][pattern_key] += 1
    
    def _extract_pattern(self, message: str) -> str:
        """Extract pattern from log message."""
        # Remove numbers and specific values
        pattern = re.sub(r'\d+', 'NUM', message)
        pattern = re.sub(r'"[^"]*"', 'STR', pattern)
        pattern = re.sub(r'\b[a-f0-9]{32}\b', 'HASH', pattern)
        
        return pattern[:100]  # Limit length
    
    def _get_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """Get feature importance for anomaly explanation."""
        feature_names = [
            "hour", "minute", "weekday",
            "is_error", "is_warning",
            "message_length", "word_count",
            "service_hash"
        ]
        
        # Simple importance based on deviation from mean
        importance = {}
        for i, name in enumerate(feature_names):
            importance[name] = abs(features[i])
        
        return importance
    
    def _get_security_action(self, category: str) -> str:
        """Get suggested action for security anomaly."""
        actions = {
            "injection": "Block request and investigate source IP",
            "traversal": "Validate input and check file access logs",
            "authentication": "Monitor for brute force attempts",
            "rate_limiting": "Check for DDoS or automated attacks"
        }
        
        return actions.get(category, "Investigate security logs")
    
    def update_baselines(self, logs: List[ParsedLog]):
        """Update baseline statistics."""
        # Group by service
        service_logs = defaultdict(list)
        for log in logs:
            service_logs[log.service].append(log)
        
        for service, s_logs in service_logs.items():
            rate = len(s_logs) / (self.window_size / 3600)
            
            if service not in self.baselines:
                self.baselines[service] = {"mean": rate, "std": 1.0, "samples": [rate]}
            else:
                self.baselines[service]["samples"].append(rate)
                
                # Keep only recent samples
                if len(self.baselines[service]["samples"]) > 100:
                    self.baselines[service]["samples"] = self.baselines[service]["samples"][-100:]
                
                # Update statistics
                self.baselines[service]["mean"] = np.mean(self.baselines[service]["samples"])
                self.baselines[service]["std"] = np.std(self.baselines[service]["samples"])
        
        # Train pattern models
        if len(logs) > 100:
            for model in self.pattern_models.values():
                model.fit(logs)
    
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
        
        return {
            "total": len(self.anomalies),
            "recent_count": len(recent),
            "by_type": dict(by_type),
            "by_severity": by_severity,
            "top_services": Counter(
                service 
                for a in self.anomalies 
                for service in a.affected_services
            ).most_common(5)
        }
