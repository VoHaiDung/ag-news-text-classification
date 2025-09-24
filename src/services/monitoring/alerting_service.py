"""
Alerting Service Implementation for AG News Text Classification
================================================================================
This module implements comprehensive alerting and notification management for
system monitoring, providing threshold-based alerts and intelligent routing.

The alerting service provides:
- Multi-channel alert delivery
- Alert aggregation and deduplication
- Severity-based routing
- Alert history and analytics

References:
    - Beyer, B., et al. (2016). Site Reliability Engineering
    - Prometheus Alerting Documentation

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import hashlib
import json

from src.services.base_service import BaseService, ServiceConfig
from src.core.exceptions import ServiceException
from src.utils.logging_config import get_logger


class AlertSeverity(Enum):
    """
    Alert severity levels.
    
    Levels:
        CRITICAL: Immediate action required
        HIGH: Urgent attention needed
        MEDIUM: Should be addressed soon
        LOW: Informational
        INFO: No action required
    """
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    INFO = 5


class AlertStatus(Enum):
    """
    Alert lifecycle status.
    
    States:
        FIRING: Alert is active
        RESOLVED: Alert has been resolved
        ACKNOWLEDGED: Alert has been acknowledged
        SILENCED: Alert is temporarily silenced
    """
    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SILENCED = "silenced"


@dataclass
class AlertRule:
    """
    Alert rule definition.
    
    Attributes:
        name: Rule name
        metric: Metric to monitor
        condition: Alert condition function
        threshold: Threshold value
        severity: Alert severity
        duration: Duration before firing
        annotations: Additional metadata
        labels: Alert labels for routing
    """
    name: str
    metric: str
    condition: Callable[[float, float], bool]
    threshold: float
    severity: AlertSeverity = AlertSeverity.MEDIUM
    duration: timedelta = timedelta(minutes=1)
    annotations: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def evaluate(self, value: float) -> bool:
        """
        Evaluate if alert should fire.
        
        Args:
            value: Current metric value
            
        Returns:
            bool: True if condition met
        """
        return self.condition(value, self.threshold)


@dataclass
class Alert:
    """
    Alert instance.
    
    Attributes:
        id: Unique alert identifier
        rule: Alert rule that triggered
        value: Metric value that triggered alert
        status: Current alert status
        severity: Alert severity
        message: Alert message
        started_at: When alert started firing
        resolved_at: When alert was resolved
        acknowledged_at: When alert was acknowledged
        acknowledged_by: Who acknowledged the alert
        annotations: Alert annotations
        labels: Alert labels
    """
    id: str
    rule: AlertRule
    value: float
    status: AlertStatus = AlertStatus.FIRING
    severity: AlertSeverity = AlertSeverity.MEDIUM
    message: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    annotations: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "id": self.id,
            "rule": self.rule.name,
            "value": self.value,
            "status": self.status.value,
            "severity": self.severity.name,
            "message": self.message,
            "started_at": self.started_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "duration": str(datetime.now() - self.started_at) if self.status == AlertStatus.FIRING else None,
            "annotations": self.annotations,
            "labels": self.labels
        }


@dataclass
class AlertChannel:
    """
    Alert notification channel.
    
    Attributes:
        name: Channel name
        type: Channel type (email, slack, webhook, etc.)
        config: Channel configuration
        severity_filter: Minimum severity to send
        enabled: Whether channel is active
    """
    name: str
    type: str
    config: Dict[str, Any]
    severity_filter: AlertSeverity = AlertSeverity.MEDIUM
    enabled: bool = True


class AlertingService(BaseService):
    """
    Service for managing alerts and notifications.
    
    This service monitors metrics, evaluates alert rules, and manages
    alert lifecycle with intelligent routing and deduplication.
    """
    
    def __init__(
        self,
        config: Optional[ServiceConfig] = None,
        evaluation_interval: int = 15,
        alert_retention_days: int = 30
    ):
        """
        Initialize alerting service.
        
        Args:
            config: Service configuration
            evaluation_interval: Rule evaluation interval in seconds
            alert_retention_days: Days to retain alert history
        """
        if config is None:
            config = ServiceConfig(name="alerting_service")
        super().__init__(config)
        
        self.evaluation_interval = evaluation_interval
        self.alert_retention = timedelta(days=alert_retention_days)
        
        # Alert storage
        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        
        # Alert channels
        self._channels: Dict[str, AlertChannel] = {}
        
        # Deduplication
        self._alert_fingerprints: Dict[str, datetime] = {}
        self._dedup_window = timedelta(minutes=5)
        
        # Silence rules
        self._silences: Dict[str, Dict[str, Any]] = {}
        
        # Background tasks
        self._evaluation_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Notification handlers
        self._notification_handlers: Dict[str, Callable] = {}
        
        # Metrics service reference
        self._metrics_service = None
        
        self.logger = get_logger("service.alerting")
    
    async def _initialize(self) -> None:
        """Initialize alerting service."""
        self.logger.info("Initializing alerting service")
        
        # Get metrics service
        from src.services.service_registry import registry
        self._metrics_service = registry.get("metrics_service")
        
        # Register default alert rules
        self._register_default_rules()
        
        # Start background tasks
        self._evaluation_task = asyncio.create_task(self._evaluation_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _start(self) -> None:
        """Start alerting service."""
        self.logger.info("Alerting service started")
    
    async def _stop(self) -> None:
        """Stop alerting service."""
        # Cancel background tasks
        for task in [self._evaluation_task, self._cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.logger.info("Alerting service stopped")
    
    async def _cleanup(self) -> None:
        """Cleanup alerting resources."""
        self._rules.clear()
        self._active_alerts.clear()
        self._channels.clear()
    
    async def _check_health(self) -> bool:
        """Check alerting service health."""
        return len(self._active_alerts) < 1000  # Max active alerts
    
    def _register_default_rules(self) -> None:
        """Register default alert rules."""
        # High CPU usage
        self.register_rule(
            AlertRule(
                name="high_cpu_usage",
                metric="cpu_usage",
                condition=lambda v, t: v > t,
                threshold=80.0,
                severity=AlertSeverity.HIGH,
                duration=timedelta(minutes=5),
                annotations={"description": "CPU usage is above 80%"}
            )
        )
        
        # High memory usage
        self.register_rule(
            AlertRule(
                name="high_memory_usage",
                metric="memory_usage",
                condition=lambda v, t: v > t,
                threshold=85.0,
                severity=AlertSeverity.HIGH,
                duration=timedelta(minutes=3),
                annotations={"description": "Memory usage is above 85%"}
            )
        )
        
        # Low accuracy
        self.register_rule(
            AlertRule(
                name="low_model_accuracy",
                metric="model_accuracy",
                condition=lambda v, t: v < t,
                threshold=90.0,
                severity=AlertSeverity.MEDIUM,
                duration=timedelta(minutes=10),
                annotations={"description": "Model accuracy dropped below 90%"}
            )
        )
        
        # High request latency
        self.register_rule(
            AlertRule(
                name="high_request_latency",
                metric="request_latency",
                condition=lambda v, t: v > t,
                threshold=1000.0,  # milliseconds
                severity=AlertSeverity.HIGH,
                duration=timedelta(minutes=2),
                annotations={"description": "Request latency above 1 second"}
            )
        )
    
    def register_rule(self, rule: AlertRule) -> None:
        """
        Register an alert rule.
        
        Args:
            rule: Alert rule to register
        """
        self._rules[rule.name] = rule
        self.logger.info(f"Registered alert rule: {rule.name}")
    
    def register_channel(self, channel: AlertChannel) -> None:
        """
        Register a notification channel.
        
        Args:
            channel: Notification channel to register
        """
        self._channels[channel.name] = channel
        
        # Register handler based on channel type
        if channel.type == "webhook":
            self._notification_handlers[channel.name] = self._send_webhook
        elif channel.type == "email":
            self._notification_handlers[channel.name] = self._send_email
        elif channel.type == "slack":
            self._notification_handlers[channel.name] = self._send_slack
        
        self.logger.info(f"Registered notification channel: {channel.name} ({channel.type})")
    
    async def _evaluation_loop(self) -> None:
        """Background loop for evaluating alert rules."""
        pending_alerts = defaultdict(list)  # Track pending alerts for duration check
        
        while True:
            try:
                await asyncio.sleep(self.evaluation_interval)
                
                if not self._metrics_service:
                    continue
                
                # Evaluate each rule
                for rule_name, rule in self._rules.items():
                    try:
                        # Get metric value
                        metric = self._metrics_service.get_metric(rule.metric)
                        if not metric:
                            continue
                        
                        value = metric.value
                        
                        # Evaluate condition
                        if rule.evaluate(value):
                            # Add to pending if not already
                            pending_key = f"{rule_name}:{rule.metric}"
                            
                            if pending_key not in pending_alerts:
                                pending_alerts[pending_key].append(datetime.now())
                            
                            # Check if duration threshold met
                            first_occurrence = pending_alerts[pending_key][0]
                            if datetime.now() - first_occurrence >= rule.duration:
                                # Fire alert
                                await self._fire_alert(rule, value)
                                # Clear pending
                                del pending_alerts[pending_key]
                        else:
                            # Condition not met, clear pending
                            pending_key = f"{rule_name}:{rule.metric}"
                            if pending_key in pending_alerts:
                                del pending_alerts[pending_key]
                            
                            # Resolve active alert if exists
                            alert_id = self._generate_alert_id(rule, {})
                            if alert_id in self._active_alerts:
                                await self._resolve_alert(alert_id)
                    
                    except Exception as e:
                        self.logger.error(f"Error evaluating rule {rule_name}: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Evaluation loop error: {e}")
    
    async def _fire_alert(self, rule: AlertRule, value: float) -> None:
        """
        Fire an alert.
        
        Args:
            rule: Alert rule that triggered
            value: Metric value
        """
        # Generate alert ID
        alert_id = self._generate_alert_id(rule, {"value": value})
        
        # Check if already firing
        if alert_id in self._active_alerts:
            return
        
        # Check for deduplication
        if self._is_duplicate(alert_id):
            self.logger.debug(f"Suppressing duplicate alert: {rule.name}")
            return
        
        # Check for silences
        if self._is_silenced(rule):
            self.logger.debug(f"Alert silenced: {rule.name}")
            return
        
        # Create alert
        alert = Alert(
            id=alert_id,
            rule=rule,
            value=value,
            severity=rule.severity,
            message=f"{rule.name}: {rule.annotations.get('description', '')}",
            annotations=rule.annotations,
            labels=rule.labels
        )
        
        # Store alert
        self._active_alerts[alert_id] = alert
        self._alert_history.append(alert)
        
        # Update fingerprint
        self._alert_fingerprints[alert_id] = datetime.now()
        
        self.logger.warning(
            f"Alert fired: {rule.name} (severity: {rule.severity.name}, value: {value})"
        )
        
        # Send notifications
        await self._send_notifications(alert)
    
    async def _resolve_alert(self, alert_id: str) -> None:
        """
        Resolve an active alert.
        
        Args:
            alert_id: Alert identifier
        """
        if alert_id not in self._active_alerts:
            return
        
        alert = self._active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        
        # Remove from active alerts
        del self._active_alerts[alert_id]
        
        self.logger.info(f"Alert resolved: {alert.rule.name}")
        
        # Send resolution notification
        await self._send_notifications(alert, is_resolution=True)
    
    async def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str
    ) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert identifier
            acknowledged_by: User acknowledging the alert
            
        Returns:
            bool: True if acknowledged successfully
        """
        if alert_id not in self._active_alerts:
            return False
        
        alert = self._active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now()
        alert.acknowledged_by = acknowledged_by
        
        self.logger.info(f"Alert acknowledged: {alert.rule.name} by {acknowledged_by}")
        return True
    
    def silence_alert(
        self,
        rule_name: str,
        duration: timedelta,
        reason: str = ""
    ) -> str:
        """
        Silence alerts for a rule.
        
        Args:
            rule_name: Rule to silence
            duration: Silence duration
            reason: Reason for silencing
            
        Returns:
            str: Silence ID
        """
        silence_id = f"silence_{datetime.now().timestamp()}"
        
        self._silences[silence_id] = {
            "rule_name": rule_name,
            "start": datetime.now(),
            "end": datetime.now() + duration,
            "reason": reason
        }
        
        self.logger.info(f"Silenced rule {rule_name} for {duration}")
        return silence_id
    
    def _generate_alert_id(self, rule: AlertRule, context: Dict[str, Any]) -> str:
        """
        Generate unique alert ID.
        
        Args:
            rule: Alert rule
            context: Alert context
            
        Returns:
            str: Alert ID
        """
        # Create fingerprint from rule and labels
        fingerprint_data = {
            "rule": rule.name,
            "labels": rule.labels
        }
        
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()
    
    def _is_duplicate(self, alert_id: str) -> bool:
        """
        Check if alert is a duplicate within dedup window.
        
        Args:
            alert_id: Alert identifier
            
        Returns:
            bool: True if duplicate
        """
        if alert_id in self._alert_fingerprints:
            last_seen = self._alert_fingerprints[alert_id]
            if datetime.now() - last_seen < self._dedup_window:
                return True
        return False
    
    def _is_silenced(self, rule: AlertRule) -> bool:
        """
        Check if rule is silenced.
        
        Args:
            rule: Alert rule
            
        Returns:
            bool: True if silenced
        """
        now = datetime.now()
        
        for silence in self._silences.values():
            if silence["rule_name"] == rule.name:
                if silence["start"] <= now <= silence["end"]:
                    return True
        
        return False
    
    async def _send_notifications(
        self,
        alert: Alert,
        is_resolution: bool = False
    ) -> None:
        """
        Send alert notifications to configured channels.
        
        Args:
            alert: Alert to notify about
            is_resolution: Whether this is a resolution notification
        """
        for channel_name, channel in self._channels.items():
            # Check if channel is enabled
            if not channel.enabled:
                continue
            
            # Check severity filter
            if alert.severity.value > channel.severity_filter.value:
                continue
            
            # Get handler
            handler = self._notification_handlers.get(channel_name)
            if not handler:
                continue
            
            try:
                await handler(channel, alert, is_resolution)
            except Exception as e:
                self.logger.error(f"Failed to send notification to {channel_name}: {e}")
    
    async def _send_webhook(
        self,
        channel: AlertChannel,
        alert: Alert,
        is_resolution: bool
    ) -> None:
        """
        Send webhook notification.
        
        Args:
            channel: Webhook channel
            alert: Alert to send
            is_resolution: Whether this is a resolution
        """
        import aiohttp
        
        url = channel.config.get("url")
        if not url:
            return
        
        payload = {
            "alert": alert.to_dict(),
            "is_resolution": is_resolution,
            "timestamp": datetime.now().isoformat()
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=channel.config.get("headers", {})
            ) as response:
                if response.status != 200:
                    self.logger.error(f"Webhook failed: {response.status}")
    
    async def _send_email(
        self,
        channel: AlertChannel,
        alert: Alert,
        is_resolution: bool
    ) -> None:
        """
        Send email notification.
        
        Args:
            channel: Email channel
            alert: Alert to send
            is_resolution: Whether this is a resolution
        """
        # Email implementation would go here
        self.logger.info(f"Email notification: {alert.rule.name} to {channel.config.get('to')}")
    
    async def _send_slack(
        self,
        channel: AlertChannel,
        alert: Alert,
        is_resolution: bool
    ) -> None:
        """
        Send Slack notification.
        
        Args:
            channel: Slack channel
            alert: Alert to send
            is_resolution: Whether this is a resolution
        """
        # Slack implementation would go here
        self.logger.info(f"Slack notification: {alert.rule.name} to {channel.config.get('channel')}")
    
    async def _cleanup_loop(self) -> None:
        """Background loop for cleaning up old alerts."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run hourly
                
                # Clean old alerts from history
                cutoff = datetime.now() - self.alert_retention
                self._alert_history = [
                    alert for alert in self._alert_history
                    if alert.started_at > cutoff
                ]
                
                # Clean old fingerprints
                self._alert_fingerprints = {
                    k: v for k, v in self._alert_fingerprints.items()
                    if v > cutoff
                }
                
                # Clean expired silences
                now = datetime.now()
                self._silences = {
                    k: v for k, v in self._silences.items()
                    if v["end"] > now
                }
                
                self.logger.info("Cleaned up old alerts and silences")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """
        Get all active alerts.
        
        Returns:
            List[Alert]: Active alerts
        """
        return list(self._active_alerts.values())
    
    def get_alert_history(
        self,
        hours: int = 24,
        severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """
        Get alert history.
        
        Args:
            hours: Hours of history to retrieve
            severity: Filter by severity
            
        Returns:
            List[Alert]: Historical alerts
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        alerts = [
            alert for alert in self._alert_history
            if alert.started_at > cutoff
        ]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return alerts
