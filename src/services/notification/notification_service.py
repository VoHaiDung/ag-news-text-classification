"""
Notification Service Implementation for AG News Text Classification
================================================================================
This module implements a unified notification service for managing multi-channel
notifications with template support and delivery tracking.

The notification service provides:
- Multi-channel notification routing
- Template-based message formatting
- Priority-based delivery
- Retry logic and failure handling

References:
    - Martin Fowler (2002). Patterns of Enterprise Application Architecture
    - Notification Design Patterns
    - Microservices Communication Patterns

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import json
import re

from src.services.base_service import BaseService, ServiceConfig
from src.core.exceptions import ServiceException
from src.utils.logging_config import get_logger


class NotificationChannel(Enum):
    """
    Available notification channels.
    
    Channels:
        EMAIL: Email notifications
        SLACK: Slack messaging
        WEBHOOK: HTTP webhooks
        SMS: SMS messaging
        PUSH: Push notifications
    """
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PUSH = "push"


class NotificationPriority(Enum):
    """
    Notification priority levels.
    
    Priorities:
        CRITICAL: Immediate delivery required
        HIGH: High priority delivery
        NORMAL: Standard delivery
        LOW: Low priority delivery
    """
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class NotificationStatus(Enum):
    """
    Notification delivery status.
    
    States:
        PENDING: Awaiting delivery
        SENDING: Currently being sent
        DELIVERED: Successfully delivered
        FAILED: Delivery failed
        RETRYING: Retrying delivery
    """
    PENDING = "pending"
    SENDING = "sending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class NotificationTemplate:
    """
    Notification message template.
    
    Attributes:
        name: Template name
        subject: Message subject template
        body: Message body template
        html_body: HTML body template (optional)
        variables: Required template variables
        channel_specific: Channel-specific templates
    """
    name: str
    subject: str
    body: str
    html_body: Optional[str] = None
    variables: Set[str] = field(default_factory=set)
    channel_specific: Dict[NotificationChannel, Dict[str, str]] = field(default_factory=dict)
    
    def render(self, variables: Dict[str, Any]) -> Dict[str, str]:
        """
        Render template with variables.
        
        Args:
            variables: Template variables
            
        Returns:
            Dict[str, str]: Rendered content
        """
        # Simple variable substitution using regex
        pattern = re.compile(r'\{\{(\w+)\}\}')
        
        def replace_var(match):
            var_name = match.group(1)
            return str(variables.get(var_name, match.group(0)))
        
        rendered = {
            "subject": pattern.sub(replace_var, self.subject),
            "body": pattern.sub(replace_var, self.body)
        }
        
        if self.html_body:
            rendered["html_body"] = pattern.sub(replace_var, self.html_body)
        
        return rendered


@dataclass
class Notification:
    """
    Notification instance.
    
    Attributes:
        id: Unique notification ID
        channels: Target channels
        recipient: Recipient identifier
        subject: Notification subject
        body: Notification body
        priority: Delivery priority
        status: Current status
        template: Template used
        variables: Template variables
        metadata: Additional metadata
        created_at: Creation timestamp
        sent_at: Send timestamp
        delivered_at: Delivery timestamp
        retry_count: Number of retry attempts
        error: Error message if failed
    """
    id: str
    channels: List[NotificationChannel]
    recipient: str
    subject: str
    body: str
    priority: NotificationPriority = NotificationPriority.NORMAL
    status: NotificationStatus = NotificationStatus.PENDING
    template: Optional[str] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    retry_count: int = 0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert notification to dictionary."""
        return {
            "id": self.id,
            "channels": [ch.value for ch in self.channels],
            "recipient": self.recipient,
            "subject": self.subject,
            "priority": self.priority.name,
            "status": self.status.value,
            "template": self.template,
            "created_at": self.created_at.isoformat(),
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "retry_count": self.retry_count,
            "error": self.error
        }


class NotificationProvider(ABC):
    """Abstract base class for notification providers."""
    
    @abstractmethod
    async def send(self, notification: Notification) -> bool:
        """Send notification."""
        pass
    
    @abstractmethod
    async def validate_recipient(self, recipient: str) -> bool:
        """Validate recipient format."""
        pass


class NotificationService(BaseService):
    """
    Unified notification service for multi-channel messaging.
    
    This service manages notification delivery across multiple channels
    with template support, priority routing, and delivery tracking.
    """
    
    def __init__(
        self,
        config: Optional[ServiceConfig] = None,
        max_retries: int = 3,
        retry_delay: int = 60,
        enable_batching: bool = True
    ):
        """
        Initialize notification service.
        
        Args:
            config: Service configuration
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries (seconds)
            enable_batching: Enable notification batching
        """
        if config is None:
            config = ServiceConfig(name="notification_service")
        super().__init__(config)
        
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_batching = enable_batching
        
        # Notification providers
        self._providers: Dict[NotificationChannel, NotificationProvider] = {}
        
        # Templates
        self._templates: Dict[str, NotificationTemplate] = {}
        
        # Notification queues by priority
        self._queues: Dict[NotificationPriority, List[Notification]] = {
            priority: [] for priority in NotificationPriority
        }
        
        # Tracking
        self._sent_notifications: Dict[str, Notification] = {}
        self._failed_notifications: List[Notification] = []
        
        # Background tasks
        self._delivery_task: Optional[asyncio.Task] = None
        
        # Statistics
        self._stats = {
            "sent": 0,
            "delivered": 0,
            "failed": 0,
            "retried": 0
        }
        
        self.logger = get_logger("service.notification")
    
    async def _initialize(self) -> None:
        """Initialize notification service."""
        self.logger.info("Initializing notification service")
        
        # Initialize providers
        await self._initialize_providers()
        
        # Register default templates
        self._register_default_templates()
        
        # Start delivery task
        self._delivery_task = asyncio.create_task(self._delivery_loop())
    
    async def _start(self) -> None:
        """Start notification service."""
        self.logger.info("Notification service started")
    
    async def _stop(self) -> None:
        """Stop notification service."""
        # Cancel delivery task
        if self._delivery_task:
            self._delivery_task.cancel()
            try:
                await self._delivery_task
            except asyncio.CancelledError:
                pass
        
        # Send remaining notifications
        await self._flush_queues()
        
        self.logger.info("Notification service stopped")
    
    async def _cleanup(self) -> None:
        """Cleanup notification resources."""
        self._providers.clear()
        self._templates.clear()
        self._queues.clear()
    
    async def _check_health(self) -> bool:
        """Check notification service health."""
        # Check if queues are not overloaded
        total_queued = sum(len(q) for q in self._queues.values())
        return total_queued < 10000
    
    async def _initialize_providers(self) -> None:
        """Initialize notification providers."""
        # Initialize email provider
        from src.services.notification.email_notifier import EmailNotifier
        self._providers[NotificationChannel.EMAIL] = EmailNotifier()
        
        # Initialize Slack provider
        from src.services.notification.slack_notifier import SlackNotifier
        self._providers[NotificationChannel.SLACK] = SlackNotifier()
        
        # Initialize webhook provider
        from src.services.notification.webhook_notifier import WebhookNotifier
        self._providers[NotificationChannel.WEBHOOK] = WebhookNotifier()
        
        self.logger.info(f"Initialized {len(self._providers)} notification providers")
    
    def _register_default_templates(self) -> None:
        """Register default notification templates."""
        # Alert template
        self.register_template(
            NotificationTemplate(
                name="alert",
                subject="Alert: {{alert_type}}",
                body="Alert triggered: {{alert_type}}\n\nDetails: {{details}}\n\nTime: {{timestamp}}",
                variables={"alert_type", "details", "timestamp"}
            )
        )
        
        # Error template
        self.register_template(
            NotificationTemplate(
                name="error",
                subject="Error: {{error_type}}",
                body="An error occurred:\n\nType: {{error_type}}\nMessage: {{error_message}}\nTime: {{timestamp}}",
                variables={"error_type", "error_message", "timestamp"}
            )
        )
        
        # Success template
        self.register_template(
            NotificationTemplate(
                name="success",
                subject="Success: {{operation}}",
                body="Operation completed successfully:\n\n{{operation}}\n\nResult: {{result}}\nTime: {{timestamp}}",
                variables={"operation", "result", "timestamp"}
            )
        )
    
    def register_template(self, template: NotificationTemplate) -> None:
        """
        Register notification template.
        
        Args:
            template: Template to register
        """
        self._templates[template.name] = template
        self.logger.info(f"Registered template: {template.name}")
    
    def register_provider(
        self,
        channel: NotificationChannel,
        provider: NotificationProvider
    ) -> None:
        """
        Register notification provider.
        
        Args:
            channel: Notification channel
            provider: Provider instance
        """
        self._providers[channel] = provider
        self.logger.info(f"Registered provider for channel: {channel.value}")
    
    async def send(
        self,
        recipient: str,
        subject: str,
        body: str,
        channels: Optional[List[NotificationChannel]] = None,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        template: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Send notification.
        
        Args:
            recipient: Recipient identifier
            subject: Notification subject
            body: Notification body
            channels: Target channels (default: all available)
            priority: Delivery priority
            template: Template name to use
            variables: Template variables
            metadata: Additional metadata
            
        Returns:
            str: Notification ID
        """
        # Use template if specified
        if template and template in self._templates:
            template_obj = self._templates[template]
            if variables:
                rendered = template_obj.render(variables)
                subject = rendered.get("subject", subject)
                body = rendered.get("body", body)
        
        # Default to all available channels
        if not channels:
            channels = list(self._providers.keys())
        
        # Create notification
        import uuid
        notification = Notification(
            id=str(uuid.uuid4()),
            channels=channels,
            recipient=recipient,
            subject=subject,
            body=body,
            priority=priority,
            template=template,
            variables=variables or {},
            metadata=metadata or {}
        )
        
        # Add to queue
        self._queues[priority].append(notification)
        
        self.logger.info(
            f"Queued notification {notification.id} for {recipient} "
            f"(priority: {priority.name})"
        )
        
        return notification.id
    
    async def _delivery_loop(self) -> None:
        """Background loop for delivering notifications."""
        while True:
            try:
                # Process queues by priority
                for priority in sorted(NotificationPriority, key=lambda p: p.value):
                    queue = self._queues[priority]
                    
                    if self.enable_batching and len(queue) > 10:
                        # Process in batches
                        batch = queue[:10]
                        queue[:10] = []
                        
                        await asyncio.gather(*[
                            self._deliver_notification(n)
                            for n in batch
                        ], return_exceptions=True)
                    elif queue:
                        # Process one at a time
                        notification = queue.pop(0)
                        await self._deliver_notification(notification)
                
                await asyncio.sleep(1)  # Small delay
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Delivery loop error: {e}")
                await asyncio.sleep(5)
    
    async def _deliver_notification(self, notification: Notification) -> None:
        """
        Deliver a single notification.
        
        Args:
            notification: Notification to deliver
        """
        notification.status = NotificationStatus.SENDING
        notification.sent_at = datetime.now()
        
        success_channels = []
        failed_channels = []
        
        # Send to each channel
        for channel in notification.channels:
            if channel not in self._providers:
                self.logger.warning(f"No provider for channel: {channel.value}")
                failed_channels.append(channel)
                continue
            
            provider = self._providers[channel]
            
            try:
                success = await provider.send(notification)
                
                if success:
                    success_channels.append(channel)
                    self._stats["sent"] += 1
                else:
                    failed_channels.append(channel)
                    
            except Exception as e:
                self.logger.error(f"Failed to send via {channel.value}: {e}")
                failed_channels.append(channel)
                notification.error = str(e)
        
        # Update status
        if success_channels and not failed_channels:
            notification.status = NotificationStatus.DELIVERED
            notification.delivered_at = datetime.now()
            self._stats["delivered"] += 1
            
            self.logger.info(
                f"Delivered notification {notification.id} via "
                f"{[ch.value for ch in success_channels]}"
            )
        elif failed_channels and notification.retry_count < self.max_retries:
            # Retry failed channels
            notification.retry_count += 1
            notification.status = NotificationStatus.RETRYING
            notification.channels = failed_channels
            
            self._stats["retried"] += 1
            
            # Re-queue with delay
            await asyncio.sleep(self.retry_delay)
            self._queues[notification.priority].append(notification)
            
            self.logger.warning(
                f"Retrying notification {notification.id} "
                f"(attempt {notification.retry_count}/{self.max_retries})"
            )
        else:
            # Final failure
            notification.status = NotificationStatus.FAILED
            self._failed_notifications.append(notification)
            self._stats["failed"] += 1
            
            self.logger.error(
                f"Failed to deliver notification {notification.id} after "
                f"{notification.retry_count} retries"
            )
        
        # Track sent notification
        self._sent_notifications[notification.id] = notification
    
    async def _flush_queues(self) -> None:
        """Flush all notification queues."""
        for priority in NotificationPriority:
            queue = self._queues[priority]
            
            while queue:
                notification = queue.pop(0)
                await self._deliver_notification(notification)
    
    def get_notification_status(self, notification_id: str) -> Optional[Notification]:
        """
        Get notification status.
        
        Args:
            notification_id: Notification ID
            
        Returns:
            Optional[Notification]: Notification instance or None
        """
        return self._sent_notifications.get(notification_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get notification statistics.
        
        Returns:
            Dict[str, Any]: Notification statistics
        """
        total_queued = sum(len(q) for q in self._queues.values())
        
        return {
            **self._stats,
            "queued": total_queued,
            "success_rate": (
                self._stats["delivered"] / 
                max(self._stats["sent"], 1)
            ) * 100 if self._stats["sent"] > 0 else 0
        }
