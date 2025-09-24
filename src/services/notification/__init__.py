"""
Notification Services Module for AG News Text Classification
================================================================================
This module provides multi-channel notification capabilities for system alerts,
events, and communications across various platforms.

The notification services enable:
- Multi-channel message delivery
- Template-based notifications
- Priority-based routing
- Delivery tracking and retry logic

References:
    - Enterprise Integration Patterns (Hohpe & Woolf)
    - Notification System Design Patterns
    - AWS SNS Best Practices

Author: Võ Hải Dũng
License: MIT
"""

from src.services.notification.notification_service import (
    NotificationService,
    NotificationChannel,
    NotificationPriority,
    NotificationStatus,
    Notification,
    NotificationTemplate
)
from src.services.notification.email_notifier import EmailNotifier
from src.services.notification.slack_notifier import SlackNotifier
from src.services.notification.webhook_notifier import WebhookNotifier

__all__ = [
    # Main Service
    "NotificationService",
    "NotificationChannel",
    "NotificationPriority",
    "NotificationStatus",
    "Notification",
    "NotificationTemplate",
    
    # Notifiers
    "EmailNotifier",
    "SlackNotifier",
    "WebhookNotifier"
]

# Module version
__version__ = "1.0.0"
