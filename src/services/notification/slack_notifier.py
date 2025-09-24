"""
Slack Notifier Implementation for AG News Text Classification
================================================================================
This module implements Slack notification delivery using the Slack Web API,
supporting rich message formatting, channels, and direct messages.

The Slack notifier provides:
- Channel and direct message support
- Rich message formatting with blocks
- Thread replies
- File attachments

References:
    - Slack API Documentation
    - Slack Block Kit Builder
    - Slack App Best Practices

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
import aiohttp
import json
from typing import Optional, Dict, Any, List
from datetime import datetime

from src.services.notification.notification_service import NotificationProvider, Notification
from src.utils.logging_config import get_logger


class SlackNotifier(NotificationProvider):
    """
    Slack notification provider using Slack Web API.
    
    This provider handles message delivery to Slack channels
    and users with rich formatting support.
    """
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        webhook_url: Optional[str] = None,
        default_channel: str = "#general",
        use_blocks: bool = True,
        max_text_length: int = 3000
    ):
        """
        Initialize Slack notifier.
        
        Args:
            bot_token: Slack bot token for API access
            webhook_url: Incoming webhook URL (alternative to bot token)
            default_channel: Default channel for notifications
            use_blocks: Use Slack blocks for rich formatting
            max_text_length: Maximum text length per message
        """
        self.bot_token = bot_token
        self.webhook_url = webhook_url
        self.default_channel = default_channel
        self.use_blocks = use_blocks
        self.max_text_length = max_text_length
        
        # Slack API endpoints
        self.api_base = "https://slack.com/api"
        self.post_message_url = f"{self.api_base}/chat.postMessage"
        self.upload_file_url = f"{self.api_base}/files.upload"
        
        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None
        
        self.logger = get_logger("notifier.slack")
    
    async def send(self, notification: Notification) -> bool:
        """
        Send Slack notification.
        
        Args:
            notification: Notification to send
            
        Returns:
            bool: True if sent successfully
        """
        try:
            # Determine target (channel or user)
            target = await self._resolve_target(notification.recipient)
            
            if not target:
                self.logger.error(f"Invalid Slack target: {notification.recipient}")
                return False
            
            # Create message payload
            payload = await self._create_payload(notification, target)
            
            # Send message
            if self.webhook_url:
                success = await self._send_webhook(payload)
            else:
                success = await self._send_api(payload)
            
            if success:
                
