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
- Rate limiting compliance
- Error recovery mechanisms

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
import os
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from src.services.notification.notification_service import (
    NotificationProvider, 
    Notification,
    NotificationPriority,
    NotificationStatus
)
from src.utils.logging_config import get_logger
from src.core.exceptions import ServiceException


class SlackMessageType(Enum):
    """Slack message types."""
    TEXT = "text"
    BLOCKS = "blocks"
    ATTACHMENT = "attachment"
    FILE = "file"


@dataclass
class SlackConfig:
    """Slack configuration parameters."""
    bot_token: Optional[str] = None
    webhook_url: Optional[str] = None
    app_token: Optional[str] = None
    signing_secret: Optional[str] = None
    default_channel: str = "#general"
    use_blocks: bool = True
    max_text_length: int = 3000
    max_blocks: int = 50
    rate_limit_delay: float = 1.0
    retry_attempts: int = 3
    retry_delay: float = 2.0
    timeout: float = 30.0


class SlackNotifier(NotificationProvider):
    """
    Slack notification provider using Slack Web API.
    
    This provider handles message delivery to Slack channels
    and users with rich formatting support.
    
    Attributes:
        config: Slack configuration
        session: Async HTTP session
        rate_limiter: Rate limiting tracker
    """
    
    def __init__(self, config: Optional[SlackConfig] = None):
        """
        Initialize Slack notifier.
        
        Args:
            config: Slack configuration object
        """
        self.config = config or SlackConfig()
        
        # Load from environment if not provided
        if not self.config.bot_token:
            self.config.bot_token = os.getenv("SLACK_BOT_TOKEN")
        if not self.config.webhook_url:
            self.config.webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        
        # Slack API endpoints
        self.api_base = "https://slack.com/api"
        self.post_message_url = f"{self.api_base}/chat.postMessage"
        self.post_ephemeral_url = f"{self.api_base}/chat.postEphemeral"
        self.update_message_url = f"{self.api_base}/chat.update"
        self.upload_file_url = f"{self.api_base}/files.upload"
        self.users_list_url = f"{self.api_base}/users.list"
        self.conversations_list_url = f"{self.api_base}/conversations.list"
        
        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self._last_request_time = 0
        self._request_count = 0
        
        # Cache for user and channel lookups
        self._user_cache: Dict[str, str] = {}
        self._channel_cache: Dict[str, str] = {}
        
        self.logger = get_logger("services.notification.slack")
    
    async def initialize(self) -> None:
        """Initialize Slack notifier resources."""
        if not self._session:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
        
        # Validate credentials
        if self.config.bot_token:
            await self._validate_bot_token()
        
        # Populate caches
        await self._populate_caches()
    
    async def cleanup(self) -> None:
        """Cleanup Slack notifier resources."""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def send(self, notification: Notification) -> bool:
        """
        Send Slack notification.
        
        Args:
            notification: Notification to send
            
        Returns:
            bool: True if sent successfully
        """
        try:
            # Initialize session if needed
            if not self._session:
                await self.initialize()
            
            # Apply rate limiting
            await self._apply_rate_limit()
            
            # Determine target (channel or user)
            target = await self._resolve_target(notification.recipient)
            
            if not target:
                self.logger.error(f"Invalid Slack target: {notification.recipient}")
                return False
            
            # Create message payload
            payload = await self._create_payload(notification, target)
            
            # Send message with retries
            success = await self._send_with_retry(payload)
            
            if success:
                self.logger.info(
                    f"Slack notification sent to {target} "
                    f"[Priority: {notification.priority.value}]"
                )
            else:
                self.logger.error(f"Failed to send Slack notification to {target}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending Slack notification: {str(e)}")
            return False
    
    async def send_batch(self, notifications: List[Notification]) -> List[bool]:
        """
        Send multiple Slack notifications.
        
        Args:
            notifications: List of notifications to send
            
        Returns:
            List of success statuses
        """
        results = []
        
        for notification in notifications:
            # Add delay between messages to avoid rate limiting
            if results:
                await asyncio.sleep(self.config.rate_limit_delay)
            
            success = await self.send(notification)
            results.append(success)
        
        return results
    
    async def _validate_bot_token(self) -> bool:
        """
        Validate Slack bot token.
        
        Returns:
            bool: True if token is valid
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.config.bot_token}",
                "Content-Type": "application/json"
            }
            
            async with self._session.post(
                f"{self.api_base}/auth.test",
                headers=headers
            ) as response:
                data = await response.json()
                
                if data.get("ok"):
                    self.logger.info(f"Slack bot authenticated as: {data.get('user')}")
                    return True
                else:
                    self.logger.error(f"Slack authentication failed: {data.get('error')}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error validating Slack token: {str(e)}")
            return False
    
    async def _populate_caches(self) -> None:
        """Populate user and channel caches."""
        if not self.config.bot_token:
            return
        
        try:
            # Populate user cache
            await self._cache_users()
            
            # Populate channel cache
            await self._cache_channels()
            
        except Exception as e:
            self.logger.warning(f"Error populating caches: {str(e)}")
    
    async def _cache_users(self) -> None:
        """Cache Slack users."""
        headers = {
            "Authorization": f"Bearer {self.config.bot_token}",
            "Content-Type": "application/json"
        }
        
        async with self._session.get(
            self.users_list_url,
            headers=headers
        ) as response:
            data = await response.json()
            
            if data.get("ok"):
                for user in data.get("members", []):
                    if not user.get("deleted", False):
                        self._user_cache[user["name"]] = user["id"]
                        if user.get("real_name"):
                            self._user_cache[user["real_name"]] = user["id"]
    
    async def _cache_channels(self) -> None:
        """Cache Slack channels."""
        headers = {
            "Authorization": f"Bearer {self.config.bot_token}",
            "Content-Type": "application/json"
        }
        
        async with self._session.get(
            self.conversations_list_url,
            headers=headers
        ) as response:
            data = await response.json()
            
            if data.get("ok"):
                for channel in data.get("channels", []):
                    self._channel_cache[channel["name"]] = channel["id"]
                    self._channel_cache[f"#{channel['name']}"] = channel["id"]
    
    async def _resolve_target(self, recipient: str) -> Optional[str]:
        """
        Resolve recipient to Slack channel or user ID.
        
        Args:
            recipient: Recipient identifier
            
        Returns:
            Resolved Slack ID or None
        """
        # Check if already a valid ID
        if recipient.startswith(("C", "U", "D", "G")):
            return recipient
        
        # Check if it's a channel
        if recipient.startswith("#"):
            # Look up in cache
            if recipient in self._channel_cache:
                return self._channel_cache[recipient]
            
            # Try without hash
            channel_name = recipient[1:]
            if channel_name in self._channel_cache:
                return self._channel_cache[channel_name]
            
            # Use as-is for public channels
            return recipient
        
        # Check if it's a user mention
        if recipient.startswith("@"):
            username = recipient[1:]
            if username in self._user_cache:
                return self._user_cache[username]
        
        # Check user cache directly
        if recipient in self._user_cache:
            return self._user_cache[recipient]
        
        # Default to configured channel
        return self.config.default_channel
    
    async def _create_payload(
        self,
        notification: Notification,
        target: str
    ) -> Dict[str, Any]:
        """
        Create Slack message payload.
        
        Args:
            notification: Notification object
            target: Target channel or user
            
        Returns:
            Slack API payload
        """
        payload = {
            "channel": target,
            "text": notification.message
        }
        
        # Add blocks for rich formatting if enabled
        if self.config.use_blocks and notification.metadata:
            blocks = await self._create_blocks(notification)
            if blocks:
                payload["blocks"] = blocks
        
        # Add thread timestamp if replying to thread
        if notification.metadata and "thread_ts" in notification.metadata:
            payload["thread_ts"] = notification.metadata["thread_ts"]
        
        # Add attachments if provided
        if notification.metadata and "attachments" in notification.metadata:
            payload["attachments"] = notification.metadata["attachments"]
        
        # Set message visibility
        if notification.metadata and notification.metadata.get("ephemeral"):
            payload["user"] = notification.metadata.get("user_id")
        
        return payload
    
    async def _create_blocks(self, notification: Notification) -> List[Dict[str, Any]]:
        """
        Create Slack blocks for rich formatting.
        
        Args:
            notification: Notification object
            
        Returns:
            List of Slack blocks
        """
        blocks = []
        
        # Header block
        header_text = notification.metadata.get("title", "Notification")
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": header_text[:150]  # Slack limit
            }
        })
        
        # Priority indicator
        priority_emoji = {
            NotificationPriority.LOW: ":white_circle:",
            NotificationPriority.MEDIUM: ":large_blue_circle:",
            NotificationPriority.HIGH: ":large_orange_circle:",
            NotificationPriority.CRITICAL: ":red_circle:"
        }
        
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"{priority_emoji.get(notification.priority, '')} "
                           f"*Priority:* {notification.priority.value}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                }
            ]
        })
        
        # Main message
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": notification.message[:self.config.max_text_length]
            }
        })
        
        # Additional fields
        if notification.metadata:
            fields = []
            
            for key, value in notification.metadata.items():
                if key not in ["title", "attachments", "thread_ts", "ephemeral", "user_id"]:
                    fields.append({
                        "type": "mrkdwn",
                        "text": f"*{key}:*\n{str(value)[:100]}"
                    })
            
            if fields:
                blocks.append({
                    "type": "section",
                    "fields": fields[:10]  # Slack limit
                })
        
        # Actions block if provided
        if notification.metadata and "actions" in notification.metadata:
            action_elements = []
            
            for action in notification.metadata["actions"][:5]:  # Slack limit
                action_elements.append({
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": action.get("text", "Action")
                    },
                    "action_id": action.get("action_id", "button"),
                    "url": action.get("url")
                })
            
            if action_elements:
                blocks.append({
                    "type": "actions",
                    "elements": action_elements
                })
        
        # Divider
        blocks.append({"type": "divider"})
        
        return blocks[:self.config.max_blocks]
    
    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting to avoid API throttling."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self.config.rate_limit_delay:
            await asyncio.sleep(self.config.rate_limit_delay - time_since_last)
        
        self._last_request_time = asyncio.get_event_loop().time()
        self._request_count += 1
    
    async def _send_with_retry(self, payload: Dict[str, Any]) -> bool:
        """
        Send message with retry logic.
        
        Args:
            payload: Message payload
            
        Returns:
            bool: True if sent successfully
        """
        for attempt in range(self.config.retry_attempts):
            try:
                if self.config.webhook_url:
                    success = await self._send_webhook(payload)
                else:
                    success = await self._send_api(payload)
                
                if success:
                    return True
                
                # Wait before retry
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    
            except Exception as e:
                self.logger.warning(f"Send attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        
        return False
    
    async def _send_webhook(self, payload: Dict[str, Any]) -> bool:
        """
        Send message via webhook.
        
        Args:
            payload: Message payload
            
        Returns:
            bool: True if sent successfully
        """
        try:
            # Webhook payloads are simpler
            webhook_payload = {
                "text": payload.get("text"),
                "blocks": payload.get("blocks"),
                "attachments": payload.get("attachments")
            }
            
            # Remove None values
            webhook_payload = {k: v for k, v in webhook_payload.items() if v is not None}
            
            async with self._session.post(
                self.config.webhook_url,
                json=webhook_payload
            ) as response:
                return response.status == 200
                
        except Exception as e:
            self.logger.error(f"Webhook send error: {str(e)}")
            return False
    
    async def _send_api(self, payload: Dict[str, Any]) -> bool:
        """
        Send message via Slack API.
        
        Args:
            payload: Message payload
            
        Returns:
            bool: True if sent successfully
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.config.bot_token}",
                "Content-Type": "application/json"
            }
            
            # Determine endpoint
            if payload.get("user") and "ephemeral" in str(payload):
                endpoint = self.post_ephemeral_url
            else:
                endpoint = self.post_message_url
            
            async with self._session.post(
                endpoint,
                headers=headers,
                json=payload
            ) as response:
                data = await response.json()
                
                if data.get("ok"):
                    return True
                else:
                    self.logger.error(f"Slack API error: {data.get('error')}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"API send error: {str(e)}")
            return False
    
    async def update_message(
        self,
        channel: str,
        timestamp: str,
        text: Optional[str] = None,
        blocks: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Update an existing Slack message.
        
        Args:
            channel: Channel ID
            timestamp: Message timestamp
            text: New text content
            blocks: New blocks content
            
        Returns:
            bool: True if updated successfully
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.config.bot_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "channel": channel,
                "ts": timestamp
            }
            
            if text:
                payload["text"] = text
            if blocks:
                payload["blocks"] = blocks
            
            async with self._session.post(
                self.update_message_url,
                headers=headers,
                json=payload
            ) as response:
                data = await response.json()
                return data.get("ok", False)
                
        except Exception as e:
            self.logger.error(f"Error updating message: {str(e)}")
            return False
    
    async def upload_file(
        self,
        channel: str,
        file_path: str,
        title: Optional[str] = None,
        comment: Optional[str] = None
    ) -> bool:
        """
        Upload a file to Slack.
        
        Args:
            channel: Target channel
            file_path: Path to file
            title: File title
            comment: Initial comment
            
        Returns:
            bool: True if uploaded successfully
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.config.bot_token}"
            }
            
            with open(file_path, "rb") as file:
                data = aiohttp.FormData()
                data.add_field("channels", channel)
                data.add_field("file", file, filename=os.path.basename(file_path))
                
                if title:
                    data.add_field("title", title)
                if comment:
                    data.add_field("initial_comment", comment)
                
                async with self._session.post(
                    self.upload_file_url,
                    headers=headers,
                    data=data
                ) as response:
                    result = await response.json()
                    return result.get("ok", False)
                    
        except Exception as e:
            self.logger.error(f"Error uploading file: {str(e)}")
            return False
