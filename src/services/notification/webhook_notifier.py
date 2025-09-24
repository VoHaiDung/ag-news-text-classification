"""
Webhook Notifier Implementation for AG News Text Classification
================================================================================
This module implements generic webhook notification delivery supporting
various webhook endpoints with customizable payloads and authentication.

The webhook notifier provides:
- Generic webhook support
- Custom payload formatting
- Multiple authentication methods
- Retry logic with exponential backoff
- Request/response logging

References:
    - Webhook Best Practices
    - HTTP Authentication Schemes (RFC 7235)
    - REST API Design Guidelines

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
import aiohttp
import json
import hmac
import hashlib
from typing import Optional, Dict, Any, List, Union, Callable
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


class WebhookAuthType(Enum):
    """Webhook authentication types."""
    NONE = "none"
    BEARER = "bearer"
    BASIC = "basic"
    API_KEY = "api_key"
    HMAC = "hmac"
    CUSTOM = "custom"


class WebhookMethod(Enum):
    """HTTP methods for webhook requests."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"


@dataclass
class WebhookConfig:
    """Webhook configuration parameters."""
    url: str
    method: WebhookMethod = WebhookMethod.POST
    auth_type: WebhookAuthType = WebhookAuthType.NONE
    auth_credentials: Optional[Dict[str, str]] = None
    headers: Optional[Dict[str, str]] = None
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    verify_ssl: bool = True
    follow_redirects: bool = True
    max_redirects: int = 5


class WebhookNotifier(NotificationProvider):
    """
    Generic webhook notification provider.
    
    This provider handles notification delivery to any webhook endpoint
    with customizable authentication and payload formatting.
    
    Attributes:
        config: Webhook configuration
        session: Async HTTP session
        payload_formatter: Custom payload formatting function
    """
    
    def __init__(
        self,
        config: WebhookConfig,
        payload_formatter: Optional[Callable[[Notification], Dict[str, Any]]] = None
    ):
        """
        Initialize webhook notifier.
        
        Args:
            config: Webhook configuration
            payload_formatter: Optional custom payload formatter
        """
        self.config = config
        self.payload_formatter = payload_formatter or self._default_payload_formatter
        
        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Statistics
        self._request_count = 0
        self._success_count = 0
        self._failure_count = 0
        
        self.logger = get_logger("services.notification.webhook")
    
    async def initialize(self) -> None:
        """Initialize webhook notifier resources."""
        if not self._session:
            connector = aiohttp.TCPConnector(
                ssl=self.config.verify_ssl,
                limit=100,
                limit_per_host=30
            )
            
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
    
    async def cleanup(self) -> None:
        """Cleanup webhook notifier resources."""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def send(self, notification: Notification) -> bool:
        """
        Send webhook notification.
        
        Args:
            notification: Notification to send
            
        Returns:
            bool: True if sent successfully
        """
        try:
            # Initialize session if needed
            if not self._session:
                await self.initialize()
            
            # Format payload
            payload = self.payload_formatter(notification)
            
            # Prepare request
            request_data = await self._prepare_request(payload, notification)
            
            # Send with retry logic
            success = await self._send_with_retry(request_data)
            
            # Update statistics
            self._request_count += 1
            if success:
                self._success_count += 1
                self.logger.info(
                    f"Webhook notification sent to {self.config.url} "
                    f"[Priority: {notification.priority.value}]"
                )
            else:
                self._failure_count += 1
                self.logger.error(f"Failed to send webhook notification to {self.config.url}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending webhook notification: {str(e)}")
            self._failure_count += 1
            return False
    
    async def send_batch(self, notifications: List[Notification]) -> List[bool]:
        """
        Send multiple webhook notifications.
        
        Args:
            notifications: List of notifications to send
            
        Returns:
            List of success statuses
        """
        tasks = []
        
        for notification in notifications:
            task = asyncio.create_task(self.send(notification))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to False
        return [
            result if isinstance(result, bool) else False
            for result in results
        ]
    
    def _default_payload_formatter(self, notification: Notification) -> Dict[str, Any]:
        """
        Default payload formatter.
        
        Args:
            notification: Notification object
            
        Returns:
            Formatted payload dictionary
        """
        payload = {
            "timestamp": datetime.now().isoformat(),
            "priority": notification.priority.value,
            "recipient": notification.recipient,
            "subject": notification.subject,
            "message": notification.message,
            "status": notification.status.value
        }
        
        # Add metadata if present
        if notification.metadata:
            payload["metadata"] = notification.metadata
        
        # Add retry information
        if notification.retry_count > 0:
            payload["retry_count"] = notification.retry_count
        
        return payload
    
    async def _prepare_request(
        self,
        payload: Dict[str, Any],
        notification: Notification
    ) -> Dict[str, Any]:
        """
        Prepare webhook request data.
        
        Args:
            payload: Formatted payload
            notification: Original notification
            
        Returns:
            Request data dictionary
        """
        # Base request data
        request_data = {
            "url": self.config.url,
            "method": self.config.method.value
        }
        
        # Headers
        headers = self.config.headers.copy() if self.config.headers else {}
        headers["Content-Type"] = "application/json"
        
        # Add authentication headers
        auth_headers = await self._get_auth_headers(payload, notification)
        headers.update(auth_headers)
        
        request_data["headers"] = headers
        
        # Body for POST/PUT/PATCH
        if self.config.method in [WebhookMethod.POST, WebhookMethod.PUT, WebhookMethod.PATCH]:
            request_data["json"] = payload
        else:
            # For GET, add as query parameters
            request_data["params"] = payload
        
        return request_data
    
    async def _get_auth_headers(
        self,
        payload: Dict[str, Any],
        notification: Notification
    ) -> Dict[str, str]:
        """
        Generate authentication headers.
        
        Args:
            payload: Request payload
            notification: Notification object
            
        Returns:
            Authentication headers
        """
        headers = {}
        
        if not self.config.auth_credentials:
            return headers
        
        if self.config.auth_type == WebhookAuthType.BEARER:
            token = self.config.auth_credentials.get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"
        
        elif self.config.auth_type == WebhookAuthType.BASIC:
            username = self.config.auth_credentials.get("username")
            password = self.config.auth_credentials.get("password")
            if username and password:
                import base64
                credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                headers["Authorization"] = f"Basic {credentials}"
        
        elif self.config.auth_type == WebhookAuthType.API_KEY:
            api_key = self.config.auth_credentials.get("api_key")
            header_name = self.config.auth_credentials.get("header_name", "X-API-Key")
            if api_key:
                headers[header_name] = api_key
        
        elif self.config.auth_type == WebhookAuthType.HMAC:
            secret = self.config.auth_credentials.get("secret")
            if secret:
                # Generate HMAC signature
                payload_str = json.dumps(payload, sort_keys=True)
                signature = hmac.new(
                    secret.encode(),
                    payload_str.encode(),
                    hashlib.sha256
                ).hexdigest()
                headers["X-Webhook-Signature"] = signature
        
        elif self.config.auth_type == WebhookAuthType.CUSTOM:
            # Custom authentication logic
            custom_headers = self.config.auth_credentials.get("custom_headers", {})
            headers.update(custom_headers)
        
        return headers
    
    async def _send_with_retry(self, request_data: Dict[str, Any]) -> bool:
        """
        Send request with retry logic.
        
        Args:
            request_data: Request data dictionary
            
        Returns:
            bool: True if sent successfully
        """
        last_exception = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                # Calculate delay for this attempt
                if attempt > 0:
                    if self.config.exponential_backoff:
                        delay = self.config.retry_delay * (2 ** (attempt - 1))
                    else:
                        delay = self.config.retry_delay
                    
                    self.logger.debug(f"Retry attempt {attempt + 1} after {delay}s delay")
                    await asyncio.sleep(delay)
                
                # Send request
                success = await self._send_request(request_data)
                
                if success:
                    return True
                    
            except Exception as e:
                last_exception = e
                self.logger.warning(
                    f"Request attempt {attempt + 1}/{self.config.retry_attempts} failed: {str(e)}"
                )
        
        if last_exception:
            self.logger.error(f"All retry attempts failed. Last error: {str(last_exception)}")
        
        return False
    
    async def _send_request(self, request_data: Dict[str, Any]) -> bool:
        """
        Send HTTP request to webhook.
        
        Args:
            request_data: Request data dictionary
            
        Returns:
            bool: True if request was successful
        """
        method = request_data["method"]
        url = request_data["url"]
        headers = request_data.get("headers", {})
        
        kwargs = {
            "headers": headers,
            "allow_redirects": self.config.follow_redirects,
            "max_redirects": self.config.max_redirects
        }
        
        # Add body or params
        if "json" in request_data:
            kwargs["json"] = request_data["json"]
        if "params" in request_data:
            kwargs["params"] = request_data["params"]
        
        async with self._session.request(method, url, **kwargs) as response:
            # Log request/response
            self.logger.debug(
                f"Webhook request: {method} {url} "
                f"Status: {response.status}"
            )
            
            # Check if successful (2xx status codes)
            if 200 <= response.status < 300:
                return True
            
            # Log error response
            try:
                error_body = await response.text()
                self.logger.error(
                    f"Webhook error response: Status={response.status}, "
                    f"Body={error_body[:500]}"
                )
            except:
                pass
            
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get webhook notifier statistics.
        
        Returns:
            Statistics dictionary
        """
        success_rate = (
            self._success_count / self._request_count
            if self._request_count > 0
            else 0.0
        )
        
        return {
            "url": self.config.url,
            "method": self.config.method.value,
            "auth_type": self.config.auth_type.value,
            "total_requests": self._request_count,
            "successful": self._success_count,
            "failed": self._failure_count,
            "success_rate": success_rate
        }
