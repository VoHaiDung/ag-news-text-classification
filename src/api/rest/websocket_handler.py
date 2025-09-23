"""
WebSocket Handler for Real-time Communication
================================================================================
Implements WebSocket support for real-time features including live predictions,
training progress monitoring, and system notifications.

This module follows the WebSocket protocol (RFC 6455) and implements
event-driven architecture for bidirectional communication.

References:
    - RFC 6455 (2011). The WebSocket Protocol
    - FastAPI WebSocket Documentation
    - Hohpe, G., & Woolf, B. (2003). Enterprise Integration Patterns: Event-Driven Consumer

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.websockets import WebSocketState

from src.api.base.auth import AuthToken, AuthenticationManager
from src.services.service_registry import ServiceRegistry
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class MessageType(Enum):
    """WebSocket message types."""
    # Client -> Server
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PREDICT = "predict"
    PING = "ping"
    
    # Server -> Client
    PREDICTION = "prediction"
    TRAINING_UPDATE = "training_update"
    MODEL_UPDATE = "model_update"
    SYSTEM_STATUS = "system_status"
    ERROR = "error"
    PONG = "pong"
    ACK = "ack"


class ConnectionManager:
    """
    Manages WebSocket connections and message routing.
    
    Implements the Observer pattern for broadcasting updates to
    multiple connected clients based on their subscriptions.
    """
    
    def __init__(self):
        """Initialize connection manager."""
        # Active connections by client ID
        self.connections: Dict[str, WebSocket] = {}
        
        # Subscriptions by topic
        self.subscriptions: Dict[str, Set[str]] = {
            "predictions": set(),
            "training": set(),
            "models": set(),
            "system": set()
        }
        
        # Client metadata
        self.client_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self._total_connections = 0
        self._messages_sent = 0
        self._messages_received = 0
    
    async def connect(
        self,
        websocket: WebSocket,
        client_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Accept and register a new connection.
        
        Args:
            websocket: WebSocket connection
            client_id: Unique client identifier
            metadata: Optional client metadata
        """
        await websocket.accept()
        
        self.connections[client_id] = websocket
        self.client_metadata[client_id] = metadata or {}
        self._total_connections += 1
        
        logger.info(f"WebSocket client connected: {client_id}")
        
        # Send welcome message
        await self.send_message(
            client_id,
            MessageType.ACK,
            {
                "message": "Connected successfully",
                "client_id": client_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    async def disconnect(self, client_id: str) -> None:
        """
        Disconnect and cleanup client.
        
        Args:
            client_id: Client identifier
        """
        if client_id in self.connections:
            # Remove from all subscriptions
            for topic, subscribers in self.subscriptions.items():
                subscribers.discard(client_id)
            
            # Remove connection
            del self.connections[client_id]
            
            # Remove metadata
            if client_id in self.client_metadata:
                del self.client_metadata[client_id]
            
            logger.info(f"WebSocket client disconnected: {client_id}")
    
    async def subscribe(
        self,
        client_id: str,
        topic: str
    ) -> bool:
        """
        Subscribe client to a topic.
        
        Args:
            client_id: Client identifier
            topic: Topic to subscribe to
            
        Returns:
            True if subscription successful
        """
        if topic not in self.subscriptions:
            logger.warning(f"Invalid subscription topic: {topic}")
            return False
        
        self.subscriptions[topic].add(client_id)
        logger.debug(f"Client {client_id} subscribed to {topic}")
        
        return True
    
    async def unsubscribe(
        self,
        client_id: str,
        topic: str
    ) -> bool:
        """
        Unsubscribe client from a topic.
        
        Args:
            client_id: Client identifier
            topic: Topic to unsubscribe from
            
        Returns:
            True if unsubscription successful
        """
        if topic in self.subscriptions:
            self.subscriptions[topic].discard(client_id)
            logger.debug(f"Client {client_id} unsubscribed from {topic}")
            return True
        
        return False
    
    async def send_message(
        self,
        client_id: str,
        message_type: MessageType,
        data: Any
    ) -> bool:
        """
        Send message to a specific client.
        
        Args:
            client_id: Client identifier
            message_type: Type of message
            data: Message data
            
        Returns:
            True if message sent successfully
        """
        if client_id not in self.connections:
            logger.warning(f"Client {client_id} not connected")
            return False
        
        websocket = self.connections[client_id]
        
        try:
            message = {
                "type": message_type.value,
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            await websocket.send_json(message)
            self._messages_sent += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to {client_id}: {str(e)}")
            await self.disconnect(client_id)
            return False
    
    async def broadcast(
        self,
        topic: str,
        message_type: MessageType,
        data: Any
    ) -> int:
        """
        Broadcast message to all subscribers of a topic.
        
        Args:
            topic: Topic to broadcast to
            message_type: Type of message
            data: Message data
            
        Returns:
            Number of clients message was sent to
        """
        if topic not in self.subscriptions:
            logger.warning(f"Invalid broadcast topic: {topic}")
            return 0
        
        subscribers = list(self.subscriptions[topic])
        sent_count = 0
        
        for client_id in subscribers:
            if await self.send_message(client_id, message_type, data):
                sent_count += 1
        
        logger.debug(f"Broadcast to {sent_count} clients on topic {topic}")
        
        return sent_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get connection manager statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "active_connections": len(self.connections),
            "total_connections": self._total_connections,
            "messages_sent": self._messages_sent,
            "messages_received": self._messages_received,
            "subscriptions": {
                topic: len(subscribers)
                for topic, subscribers in self.subscriptions.items()
            }
        }


# Global connection manager instance
connection_manager = ConnectionManager()


class WebSocketHandler:
    """
    Handler for WebSocket connections and message processing.
    
    Implements business logic for real-time features including
    authentication, message validation, and event handling.
    """
    
    def __init__(
        self,
        websocket: WebSocket,
        service_registry: ServiceRegistry,
        auth_manager: Optional[AuthenticationManager] = None
    ):
        """
        Initialize WebSocket handler.
        
        Args:
            websocket: WebSocket connection
            service_registry: Service registry
            auth_manager: Optional authentication manager
        """
        self.websocket = websocket
        self.service_registry = service_registry
        self.auth_manager = auth_manager
        self.client_id = str(uuid.uuid4())
        self.authenticated = False
        self.user_info: Optional[AuthToken] = None
    
    async def handle_connection(
        self,
        token: Optional[str] = None
    ) -> None:
        """
        Handle WebSocket connection lifecycle.
        
        Args:
            token: Optional authentication token
        """
        try:
            # Authenticate if token provided
            if token and self.auth_manager:
                await self._authenticate(token)
            
            # Connect client
            await connection_manager.connect(
                self.websocket,
                self.client_id,
                metadata={
                    "authenticated": self.authenticated,
                    "user_id": self.user_info.user_id if self.user_info else None
                }
            )
            
            # Handle messages
            await self._message_loop()
            
        except WebSocketDisconnect:
            logger.info(f"WebSocket client {self.client_id} disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}", exc_info=True)
            await self._send_error(str(e))
        finally:
            await connection_manager.disconnect(self.client_id)
    
    async def _authenticate(self, token: str) -> None:
        """
        Authenticate WebSocket connection.
        
        Args:
            token: Authentication token
            
        Raises:
            Exception: If authentication fails
        """
        try:
            self.user_info = await self.auth_manager.jwt_auth.validate_token(token)
            self.authenticated = True
            logger.info(f"WebSocket client {self.client_id} authenticated as {self.user_info.user_id}")
        except Exception as e:
            logger.warning(f"WebSocket authentication failed: {str(e)}")
            raise Exception("Authentication failed")
    
    async def _message_loop(self) -> None:
        """Process incoming messages."""
        while True:
            try:
                # Receive message
                message = await self.websocket.receive_json()
                connection_manager._messages_received += 1
                
                # Validate message structure
                if not isinstance(message, dict) or "type" not in message:
                    await self._send_error("Invalid message format")
                    continue
                
                message_type = message.get("type")
                data = message.get("data", {})
                
                # Process message based on type
                if message_type == MessageType.SUBSCRIBE.value:
                    await self._handle_subscribe(data)
                    
                elif message_type == MessageType.UNSUBSCRIBE.value:
                    await self._handle_unsubscribe(data)
                    
                elif message_type == MessageType.PREDICT.value:
                    await self._handle_predict(data)
                    
                elif message_type == MessageType.PING.value:
                    await self._handle_ping()
                    
                else:
                    await self._send_error(f"Unknown message type: {message_type}")
                    
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await self._send_error("Invalid JSON")
            except Exception as e:
                logger.error(f"Message processing error: {str(e)}")
                await self._send_error(str(e))
    
    async def _handle_subscribe(self, data: Dict[str, Any]) -> None:
        """
        Handle subscription request.
        
        Args:
            data: Subscription data
        """
        topic = data.get("topic")
        
        if not topic:
            await self._send_error("Topic required for subscription")
            return
        
        # Check authorization for topic
        if topic == "training" and not self.authenticated:
            await self._send_error("Authentication required for training updates")
            return
        
        # Subscribe
        success = await connection_manager.subscribe(self.client_id, topic)
        
        if success:
            await connection_manager.send_message(
                self.client_id,
                MessageType.ACK,
                {"message": f"Subscribed to {topic}"}
            )
        else:
            await self._send_error(f"Failed to subscribe to {topic}")
    
    async def _handle_unsubscribe(self, data: Dict[str, Any]) -> None:
        """
        Handle unsubscription request.
        
        Args:
            data: Unsubscription data
        """
        topic = data.get("topic")
        
        if not topic:
            await self._send_error("Topic required for unsubscription")
            return
        
        # Unsubscribe
        success = await connection_manager.unsubscribe(self.client_id, topic)
        
        if success:
            await connection_manager.send_message(
                self.client_id,
                MessageType.ACK,
                {"message": f"Unsubscribed from {topic}"}
            )
    
    async def _handle_predict(self, data: Dict[str, Any]) -> None:
        """
        Handle real-time prediction request.
        
        Args:
            data: Prediction request data
        """
        text = data.get("text")
        model_name = data.get("model")
        
        if not text:
            await self._send_error("Text required for prediction")
            return
        
        try:
            # Get prediction service
            prediction_service = self.service_registry.get_service("prediction_service")
            
            if not prediction_service:
                await self._send_error("Prediction service unavailable")
                return
            
            # Perform prediction
            result = await prediction_service.predict(
                text=text,
                model_name=model_name,
                return_probabilities=True
            )
            
            # Send result
            await connection_manager.send_message(
                self.client_id,
                MessageType.PREDICTION,
                {
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "category": result.category,
                    "confidence": result.confidence,
                    "probabilities": result.probabilities,
                    "model": result.model_name,
                    "processing_time": result.processing_time
                }
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            await self._send_error(f"Prediction failed: {str(e)}")
    
    async def _handle_ping(self) -> None:
        """Handle ping message."""
        await connection_manager.send_message(
            self.client_id,
            MessageType.PONG,
            {"message": "pong"}
        )
    
    async def _send_error(self, error_message: str) -> None:
        """
        Send error message to client.
        
        Args:
            error_message: Error message
        """
        await connection_manager.send_message(
            self.client_id,
            MessageType.ERROR,
            {"error": error_message}
        )


async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None, description="Authentication token"),
    service_registry: ServiceRegistry = None,
    auth_manager: AuthenticationManager = None
):
    """
    Main WebSocket endpoint.
    
    Args:
        websocket: WebSocket connection
        token: Optional authentication token
        service_registry: Service registry
        auth_manager: Authentication manager
    """
    handler = WebSocketHandler(websocket, service_registry, auth_manager)
    await handler.handle_connection(token)


class WebSocketNotifier:
    """
    Service for sending notifications via WebSocket.
    
    Used by other services to push real-time updates to clients.
    """
    
    @staticmethod
    async def notify_training_update(
        job_id: str,
        status: str,
        progress: float,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Notify training progress update.
        
        Args:
            job_id: Training job ID
            status: Job status
            progress: Progress percentage
            metrics: Optional metrics
        """
        await connection_manager.broadcast(
            "training",
            MessageType.TRAINING_UPDATE,
            {
                "job_id": job_id,
                "status": status,
                "progress": progress,
                "metrics": metrics or {}
            }
        )
    
    @staticmethod
    async def notify_model_update(
        model_name: str,
        action: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Notify model update.
        
        Args:
            model_name: Model name
            action: Update action (loaded, unloaded, updated)
            details: Optional details
        """
        await connection_manager.broadcast(
            "models",
            MessageType.MODEL_UPDATE,
            {
                "model": model_name,
                "action": action,
                "details": details or {}
            }
        )
    
    @staticmethod
    async def notify_system_status(
        status: str,
        message: str,
        severity: str = "info"
    ) -> None:
        """
        Notify system status update.
        
        Args:
            status: System status
            message: Status message
            severity: Message severity (info, warning, error)
        """
        await connection_manager.broadcast(
            "system",
            MessageType.SYSTEM_STATUS,
            {
                "status": status,
                "message": message,
                "severity": severity
            }
        )
