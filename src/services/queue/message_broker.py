"""
Message Broker Implementation for AG News Text Classification
================================================================================
This module implements a message broker for inter-service communication,
supporting publish-subscribe and point-to-point messaging patterns.

The message broker provides:
- Topic-based publish-subscribe
- Message routing and filtering
- Durable subscriptions
- Message acknowledgment and redelivery

References:
    - Hohpe, G., & Woolf, B. (2003). Enterprise Integration Patterns
    - RabbitMQ in Depth (Manning)
    - Apache Kafka Documentation

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
import uuid
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque

from src.services.base_service import BaseService, ServiceConfig
from src.core.exceptions import ServiceException
from src.utils.logging_config import get_logger


class MessageType(Enum):
    """
    Message types for routing.
    
    Types:
        COMMAND: Direct command to service
        EVENT: Event notification
        QUERY: Query request
        RESPONSE: Query response
        BROADCAST: Broadcast to all subscribers
    """
    COMMAND = "command"
    EVENT = "event"
    QUERY = "query"
    RESPONSE = "response"
    BROADCAST = "broadcast"


class DeliveryMode(Enum):
    """
    Message delivery modes.
    
    Modes:
        AT_MOST_ONCE: Message may be lost but never duplicated
        AT_LEAST_ONCE: Message never lost but may be duplicated
        EXACTLY_ONCE: Message delivered exactly once
    """
    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"


@dataclass
class Message:
    """
    Message for broker communication.
    
    Attributes:
        id: Unique message identifier
        type: Message type
        topic: Message topic/channel
        payload: Message payload
        headers: Message headers/metadata
        timestamp: Message timestamp
        correlation_id: For request-response correlation
        reply_to: Reply topic for responses
        expiration: Message expiration time
        delivery_mode: Delivery guarantee mode
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.EVENT
    topic: str = ""
    payload: Any = None
    headers: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    expiration: Optional[datetime] = None
    delivery_mode: DeliveryMode = DeliveryMode.AT_LEAST_ONCE
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.expiration:
            return datetime.now() > self.expiration
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "topic": self.topic,
            "headers": self.headers,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "expiration": self.expiration.isoformat() if self.expiration else None,
            "delivery_mode": self.delivery_mode.value
        }


@dataclass
class Subscription:
    """
    Topic subscription.
    
    Attributes:
        id: Subscription identifier
        topic_pattern: Topic pattern (supports wildcards)
        handler: Message handler function
        filter_predicate: Optional message filter
        durable: Whether subscription survives broker restart
        auto_ack: Automatic acknowledgment
        max_retries: Maximum delivery retries
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic_pattern: str = ""
    handler: Optional[Callable] = None
    filter_predicate: Optional[Callable[[Message], bool]] = None
    durable: bool = False
    auto_ack: bool = True
    max_retries: int = 3


@dataclass
class Exchange:
    """
    Message exchange for routing.
    
    Attributes:
        name: Exchange name
        type: Exchange type (direct, topic, fanout)
        durable: Whether exchange survives restart
        auto_delete: Delete when no longer used
    """
    name: str
    type: str = "topic"  # direct, topic, fanout
    durable: bool = True
    auto_delete: bool = False


@dataclass
class Queue:
    """
    Message queue.
    
    Attributes:
        name: Queue name
        durable: Whether queue survives restart
        exclusive: Exclusive to connection
        auto_delete: Delete when no longer used
        max_length: Maximum queue length
        ttl: Message time-to-live
    """
    name: str
    durable: bool = True
    exclusive: bool = False
    auto_delete: bool = False
    max_length: Optional[int] = None
    ttl: Optional[timedelta] = None


class MessageBroker(BaseService):
    """
    Message broker for inter-service communication.
    
    This service provides publish-subscribe and point-to-point messaging
    with support for topics, routing, and durable subscriptions.
    """
    
    def __init__(
        self,
        config: Optional[ServiceConfig] = None,
        max_queue_size: int = 10000,
        enable_persistence: bool = True
    ):
        """
        Initialize message broker.
        
        Args:
            config: Service configuration
            max_queue_size: Maximum queue size
            enable_persistence: Enable message persistence
        """
        if config is None:
            config = ServiceConfig(name="message_broker")
        super().__init__(config)
        
        self.max_queue_size = max_queue_size
        self.enable_persistence = enable_persistence
        
        # Exchanges and queues
        self._exchanges: Dict[str, Exchange] = {}
        self._queues: Dict[str, deque] = {}
        self._bindings: Dict[str, List[str]] = defaultdict(list)  # exchange -> queues
        
        # Subscriptions
        self._subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        self._subscription_queues: Dict[str, deque] = {}  # subscription -> message queue
        
        # Message tracking
        self._unacknowledged: Dict[str, Message] = {}
        self._message_retries: Dict[str, int] = defaultdict(int)
        
        # Background tasks
        self._delivery_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self._stats = defaultdict(int)
        
        self.logger = get_logger("service.message_broker")
    
    async def _initialize(self) -> None:
        """Initialize message broker."""
        self.logger.info("Initializing message broker")
        
        # Create default exchanges
        self._create_default_exchanges()
        
        # Start background tasks
        self._delivery_task = asyncio.create_task(self._delivery_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _start(self) -> None:
        """Start message broker service."""
        self.logger.info("Message broker started")
    
    async def _stop(self) -> None:
        """Stop message broker service."""
        # Cancel background tasks
        for task in [self._delivery_task, self._cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.logger.info("Message broker stopped")
    
    async def _cleanup(self) -> None:
        """Cleanup broker resources."""
        self._exchanges.clear()
        self._queues.clear()
        self._subscriptions.clear()
    
    async def _check_health(self) -> bool:
        """Check broker health."""
        total_messages = sum(len(q) for q in self._queues.values())
        return total_messages < self.max_queue_size
    
    def _create_default_exchanges(self) -> None:
        """Create default exchanges."""
        # Direct exchange for point-to-point
        self.declare_exchange(Exchange(
            name="direct",
            type="direct",
            durable=True
        ))
        
        # Topic exchange for pub-sub
        self.declare_exchange(Exchange(
            name="topic",
            type="topic",
            durable=True
        ))
        
        # Fanout exchange for broadcast
        self.declare_exchange(Exchange(
            name="fanout",
            type="fanout",
            durable=True
        ))
    
    def declare_exchange(self, exchange: Exchange) -> None:
        """
        Declare an exchange.
        
        Args:
            exchange: Exchange to declare
        """
        self._exchanges[exchange.name] = exchange
        self.logger.info(f"Declared exchange: {exchange.name} ({exchange.type})")
    
    def declare_queue(self, queue: Queue) -> None:
        """
        Declare a queue.
        
        Args:
            queue: Queue to declare
        """
        if queue.name not in self._queues:
            self._queues[queue.name] = deque(maxlen=queue.max_length)
        
        self.logger.info(f"Declared queue: {queue.name}")
    
    def bind_queue(
        self,
        queue_name: str,
        exchange_name: str,
        routing_key: str = ""
    ) -> None:
        """
        Bind queue to exchange.
        
        Args:
            queue_name: Queue name
            exchange_name: Exchange name
            routing_key: Routing key for binding
        """
        if exchange_name not in self._exchanges:
            raise ServiceException(f"Exchange not found: {exchange_name}")
        
        binding_key = f"{exchange_name}:{routing_key}"
        self._bindings[binding_key].append(queue_name)
        
        self.logger.info(
            f"Bound queue {queue_name} to exchange {exchange_name} "
            f"with routing key '{routing_key}'"
        )
    
    async def publish(
        self,
        message: Message,
        exchange: str = "topic",
        routing_key: str = ""
    ) -> bool:
        """
        Publish a message.
        
        Args:
            message: Message to publish
            exchange: Exchange name
            routing_key: Routing key
            
        Returns:
            bool: True if published successfully
        """
        try:
            # Check exchange exists
            if exchange not in self._exchanges:
                raise ServiceException(f"Exchange not found: {exchange}")
            
            exchange_obj = self._exchanges[exchange]
            
            # Route message based on exchange type
            if exchange_obj.type == "direct":
                # Direct routing
                queues = self._bindings.get(f"{exchange}:{routing_key}", [])
            elif exchange_obj.type == "topic":
                # Topic routing with wildcards
                queues = self._match_topic_queues(exchange, routing_key)
            elif exchange_obj.type == "fanout":
                # Fanout to all bound queues
                queues = []
                for key, queue_list in self._bindings.items():
                    if key.startswith(f"{exchange}:"):
                        queues.extend(queue_list)
            else:
                queues = []
            
            # Deliver to queues
            for queue_name in queues:
                if queue_name in self._queues:
                    self._queues[queue_name].append(message)
            
            # Deliver to topic subscribers
            await self._deliver_to_subscribers(message, routing_key or message.topic)
            
            # Update statistics
            self._stats["published"] += 1
            self._stats[f"exchange_{exchange}"] += 1
            
            self.logger.debug(
                f"Published message {message.id} to {exchange} "
                f"with routing key '{routing_key}'"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to publish message: {e}")
            return False
    
    def subscribe(
        self,
        topic_pattern: str,
        handler: Callable,
        filter_predicate: Optional[Callable[[Message], bool]] = None,
        durable: bool = False,
        auto_ack: bool = True
    ) -> str:
        """
        Subscribe to topic.
        
        Args:
            topic_pattern: Topic pattern (supports wildcards)
            handler: Message handler
            filter_predicate: Optional message filter
            durable: Durable subscription
            auto_ack: Automatic acknowledgment
            
        Returns:
            str: Subscription ID
        """
        subscription = Subscription(
            topic_pattern=topic_pattern,
            handler=handler,
            filter_predicate=filter_predicate,
            durable=durable,
            auto_ack=auto_ack
        )
        
        self._subscriptions[topic_pattern].append(subscription)
        self._subscription_queues[subscription.id] = deque(maxlen=1000)
        
        self._stats["subscriptions"] += 1
        
        self.logger.info(f"Created subscription: {subscription.id} for pattern '{topic_pattern}'")
        return subscription.id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from topic.
        
        Args:
            subscription_id: Subscription ID
            
        Returns:
            bool: True if unsubscribed successfully
        """
        for pattern, subs in self._subscriptions.items():
            for sub in subs:
                if sub.id == subscription_id:
                    subs.remove(sub)
                    del self._subscription_queues[subscription_id]
                    
                    self._stats["subscriptions"] -= 1
                    
                    self.logger.info(f"Removed subscription: {subscription_id}")
                    return True
        
        return False
    
    async def _deliver_to_subscribers(self, message: Message, topic: str) -> None:
        """
        Deliver message to matching subscribers.
        
        Args:
            message: Message to deliver
            topic: Message topic
        """
        delivered = 0
        
        for pattern, subscriptions in self._subscriptions.items():
            if self._matches_pattern(topic, pattern):
                for subscription in subscriptions:
                    # Apply filter if present
                    if subscription.filter_predicate:
                        try:
                            if not subscription.filter_predicate(message):
                                continue
                        except Exception as e:
                            self.logger.error(f"Filter predicate error: {e}")
                            continue
                    
                    # Queue message for delivery
                    self._subscription_queues[subscription.id].append(message)
                    delivered += 1
        
        if delivered > 0:
            self.logger.debug(f"Queued message {message.id} for {delivered} subscribers")
    
    def _matches_pattern(self, topic: str, pattern: str) -> bool:
        """
        Check if topic matches pattern.
        
        Args:
            topic: Topic to match
            pattern: Pattern with wildcards (* and #)
            
        Returns:
            bool: True if matches
        """
        # Convert pattern to regex
        # * matches single word, # matches multiple words
        import re
        
        regex_pattern = pattern.replace(".", r"\.")
        regex_pattern = regex_pattern.replace("*", r"[^.]+")
        regex_pattern = regex_pattern.replace("#", r".*")
        regex_pattern = f"^{regex_pattern}$"
        
        return bool(re.match(regex_pattern, topic))
    
    def _match_topic_queues(self, exchange: str, routing_key: str) -> List[str]:
        """
        Match queues for topic routing.
        
        Args:
            exchange: Exchange name
            routing_key: Routing key
            
        Returns:
            List[str]: Matching queue names
        """
        matched_queues = []
        
        for binding_key, queues in self._bindings.items():
            if binding_key.startswith(f"{exchange}:"):
                pattern = binding_key.split(":", 1)[1]
                if self._matches_pattern(routing_key, pattern):
                    matched_queues.extend(queues)
        
        return matched_queues
    
    async def _delivery_loop(self) -> None:
        """Background loop for message delivery."""
        while True:
            try:
                # Deliver messages to subscribers
                for sub_id, queue in self._subscription_queues.items():
                    if not queue:
                        continue
                    
                    # Find subscription
                    subscription = None
                    for subs in self._subscriptions.values():
                        for sub in subs:
                            if sub.id == sub_id:
                                subscription = sub
                                break
                        if subscription:
                            break
                    
                    if not subscription:
                        continue
                    
                    # Process queued messages
                    while queue:
                        message = queue.popleft()
                        
                        # Check expiration
                        if message.is_expired():
                            self._stats["expired"] += 1
                            continue
                        
                        # Deliver message
                        await self._deliver_message(message, subscription)
                
                await asyncio.sleep(0.1)  # Small delay
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Delivery loop error: {e}")
                await asyncio.sleep(1)
    
    async def _deliver_message(
        self,
        message: Message,
        subscription: Subscription
    ) -> None:
        """
        Deliver message to subscriber.
        
        Args:
            message: Message to deliver
            subscription: Subscription
        """
        try:
            # Track for acknowledgment
            if not subscription.auto_ack:
                self._unacknowledged[message.id] = message
            
            # Call handler
            if asyncio.iscoroutinefunction(subscription.handler):
                await subscription.handler(message)
            else:
                await asyncio.get_event_loop().run_in_executor(
                    None, subscription.handler, message
                )
            
            # Auto-acknowledge if configured
            if subscription.auto_ack:
                self._stats["delivered"] += 1
            
        except Exception as e:
            self.logger.error(f"Message delivery failed: {e}")
            
            # Handle retry
            self._message_retries[message.id] += 1
            
            if self._message_retries[message.id] < subscription.max_retries:
                # Re-queue for retry
                self._subscription_queues[subscription.id].append(message)
            else:
                # Move to dead letter
                self._stats["failed"] += 1
                self.logger.error(
                    f"Message {message.id} failed after "
                    f"{subscription.max_retries} retries"
                )
    
    def acknowledge(self, message_id: str) -> bool:
        """
        Acknowledge message delivery.
        
        Args:
            message_id: Message ID
            
        Returns:
            bool: True if acknowledged
        """
        if message_id in self._unacknowledged:
            del self._unacknowledged[message_id]
            self._stats["delivered"] += 1
            return True
        
        return False
    
    async def _cleanup_loop(self) -> None:
        """Background loop for cleanup."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Clean expired messages
                for queue in self._queues.values():
                    expired = []
                    for i, message in enumerate(queue):
                        if message.is_expired():
                            expired.append(i)
                    
                    # Remove expired messages (in reverse order)
                    for i in reversed(expired):
                        del queue[i]
                        self._stats["expired"] += 1
                
                # Clean old retry counters
                cutoff = datetime.now() - timedelta(hours=1)
                old_retries = []
                
                for msg_id in self._message_retries:
                    if msg_id not in self._unacknowledged:
                        old_retries.append(msg_id)
                
                for msg_id in old_retries:
                    del self._message_retries[msg_id]
                
                self.logger.debug("Cleaned up expired messages and old retry counters")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get broker statistics.
        
        Returns:
            Dict[str, Any]: Broker statistics
        """
        total_queued = sum(len(q) for q in self._queues.values())
        total_subscription_queued = sum(
            len(q) for q in self._subscription_queues.values()
        )
        
        return {
            "exchanges": len(self._exchanges),
            "queues": len(self._queues),
            "subscriptions": sum(len(s) for s in self._subscriptions.values()),
            "total_queued": total_queued,
            "subscription_queued": total_subscription_queued,
            "unacknowledged": len(self._unacknowledged),
            "published": self._stats["published"],
            "delivered": self._stats["delivered"],
            "failed": self._stats["failed"],
            "expired": self._stats["expired"]
        }
