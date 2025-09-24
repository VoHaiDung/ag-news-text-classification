"""
Queue Services Module for AG News Text Classification
================================================================================
This module implements asynchronous task queuing and message broker functionality
for distributed processing and service communication.

The queue services provide:
- Task queue management
- Message broker for inter-service communication
- Job processing and scheduling
- Dead letter queue handling

References:
    - Hohpe, G., & Woolf, B. (2003). Enterprise Integration Patterns
    - RabbitMQ Documentation

Author: Võ Hải Dũng
License: MIT
"""

from src.services.queue.task_queue import (
    TaskQueue,
    Task,
    TaskStatus,
    TaskPriority
)
from src.services.queue.message_broker import (
    MessageBroker,
    Message,
    MessageType,
    Exchange,
    Queue
)
from src.services.queue.job_processor import (
    JobProcessor,
    ProcessingStrategy,
    ProcessingResult
)

__all__ = [
    "TaskQueue",
    "Task",
    "TaskStatus",
    "TaskPriority",
    "MessageBroker",
    "Message",
    "MessageType",
    "Exchange",
    "Queue",
    "JobProcessor",
    "ProcessingStrategy",
    "ProcessingResult"
]
