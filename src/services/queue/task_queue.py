"""
Task Queue Implementation for AG News Text Classification
================================================================================
This module implements a distributed task queue system for asynchronous task
processing, supporting priority-based execution and retry mechanisms.

The task queue provides:
- Priority-based task scheduling
- Distributed task processing
- Retry and dead letter queue handling
- Task result persistence

References:
    - Celery Documentation: Distributed Task Queue
    - Amazon SQS Developer Guide
    - Tanenbaum, A. S., & Van Steen, M. (2017). Distributed Systems

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
import heapq
import uuid
import pickle
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

from src.services.base_service import BaseService, ServiceConfig
from src.core.exceptions import ServiceException
from src.utils.logging_config import get_logger


class TaskStatus(Enum):
    """
    Task execution status enumeration.
    
    States:
        PENDING: Task is queued for execution
        RUNNING: Task is currently being processed
        COMPLETED: Task completed successfully
        FAILED: Task execution failed
        RETRYING: Task is being retried
        CANCELLED: Task was cancelled
        EXPIRED: Task expired before execution
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class TaskPriority(Enum):
    """
    Task priority levels for execution ordering.
    
    Priorities (lower value = higher priority):
        CRITICAL: Execute immediately
        HIGH: High priority execution
        NORMAL: Standard priority
        LOW: Low priority execution
        DEFERRED: Execute when resources available
    """
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    DEFERRED = 4


@dataclass
class Task:
    """
    Task definition for queue processing.
    
    Attributes:
        id: Unique task identifier
        name: Task name
        payload: Task data/parameters
        priority: Task priority level
        status: Current task status
        handler: Task handler function name
        result: Task execution result
        error: Error message if failed
        created_at: Task creation timestamp
        started_at: Task execution start time
        completed_at: Task completion time
        expire_at: Task expiration time
        retry_count: Number of retry attempts
        max_retries: Maximum retry attempts allowed
        retry_delay: Delay between retries
        metadata: Additional task metadata
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    handler: Optional[str] = None
    result: Any = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    expire_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    retry_delay: int = 5  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Comparison for priority queue ordering."""
        return (self.priority.value, self.created_at) < (other.priority.value, other.created_at)
    
    def is_expired(self) -> bool:
        """Check if task has expired."""
        if self.expire_at:
            return datetime.now() > self.expire_at
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "priority": self.priority.name,
            "status": self.status.value,
            "handler": self.handler,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "expire_at": self.expire_at.isoformat() if self.expire_at else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "metadata": self.metadata
        }


class TaskQueue(BaseService):
    """
    Distributed task queue for asynchronous processing.
    
    This service manages task queuing, distribution, and execution with
    support for priorities, retries, and dead letter queue handling.
    """
    
    def __init__(
        self,
        config: Optional[ServiceConfig] = None,
        max_workers: int = 10,
        max_queue_size: int = 10000,
        enable_persistence: bool = True
    ):
        """
        Initialize task queue.
        
        Args:
            config: Service configuration
            max_workers: Maximum concurrent workers
            max_queue_size: Maximum queue size
            enable_persistence: Enable task persistence
        """
        if config is None:
            config = ServiceConfig(name="task_queue")
        super().__init__(config)
        
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.enable_persistence = enable_persistence
        
        # Task storage
        self._pending_queue: List[Task] = []  # Priority queue
        self._running_tasks: Dict[str, Task] = {}
        self._completed_tasks: Dict[str, Task] = {}
        self._failed_tasks: Dict[str, Task] = {}
        self._dead_letter_queue: List[Task] = []
        
        # Task handlers
        self._handlers: Dict[str, Callable] = {}
        
        # Worker management
        self._workers: List[asyncio.Task] = []
        self._worker_semaphore = asyncio.Semaphore(max_workers)
        self._task_available = asyncio.Event()
        
        # Statistics
        self._stats = defaultdict(int)
        
        # Persistence
        if enable_persistence:
            from pathlib import Path
            self._storage_path = Path("outputs/task_queue")
            self._storage_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger("service.task_queue")
    
    async def _initialize(self) -> None:
        """Initialize task queue resources."""
        self.logger.info(f"Initializing task queue with {self.max_workers} workers")
        
        # Load persisted tasks if enabled
        if self.enable_persistence:
            await self._load_persisted_tasks()
        
        # Start worker pool
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker_loop(i))
            self._workers.append(worker)
    
    async def _start(self) -> None:
        """Start task queue service."""
        self.logger.info("Task queue started")
    
    async def _stop(self) -> None:
        """Stop task queue service."""
        # Signal workers to stop
        self._task_available.set()
        
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
        
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        
        # Persist remaining tasks
        if self.enable_persistence:
            await self._persist_tasks()
        
        self.logger.info("Task queue stopped")
    
    async def _cleanup(self) -> None:
        """Cleanup task queue resources."""
        self._pending_queue.clear()
        self._running_tasks.clear()
        self._workers.clear()
    
    async def _check_health(self) -> bool:
        """Check task queue health."""
        return len(self._pending_queue) < self.max_queue_size
    
    def register_handler(self, name: str, handler: Callable) -> None:
        """
        Register a task handler.
        
        Args:
            name: Handler name
            handler: Handler function
        """
        self._handlers[name] = handler
        self.logger.info(f"Registered task handler: {name}")
    
    async def submit_task(
        self,
        name: str,
        payload: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        handler: Optional[str] = None,
        expire_after: Optional[timedelta] = None,
        max_retries: int = 3
    ) -> str:
        """
        Submit a task to the queue.
        
        Args:
            name: Task name
            payload: Task payload
            priority: Task priority
            handler: Handler name
            expire_after: Expiration duration
            max_retries: Maximum retry attempts
            
        Returns:
            str: Task ID
            
        Raises:
            ServiceException: If queue is full
        """
        if len(self._pending_queue) >= self.max_queue_size:
            raise ServiceException("Task queue is full")
        
        # Create task
        task = Task(
            name=name,
            payload=payload or {},
            priority=priority,
            handler=handler,
            expire_at=datetime.now() + expire_after if expire_after else None,
            max_retries=max_retries
        )
        
        # Add to queue
        heapq.heappush(self._pending_queue, task)
        self._task_available.set()
        
        # Update statistics
        self._stats["submitted"] += 1
        self._stats[f"priority_{priority.name}"] += 1
        
        self.logger.debug(f"Submitted task: {task.id} ({name})")
        return task.id
    
    async def _worker_loop(self, worker_id: int) -> None:
        """
        Worker loop for processing tasks.
        
        Args:
            worker_id: Worker identifier
        """
        self.logger.debug(f"Worker {worker_id} started")
        
        while True:
            try:
                # Wait for task availability
                await self._task_available.wait()
                
                # Get next task
                task = await self._get_next_task()
                
                if task:
                    async with self._worker_semaphore:
                        await self._process_task(task, worker_id)
                
            except asyncio.CancelledError:
                self.logger.debug(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)
    
    async def _get_next_task(self) -> Optional[Task]:
        """
        Get next task from queue.
        
        Returns:
            Optional[Task]: Next task or None
        """
        while self._pending_queue:
            # Get highest priority task
            task = heapq.heappop(self._pending_queue)
            
            # Check expiration
            if task.is_expired():
                task.status = TaskStatus.EXPIRED
                self._failed_tasks[task.id] = task
                self._stats["expired"] += 1
                continue
            
            # Clear event if queue is empty
            if not self._pending_queue:
                self._task_available.clear()
            
            return task
        
        self._task_available.clear()
        return None
    
    async def _process_task(self, task: Task, worker_id: int) -> None:
        """
        Process a single task.
        
        Args:
            task: Task to process
            worker_id: Worker identifier
        """
        self.logger.debug(f"Worker {worker_id} processing task {task.id}")
        
        # Update task status
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        self._running_tasks[task.id] = task
        
        try:
            # Get handler
            handler = self._handlers.get(task.handler or task.name)
            
            if not handler:
                raise ServiceException(f"No handler found for task: {task.name}")
            
            # Execute handler
            if asyncio.iscoroutinefunction(handler):
                result = await handler(task.payload)
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, handler, task.payload
                )
            
            # Update task with result
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            # Move to completed
            del self._running_tasks[task.id]
            self._completed_tasks[task.id] = task
            
            # Update statistics
            self._stats["completed"] += 1
            duration = (task.completed_at - task.started_at).total_seconds()
            self._stats["total_duration"] += duration
            
            self.logger.info(
                f"Task {task.id} completed in {duration:.2f}s"
            )
            
        except Exception as e:
            # Handle task failure
            task.error = str(e)
            task.completed_at = datetime.now()
            
            del self._running_tasks[task.id]
            
            # Check retry
            if task.retry_count < task.max_retries:
                await self._retry_task(task)
            else:
                # Move to failed/dead letter queue
                task.status = TaskStatus.FAILED
                self._failed_tasks[task.id] = task
                self._dead_letter_queue.append(task)
                
                self._stats["failed"] += 1
                
                self.logger.error(
                    f"Task {task.id} failed after {task.retry_count} retries: {e}"
                )
    
    async def _retry_task(self, task: Task) -> None:
        """
        Retry a failed task.
        
        Args:
            task: Task to retry
        """
        task.retry_count += 1
        task.status = TaskStatus.RETRYING
        
        # Calculate retry delay with exponential backoff
        delay = task.retry_delay * (2 ** (task.retry_count - 1))
        
        self.logger.info(
            f"Retrying task {task.id} (attempt {task.retry_count}/"
            f"{task.max_retries}) after {delay}s"
        )
        
        # Schedule retry
        await asyncio.sleep(delay)
        
        # Re-queue task
        task.status = TaskStatus.PENDING
        heapq.heappush(self._pending_queue, task)
        self._task_available.set()
        
        self._stats["retried"] += 1
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or running task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            bool: True if cancelled successfully
        """
        # Check pending queue
        for i, task in enumerate(self._pending_queue):
            if task.id == task_id:
                task.status = TaskStatus.CANCELLED
                del self._pending_queue[i]
                heapq.heapify(self._pending_queue)
                
                self._stats["cancelled"] += 1
                self.logger.info(f"Cancelled pending task: {task_id}")
                return True
        
        # Check running tasks (mark for cancellation)
        if task_id in self._running_tasks:
            task = self._running_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            
            self._stats["cancelled"] += 1
            self.logger.info(f"Marked running task for cancellation: {task_id}")
            return True
        
        return False
    
    def get_task_status(self, task_id: str) -> Optional[Task]:
        """
        Get task status.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Optional[Task]: Task instance or None
        """
        # Check all queues
        for queue in [self._running_tasks, self._completed_tasks, self._failed_tasks]:
            if task_id in queue:
                return queue[task_id]
        
        # Check pending queue
        for task in self._pending_queue:
            if task.id == task_id:
                return task
        
        return None
    
    async def process_dead_letter_queue(
        self,
        handler: Optional[Callable] = None
    ) -> int:
        """
        Process tasks in dead letter queue.
        
        Args:
            handler: Custom handler for dead letter tasks
            
        Returns:
            int: Number of tasks processed
        """
        processed = 0
        
        while self._dead_letter_queue:
            task = self._dead_letter_queue.pop(0)
            
            if handler:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(task)
                    else:
                        handler(task)
                    processed += 1
                except Exception as e:
                    self.logger.error(f"Dead letter handler error: {e}")
            else:
                # Default: log and discard
                self.logger.warning(
                    f"Discarding dead letter task: {task.id} ({task.name})"
                )
                processed += 1
        
        return processed
    
    async def _persist_tasks(self) -> None:
        """Persist tasks to storage."""
        if not self.enable_persistence:
            return
        
        try:
            # Persist pending tasks
            pending_path = self._storage_path / "pending_tasks.pkl"
            with open(pending_path, "wb") as f:
                pickle.dump(list(self._pending_queue), f)
            
            # Persist failed tasks
            failed_path = self._storage_path / "failed_tasks.pkl"
            with open(failed_path, "wb") as f:
                pickle.dump(self._failed_tasks, f)
            
            self.logger.info("Persisted tasks to storage")
            
        except Exception as e:
            self.logger.error(f"Failed to persist tasks: {e}")
    
    async def _load_persisted_tasks(self) -> None:
        """Load persisted tasks from storage."""
        if not self.enable_persistence:
            return
        
        try:
            # Load pending tasks
            pending_path = self._storage_path / "pending_tasks.pkl"
            if pending_path.exists():
                with open(pending_path, "rb") as f:
                    tasks = pickle.load(f)
                    for task in tasks:
                        heapq.heappush(self._pending_queue, task)
                    if tasks:
                        self._task_available.set()
                    self.logger.info(f"Loaded {len(tasks)} pending tasks")
            
            # Load failed tasks
            failed_path = self._storage_path / "failed_tasks.pkl"
            if failed_path.exists():
                with open(failed_path, "rb") as f:
                    self._failed_tasks = pickle.load(f)
                    self.logger.info(f"Loaded {len(self._failed_tasks)} failed tasks")
                    
        except Exception as e:
            self.logger.error(f"Failed to load persisted tasks: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get queue statistics.
        
        Returns:
            Dict[str, Any]: Queue statistics
        """
        avg_duration = 0
        if self._stats["completed"] > 0:
            avg_duration = self._stats["total_duration"] / self._stats["completed"]
        
        return {
            "pending": len(self._pending_queue),
            "running": len(self._running_tasks),
            "completed": len(self._completed_tasks),
            "failed": len(self._failed_tasks),
            "dead_letter": len(self._dead_letter_queue),
            "submitted_total": self._stats["submitted"],
            "completed_total": self._stats["completed"],
            "failed_total": self._stats["failed"],
            "retried_total": self._stats["retried"],
            "expired_total": self._stats["expired"],
            "cancelled_total": self._stats["cancelled"],
            "avg_duration_seconds": avg_duration,
            "workers": self.max_workers,
            "queue_usage": len(self._pending_queue) / self.max_queue_size * 100
        }
