"""
Job Scheduler Implementation for AG News Text Classification
================================================================================
This module implements job scheduling and execution management for asynchronous
tasks, supporting cron-like scheduling and priority-based execution.

The scheduler provides:
- Cron and interval-based scheduling
- Priority queue execution
- Job dependency management
- Retry and failure handling

References:
    - Tanenbaum, A. S., & Bos, H. (2014). Modern Operating Systems
    - Quartz Scheduler Documentation

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
import heapq
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from croniter import croniter
import uuid

from src.services.base_service import BaseService, ServiceConfig
from src.core.exceptions import ServiceException
from src.utils.logging_config import get_logger


class JobStatus(Enum):
    """
    Job execution status enumeration.
    
    States:
        PENDING: Job is scheduled but not started
        RUNNING: Job is currently executing
        COMPLETED: Job completed successfully
        FAILED: Job execution failed
        CANCELLED: Job was cancelled
        RETRYING: Job is being retried after failure
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class JobPriority(Enum):
    """
    Job priority levels for execution ordering.
    
    Levels:
        CRITICAL: Highest priority (0)
        HIGH: High priority (1)
        NORMAL: Normal priority (2)
        LOW: Low priority (3)
        BACKGROUND: Lowest priority (4)
    """
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class Schedule:
    """
    Schedule configuration for job execution.
    
    Attributes:
        cron_expression: Cron expression for scheduling
        interval_seconds: Fixed interval in seconds
        start_time: Start time for scheduling
        end_time: End time for scheduling
        max_runs: Maximum number of runs
        timezone: Timezone for cron expressions
    """
    cron_expression: Optional[str] = None
    interval_seconds: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_runs: Optional[int] = None
    timezone: str = "UTC"
    
    def validate(self) -> bool:
        """Validate schedule configuration."""
        if not self.cron_expression and not self.interval_seconds:
            raise ValueError("Either cron_expression or interval_seconds must be specified")
        
        if self.cron_expression:
            try:
                croniter(self.cron_expression)
            except Exception as e:
                raise ValueError(f"Invalid cron expression: {e}")
        
        if self.interval_seconds and self.interval_seconds <= 0:
            raise ValueError("Interval must be positive")
        
        return True
    
    def get_next_run_time(self, last_run: Optional[datetime] = None) -> Optional[datetime]:
        """
        Calculate next run time based on schedule.
        
        Args:
            last_run: Last execution time
            
        Returns:
            datetime: Next scheduled run time or None
        """
        now = datetime.now()
        
        # Check schedule boundaries
        if self.start_time and now < self.start_time:
            return self.start_time
        
        if self.end_time and now > self.end_time:
            return None
        
        # Calculate next run
        if self.cron_expression:
            base_time = last_run or now
            cron = croniter(self.cron_expression, base_time)
            next_time = cron.get_next(datetime)
            
            if self.end_time and next_time > self.end_time:
                return None
            
            return next_time
        
        elif self.interval_seconds:
            if last_run:
                next_time = last_run + timedelta(seconds=self.interval_seconds)
            else:
                next_time = now
            
            if self.end_time and next_time > self.end_time:
                return None
            
            return next_time
        
        return None


@dataclass
class Job:
    """
    Job definition with execution configuration.
    
    Attributes:
        id: Unique job identifier
        name: Job name
        task: Callable to execute
        args: Positional arguments
        kwargs: Keyword arguments
        schedule: Schedule configuration
        priority: Job priority
        dependencies: Job IDs that must complete first
        retry_policy: Retry configuration
        timeout: Execution timeout in seconds
        metadata: Additional job metadata
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    task: Optional[Callable] = None
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    schedule: Optional[Schedule] = None
    priority: JobPriority = JobPriority.NORMAL
    dependencies: List[str] = field(default_factory=list)
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {
        "max_retries": 3,
        "retry_delay": 5,
        "exponential_backoff": True
    })
    timeout: int = 300
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate job configuration."""
        if not self.name:
            raise ValueError("Job name is required")
        if not callable(self.task):
            raise ValueError("Job task must be callable")
        if self.schedule:
            self.schedule.validate()
        return True


@dataclass
class JobExecution:
    """
    Job execution instance with runtime information.
    
    Attributes:
        job: Job being executed
        status: Current execution status
        started_at: Execution start time
        completed_at: Execution completion time
        result: Execution result
        error: Error message if failed
        retry_count: Number of retry attempts
        run_count: Total number of runs
    """
    job: Job
    status: JobStatus = JobStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    run_count: int = 0
    
    def __lt__(self, other):
        """Comparison for priority queue ordering."""
        return (self.job.priority.value, self.job.id) < (other.job.priority.value, other.job.id)


class JobScheduler(BaseService):
    """
    Scheduler for managing and executing jobs with various scheduling strategies.
    
    This service provides comprehensive job scheduling with support for
    cron expressions, priorities, dependencies, and retry mechanisms.
    """
    
    def __init__(
        self,
        config: Optional[ServiceConfig] = None,
        max_workers: int = 10,
        check_interval: int = 1
    ):
        """
        Initialize job scheduler.
        
        Args:
            config: Service configuration
            max_workers: Maximum concurrent workers
            check_interval: Interval for checking scheduled jobs (seconds)
        """
        if config is None:
            config = ServiceConfig(name="job_scheduler")
        super().__init__(config)
        
        self.max_workers = max_workers
        self.check_interval = check_interval
        
        # Job storage
        self._jobs: Dict[str, Job] = {}
        self._executions: Dict[str, JobExecution] = {}
        
        # Execution management
        self._pending_queue: List[JobExecution] = []  # Priority queue
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._completed_jobs: Dict[str, JobExecution] = {}
        
        # Background tasks
        self._scheduler_task: Optional[asyncio.Task] = None
        self._worker_tasks: List[asyncio.Task] = []
        
        # Synchronization
        self._queue_lock = asyncio.Lock()
        self._job_available = asyncio.Event()
        
        self.logger = get_logger("service.job_scheduler")
    
    async def _initialize(self) -> None:
        """Initialize job scheduler resources."""
        self.logger.info(f"Initializing job scheduler with {self.max_workers} workers")
        
        # Start background tasks
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        # Start worker tasks
        for i in range(self.max_workers):
            worker_task = asyncio.create_task(self._worker_loop(i))
            self._worker_tasks.append(worker_task)
    
    async def _start(self) -> None:
        """Start job scheduler service."""
        self.logger.info("Job scheduler started")
    
    async def _stop(self) -> None:
        """Stop job scheduler service."""
        # Cancel scheduler task
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Cancel worker tasks
        for task in self._worker_tasks:
            task.cancel()
        
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        # Cancel running job tasks
        for task in self._running_tasks.values():
            task.cancel()
        
        if self._running_tasks:
            await asyncio.gather(*self._running_tasks.values(), return_exceptions=True)
        
        self.logger.info("Job scheduler stopped")
    
    async def _cleanup(self) -> None:
        """Cleanup scheduler resources."""
        self._pending_queue.clear()
        self._running_tasks.clear()
        self._worker_tasks.clear()
    
    async def _check_health(self) -> bool:
        """Check scheduler health."""
        return len(self._running_tasks) <= self.max_workers
    
    def register_job(self, job: Job) -> str:
        """
        Register a job with the scheduler.
        
        Args:
            job: Job to register
            
        Returns:
            str: Job ID
            
        Raises:
            ValueError: If job is invalid
        """
        job.validate()
        
        if not job.id:
            job.id = str(uuid.uuid4())
        
        self._jobs[job.id] = job
        
        # Create execution instance
        execution = JobExecution(job=job)
        self._executions[job.id] = execution
        
        # Schedule job
        if job.schedule:
            self.logger.info(f"Registered scheduled job: {job.name} ({job.id})")
        else:
            # One-time job, add to queue
            asyncio.create_task(self._enqueue_job(execution))
            self.logger.info(f"Registered one-time job: {job.name} ({job.id})")
        
        return job.id
    
    async def submit_job(
        self,
        task: Callable,
        args: Optional[Tuple] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        priority: JobPriority = JobPriority.NORMAL,
        dependencies: Optional[List[str]] = None
    ) -> str:
        """
        Submit a one-time job for execution.
        
        Args:
            task: Task to execute
            args: Task arguments
            kwargs: Task keyword arguments
            name: Job name
            priority: Job priority
            dependencies: Job dependencies
            
        Returns:
            str: Job ID
        """
        job = Job(
            name=name or task.__name__,
            task=task,
            args=args or (),
            kwargs=kwargs or {},
            priority=priority,
            dependencies=dependencies or []
        )
        
        return self.register_job(job)
    
    async def _scheduler_loop(self) -> None:
        """Background loop for checking and scheduling jobs."""
        while True:
            try:
                await asyncio.sleep(self.check_interval)
                await self._check_scheduled_jobs()
                await self._check_dependencies()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Scheduler loop error: {e}")
    
    async def _check_scheduled_jobs(self) -> None:
        """Check for jobs that need to be scheduled."""
        now = datetime.now()
        
        for job_id, execution in self._executions.items():
            job = execution.job
            
            # Skip if not scheduled or already running
            if not job.schedule or execution.status == JobStatus.RUNNING:
                continue
            
            # Check if max runs reached
            if job.schedule.max_runs and execution.run_count >= job.schedule.max_runs:
                continue
            
            # Get next run time
            last_run = execution.completed_at
            next_run = job.schedule.get_next_run_time(last_run)
            
            if next_run and next_run <= now:
                # Time to run the job
                new_execution = JobExecution(job=job)
                new_execution.run_count = execution.run_count + 1
                
                await self._enqueue_job(new_execution)
                
                # Update original execution
                execution.run_count = new_execution.run_count
                
                self.logger.debug(f"Scheduled job {job.name} for execution")
    
    async def _check_dependencies(self) -> None:
        """Check job dependencies and release blocked jobs."""
        async with self._queue_lock:
            ready_jobs = []
            
            for execution in list(self._pending_queue):
                if execution.status != JobStatus.PENDING:
                    continue
                
                # Check if all dependencies are completed
                dependencies_met = True
                for dep_id in execution.job.dependencies:
                    if dep_id in self._completed_jobs:
                        dep_execution = self._completed_jobs[dep_id]
                        if dep_execution.status != JobStatus.COMPLETED:
                            dependencies_met = False
                            break
                    else:
                        dependencies_met = False
                        break
                
                if dependencies_met and execution.job.dependencies:
                    ready_jobs.append(execution)
            
            # Remove ready jobs from pending and re-add without dependencies
            for execution in ready_jobs:
                self._pending_queue.remove(execution)
                execution.job.dependencies = []
                heapq.heappush(self._pending_queue, execution)
                self._job_available.set()
    
    async def _enqueue_job(self, execution: JobExecution) -> None:
        """
        Add job to execution queue.
        
        Args:
            execution: Job execution instance
        """
        async with self._queue_lock:
            heapq.heappush(self._pending_queue, execution)
            self._job_available.set()
            
            self.logger.debug(
                f"Enqueued job {execution.job.name} with priority "
                f"{execution.job.priority.name}"
            )
    
    async def _worker_loop(self, worker_id: int) -> None:
        """
        Worker loop for processing jobs.
        
        Args:
            worker_id: Worker identifier
        """
        self.logger.debug(f"Worker {worker_id} started")
        
        while True:
            try:
                # Wait for job availability
                await self._job_available.wait()
                
                # Get next job from queue
                execution = await self._get_next_job()
                
                if execution:
                    self.logger.debug(
                        f"Worker {worker_id} executing job {execution.job.name}"
                    )
                    
                    # Execute job
                    task = asyncio.create_task(self._execute_job(execution))
                    self._running_tasks[execution.job.id] = task
                    
                    # Wait for completion
                    await task
                    
                    # Remove from running tasks
                    self._running_tasks.pop(execution.job.id, None)
                    
                    # Move to completed
                    self._completed_jobs[execution.job.id] = execution
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
    
    async def _get_next_job(self) -> Optional[JobExecution]:
        """
        Get next job from priority queue.
        
        Returns:
            JobExecution: Next job to execute or None
        """
        async with self._queue_lock:
            # Find job without dependencies
            for i, execution in enumerate(self._pending_queue):
                if not execution.job.dependencies:
                    # Remove from queue
                    self._pending_queue.pop(i)
                    heapq.heapify(self._pending_queue)
                    
                    # Clear event if queue is empty
                    if not self._pending_queue:
                        self._job_available.clear()
                    
                    return execution
            
            # No jobs without dependencies
            self._job_available.clear()
            return None
    
    async def _execute_job(self, execution: JobExecution) -> None:
        """
        Execute a single job.
        
        Args:
            execution: Job execution instance
        """
        execution.status = JobStatus.RUNNING
        execution.started_at = datetime.now()
        
        try:
            # Execute with timeout
            if asyncio.iscoroutinefunction(execution.job.task):
                result = await asyncio.wait_for(
                    execution.job.task(*execution.job.args, **execution.job.kwargs),
                    timeout=execution.job.timeout
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        execution.job.task,
                        *execution.job.args,
                        **execution.job.kwargs
                    ),
                    timeout=execution.job.timeout
                )
            
            execution.result = result
            execution.status = JobStatus.COMPLETED
            execution.completed_at = datetime.now()
            
            self.logger.info(
                f"Job {execution.job.name} completed successfully in "
                f"{(execution.completed_at - execution.started_at).total_seconds():.2f}s"
            )
            
        except asyncio.TimeoutError:
            execution.status = JobStatus.FAILED
            execution.error = f"Job timed out after {execution.job.timeout} seconds"
            execution.completed_at = datetime.now()
            
            await self._handle_job_failure(execution)
            
        except Exception as e:
            execution.status = JobStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.now()
            
            self.logger.error(f"Job {execution.job.name} failed: {e}")
            
            await self._handle_job_failure(execution)
    
    async def _handle_job_failure(self, execution: JobExecution) -> None:
        """
        Handle job failure with retry logic.
        
        Args:
            execution: Failed job execution
        """
        retry_policy = execution.job.retry_policy
        
        if execution.retry_count < retry_policy["max_retries"]:
            execution.retry_count += 1
            execution.status = JobStatus.RETRYING
            
            # Calculate retry delay
            delay = retry_policy["retry_delay"]
            if retry_policy.get("exponential_backoff"):
                delay = delay * (2 ** (execution.retry_count - 1))
            
            self.logger.info(
                f"Retrying job {execution.job.name} "
                f"(attempt {execution.retry_count}/{retry_policy['max_retries']}) "
                f"after {delay}s delay"
            )
            
            # Schedule retry
            await asyncio.sleep(delay)
            execution.status = JobStatus.PENDING
            await self._enqueue_job(execution)
        else:
            self.logger.error(
                f"Job {execution.job.name} failed after "
                f"{retry_policy['max_retries']} retries"
            )
    
    def get_job_status(self, job_id: str) -> Optional[JobExecution]:
        """
        Get job execution status.
        
        Args:
            job_id: Job identifier
            
        Returns:
            JobExecution: Job execution instance or None
        """
        return self._executions.get(job_id) or self._completed_jobs.get(job_id)
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending or running job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            bool: True if cancelled successfully
        """
        # Check if job is running
        if job_id in self._running_tasks:
            task = self._running_tasks[job_id]
            task.cancel()
            
            if job_id in self._executions:
                self._executions[job_id].status = JobStatus.CANCELLED
            
            self.logger.info(f"Cancelled running job: {job_id}")
            return True
        
        # Check if job is pending
        async with self._queue_lock:
            for execution in self._pending_queue:
                if execution.job.id == job_id:
                    self._pending_queue.remove(execution)
                    heapq.heapify(self._pending_queue)
                    execution.status = JobStatus.CANCELLED
                    
                    self.logger.info(f"Cancelled pending job: {job_id}")
                    return True
        
        return False
