"""
Base Service Implementation for AG News Text Classification
================================================================================
This module provides the abstract base class for all services in the system,
implementing common functionality and enforcing interface contracts.

The base service follows the Template Method pattern and provides lifecycle
management, health checking, and configuration handling.

References:
    - Gamma, E., et al. (1994). Design Patterns: Elements of Reusable Object-Oriented Software
    - Martin, R. C. (2017). Clean Architecture: A Craftsman's Guide to Software Structure

Author: Võ Hải Dũng
License: MIT
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import threading
from contextlib import contextmanager

from src.core.exceptions import ServiceException
from src.utils.logging_config import get_logger


class ServiceStatus(Enum):
    """
    Enumeration of service lifecycle states.
    
    States follow the standard service lifecycle:
    - INITIALIZING: Service is being configured
    - READY: Service is configured but not started
    - RUNNING: Service is actively processing
    - PAUSED: Service is temporarily suspended
    - STOPPING: Service is shutting down
    - STOPPED: Service has been terminated
    - ERROR: Service encountered a critical error
    """
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ServiceConfig:
    """
    Configuration container for service instances.
    
    Attributes:
        name: Unique service identifier
        version: Service version string
        timeout: Maximum execution time in seconds
        retry_policy: Retry configuration
        dependencies: List of required service names
        metadata: Additional service metadata
    """
    name: str
    version: str = "1.0.0"
    timeout: int = 300
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {
        "max_retries": 3,
        "backoff_factor": 2,
        "max_backoff": 60
    })
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.name:
            raise ValueError("Service name is required")
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        if self.retry_policy["max_retries"] < 0:
            raise ValueError("Max retries must be non-negative")
        return True


@dataclass
class ServiceHealth:
    """
    Health status information for service monitoring.
    
    Attributes:
        status: Current service status
        is_healthy: Overall health indicator
        last_check: Timestamp of last health check
        metrics: Service-specific metrics
        errors: Recent error messages
    """
    status: ServiceStatus
    is_healthy: bool
    last_check: datetime
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert health status to dictionary representation."""
        return {
            "status": self.status.value,
            "is_healthy": self.is_healthy,
            "last_check": self.last_check.isoformat(),
            "metrics": self.metrics,
            "errors": self.errors[-10:]  # Keep last 10 errors
        }


class BaseService(ABC):
    """
    Abstract base class for all services in the system.
    
    This class provides common functionality for service lifecycle management,
    health monitoring, and configuration handling. Subclasses must implement
    the abstract methods to provide service-specific behavior.
    
    The service follows a standard lifecycle:
    1. Initialize with configuration
    2. Setup dependencies and resources
    3. Start processing
    4. Handle requests/tasks
    5. Cleanup and shutdown
    """
    
    def __init__(self, config: ServiceConfig):
        """
        Initialize base service with configuration.
        
        Args:
            config: Service configuration object
            
        Raises:
            ValueError: If configuration is invalid
        """
        config.validate()
        self.config = config
        self.logger = get_logger(f"service.{config.name}")
        
        # Service state management
        self._status = ServiceStatus.INITIALIZING
        self._health = ServiceHealth(
            status=self._status,
            is_healthy=False,
            last_check=datetime.now()
        )
        
        # Thread safety
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        
        # Lifecycle hooks
        self._startup_hooks: List[Callable] = []
        self._shutdown_hooks: List[Callable] = []
        
        # Performance metrics
        self._start_time: Optional[datetime] = None
        self._request_count = 0
        self._error_count = 0
        
        self.logger.info(f"Initialized service: {config.name} v{config.version}")
    
    @property
    def status(self) -> ServiceStatus:
        """Get current service status."""
        with self._lock:
            return self._status
    
    @status.setter
    def status(self, value: ServiceStatus):
        """Set service status with logging."""
        with self._lock:
            old_status = self._status
            self._status = value
            self._health.status = value
            self.logger.info(f"Service status changed: {old_status.value} -> {value.value}")
    
    @property
    def is_running(self) -> bool:
        """Check if service is currently running."""
        return self.status == ServiceStatus.RUNNING
    
    @property
    def uptime(self) -> Optional[timedelta]:
        """Calculate service uptime."""
        if self._start_time:
            return datetime.now() - self._start_time
        return None
    
    def add_startup_hook(self, hook: Callable) -> None:
        """
        Register a startup hook to be executed when service starts.
        
        Args:
            hook: Callable to execute on startup
        """
        self._startup_hooks.append(hook)
        self.logger.debug(f"Added startup hook: {hook.__name__}")
    
    def add_shutdown_hook(self, hook: Callable) -> None:
        """
        Register a shutdown hook to be executed when service stops.
        
        Args:
            hook: Callable to execute on shutdown
        """
        self._shutdown_hooks.append(hook)
        self.logger.debug(f"Added shutdown hook: {hook.__name__}")
    
    async def start(self) -> None:
        """
        Start the service and begin processing.
        
        This method orchestrates the service startup sequence:
        1. Execute startup hooks
        2. Initialize resources
        3. Start processing
        4. Update status
        
        Raises:
            ServiceException: If startup fails
        """
        try:
            self.logger.info(f"Starting service: {self.config.name}")
            self.status = ServiceStatus.INITIALIZING
            
            # Execute startup hooks
            for hook in self._startup_hooks:
                await self._execute_hook(hook, "startup")
            
            # Initialize service-specific resources
            await self._initialize()
            
            # Start processing
            self.status = ServiceStatus.RUNNING
            self._start_time = datetime.now()
            self._health.is_healthy = True
            
            # Start service-specific processing
            await self._start()
            
            self.logger.info(f"Service started successfully: {self.config.name}")
            
        except Exception as e:
            self.status = ServiceStatus.ERROR
            self._health.is_healthy = False
            self._health.errors.append(str(e))
            self.logger.error(f"Failed to start service: {e}")
            raise ServiceException(f"Service startup failed: {e}")
    
    async def stop(self) -> None:
        """
        Stop the service gracefully.
        
        This method orchestrates the service shutdown sequence:
        1. Signal stop to processing threads
        2. Wait for graceful completion
        3. Cleanup resources
        4. Execute shutdown hooks
        
        Raises:
            ServiceException: If shutdown fails
        """
        try:
            self.logger.info(f"Stopping service: {self.config.name}")
            self.status = ServiceStatus.STOPPING
            self._stop_event.set()
            
            # Stop service-specific processing
            await self._stop()
            
            # Cleanup resources
            await self._cleanup()
            
            # Execute shutdown hooks
            for hook in self._shutdown_hooks:
                await self._execute_hook(hook, "shutdown")
            
            self.status = ServiceStatus.STOPPED
            self._health.is_healthy = False
            self.logger.info(f"Service stopped successfully: {self.config.name}")
            
        except Exception as e:
            self.status = ServiceStatus.ERROR
            self._health.errors.append(str(e))
            self.logger.error(f"Failed to stop service: {e}")
            raise ServiceException(f"Service shutdown failed: {e}")
    
    async def restart(self) -> None:
        """
        Restart the service by stopping and starting it.
        
        Raises:
            ServiceException: If restart fails
        """
        self.logger.info(f"Restarting service: {self.config.name}")
        await self.stop()
        await asyncio.sleep(1)  # Brief pause between stop and start
        await self.start()
    
    async def health_check(self) -> ServiceHealth:
        """
        Perform health check and return status.
        
        Returns:
            ServiceHealth: Current health status
        """
        try:
            # Update last check time
            self._health.last_check = datetime.now()
            
            # Perform service-specific health check
            is_healthy = await self._check_health()
            self._health.is_healthy = is_healthy and self.is_running
            
            # Update metrics
            self._health.metrics.update({
                "uptime_seconds": self.uptime.total_seconds() if self.uptime else 0,
                "request_count": self._request_count,
                "error_count": self._error_count,
                "error_rate": self._error_count / max(self._request_count, 1)
            })
            
            return self._health
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self._health.is_healthy = False
            self._health.errors.append(str(e))
            return self._health
    
    @contextmanager
    def track_request(self):
        """
        Context manager for tracking service requests.
        
        Usage:
            with service.track_request():
                # Process request
                pass
        """
        self._request_count += 1
        try:
            yield
        except Exception as e:
            self._error_count += 1
            self._health.errors.append(str(e))
            raise
    
    async def _execute_hook(self, hook: Callable, hook_type: str) -> None:
        """
        Execute a lifecycle hook with error handling.
        
        Args:
            hook: Callable to execute
            hook_type: Type of hook (startup/shutdown)
        """
        try:
            if asyncio.iscoroutinefunction(hook):
                await hook()
            else:
                hook()
        except Exception as e:
            self.logger.error(f"Error executing {hook_type} hook {hook.__name__}: {e}")
    
    @abstractmethod
    async def _initialize(self) -> None:
        """
        Initialize service-specific resources.
        
        This method should set up any resources needed by the service,
        such as database connections, model loading, etc.
        """
        pass
    
    @abstractmethod
    async def _start(self) -> None:
        """
        Start service-specific processing.
        
        This method should begin the main processing loop or task handling
        for the service.
        """
        pass
    
    @abstractmethod
    async def _stop(self) -> None:
        """
        Stop service-specific processing.
        
        This method should gracefully stop any ongoing processing and
        ensure all tasks are completed or cancelled appropriately.
        """
        pass
    
    @abstractmethod
    async def _cleanup(self) -> None:
        """
        Cleanup service-specific resources.
        
        This method should release any resources held by the service,
        such as closing connections, freeing memory, etc.
        """
        pass
    
    @abstractmethod
    async def _check_health(self) -> bool:
        """
        Perform service-specific health check.
        
        Returns:
            bool: True if service is healthy
        """
        pass
