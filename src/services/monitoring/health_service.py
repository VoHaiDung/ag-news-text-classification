"""
Health Service Implementation for AG News Text Classification
================================================================================
This module implements comprehensive health monitoring for all system components,
providing health checks, status aggregation, and failure detection.

The health service provides:
- Component health checks
- Dependency verification
- Resource monitoring
- Health status aggregation

References:
    - Google SRE Book: Monitoring Distributed Systems
    - Health Check Response Format for HTTP APIs (RFC)

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
import psutil
import aiohttp
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from src.services.base_service import BaseService, ServiceConfig, ServiceHealth
from src.core.exceptions import ServiceException
from src.utils.logging_config import get_logger


class HealthStatus(Enum):
    """
    Health status levels.
    
    Levels:
        HEALTHY: All checks passing
        DEGRADED: Some non-critical checks failing
        UNHEALTHY: Critical checks failing
        UNKNOWN: Unable to determine status
    """
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """
    Health status for a system component.
    
    Attributes:
        name: Component name
        status: Health status
        checks: Individual check results
        message: Status message
        last_check: Last check timestamp
        metadata: Additional health metadata
    """
    name: str
    status: HealthStatus
    checks: Dict[str, bool] = field(default_factory=dict)
    message: str = ""
    last_check: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "status": self.status.value,
            "checks": self.checks,
            "message": self.message,
            "last_check": self.last_check.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class HealthCheck:
    """
    Health check definition.
    
    Attributes:
        name: Check name
        component: Component to check
        check_fn: Check function
        critical: Whether check is critical
        timeout: Check timeout in seconds
        interval: Check interval in seconds
    """
    name: str
    component: str
    check_fn: Callable
    critical: bool = True
    timeout: int = 5
    interval: int = 30


class HealthService(BaseService):
    """
    Service for monitoring system health and component status.
    
    This service performs periodic health checks on all system components
    and provides aggregated health status.
    """
    
    def __init__(
        self,
        config: Optional[ServiceConfig] = None,
        check_interval: int = 30,
        enable_auto_recovery: bool = True
    ):
        """
        Initialize health service.
        
        Args:
            config: Service configuration
            check_interval: Default check interval in seconds
            enable_auto_recovery: Enable automatic recovery attempts
        """
        if config is None:
            config = ServiceConfig(name="health_service")
        super().__init__(config)
        
        self.check_interval = check_interval
        self.enable_auto_recovery = enable_auto_recovery
        
        # Health checks registry
        self._health_checks: List[HealthCheck] = []
        self._component_health: Dict[str, ComponentHealth] = {}
        
        # Background check task
        self._check_task: Optional[asyncio.Task] = None
        
        # Health callbacks
        self._status_callbacks: Dict[HealthStatus, List[Callable]] = {
            status: [] for status in HealthStatus
        }
        
        # Resource thresholds
        self.resource_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0
        }
        
        self.logger = get_logger("service.health")
    
    async def _initialize(self) -> None:
        """Initialize health service."""
        self.logger.info("Initializing health service")
        
        # Register default health checks
        self._register_default_checks()
        
        # Start background health check task
        self._check_task = asyncio.create_task(self._health_check_loop())
    
    async def _start(self) -> None:
        """Start health service."""
        self.logger.info("Health service started")
    
    async def _stop(self) -> None:
        """Stop health service."""
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Health service stopped")
    
    async def _cleanup(self) -> None:
        """Cleanup health service resources."""
        self._health_checks.clear()
        self._component_health.clear()
    
    async def _check_health(self) -> bool:
        """
        Check health service's own health.
        
        Returns:
            bool: True if healthy
        """
        return len(self._health_checks) > 0
    
    def _register_default_checks(self) -> None:
        """Register default system health checks."""
        # CPU check
        self.register_check(
            HealthCheck(
                name="cpu_usage",
                component="system",
                check_fn=self._check_cpu,
                critical=False,
                interval=10
            )
        )
        
        # Memory check
        self.register_check(
            HealthCheck(
                name="memory_usage",
                component="system",
                check_fn=self._check_memory,
                critical=True,
                interval=10
            )
        )
        
        # Disk check
        self.register_check(
            HealthCheck(
                name="disk_usage",
                component="system",
                check_fn=self._check_disk,
                critical=True,
                interval=60
            )
        )
        
        # Service registry check
        self.register_check(
            HealthCheck(
                name="service_registry",
                component="services",
                check_fn=self._check_service_registry,
                critical=True,
                interval=30
            )
        )
    
    def register_check(self, check: HealthCheck) -> None:
        """
        Register a health check.
        
        Args:
            check: Health check to register
        """
        self._health_checks.append(check)
        
        # Initialize component health if not exists
        if check.component not in self._component_health:
            self._component_health[check.component] = ComponentHealth(
                name=check.component,
                status=HealthStatus.UNKNOWN
            )
        
        self.logger.info(f"Registered health check: {check.name} for {check.component}")
    
    async def _health_check_loop(self) -> None:
        """Background loop for performing health checks."""
        check_timers = {}
        
        while True:
            try:
                current_time = datetime.now()
                
                for check in self._health_checks:
                    # Check if it's time to run this check
                    last_run = check_timers.get(check.name)
                    
                    if last_run is None or \
                       (current_time - last_run).total_seconds() >= check.interval:
                        # Run check asynchronously
                        asyncio.create_task(self._run_health_check(check))
                        check_timers[check.name] = current_time
                
                # Aggregate health status
                await self._aggregate_health()
                
                # Sleep for a short interval
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(5)
    
    async def _run_health_check(self, check: HealthCheck) -> None:
        """
        Run a single health check.
        
        Args:
            check: Health check to run
        """
        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                self._execute_check(check.check_fn),
                timeout=check.timeout
            )
            
            # Update component health
            component = self._component_health[check.component]
            component.checks[check.name] = result
            component.last_check = datetime.now()
            
            if not result:
                self.logger.warning(
                    f"Health check failed: {check.name} for {check.component}"
                )
                
                if check.critical:
                    component.status = HealthStatus.UNHEALTHY
                    component.message = f"Critical check '{check.name}' failed"
                else:
                    if component.status == HealthStatus.HEALTHY:
                        component.status = HealthStatus.DEGRADED
                        component.message = f"Non-critical check '{check.name}' failed"
            
        except asyncio.TimeoutError:
            self.logger.error(f"Health check timed out: {check.name}")
            component = self._component_health[check.component]
            component.checks[check.name] = False
            component.status = HealthStatus.UNHEALTHY
            component.message = f"Check '{check.name}' timed out"
            
        except Exception as e:
            self.logger.error(f"Health check error for {check.name}: {e}")
            component = self._component_health[check.component]
            component.checks[check.name] = False
            component.status = HealthStatus.UNKNOWN
            component.message = f"Check '{check.name}' error: {e}"
    
    async def _execute_check(self, check_fn: Callable) -> bool:
        """
        Execute a check function.
        
        Args:
            check_fn: Check function to execute
            
        Returns:
            bool: Check result
        """
        if asyncio.iscoroutinefunction(check_fn):
            return await check_fn()
        else:
            return check_fn()
    
    async def _aggregate_health(self) -> None:
        """Aggregate component health into overall status."""
        if not self._component_health:
            return
        
        # Count statuses
        status_counts = {status: 0 for status in HealthStatus}
        
        for component in self._component_health.values():
            status_counts[component.status] += 1
        
        # Determine overall status
        if status_counts[HealthStatus.UNHEALTHY] > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED] > 0:
            overall_status = HealthStatus.DEGRADED
        elif status_counts[HealthStatus.UNKNOWN] == len(self._component_health):
            overall_status = HealthStatus.UNKNOWN
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Update service health
        self._health.status = ServiceStatus.ERROR if overall_status == HealthStatus.UNHEALTHY \
            else ServiceStatus.RUNNING
        self._health.is_healthy = overall_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        
        # Trigger callbacks
        await self._trigger_status_callbacks(overall_status)
        
        # Attempt auto-recovery if enabled
        if self.enable_auto_recovery and overall_status == HealthStatus.UNHEALTHY:
            await self._attempt_recovery()
    
    async def _trigger_status_callbacks(self, status: HealthStatus) -> None:
        """
        Trigger callbacks for status change.
        
        Args:
            status: Current health status
        """
        for callback in self._status_callbacks[status]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(status, self._component_health)
                else:
                    callback(status, self._component_health)
            except Exception as e:
                self.logger.error(f"Status callback error: {e}")
    
    async def _attempt_recovery(self) -> None:
        """Attempt to recover unhealthy components."""
        self.logger.info("Attempting auto-recovery for unhealthy components")
        
        # Get service registry
        from src.services.service_registry import registry
        
        for component_name, health in self._component_health.items():
            if health.status == HealthStatus.UNHEALTHY:
                # Try to restart service if it's a registered service
                service = registry.get(component_name)
                if service:
                    try:
                        self.logger.info(f"Restarting unhealthy service: {component_name}")
                        await service.restart()
                    except Exception as e:
                        self.logger.error(f"Failed to restart {component_name}: {e}")
    
    async def _check_cpu(self) -> bool:
        """
        Check CPU usage.
        
        Returns:
            bool: True if within threshold
        """
        cpu_percent = psutil.cpu_percent(interval=1)
        
        self._component_health["system"].metadata["cpu_percent"] = cpu_percent
        
        return cpu_percent < self.resource_thresholds["cpu_percent"]
    
    async def _check_memory(self) -> bool:
        """
        Check memory usage.
        
        Returns:
            bool: True if within threshold
        """
        memory = psutil.virtual_memory()
        
        self._component_health["system"].metadata.update({
            "memory_percent": memory.percent,
            "memory_available": memory.available,
            "memory_total": memory.total
        })
        
        return memory.percent < self.resource_thresholds["memory_percent"]
    
    async def _check_disk(self) -> bool:
        """
        Check disk usage.
        
        Returns:
            bool: True if within threshold
        """
        disk = psutil.disk_usage("/")
        
        self._component_health["system"].metadata.update({
            "disk_percent": disk.percent,
            "disk_free": disk.free,
            "disk_total": disk.total
        })
        
        return disk.percent < self.resource_thresholds["disk_percent"]
    
    async def _check_service_registry(self) -> bool:
        """
        Check service registry health.
        
        Returns:
            bool: True if healthy
        """
        from src.services.service_registry import registry
        
        try:
            # Check all registered services
            all_healthy = True
            service_statuses = await registry.health_check_all()
            
            for name, health in service_statuses.items():
                if not health.is_healthy:
                    all_healthy = False
                    self.logger.warning(f"Service unhealthy: {name}")
            
            return all_healthy
            
        except Exception as e:
            self.logger.error(f"Service registry check failed: {e}")
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get current health status.
        
        Returns:
            Dict[str, Any]: Health status summary
        """
        # Determine overall status
        overall_status = HealthStatus.HEALTHY
        
        for component in self._component_health.values():
            if component.status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
                break
            elif component.status == HealthStatus.DEGRADED:
                overall_status = HealthStatus.DEGRADED
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "components": {
                name: health.to_dict()
                for name, health in self._component_health.items()
            },
            "checks_total": len(self._health_checks),
            "checks_passing": sum(
                1 for c in self._component_health.values()
                if c.status == HealthStatus.HEALTHY
            )
        }
    
    def add_status_callback(
        self,
        status: HealthStatus,
        callback: Callable
    ) -> None:
        """
        Add callback for status changes.
        
        Args:
            status: Status to trigger on
            callback: Callback function
        """
        self._status_callbacks[status].append(callback)
        self.logger.debug(f"Added status callback for {status.value}")
