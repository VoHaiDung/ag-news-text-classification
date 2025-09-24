"""
Service Registry Implementation for AG News Text Classification
================================================================================
This module implements a centralized service registry following the Service
Locator pattern, managing service discovery and dependency injection.

The registry provides:
- Service registration and discovery
- Dependency resolution
- Lifecycle management
- Health monitoring

References:
    - Fowler, M. (2002). Patterns of Enterprise Application Architecture
    - Buschmann, F., et al. (1996). Pattern-Oriented Software Architecture

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
from typing import Dict, Type, Optional, List, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import threading
from contextlib import asynccontextmanager

from src.services.base_service import BaseService, ServiceStatus, ServiceHealth, ServiceConfig
from src.core.exceptions import ServiceException
from src.utils.logging_config import get_logger


@dataclass
class ServiceDescriptor:
    """
    Descriptor containing service metadata and instance.
    
    Attributes:
        service_class: Service class type
        instance: Service instance
        config: Service configuration
        dependencies: Set of dependency service names
        dependents: Set of dependent service names
        registered_at: Registration timestamp
    """
    service_class: Type[BaseService]
    instance: Optional[BaseService] = None
    config: Optional[ServiceConfig] = None
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    registered_at: datetime = field(default_factory=datetime.now)


class ServiceRegistry:
    """
    Centralized service registry for managing service lifecycle and dependencies.
    
    The registry implements the Service Locator pattern and provides:
    - Service registration with dependency tracking
    - Automatic dependency resolution
    - Coordinated startup/shutdown
    - Health monitoring across all services
    
    Thread-safe implementation supporting concurrent access.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """
        Implement singleton pattern for global registry access.
        
        Returns:
            ServiceRegistry: Singleton instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize service registry."""
        if not hasattr(self, '_initialized'):
            self._services: Dict[str, ServiceDescriptor] = {}
            self._lock = threading.RLock()
            self.logger = get_logger("service.registry")
            self._initialized = True
            self.logger.info("Service registry initialized")
    
    def register(
        self,
        name: str,
        service_class: Type[BaseService],
        config: Optional[ServiceConfig] = None,
        dependencies: Optional[List[str]] = None
    ) -> None:
        """
        Register a service class with the registry.
        
        Args:
            name: Unique service identifier
            service_class: Service class to register
            config: Service configuration
            dependencies: List of required service names
            
        Raises:
            ValueError: If service is already registered
            TypeError: If service_class is not a BaseService subclass
        """
        with self._lock:
            if name in self._services:
                raise ValueError(f"Service already registered: {name}")
            
            if not issubclass(service_class, BaseService):
                raise TypeError(f"Service must inherit from BaseService: {service_class}")
            
            # Create configuration if not provided
            if config is None:
                config = ServiceConfig(name=name)
            
            # Create service descriptor
            descriptor = ServiceDescriptor(
                service_class=service_class,
                config=config,
                dependencies=set(dependencies or [])
            )
            
            # Update dependency graph
            for dep_name in descriptor.dependencies:
                if dep_name in self._services:
                    self._services[dep_name].dependents.add(name)
            
            self._services[name] = descriptor
            self.logger.info(f"Registered service: {name} ({service_class.__name__})")
    
    def unregister(self, name: str) -> None:
        """
        Unregister a service from the registry.
        
        Args:
            name: Service name to unregister
            
        Raises:
            ValueError: If service is not registered or has dependents
        """
        with self._lock:
            if name not in self._services:
                raise ValueError(f"Service not registered: {name}")
            
            descriptor = self._services[name]
            
            # Check for dependents
            if descriptor.dependents:
                raise ValueError(
                    f"Cannot unregister service with dependents: {descriptor.dependents}"
                )
            
            # Stop service if running
            if descriptor.instance and descriptor.instance.is_running:
                asyncio.create_task(descriptor.instance.stop())
            
            # Update dependency graph
            for dep_name in descriptor.dependencies:
                if dep_name in self._services:
                    self._services[dep_name].dependents.discard(name)
            
            del self._services[name]
            self.logger.info(f"Unregistered service: {name}")
    
    def get(self, name: str) -> Optional[BaseService]:
        """
        Get a service instance by name.
        
        Args:
            name: Service name
            
        Returns:
            BaseService: Service instance or None if not found
        """
        with self._lock:
            descriptor = self._services.get(name)
            return descriptor.instance if descriptor else None
    
    def get_or_create(self, name: str) -> BaseService:
        """
        Get or create a service instance.
        
        Args:
            name: Service name
            
        Returns:
            BaseService: Service instance
            
        Raises:
            ValueError: If service is not registered
        """
        with self._lock:
            if name not in self._services:
                raise ValueError(f"Service not registered: {name}")
            
            descriptor = self._services[name]
            
            if descriptor.instance is None:
                descriptor.instance = descriptor.service_class(descriptor.config)
                self.logger.info(f"Created service instance: {name}")
            
            return descriptor.instance
    
    async def start_service(self, name: str) -> None:
        """
        Start a service and its dependencies.
        
        Args:
            name: Service name to start
            
        Raises:
            ServiceException: If service startup fails
        """
        # Get topological order for dependency resolution
        start_order = self._get_dependency_order([name])
        
        for service_name in start_order:
            service = self.get_or_create(service_name)
            if not service.is_running:
                self.logger.info(f"Starting service: {service_name}")
                await service.start()
    
    async def stop_service(self, name: str) -> None:
        """
        Stop a service and its dependents.
        
        Args:
            name: Service name to stop
            
        Raises:
            ServiceException: If service shutdown fails
        """
        # Get reverse topological order for dependent resolution
        stop_order = self._get_dependent_order([name])
        
        for service_name in stop_order:
            service = self.get(service_name)
            if service and service.is_running:
                self.logger.info(f"Stopping service: {service_name}")
                await service.stop()
    
    async def start_all(self) -> None:
        """
        Start all registered services in dependency order.
        
        Raises:
            ServiceException: If any service fails to start
        """
        self.logger.info("Starting all services")
        start_order = self._get_dependency_order(list(self._services.keys()))
        
        for name in start_order:
            service = self.get_or_create(name)
            if not service.is_running:
                await service.start()
        
        self.logger.info("All services started successfully")
    
    async def stop_all(self) -> None:
        """
        Stop all running services in reverse dependency order.
        
        Raises:
            ServiceException: If any service fails to stop
        """
        self.logger.info("Stopping all services")
        stop_order = self._get_dependency_order(list(self._services.keys()))
        stop_order.reverse()
        
        for name in stop_order:
            service = self.get(name)
            if service and service.is_running:
                await service.stop()
        
        self.logger.info("All services stopped successfully")
    
    async def health_check_all(self) -> Dict[str, ServiceHealth]:
        """
        Perform health check on all services.
        
        Returns:
            Dict[str, ServiceHealth]: Health status for each service
        """
        health_status = {}
        
        for name, descriptor in self._services.items():
            if descriptor.instance:
                health = await descriptor.instance.health_check()
                health_status[name] = health
            else:
                # Service not instantiated
                health_status[name] = ServiceHealth(
                    status=ServiceStatus.STOPPED,
                    is_healthy=False,
                    last_check=datetime.now()
                )
        
        return health_status
    
    def get_service_graph(self) -> Dict[str, Dict[str, Any]]:
        """
        Get service dependency graph.
        
        Returns:
            Dict: Service graph with dependencies and dependents
        """
        graph = {}
        
        with self._lock:
            for name, descriptor in self._services.items():
                graph[name] = {
                    "class": descriptor.service_class.__name__,
                    "status": descriptor.instance.status.value if descriptor.instance else "not_created",
                    "dependencies": list(descriptor.dependencies),
                    "dependents": list(descriptor.dependents),
                    "registered_at": descriptor.registered_at.isoformat()
                }
        
        return graph
    
    def _get_dependency_order(self, services: List[str]) -> List[str]:
        """
        Get topological order for service dependencies.
        
        Args:
            services: List of service names
            
        Returns:
            List[str]: Services in dependency order
            
        Raises:
            ValueError: If circular dependency detected
        """
        visited = set()
        stack = []
        temp_visited = set()
        
        def visit(name: str):
            if name in temp_visited:
                raise ValueError(f"Circular dependency detected involving: {name}")
            if name in visited:
                return
            
            temp_visited.add(name)
            
            if name in self._services:
                for dep in self._services[name].dependencies:
                    if dep in services:
                        visit(dep)
            
            temp_visited.remove(name)
            visited.add(name)
            stack.append(name)
        
        for service in services:
            if service not in visited:
                visit(service)
        
        return stack
    
    def _get_dependent_order(self, services: List[str]) -> List[str]:
        """
        Get reverse topological order for service dependents.
        
        Args:
            services: List of service names
            
        Returns:
            List[str]: Services in dependent order
        """
        result = []
        visited = set()
        
        def visit(name: str):
            if name in visited:
                return
            visited.add(name)
            
            if name in self._services:
                for dependent in self._services[name].dependents:
                    visit(dependent)
            
            result.append(name)
        
        for service in services:
            visit(service)
        
        return result
    
    @asynccontextmanager
    async def service_context(self, *service_names: str):
        """
        Context manager for service lifecycle management.
        
        Args:
            *service_names: Service names to manage
            
        Usage:
            async with registry.service_context("service1", "service2"):
                # Services are started
                # Use services
                pass
            # Services are stopped
        """
        started_services = []
        
        try:
            # Start services
            for name in service_names:
                await self.start_service(name)
                started_services.append(name)
            
            yield self
            
        finally:
            # Stop services in reverse order
            for name in reversed(started_services):
                try:
                    await self.stop_service(name)
                except Exception as e:
                    self.logger.error(f"Error stopping service {name}: {e}")
    
    def clear(self) -> None:
        """Clear all registered services."""
        with self._lock:
            self._services.clear()
            self.logger.info("Service registry cleared")


# Global registry instance
registry = ServiceRegistry()
