"""
Service Utilities for AG News Text Classification System
================================================================================
This module provides utility functions and classes for service management,
communication, and coordination. It implements common patterns for microservices
architecture including service discovery, health checking, and resilience patterns.

The utilities follow best practices for distributed systems and microservices
as outlined in cloud-native computing principles.

References:
    - Richardson, C. (2018). Microservices Patterns: With Examples in Java
    - Newman, S. (2021). Building Microservices: Designing Fine-Grained Systems
    - Kleppmann, M. (2017). Designing Data-Intensive Applications

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
import functools
import logging
import time
from typing import Any, Dict, List, Optional, Callable, TypeVar, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import random
from contextlib import asynccontextmanager
import aiohttp
import grpc
from concurrent.futures import ThreadPoolExecutor
import consul
import redis
from prometheus_client import Counter, Histogram, Gauge
import circuitbreaker

from ..core.exceptions import ServiceError, ValidationError
from ..core.registry import ServiceRegistry

# Configure logging
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

# Metrics
service_call_counter = Counter(
    'service_calls_total',
    'Total number of service calls',
    ['service', 'method', 'status']
)
service_latency_histogram = Histogram(
    'service_call_duration_seconds',
    'Service call duration',
    ['service', 'method']
)
circuit_breaker_state = Gauge(
    'circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=open, 2=half-open)',
    ['service']
)


class ServiceState(Enum):
    """Service state enumeration"""
    STARTING = "starting"
    READY = "ready"
    UNHEALTHY = "unhealthy"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class ServiceConfig:
    """
    Service configuration container
    
    Attributes:
        name: Service name
        host: Service host
        port: Service port
        protocol: Communication protocol
        timeout: Request timeout in seconds
        retries: Number of retry attempts
        circuit_breaker_threshold: Circuit breaker failure threshold
        health_check_interval: Health check interval in seconds
    """
    name: str
    host: str = "localhost"
    port: int = 8000
    protocol: str = "http"
    timeout: int = 30
    retries: int = 3
    circuit_breaker_threshold: int = 5
    health_check_interval: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)


class ServiceDiscovery:
    """
    Service discovery implementation using Consul
    
    Provides service registration, discovery, and health checking
    capabilities for microservices architecture.
    """
    
    def __init__(self, consul_host: str = "localhost", consul_port: int = 8500):
        """
        Initialize service discovery
        
        Args:
            consul_host: Consul server host
            consul_port: Consul server port
        """
        self.consul = consul.Consul(host=consul_host, port=consul_port)
        self._service_cache: Dict[str, List[ServiceConfig]] = {}
        self._cache_ttl = 30  # Cache TTL in seconds
        self._last_cache_update: Dict[str, datetime] = {}
    
    def register_service(self, config: ServiceConfig) -> bool:
        """
        Register service with Consul
        
        Args:
            config: Service configuration
            
        Returns:
            Registration success status
        """
        try:
            # Prepare service definition
            service_id = f"{config.name}-{config.host}-{config.port}"
            
            # Register service
            self.consul.agent.service.register(
                name=config.name,
                service_id=service_id,
                address=config.host,
                port=config.port,
                tags=[config.protocol],
                meta=config.metadata,
                check=consul.Check.http(
                    f"{config.protocol}://{config.host}:{config.port}/health",
                    interval=f"{config.health_check_interval}s",
                    timeout="5s",
                    deregister_critical_service_after="1m"
                )
            )
            
            logger.info(f"Service {service_id} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register service {config.name}: {e}")
            return False
    
    def discover_service(self, service_name: str, 
                        use_cache: bool = True) -> List[ServiceConfig]:
        """
        Discover service instances
        
        Args:
            service_name: Name of service to discover
            use_cache: Whether to use cached results
            
        Returns:
            List of service configurations
        """
        # Check cache
        if use_cache and self._is_cache_valid(service_name):
            return self._service_cache.get(service_name, [])
        
        try:
            # Query Consul
            _, services = self.consul.health.service(
                service_name, 
                passing=True
            )
            
            # Parse service instances
            configs = []
            for service in services:
                svc = service['Service']
                config = ServiceConfig(
                    name=svc['Service'],
                    host=svc['Address'],
                    port=svc['Port'],
                    protocol=svc['Tags'][0] if svc['Tags'] else 'http',
                    metadata=svc.get('Meta', {})
                )
                configs.append(config)
            
            # Update cache
            self._service_cache[service_name] = configs
            self._last_cache_update[service_name] = datetime.now()
            
            return configs
            
        except Exception as e:
            logger.error(f"Failed to discover service {service_name}: {e}")
            return []
    
    def deregister_service(self, service_id: str) -> bool:
        """
        Deregister service from Consul
        
        Args:
            service_id: Service identifier
            
        Returns:
            Deregistration success status
        """
        try:
            self.consul.agent.service.deregister(service_id)
            logger.info(f"Service {service_id} deregistered successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to deregister service {service_id}: {e}")
            return False
    
    def _is_cache_valid(self, service_name: str) -> bool:
        """Check if cache is still valid"""
        if service_name not in self._last_cache_update:
            return False
        
        age = datetime.now() - self._last_cache_update[service_name]
        return age.total_seconds() < self._cache_ttl


class LoadBalancer:
    """
    Load balancer for service instances
    
    Implements various load balancing strategies for distributing
    requests across multiple service instances.
    """
    
    def __init__(self, strategy: str = "round_robin"):
        """
        Initialize load balancer
        
        Args:
            strategy: Load balancing strategy
        """
        self.strategy = strategy
        self._round_robin_counters: Dict[str, int] = {}
        self._instance_weights: Dict[str, Dict[str, float]] = {}
    
    def select_instance(self, instances: List[ServiceConfig], 
                       service_name: str) -> Optional[ServiceConfig]:
        """
        Select service instance based on strategy
        
        Args:
            instances: Available service instances
            service_name: Service name for tracking
            
        Returns:
            Selected service configuration
        """
        if not instances:
            return None
        
        if self.strategy == "round_robin":
            return self._round_robin(instances, service_name)
        elif self.strategy == "random":
            return random.choice(instances)
        elif self.strategy == "least_connections":
            return self._least_connections(instances)
        elif self.strategy == "weighted":
            return self._weighted_selection(instances, service_name)
        else:
            # Default to first instance
            return instances[0]
    
    def _round_robin(self, instances: List[ServiceConfig], 
                    service_name: str) -> ServiceConfig:
        """Round-robin selection"""
        counter = self._round_robin_counters.get(service_name, 0)
        selected = instances[counter % len(instances)]
        self._round_robin_counters[service_name] = counter + 1
        return selected
    
    def _least_connections(self, instances: List[ServiceConfig]) -> ServiceConfig:
        """Least connections selection (placeholder)"""
        # In production, this would track actual connection counts
        return random.choice(instances)
    
    def _weighted_selection(self, instances: List[ServiceConfig],
                          service_name: str) -> ServiceConfig:
        """Weighted random selection"""
        if service_name not in self._instance_weights:
            # Initialize equal weights
            self._instance_weights[service_name] = {
                f"{i.host}:{i.port}": 1.0 for i in instances
            }
        
        weights = self._instance_weights[service_name]
        total_weight = sum(weights.values())
        
        r = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for instance in instances:
            key = f"{instance.host}:{instance.port}"
            cumulative_weight += weights.get(key, 1.0)
            if r <= cumulative_weight:
                return instance
        
        return instances[-1]


class CircuitBreaker:
    """
    Circuit breaker pattern implementation
    
    Provides fault tolerance by preventing cascading failures
    in distributed systems.
    """
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening
            recovery_timeout: Timeout before attempting recovery
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function with circuit breaker protection
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            ServiceError: If circuit is open
        """
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise ServiceError("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    async def async_call(self, func: Callable, *args, **kwargs) -> Any:
        """Async version of call"""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise ServiceError("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning("Circuit breaker opened")
    
    def _should_attempt_reset(self) -> bool:
        """Check if should attempt reset"""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[type, ...] = (Exception,)
) -> Callable:
    """
    Decorator for retry with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries
        backoff_factor: Backoff multiplication factor
        max_delay: Maximum delay between retries
        exceptions: Exceptions to catch and retry
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay} seconds..."
                        )
                        await asyncio.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        logger.error(f"All retry attempts failed: {e}")
            
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay} seconds..."
                        )
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        logger.error(f"All retry attempts failed: {e}")
            
            raise last_exception
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class ServiceProxy:
    """
    Proxy for service communication
    
    Provides a unified interface for communicating with services
    using different protocols (HTTP, gRPC).
    """
    
    def __init__(self, 
                 service_name: str,
                 discovery: Optional[ServiceDiscovery] = None,
                 load_balancer: Optional[LoadBalancer] = None):
        """
        Initialize service proxy
        
        Args:
            service_name: Target service name
            discovery: Service discovery instance
            load_balancer: Load balancer instance
        """
        self.service_name = service_name
        self.discovery = discovery or ServiceDiscovery()
        self.load_balancer = load_balancer or LoadBalancer()
        self.circuit_breaker = CircuitBreaker()
    
    @retry_with_backoff(max_retries=3, exceptions=(aiohttp.ClientError,))
    async def call_http(self, 
                        method: str,
                        path: str,
                        **kwargs) -> Dict[str, Any]:
        """
        Make HTTP call to service
        
        Args:
            method: HTTP method
            path: Request path
            **kwargs: Additional request parameters
            
        Returns:
            Response data
        """
        # Discover service instances
        instances = self.discovery.discover_service(self.service_name)
        if not instances:
            raise ServiceError(f"No instances found for service {self.service_name}")
        
        # Select instance
        instance = self.load_balancer.select_instance(instances, self.service_name)
        
        # Build URL
        url = f"{instance.protocol}://{instance.host}:{instance.port}{path}"
        
        # Make request with circuit breaker
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            try:
                async with session.request(method, url, **kwargs) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    # Record metrics
                    service_call_counter.labels(
                        service=self.service_name,
                        method=method,
                        status="success"
                    ).inc()
                    service_latency_histogram.labels(
                        service=self.service_name,
                        method=method
                    ).observe(time.time() - start_time)
                    
                    return data
                    
            except Exception as e:
                service_call_counter.labels(
                    service=self.service_name,
                    method=method,
                    status="failure"
                ).inc()
                raise ServiceError(f"Service call failed: {e}")
    
    async def call_grpc(self, 
                       method: str,
                       request: Any,
                       **kwargs) -> Any:
        """
        Make gRPC call to service
        
        Args:
            method: gRPC method name
            request: Request message
            **kwargs: Additional parameters
            
        Returns:
            Response message
        """
        # Discover service instances
        instances = self.discovery.discover_service(self.service_name)
        if not instances:
            raise ServiceError(f"No instances found for service {self.service_name}")
        
        # Select instance
        instance = self.load_balancer.select_instance(instances, self.service_name)
        
        # Create channel and stub
        channel = grpc.aio.insecure_channel(f"{instance.host}:{instance.port}")
        
        try:
            # Get service stub and method
            # This would be dynamically resolved in production
            stub = self._get_grpc_stub(channel)
            grpc_method = getattr(stub, method)
            
            # Make call
            response = await grpc_method(request, **kwargs)
            return response
            
        finally:
            await channel.close()
    
    def _get_grpc_stub(self, channel):
        """Get gRPC stub for service (placeholder)"""
        # In production, this would dynamically load the appropriate stub
        raise NotImplementedError("gRPC stub loading not implemented")


@asynccontextmanager
async def service_context(service_name: str, 
                         config: Optional[ServiceConfig] = None):
    """
    Context manager for service lifecycle
    
    Args:
        service_name: Service name
        config: Service configuration
        
    Yields:
        Service context
    """
    discovery = ServiceDiscovery()
    config = config or ServiceConfig(name=service_name)
    
    try:
        # Register service
        discovery.register_service(config)
        logger.info(f"Service {service_name} started")
        
        yield config
        
    finally:
        # Deregister service
        service_id = f"{config.name}-{config.host}-{config.port}"
        discovery.deregister_service(service_id)
        logger.info(f"Service {service_name} stopped")


def create_service_health_checker(
    services: List[str],
    check_interval: int = 30
) -> Callable:
    """
    Create health checker for multiple services
    
    Args:
        services: List of service names to check
        check_interval: Check interval in seconds
        
    Returns:
        Health check function
    """
    discovery = ServiceDiscovery()
    
    async def health_check() -> Dict[str, Any]:
        """Perform health check on all services"""
        results = {}
        
        for service_name in services:
            instances = discovery.discover_service(service_name)
            
            service_health = {
                "healthy": len(instances) > 0,
                "instance_count": len(instances),
                "instances": []
            }
            
            for instance in instances:
                # Check instance health
                instance_health = await _check_instance_health(instance)
                service_health["instances"].append({
                    "host": instance.host,
                    "port": instance.port,
                    "healthy": instance_health
                })
            
            results[service_name] = service_health
        
        return {
            "timestamp": datetime.now().isoformat(),
            "services": results,
            "overall_health": all(
                s["healthy"] for s in results.values()
            )
        }
    
    return health_check


async def _check_instance_health(instance: ServiceConfig) -> bool:
    """Check health of service instance"""
    url = f"{instance.protocol}://{instance.host}:{instance.port}/health"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=5) as response:
                return response.status == 200
    except:
        return False


# Export utilities
__all__ = [
    'ServiceState',
    'ServiceConfig',
    'ServiceDiscovery',
    'LoadBalancer',
    'CircuitBreaker',
    'retry_with_backoff',
    'ServiceProxy',
    'service_context',
    'create_service_health_checker'
]
