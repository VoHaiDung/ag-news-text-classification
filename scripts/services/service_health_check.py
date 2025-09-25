"""
Service Health Check Script for AG News Text Classification System
================================================================================
This script performs comprehensive health checks on all services including
API endpoints, backend services, databases, and infrastructure components.
It implements health check patterns following SRE best practices.

The health checking methodology follows Site Reliability Engineering principles
for monitoring distributed systems and ensuring service availability.

References:
    - Google SRE Book: Site Reliability Engineering (2016)
    - Distributed Systems Observability (O'Reilly, 2018)
    - Health Check API Pattern (Microsoft Azure Architecture Center)

Author: Võ Hải Dũng
License: MIT
"""

import sys
import os
import time
import json
import asyncio
import argparse
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
import logging
from dataclasses import dataclass, field, asdict
import concurrent.futures

import requests
import grpc
import psutil
import redis
import psycopg2
from prometheus_client import CollectorRegistry, Gauge, generate_latest
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.grpc.compiled import health_pb2, health_pb2_grpc
from src.core.exceptions import ServiceError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rich console for output
console = Console()


class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceHealth:
    """Service health information"""
    name: str
    status: HealthStatus
    response_time_ms: float = 0
    error_message: Optional[str] = None
    checks: Dict[str, bool] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    last_check: datetime = field(default_factory=datetime.now)


@dataclass
class DependencyHealth:
    """External dependency health information"""
    name: str
    type: str  # database, cache, queue, external_api
    status: HealthStatus
    latency_ms: float = 0
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """
    Comprehensive health checker for all system components
    
    This class implements health checking patterns for microservices,
    including liveness, readiness, and dependency checks.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize health checker
        
        Args:
            config_path: Path to health check configuration
        """
        self.config = self.load_configuration(config_path)
        self.services: Dict[str, ServiceHealth] = {}
        self.dependencies: Dict[str, DependencyHealth] = {}
        self.metrics_registry = CollectorRegistry()
        self._setup_metrics()
    
    def load_configuration(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load health check configuration"""
        default_config = {
            "services": {
                "rest-api": {
                    "url": "http://localhost:8000",
                    "endpoints": {
                        "health": "/health",
                        "ready": "/health/ready",
                        "live": "/health/live"
                    },
                    "timeout": 5,
                    "critical": True
                },
                "grpc-api": {
                    "host": "localhost",
                    "port": 50051,
                    "timeout": 5,
                    "critical": True
                },
                "graphql-api": {
                    "url": "http://localhost:4000",
                    "endpoints": {
                        "health": "/health"
                    },
                    "timeout": 5,
                    "critical": False
                },
                "prediction-service": {
                    "url": "http://localhost:8001",
                    "endpoints": {
                        "health": "/health"
                    },
                    "timeout": 10,
                    "critical": True
                },
                "training-service": {
                    "url": "http://localhost:8002",
                    "endpoints": {
                        "health": "/health"
                    },
                    "timeout": 10,
                    "critical": False
                },
                "data-service": {
                    "url": "http://localhost:8003",
                    "endpoints": {
                        "health": "/health"
                    },
                    "timeout": 5,
                    "critical": True
                },
                "monitoring-service": {
                    "url": "http://localhost:9090",
                    "endpoints": {
                        "health": "/-/healthy"
                    },
                    "timeout": 5,
                    "critical": False
                }
            },
            "dependencies": {
                "postgresql": {
                    "type": "database",
                    "host": "localhost",
                    "port": 5432,
                    "database": "agnews",
                    "critical": True
                },
                "redis": {
                    "type": "cache",
                    "host": "localhost",
                    "port": 6379,
                    "critical": False
                },
                "rabbitmq": {
                    "type": "queue",
                    "host": "localhost",
                    "port": 5672,
                    "critical": False
                }
            },
            "thresholds": {
                "response_time_warning_ms": 1000,
                "response_time_critical_ms": 5000,
                "error_rate_warning": 0.01,
                "error_rate_critical": 0.05,
                "cpu_warning_percent": 70,
                "cpu_critical_percent": 90,
                "memory_warning_percent": 80,
                "memory_critical_percent": 95,
                "disk_warning_percent": 80,
                "disk_critical_percent": 90
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config:
                    self._merge_configs(default_config, loaded_config)
        
        return default_config
    
    def _merge_configs(self, base: Dict, update: Dict):
        """Recursively merge configuration dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def _setup_metrics(self):
        """Setup Prometheus metrics"""
        self.service_health_gauge = Gauge(
            'service_health_status',
            'Service health status (1=healthy, 0.5=degraded, 0=unhealthy)',
            ['service_name'],
            registry=self.metrics_registry
        )
        
        self.service_response_time_gauge = Gauge(
            'service_response_time_ms',
            'Service response time in milliseconds',
            ['service_name'],
            registry=self.metrics_registry
        )
        
        self.dependency_health_gauge = Gauge(
            'dependency_health_status',
            'Dependency health status',
            ['dependency_name', 'dependency_type'],
            registry=self.metrics_registry
        )
        
        self.system_cpu_gauge = Gauge(
            'system_cpu_percent',
            'System CPU usage percentage',
            registry=self.metrics_registry
        )
        
        self.system_memory_gauge = Gauge(
            'system_memory_percent',
            'System memory usage percentage',
            registry=self.metrics_registry
        )
    
    async def check_all_services(self) -> Dict[str, ServiceHealth]:
        """
        Check health of all configured services
        
        Returns:
            Dictionary of service health statuses
        """
        tasks = []
        for service_name, service_config in self.config["services"].items():
            if "url" in service_config:
                tasks.append(self._check_http_service(service_name, service_config))
            elif "host" in service_config and "port" in service_config:
                tasks.append(self._check_grpc_service(service_name, service_config))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, ServiceHealth):
                self.services[result.name] = result
                self._update_service_metrics(result)
            elif isinstance(result, Exception):
                logger.error(f"Health check error: {result}")
        
        return self.services
    
    async def _check_http_service(self, name: str, config: Dict[str, Any]) -> ServiceHealth:
        """Check HTTP-based service health"""
        health = ServiceHealth(
            name=name,
            status=HealthStatus.UNKNOWN
        )
        
        base_url = config["url"]
        endpoints = config.get("endpoints", {"health": "/health"})
        timeout = config.get("timeout", 5)
        
        start_time = time.time()
        all_checks_passed = True
        
        for check_name, endpoint in endpoints.items():
            try:
                response = await self._make_http_request(
                    f"{base_url}{endpoint}",
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    health.checks[check_name] = True
                    
                    # Parse response for additional metrics
                    try:
                        data = response.json()
                        if "metrics" in data:
                            health.metrics.update(data["metrics"])
                    except:
                        pass
                else:
                    health.checks[check_name] = False
                    all_checks_passed = False
                    health.error_message = f"{check_name} returned {response.status_code}"
                    
            except Exception as e:
                health.checks[check_name] = False
                all_checks_passed = False
                health.error_message = str(e)
        
        health.response_time_ms = (time.time() - start_time) * 1000
        
        # Determine overall status
        if all_checks_passed:
            health.status = HealthStatus.HEALTHY
        elif any(health.checks.values()):
            health.status = HealthStatus.DEGRADED
        else:
            health.status = HealthStatus.UNHEALTHY
        
        return health
    
    async def _make_http_request(self, url: str, timeout: int = 5):
        """Make async HTTP request"""
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(
            None,
            lambda: requests.get(url, timeout=timeout)
        )
        return await future
    
    async def _check_grpc_service(self, name: str, config: Dict[str, Any]) -> ServiceHealth:
        """Check gRPC service health"""
        health = ServiceHealth(
            name=name,
            status=HealthStatus.UNKNOWN
        )
        
        host = config["host"]
        port = config["port"]
        timeout = config.get("timeout", 5)
        
        start_time = time.time()
        
        try:
            # Create gRPC channel
            channel = grpc.aio.insecure_channel(f"{host}:{port}")
            
            # Create health stub
            stub = health_pb2_grpc.HealthServiceStub(channel)
            
            # Perform health check
            request = health_pb2.HealthCheckRequest(service="")
            response = await stub.Check(request, timeout=timeout)
            
            health.response_time_ms = (time.time() - start_time) * 1000
            
            if response.status == health_pb2.HealthCheckResponse.SERVING:
                health.status = HealthStatus.HEALTHY
                health.checks["grpc_health"] = True
            else:
                health.status = HealthStatus.UNHEALTHY
                health.checks["grpc_health"] = False
                health.error_message = f"Service status: {response.status}"
            
            await channel.close()
            
        except Exception as e:
            health.status = HealthStatus.UNHEALTHY
            health.checks["grpc_health"] = False
            health.error_message = str(e)
            health.response_time_ms = (time.time() - start_time) * 1000
        
        return health
    
    async def check_dependencies(self) -> Dict[str, DependencyHealth]:
        """
        Check health of external dependencies
        
        Returns:
            Dictionary of dependency health statuses
        """
        tasks = []
        
        for dep_name, dep_config in self.config["dependencies"].items():
            dep_type = dep_config["type"]
            
            if dep_type == "database":
                tasks.append(self._check_database(dep_name, dep_config))
            elif dep_type == "cache":
                tasks.append(self._check_redis(dep_name, dep_config))
            elif dep_type == "queue":
                tasks.append(self._check_rabbitmq(dep_name, dep_config))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, DependencyHealth):
                self.dependencies[result.name] = result
                self._update_dependency_metrics(result)
            elif isinstance(result, Exception):
                logger.error(f"Dependency check error: {result}")
        
        return self.dependencies
    
    async def _check_database(self, name: str, config: Dict[str, Any]) -> DependencyHealth:
        """Check PostgreSQL database health"""
        health = DependencyHealth(
            name=name,
            type="database",
            status=HealthStatus.UNKNOWN
        )
        
        start_time = time.time()
        
        try:
            # Connect to database
            conn = psycopg2.connect(
                host=config["host"],
                port=config["port"],
                database=config.get("database", "agnews"),
                user=os.environ.get("DB_USER", "postgres"),
                password=os.environ.get("DB_PASSWORD", ""),
                connect_timeout=5
            )
            
            cursor = conn.cursor()
            
            # Check database is accessible
            cursor.execute("SELECT 1")
            cursor.fetchone()
            
            # Get database statistics
            cursor.execute("""
                SELECT 
                    numbackends as connections,
                    xact_commit as commits,
                    xact_rollback as rollbacks,
                    blks_read as blocks_read,
                    blks_hit as blocks_hit
                FROM pg_stat_database 
                WHERE datname = %s
            """, (config.get("database", "agnews"),))
            
            stats = cursor.fetchone()
            if stats:
                health.details = {
                    "connections": stats[0],
                    "commits": stats[1],
                    "rollbacks": stats[2],
                    "cache_hit_ratio": stats[4] / (stats[3] + stats[4]) if (stats[3] + stats[4]) > 0 else 0
                }
            
            cursor.close()
            conn.close()
            
            health.status = HealthStatus.HEALTHY
            health.latency_ms = (time.time() - start_time) * 1000
            
        except Exception as e:
            health.status = HealthStatus.UNHEALTHY
            health.error_message = str(e)
            health.latency_ms = (time.time() - start_time) * 1000
        
        return health
    
    async def _check_redis(self, name: str, config: Dict[str, Any]) -> DependencyHealth:
        """Check Redis cache health"""
        health = DependencyHealth(
            name=name,
            type="cache",
            status=HealthStatus.UNKNOWN
        )
        
        start_time = time.time()
        
        try:
            # Connect to Redis
            r = redis.Redis(
                host=config["host"],
                port=config["port"],
                socket_connect_timeout=5,
                decode_responses=True
            )
            
            # Ping Redis
            r.ping()
            
            # Get Redis info
            info = r.info()
            health.details = {
                "version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory_mb": info.get("used_memory") / 1024 / 1024 if info.get("used_memory") else 0,
                "total_commands": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses")
            }
            
            # Calculate hit rate
            hits = info.get("keyspace_hits", 0)
            misses = info.get("keyspace_misses", 0)
            if hits + misses > 0:
                health.details["hit_rate"] = hits / (hits + misses)
            
            health.status = HealthStatus.HEALTHY
            health.latency_ms = (time.time() - start_time) * 1000
            
        except Exception as e:
            health.status = HealthStatus.UNHEALTHY
            health.error_message = str(e)
            health.latency_ms = (time.time() - start_time) * 1000
        
        return health
    
    async def _check_rabbitmq(self, name: str, config: Dict[str, Any]) -> DependencyHealth:
        """Check RabbitMQ queue health"""
        health = DependencyHealth(
            name=name,
            type="queue",
            status=HealthStatus.UNKNOWN
        )
        
        start_time = time.time()
        
        try:
            # Check RabbitMQ management API
            management_url = f"http://{config['host']}:15672/api/overview"
            auth = (
                os.environ.get("RABBITMQ_USER", "guest"),
                os.environ.get("RABBITMQ_PASSWORD", "guest")
            )
            
            response = requests.get(management_url, auth=auth, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                health.details = {
                    "version": data.get("rabbitmq_version"),
                    "messages": data.get("queue_totals", {}).get("messages", 0),
                    "messages_ready": data.get("queue_totals", {}).get("messages_ready", 0),
                    "publish_rate": data.get("message_stats", {}).get("publish_details", {}).get("rate", 0),
                    "deliver_rate": data.get("message_stats", {}).get("deliver_get_details", {}).get("rate", 0)
                }
                health.status = HealthStatus.HEALTHY
            else:
                health.status = HealthStatus.UNHEALTHY
                health.error_message = f"Management API returned {response.status_code}"
            
            health.latency_ms = (time.time() - start_time) * 1000
            
        except Exception as e:
            health.status = HealthStatus.UNHEALTHY
            health.error_message = str(e)
            health.latency_ms = (time.time() - start_time) * 1000
        
        return health
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource utilization"""
        resources = {
            "cpu": {
                "usage_percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "load_average": os.getloadavg()
            },
            "memory": {
                "total_gb": psutil.virtual_memory().total / 1024**3,
                "available_gb": psutil.virtual_memory().available / 1024**3,
                "used_percent": psutil.virtual_memory().percent
            },
            "disk": {},
            "network": {}
        }
        
        # Disk usage
        for partition in psutil.disk_partitions():
            if partition.mountpoint == '/':
                usage = psutil.disk_usage(partition.mountpoint)
                resources["disk"]["root"] = {
                    "total_gb": usage.total / 1024**3,
                    "used_gb": usage.used / 1024**3,
                    "free_gb": usage.free / 1024**3,
                    "used_percent": usage.percent
                }
        
        # Network statistics
        net_io = psutil.net_io_counters()
        resources["network"] = {
            "bytes_sent_mb": net_io.bytes_sent / 1024**2,
            "bytes_recv_mb": net_io.bytes_recv / 1024**2,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
            "errors": net_io.errin + net_io.errout,
            "drops": net_io.dropin + net_io.dropout
        }
        
        # Update metrics
        self.system_cpu_gauge.set(resources["cpu"]["usage_percent"])
        self.system_memory_gauge.set(resources["memory"]["used_percent"])
        
        return resources
    
    def _update_service_metrics(self, health: ServiceHealth):
        """Update Prometheus metrics for service"""
        status_value = {
            HealthStatus.HEALTHY: 1.0,
            HealthStatus.DEGRADED: 0.5,
            HealthStatus.UNHEALTHY: 0.0,
            HealthStatus.UNKNOWN: -1.0
        }
        
        self.service_health_gauge.labels(
            service_name=health.name
        ).set(status_value[health.status])
        
        self.service_response_time_gauge.labels(
            service_name=health.name
        ).set(health.response_time_ms)
    
    def _update_dependency_metrics(self, health: DependencyHealth):
        """Update Prometheus metrics for dependency"""
        status_value = {
            HealthStatus.HEALTHY: 1.0,
            HealthStatus.DEGRADED: 0.5,
            HealthStatus.UNHEALTHY: 0.0,
            HealthStatus.UNKNOWN: -1.0
        }
        
        self.dependency_health_gauge.labels(
            dependency_name=health.name,
            dependency_type=health.type
        ).set(status_value[health.status])
    
    def evaluate_overall_health(self) -> HealthStatus:
        """
        Evaluate overall system health based on all checks
        
        Returns:
            Overall health status
        """
        critical_services_healthy = True
        all_services_healthy = True
        
        # Check services
        for service_name, service_health in self.services.items():
            is_critical = self.config["services"].get(service_name, {}).get("critical", False)
            
            if service_health.status == HealthStatus.UNHEALTHY:
                all_services_healthy = False
                if is_critical:
                    critical_services_healthy = False
            elif service_health.status == HealthStatus.DEGRADED:
                all_services_healthy = False
        
        # Check dependencies
        for dep_name, dep_health in self.dependencies.items():
            is_critical = self.config["dependencies"].get(dep_name, {}).get("critical", False)
            
            if dep_health.status == HealthStatus.UNHEALTHY:
                all_services_healthy = False
                if is_critical:
                    critical_services_healthy = False
        
        # Check system resources
        resources = self.check_system_resources()
        thresholds = self.config["thresholds"]
        
        if resources["cpu"]["usage_percent"] > thresholds["cpu_critical_percent"]:
            all_services_healthy = False
        
        if resources["memory"]["used_percent"] > thresholds["memory_critical_percent"]:
            all_services_healthy = False
        
        # Determine overall status
        if not critical_services_healthy:
            return HealthStatus.UNHEALTHY
        elif not all_services_healthy:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def generate_report(self) -> str:
        """Generate health check report"""
        report = []
        report.append("=" * 80)
        report.append(f"HEALTH CHECK REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        
        # Overall status
        overall_status = self.evaluate_overall_health()
        report.append(f"\nOVERALL STATUS: {overall_status.value.upper()}")
        report.append("")
        
        # Services
        report.append("SERVICES:")
        report.append("-" * 40)
        for name, health in sorted(self.services.items()):
            status_symbol = {
                HealthStatus.HEALTHY: "✓",
                HealthStatus.DEGRADED: "⚠",
                HealthStatus.UNHEALTHY: "✗",
                HealthStatus.UNKNOWN: "?"
            }[health.status]
            
            report.append(f"  {status_symbol} {name:20} {health.status.value:10} "
                         f"{health.response_time_ms:6.1f}ms")
            
            if health.error_message:
                report.append(f"    Error: {health.error_message}")
        
        # Dependencies
        report.append("\nDEPENDENCIES:")
        report.append("-" * 40)
        for name, health in sorted(self.dependencies.items()):
            status_symbol = {
                HealthStatus.HEALTHY: "✓",
                HealthStatus.DEGRADED: "⚠",
                HealthStatus.UNHEALTHY: "✗",
                HealthStatus.UNKNOWN: "?"
            }[health.status]
            
            report.append(f"  {status_symbol} {name:20} {health.status.value:10} "
                         f"{health.latency_ms:6.1f}ms")
            
            if health.error_message:
                report.append(f"    Error: {health.error_message}")
        
        # System resources
        resources = self.check_system_resources()
        report.append("\nSYSTEM RESOURCES:")
        report.append("-" * 40)
        report.append(f"  CPU Usage:      {resources['cpu']['usage_percent']:.1f}%")
        report.append(f"  Memory Usage:   {resources['memory']['used_percent']:.1f}%")
        
        if "root" in resources["disk"]:
            report.append(f"  Disk Usage:     {resources['disk']['root']['used_percent']:.1f}%")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def export_metrics(self) -> bytes:
        """Export metrics in Prometheus format"""
        return generate_latest(self.metrics_registry)


async def continuous_monitoring(checker: HealthChecker, interval: int = 30):
    """
    Run continuous health monitoring
    
    Args:
        checker: HealthChecker instance
        interval: Check interval in seconds
    """
    console.print("[bold blue]Starting continuous health monitoring[/bold blue]")
    console.print(f"Check interval: {interval} seconds\n")
    
    while True:
        try:
            # Run health checks
            await checker.check_all_services()
            await checker.check_dependencies()
            
            # Generate and display report
            table = Table(title=f"Health Status - {datetime.now().strftime('%H:%M:%S')}")
            table.add_column("Component", style="cyan")
            table.add_column("Type", style="yellow")
            table.add_column("Status", style="green")
            table.add_column("Response Time", style="magenta")
            
            # Add services
            for name, health in sorted(checker.services.items()):
                status_color = {
                    HealthStatus.HEALTHY: "green",
                    HealthStatus.DEGRADED: "yellow",
                    HealthStatus.UNHEALTHY: "red",
                    HealthStatus.UNKNOWN: "grey"
                }[health.status]
                
                table.add_row(
                    name,
                    "Service",
                    f"[{status_color}]{health.status.value}[/{status_color}]",
                    f"{health.response_time_ms:.1f}ms"
                )
            
            # Add dependencies
            for name, health in sorted(checker.dependencies.items()):
                status_color = {
                    HealthStatus.HEALTHY: "green",
                    HealthStatus.DEGRADED: "yellow",
                    HealthStatus.UNHEALTHY: "red",
                    HealthStatus.UNKNOWN: "grey"
                }[health.status]
                
                table.add_row(
                    name,
                    health.type.capitalize(),
                    f"[{status_color}]{health.status.value}[/{status_color}]",
                    f"{health.latency_ms:.1f}ms"
                )
            
            # Display
            console.clear()
            console.print(table)
            
            # Show overall status
            overall = checker.evaluate_overall_health()
            status_color = {
                HealthStatus.HEALTHY: "green",
                HealthStatus.DEGRADED: "yellow",
                HealthStatus.UNHEALTHY: "red",
                HealthStatus.UNKNOWN: "grey"
            }[overall]
            
            console.print(f"\n[bold {status_color}]Overall Status: {overall.value.upper()}[/bold {status_color}]")
            
            # Show resource usage
            resources = checker.check_system_resources()
            console.print(f"\nCPU: {resources['cpu']['usage_percent']:.1f}% | "
                         f"Memory: {resources['memory']['used_percent']:.1f}% | "
                         f"Load: {resources['cpu']['load_average'][0]:.2f}")
            
            # Wait for next check
            await asyncio.sleep(interval)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped by user[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error during health check: {e}[/red]")
            await asyncio.sleep(interval)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Health check for AG News Text Classification services"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to health check configuration file"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run health check once and exit"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Check interval in seconds for continuous monitoring"
    )
    parser.add_argument(
        "--export-metrics",
        action="store_true",
        help="Export metrics in Prometheus format"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for health report"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    
    args = parser.parse_args()
    
    # Initialize health checker
    checker = HealthChecker(args.config)
    
    # Run health checks
    if args.once:
        # Run once
        asyncio.run(checker.check_all_services())
        asyncio.run(checker.check_dependencies())
        
        if args.json:
            # Output JSON
            result = {
                "timestamp": datetime.now().isoformat(),
                "overall_status": checker.evaluate_overall_health().value,
                "services": {
                    name: {
                        "status": health.status.value,
                        "response_time_ms": health.response_time_ms,
                        "error": health.error_message
                    }
                    for name, health in checker.services.items()
                },
                "dependencies": {
                    name: {
                        "status": health.status.value,
                        "latency_ms": health.latency_ms,
                        "error": health.error_message
                    }
                    for name, health in checker.dependencies.items()
                },
                "resources": checker.check_system_resources()
            }
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
            else:
                print(json.dumps(result, indent=2, default=str))
        else:
            # Output text report
            report = checker.generate_report()
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(report)
            else:
                print(report)
        
        if args.export_metrics:
            # Export Prometheus metrics
            metrics = checker.export_metrics()
            metrics_file = args.output.replace('.txt', '.prom') if args.output else 'metrics.prom'
            with open(metrics_file, 'wb') as f:
                f.write(metrics)
            print(f"Metrics exported to {metrics_file}")
    else:
        # Run continuous monitoring
        try:
            asyncio.run(continuous_monitoring(checker, args.interval))
        except KeyboardInterrupt:
            console.print("\n[yellow]Health monitoring stopped[/yellow]")


if __name__ == "__main__":
    main()
