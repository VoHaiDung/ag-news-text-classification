"""
Service Diagnostics Script for AG News Text Classification System
================================================================================
This script performs comprehensive diagnostics on services to identify issues,
bottlenecks, and optimization opportunities. It implements diagnostic patterns
following best practices for distributed systems troubleshooting.

The diagnostic methodology follows principles from production debugging and
performance analysis literature for microservices architectures.

References:
    - Brendan Gregg's Systems Performance (2020)
    - Distributed Systems Observability (O'Reilly, 2018)
    - Debugging Production Systems (ACM Queue, 2019)

Author: Võ Hải Dũng
License: MIT
"""

import sys
import os
import time
import json
import traceback
import asyncio
import argparse
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import statistics

import psutil
import requests
import grpc
import numpy as np
from prometheus_client.parser import text_string_to_metric_families
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.exceptions import ServiceError
from src.utils.profiling_utils import profile_memory, profile_cpu

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rich console
console = Console()


class DiagnosticLevel(Enum):
    """Diagnostic severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DiagnosticResult:
    """Single diagnostic result"""
    category: str
    check: str
    level: DiagnosticLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendation: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ServiceDiagnostics:
    """Complete service diagnostics"""
    service_name: str
    status: str
    results: List[DiagnosticResult] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class ServiceDiagnostician:
    """
    Comprehensive service diagnostics tool
    
    Performs deep analysis of service health, performance, and configuration
    to identify issues and provide actionable recommendations.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize diagnostician
        
        Args:
            config_path: Path to diagnostic configuration
        """
        self.config = self.load_configuration(config_path)
        self.diagnostics: Dict[str, ServiceDiagnostics] = {}
        self.system_metrics = {}
    
    def load_configuration(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load diagnostic configuration"""
        default_config = {
            "services": {
                "rest-api": {
                    "url": "http://localhost:8000",
                    "metrics_endpoint": "/metrics",
                    "type": "http"
                },
                "grpc-api": {
                    "host": "localhost",
                    "port": 50051,
                    "type": "grpc"
                },
                "prediction-service": {
                    "url": "http://localhost:8001",
                    "metrics_endpoint": "/metrics",
                    "type": "http"
                },
                "training-service": {
                    "url": "http://localhost:8002",
                    "metrics_endpoint": "/metrics",
                    "type": "http"
                },
                "data-service": {
                    "url": "http://localhost:8003",
                    "metrics_endpoint": "/metrics",
                    "type": "http"
                }
            },
            "checks": {
                "connectivity": True,
                "performance": True,
                "resources": True,
                "configuration": True,
                "dependencies": True,
                "logs": True,
                "metrics": True,
                "security": True
            },
            "thresholds": {
                "response_time_ms": {
                    "warning": 500,
                    "critical": 2000
                },
                "error_rate": {
                    "warning": 0.01,
                    "critical": 0.05
                },
                "cpu_percent": {
                    "warning": 70,
                    "critical": 90
                },
                "memory_percent": {
                    "warning": 80,
                    "critical": 95
                },
                "disk_percent": {
                    "warning": 80,
                    "critical": 90
                },
                "connection_pool": {
                    "warning": 0.8,
                    "critical": 0.95
                }
            },
            "performance_tests": {
                "latency_test_requests": 100,
                "throughput_test_duration": 10,
                "concurrent_connections": 10
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    self._merge_configs(default_config, loaded)
        
        return default_config
    
    def _merge_configs(self, base: Dict, update: Dict):
        """Merge configuration dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    async def diagnose_all_services(self) -> Dict[str, ServiceDiagnostics]:
        """
        Run diagnostics on all configured services
        
        Returns:
            Dictionary of service diagnostics
        """
        console.print("[bold blue]Running Service Diagnostics[/bold blue]\n")
        
        # Collect system metrics first
        self.system_metrics = self._collect_system_metrics()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for service_name, service_config in self.config["services"].items():
                task = progress.add_task(f"Diagnosing {service_name}...", total=1)
                
                diagnostics = await self._diagnose_service(service_name, service_config)
                self.diagnostics[service_name] = diagnostics
                
                progress.update(task, completed=1)
        
        return self.diagnostics
    
    async def _diagnose_service(self, name: str, config: Dict[str, Any]) -> ServiceDiagnostics:
        """Diagnose a single service"""
        diagnostics = ServiceDiagnostics(service_name=name, status="unknown")
        
        checks = self.config["checks"]
        
        try:
            # Connectivity check
            if checks.get("connectivity", True):
                await self._check_connectivity(diagnostics, config)
            
            # Performance check
            if checks.get("performance", True):
                await self._check_performance(diagnostics, config)
            
            # Resource usage check
            if checks.get("resources", True):
                self._check_resources(diagnostics, name)
            
            # Configuration check
            if checks.get("configuration", True):
                self._check_configuration(diagnostics, config)
            
            # Dependencies check
            if checks.get("dependencies", True):
                await self._check_dependencies(diagnostics, config)
            
            # Logs analysis
            if checks.get("logs", True):
                self._analyze_logs(diagnostics, name)
            
            # Metrics analysis
            if checks.get("metrics", True):
                await self._analyze_metrics(diagnostics, config)
            
            # Security check
            if checks.get("security", True):
                self._check_security(diagnostics, config)
            
            # Determine overall status
            diagnostics.status = self._determine_status(diagnostics)
            
        except Exception as e:
            diagnostics.errors.append(str(e))
            diagnostics.status = "error"
            logger.error(f"Error diagnosing {name}: {e}")
        
        return diagnostics
    
    async def _check_connectivity(self, diagnostics: ServiceDiagnostics, config: Dict[str, Any]):
        """Check service connectivity"""
        service_type = config.get("type", "http")
        
        if service_type == "http":
            url = config["url"]
            try:
                response = requests.get(f"{url}/health", timeout=5)
                
                if response.status_code == 200:
                    diagnostics.results.append(DiagnosticResult(
                        category="Connectivity",
                        check="HTTP Health Check",
                        level=DiagnosticLevel.INFO,
                        message="Service is reachable",
                        details={"status_code": response.status_code}
                    ))
                else:
                    diagnostics.results.append(DiagnosticResult(
                        category="Connectivity",
                        check="HTTP Health Check",
                        level=DiagnosticLevel.WARNING,
                        message=f"Service returned status {response.status_code}",
                        details={"status_code": response.status_code},
                        recommendation="Check service health endpoint"
                    ))
                    
            except requests.exceptions.ConnectionError:
                diagnostics.results.append(DiagnosticResult(
                    category="Connectivity",
                    check="HTTP Connection",
                    level=DiagnosticLevel.CRITICAL,
                    message="Cannot connect to service",
                    recommendation="Check if service is running and network is accessible"
                ))
            except requests.exceptions.Timeout:
                diagnostics.results.append(DiagnosticResult(
                    category="Connectivity",
                    check="HTTP Timeout",
                    level=DiagnosticLevel.ERROR,
                    message="Connection timeout",
                    recommendation="Check service load and network latency"
                ))
                
        elif service_type == "grpc":
            host = config["host"]
            port = config["port"]
            
            try:
                channel = grpc.insecure_channel(f"{host}:{port}")
                grpc.channel_ready_future(channel).result(timeout=5)
                
                diagnostics.results.append(DiagnosticResult(
                    category="Connectivity",
                    check="gRPC Connection",
                    level=DiagnosticLevel.INFO,
                    message="gRPC service is reachable"
                ))
                
                channel.close()
                
            except grpc.FutureTimeoutError:
                diagnostics.results.append(DiagnosticResult(
                    category="Connectivity",
                    check="gRPC Connection",
                    level=DiagnosticLevel.CRITICAL,
                    message="Cannot connect to gRPC service",
                    recommendation="Check if gRPC service is running on the specified port"
                ))
    
    async def _check_performance(self, diagnostics: ServiceDiagnostics, config: Dict[str, Any]):
        """Check service performance"""
        if config.get("type") != "http":
            return
        
        url = config["url"]
        test_config = self.config["performance_tests"]
        thresholds = self.config["thresholds"]["response_time_ms"]
        
        # Latency test
        latencies = []
        errors = 0
        
        for _ in range(test_config["latency_test_requests"]):
            try:
                start = time.time()
                response = requests.get(f"{url}/health", timeout=10)
                latency = (time.time() - start) * 1000
                
                if response.status_code == 200:
                    latencies.append(latency)
                else:
                    errors += 1
                    
            except Exception:
                errors += 1
        
        if latencies:
            avg_latency = statistics.mean(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else max(latencies)
            p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) > 100 else max(latencies)
            
            diagnostics.performance["avg_latency_ms"] = avg_latency
            diagnostics.performance["p95_latency_ms"] = p95_latency
            diagnostics.performance["p99_latency_ms"] = p99_latency
            diagnostics.performance["error_rate"] = errors / test_config["latency_test_requests"]
            
            # Check against thresholds
            if p95_latency > thresholds["critical"]:
                diagnostics.results.append(DiagnosticResult(
                    category="Performance",
                    check="Response Time",
                    level=DiagnosticLevel.CRITICAL,
                    message=f"P95 latency ({p95_latency:.1f}ms) exceeds critical threshold",
                    details={"p95_ms": p95_latency, "threshold_ms": thresholds["critical"]},
                    recommendation="Investigate service performance bottlenecks"
                ))
            elif p95_latency > thresholds["warning"]:
                diagnostics.results.append(DiagnosticResult(
                    category="Performance",
                    check="Response Time",
                    level=DiagnosticLevel.WARNING,
                    message=f"P95 latency ({p95_latency:.1f}ms) exceeds warning threshold",
                    details={"p95_ms": p95_latency, "threshold_ms": thresholds["warning"]},
                    recommendation="Monitor service performance"
                ))
            else:
                diagnostics.results.append(DiagnosticResult(
                    category="Performance",
                    check="Response Time",
                    level=DiagnosticLevel.INFO,
                    message=f"Response times are within acceptable range",
                    details={"avg_ms": avg_latency, "p95_ms": p95_latency}
                ))
    
    def _check_resources(self, diagnostics: ServiceDiagnostics, service_name: str):
        """Check service resource usage"""
        thresholds = self.config["thresholds"]
        
        # Find service process
        service_process = None
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if service_name in cmdline:
                    service_process = proc
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if service_process:
            try:
                # CPU usage
                cpu_percent = service_process.cpu_percent(interval=1)
                
                if cpu_percent > thresholds["cpu_percent"]["critical"]:
                    level = DiagnosticLevel.CRITICAL
                    recommendation = "Service is consuming excessive CPU. Check for infinite loops or inefficient algorithms"
                elif cpu_percent > thresholds["cpu_percent"]["warning"]:
                    level = DiagnosticLevel.WARNING
                    recommendation = "High CPU usage detected. Monitor for sustained high usage"
                else:
                    level = DiagnosticLevel.INFO
                    recommendation = None
                
                diagnostics.results.append(DiagnosticResult(
                    category="Resources",
                    check="CPU Usage",
                    level=level,
                    message=f"CPU usage: {cpu_percent:.1f}%",
                    details={"cpu_percent": cpu_percent},
                    recommendation=recommendation
                ))
                
                # Memory usage
                memory_info = service_process.memory_info()
                memory_percent = service_process.memory_percent()
                
                if memory_percent > thresholds["memory_percent"]["critical"]:
                    level = DiagnosticLevel.CRITICAL
                    recommendation = "Critical memory usage. Check for memory leaks"
                elif memory_percent > thresholds["memory_percent"]["warning"]:
                    level = DiagnosticLevel.WARNING
                    recommendation = "High memory usage. Monitor for memory leaks"
                else:
                    level = DiagnosticLevel.INFO
                    recommendation = None
                
                diagnostics.results.append(DiagnosticResult(
                    category="Resources",
                    check="Memory Usage",
                    level=level,
                    message=f"Memory usage: {memory_percent:.1f}% ({memory_info.rss / 1024**2:.1f} MB)",
                    details={
                        "memory_percent": memory_percent,
                        "rss_mb": memory_info.rss / 1024**2,
                        "vms_mb": memory_info.vms / 1024**2
                    },
                    recommendation=recommendation
                ))
                
                # Open files/connections
                open_files = len(service_process.open_files())
                connections = len(service_process.connections())
                
                diagnostics.metrics["open_files"] = open_files
                diagnostics.metrics["connections"] = connections
                
                if open_files > 1000:
                    diagnostics.results.append(DiagnosticResult(
                        category="Resources",
                        check="Open Files",
                        level=DiagnosticLevel.WARNING,
                        message=f"High number of open files: {open_files}",
                        recommendation="Check for file descriptor leaks"
                    ))
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                diagnostics.errors.append(f"Cannot access process information: {e}")
        else:
            diagnostics.results.append(DiagnosticResult(
                category="Resources",
                check="Process Status",
                level=DiagnosticLevel.ERROR,
                message="Service process not found",
                recommendation="Check if service is running"
            ))
    
    def _check_configuration(self, diagnostics: ServiceDiagnostics, config: Dict[str, Any]):
        """Check service configuration"""
        # Check for common configuration issues
        
        # Check if running in production mode
        if config.get("debug", False):
            diagnostics.results.append(DiagnosticResult(
                category="Configuration",
                check="Debug Mode",
                level=DiagnosticLevel.WARNING,
                message="Service is running in debug mode",
                recommendation="Disable debug mode in production"
            ))
        
        # Check timeout settings
        timeout = config.get("timeout", 30)
        if timeout > 60:
            diagnostics.results.append(DiagnosticResult(
                category="Configuration",
                check="Timeout Settings",
                level=DiagnosticLevel.WARNING,
                message=f"Long timeout configured: {timeout}s",
                recommendation="Consider reducing timeout for better failure detection"
            ))
        
        # Check for missing configurations
        required_configs = ["url", "type"] if config.get("type") == "http" else ["host", "port", "type"]
        missing = [key for key in required_configs if key not in config]
        
        if missing:
            diagnostics.results.append(DiagnosticResult(
                category="Configuration",
                check="Required Configuration",
                level=DiagnosticLevel.ERROR,
                message=f"Missing required configuration: {missing}",
                recommendation="Add missing configuration parameters"
            ))
    
    async def _check_dependencies(self, diagnostics: ServiceDiagnostics, config: Dict[str, Any]):
        """Check service dependencies"""
        # This would check database connections, external APIs, etc.
        # For now, we'll check basic network dependencies
        
        # Check database connectivity (example)
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=os.environ.get("DB_HOST", "localhost"),
                port=os.environ.get("DB_PORT", 5432),
                database=os.environ.get("DB_NAME", "agnews"),
                user=os.environ.get("DB_USER", "postgres"),
                password=os.environ.get("DB_PASSWORD", ""),
                connect_timeout=5
            )
            conn.close()
            
            diagnostics.results.append(DiagnosticResult(
                category="Dependencies",
                check="Database Connection",
                level=DiagnosticLevel.INFO,
                message="Database connection successful"
            ))
        except Exception as e:
            diagnostics.results.append(DiagnosticResult(
                category="Dependencies",
                check="Database Connection",
                level=DiagnosticLevel.ERROR,
                message="Cannot connect to database",
                details={"error": str(e)},
                recommendation="Check database configuration and availability"
            ))
        
        # Check Redis connectivity (example)
        try:
            import redis
            r = redis.Redis(
                host=os.environ.get("REDIS_HOST", "localhost"),
                port=os.environ.get("REDIS_PORT", 6379),
                socket_connect_timeout=5
            )
            r.ping()
            
            diagnostics.results.append(DiagnosticResult(
                category="Dependencies",
                check="Redis Connection",
                level=DiagnosticLevel.INFO,
                message="Redis connection successful"
            ))
        except Exception as e:
            diagnostics.results.append(DiagnosticResult(
                category="Dependencies",
                check="Redis Connection",
                level=DiagnosticLevel.WARNING,
                message="Cannot connect to Redis",
                details={"error": str(e)},
                recommendation="Check Redis configuration if caching is required"
            ))
    
    def _analyze_logs(self, diagnostics: ServiceDiagnostics, service_name: str):
        """Analyze service logs for issues"""
        log_file = Path(PROJECT_ROOT) / "logs" / "services" / f"{service_name}.log"
        
        if not log_file.exists():
            diagnostics.results.append(DiagnosticResult(
                category="Logs",
                check="Log File",
                level=DiagnosticLevel.WARNING,
                message="Log file not found",
                recommendation="Check logging configuration"
            ))
            return
        
        try:
            # Analyze recent logs
            with open(log_file, 'r') as f:
                lines = f.readlines()[-1000:]  # Last 1000 lines
            
            error_count = sum(1 for line in lines if 'ERROR' in line)
            warning_count = sum(1 for line in lines if 'WARNING' in line)
            
            diagnostics.metrics["log_errors"] = error_count
            diagnostics.metrics["log_warnings"] = warning_count
            
            if error_count > 100:
                diagnostics.results.append(DiagnosticResult(
                    category="Logs",
                    check="Error Rate",
                    level=DiagnosticLevel.CRITICAL,
                    message=f"High error rate in logs: {error_count} errors",
                    recommendation="Investigate error logs for root cause"
                ))
            elif error_count > 10:
                diagnostics.results.append(DiagnosticResult(
                    category="Logs",
                    check="Error Rate",
                    level=DiagnosticLevel.WARNING,
                    message=f"Moderate errors in logs: {error_count} errors",
                    recommendation="Review error logs"
                ))
            
            # Look for specific patterns
            oom_errors = sum(1 for line in lines if 'out of memory' in line.lower())
            if oom_errors > 0:
                diagnostics.results.append(DiagnosticResult(
                    category="Logs",
                    check="Memory Issues",
                    level=DiagnosticLevel.CRITICAL,
                    message=f"Out of memory errors detected: {oom_errors}",
                    recommendation="Increase memory allocation or optimize memory usage"
                ))
                
        except Exception as e:
            diagnostics.errors.append(f"Error analyzing logs: {e}")
    
    async def _analyze_metrics(self, diagnostics: ServiceDiagnostics, config: Dict[str, Any]):
        """Analyze service metrics"""
        if config.get("type") != "http":
            return
        
        metrics_url = config["url"] + config.get("metrics_endpoint", "/metrics")
        
        try:
            response = requests.get(metrics_url, timeout=5)
            
            if response.status_code == 200:
                # Parse Prometheus metrics
                metrics = {}
                for family in text_string_to_metric_families(response.text):
                    for sample in family.samples:
                        metrics[sample.name] = sample.value
                
                diagnostics.metrics.update(metrics)
                
                # Check specific metrics
                error_rate = metrics.get("http_requests_failed_total", 0) / max(metrics.get("http_requests_total", 1), 1)
                
                if error_rate > self.config["thresholds"]["error_rate"]["critical"]:
                    diagnostics.results.append(DiagnosticResult(
                        category="Metrics",
                        check="Error Rate",
                        level=DiagnosticLevel.CRITICAL,
                        message=f"Critical error rate: {error_rate:.2%}",
                        recommendation="Investigate cause of failures"
                    ))
                elif error_rate > self.config["thresholds"]["error_rate"]["warning"]:
                    diagnostics.results.append(DiagnosticResult(
                        category="Metrics",
                        check="Error Rate",
                        level=DiagnosticLevel.WARNING,
                        message=f"Elevated error rate: {error_rate:.2%}",
                        recommendation="Monitor error trends"
                    ))
                    
        except Exception as e:
            diagnostics.errors.append(f"Error fetching metrics: {e}")
    
    def _check_security(self, diagnostics: ServiceDiagnostics, config: Dict[str, Any]):
        """Check security configuration"""
        # Check for common security issues
        
        # Check if using HTTPS
        if config.get("type") == "http" and not config.get("url", "").startswith("https"):
            diagnostics.results.append(DiagnosticResult(
                category="Security",
                check="HTTPS",
                level=DiagnosticLevel.WARNING,
                message="Service not using HTTPS",
                recommendation="Enable HTTPS for production deployments"
            ))
        
        # Check for authentication
        if not config.get("auth_required", False):
            diagnostics.results.append(DiagnosticResult(
                category="Security",
                check="Authentication",
                level=DiagnosticLevel.WARNING,
                message="No authentication configured",
                recommendation="Consider enabling authentication for sensitive endpoints"
            ))
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect overall system metrics"""
        return {
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "load_avg": os.getloadavg()
            },
            "memory": {
                "total_gb": psutil.virtual_memory().total / 1024**3,
                "available_gb": psutil.virtual_memory().available / 1024**3,
                "percent": psutil.virtual_memory().percent
            },
            "disk": {
                "total_gb": psutil.disk_usage('/').total / 1024**3,
                "free_gb": psutil.disk_usage('/').free / 1024**3,
                "percent": psutil.disk_usage('/').percent
            },
            "network": {
                "connections": len(psutil.net_connections()),
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_recv": psutil.net_io_counters().bytes_recv
            }
        }
    
    def _determine_status(self, diagnostics: ServiceDiagnostics) -> str:
        """Determine overall service status from diagnostic results"""
        if any(r.level == DiagnosticLevel.CRITICAL for r in diagnostics.results):
            return "critical"
        elif any(r.level == DiagnosticLevel.ERROR for r in diagnostics.results):
            return "error"
        elif any(r.level == DiagnosticLevel.WARNING for r in diagnostics.results):
            return "warning"
        else:
            return "healthy"
    
    def generate_report(self) -> str:
        """Generate comprehensive diagnostic report"""
        report = []
        report.append("=" * 80)
        report.append(f"SERVICE DIAGNOSTICS REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        
        # System overview
        report.append("\nSYSTEM OVERVIEW:")
        report.append("-" * 40)
        report.append(f"CPU Usage: {self.system_metrics['cpu']['percent']:.1f}%")
        report.append(f"Memory Usage: {self.system_metrics['memory']['percent']:.1f}%")
        report.append(f"Disk Usage: {self.system_metrics['disk']['percent']:.1f}%")
        report.append(f"Network Connections: {self.system_metrics['network']['connections']}")
        
        # Service diagnostics
        for service_name, diag in self.diagnostics.items():
            report.append(f"\n\nSERVICE: {service_name}")
            report.append("=" * 60)
            report.append(f"Status: {diag.status.upper()}")
            
            # Group results by category
            categories = {}
            for result in diag.results:
                if result.category not in categories:
                    categories[result.category] = []
                categories[result.category].append(result)
            
            for category, results in categories.items():
                report.append(f"\n{category}:")
                report.append("-" * 30)
                
                for result in results:
                    level_symbol = {
                        DiagnosticLevel.INFO: "[INFO]",
                        DiagnosticLevel.WARNING: "[WARN]",
                        DiagnosticLevel.ERROR: "[ERROR]",
                        DiagnosticLevel.CRITICAL: "[CRIT]"
                    }[result.level]
                    
                    report.append(f"  {level_symbol} {result.check}: {result.message}")
                    
                    if result.recommendation:
                        report.append(f"    → Recommendation: {result.recommendation}")
            
            # Performance metrics
            if diag.performance:
                report.append("\nPerformance Metrics:")
                report.append("-" * 30)
                for key, value in diag.performance.items():
                    report.append(f"  {key}: {value:.2f}")
            
            # Errors
            if diag.errors:
                report.append("\nErrors:")
                report.append("-" * 30)
                for error in diag.errors:
                    report.append(f"  - {error}")
        
        # Summary
        report.append("\n\nSUMMARY:")
        report.append("=" * 60)
        
        critical_count = sum(
            1 for d in self.diagnostics.values()
            for r in d.results
            if r.level == DiagnosticLevel.CRITICAL
        )
        error_count = sum(
            1 for d in self.diagnostics.values()
            for r in d.results
            if r.level == DiagnosticLevel.ERROR
        )
        warning_count = sum(
            1 for d in self.diagnostics.values()
            for r in d.results
            if r.level == DiagnosticLevel.WARNING
        )
        
        report.append(f"Critical Issues: {critical_count}")
        report.append(f"Errors: {error_count}")
        report.append(f"Warnings: {warning_count}")
        
        if critical_count > 0:
            report.append("\n⚠ IMMEDIATE ACTION REQUIRED: Critical issues detected")
        elif error_count > 0:
            report.append("\n⚠ ACTION REQUIRED: Errors detected")
        elif warning_count > 0:
            report.append("\n⚠ ATTENTION: Warnings detected")
        else:
            report.append("\n✓ All services operating normally")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Service diagnostics for AG News Text Classification"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to diagnostic configuration file"
    )
    parser.add_argument(
        "--services",
        nargs="+",
        help="Specific services to diagnose"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for diagnostic report"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize diagnostician
    diagnostician = ServiceDiagnostician(args.config)
    
    # Filter services if specified
    if args.services:
        diagnostician.config["services"] = {
            k: v for k, v in diagnostician.config["services"].items()
            if k in args.services
        }
    
    # Run diagnostics
    await diagnostician.diagnose_all_services()
    
    if args.json:
        # Output JSON
        result = {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": diagnostician.system_metrics,
            "services": {}
        }
        
        for service_name, diag in diagnostician.diagnostics.items():
            result["services"][service_name] = {
                "status": diag.status,
                "results": [
                    {
                        "category": r.category,
                        "check": r.check,
                        "level": r.level.value,
                        "message": r.message,
                        "details": r.details,
                        "recommendation": r.recommendation
                    }
                    for r in diag.results
                ],
                "metrics": diag.metrics,
                "performance": diag.performance,
                "errors": diag.errors
            }
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            console.print(f"[green]Diagnostics saved to {args.output}[/green]")
        else:
            print(json.dumps(result, indent=2, default=str))
    else:
        # Output text report
        report = diagnostician.generate_report()
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            console.print(f"[green]Report saved to {args.output}[/green]")
        else:
            print(report)


if __name__ == "__main__":
    asyncio.run(main())
