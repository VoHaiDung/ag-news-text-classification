"""
Start All Services Script for AG News Text Classification System
================================================================================
This script orchestrates the startup of all API services including REST, gRPC,
and GraphQL endpoints. It handles service dependencies, health checks, and
graceful startup/shutdown procedures.

The script implements service orchestration patterns following microservices
best practices for distributed system initialization.

References:
    - Newman, S. (2021). Building Microservices: Designing Fine-Grained Systems
    - Kleppmann, M. (2017). Designing Data-Intensive Applications
    - Richardson, C. (2018). Microservices Patterns

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
import sys
import os
import signal
import time
import argparse
import yaml
import subprocess
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import requests
import grpc
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.service_utils import ServiceDiscovery, ServiceConfig, ServiceState
from src.core.exceptions import ServiceError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rich console for better output
console = Console()


class ServiceManager:
    """
    Manages the lifecycle of all API services
    
    This class orchestrates the startup, monitoring, and shutdown of
    multiple services with proper dependency management and health checking.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize service manager
        
        Args:
            config_path: Path to services configuration file
        """
        self.config_path = config_path or str(
            PROJECT_ROOT / "configs" / "services" / "services.yaml"
        )
        self.services: Dict[str, Dict[str, Any]] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self.discovery = ServiceDiscovery()
        self.shutdown_event = asyncio.Event()
        self.load_configuration()
    
    def load_configuration(self):
        """Load services configuration from YAML file"""
        # Default configuration if file doesn't exist
        default_config = {
            "services": {
                "rest-api": {
                    "enabled": True,
                    "command": "python -m uvicorn src.api.rest.app:app",
                    "host": "0.0.0.0",
                    "port": 8000,
                    "env": {
                        "API_VERSION": "v1",
                        "WORKERS": "4"
                    },
                    "health_check": "http://localhost:8000/health",
                    "dependencies": [],
                    "startup_timeout": 30
                },
                "grpc-server": {
                    "enabled": True,
                    "command": "python src/api/grpc/server.py",
                    "host": "0.0.0.0",
                    "port": 50051,
                    "env": {},
                    "health_check": "grpc://localhost:50051/health",
                    "dependencies": [],
                    "startup_timeout": 30
                },
                "graphql-server": {
                    "enabled": True,
                    "command": "python src/api/graphql/server.py",
                    "host": "0.0.0.0",
                    "port": 4000,
                    "env": {},
                    "health_check": "http://localhost:4000/health",
                    "dependencies": [],
                    "startup_timeout": 30
                },
                "prediction-service": {
                    "enabled": True,
                    "command": "python src/services/core/prediction_service.py",
                    "host": "0.0.0.0",
                    "port": 8001,
                    "env": {},
                    "health_check": "http://localhost:8001/health",
                    "dependencies": [],
                    "startup_timeout": 60
                },
                "training-service": {
                    "enabled": True,
                    "command": "python src/services/core/training_service.py",
                    "host": "0.0.0.0",
                    "port": 8002,
                    "env": {},
                    "health_check": "http://localhost:8002/health",
                    "dependencies": ["prediction-service"],
                    "startup_timeout": 60
                },
                "data-service": {
                    "enabled": True,
                    "command": "python src/services/core/data_service.py",
                    "host": "0.0.0.0",
                    "port": 8003,
                    "env": {},
                    "health_check": "http://localhost:8003/health",
                    "dependencies": [],
                    "startup_timeout": 30
                },
                "monitoring-service": {
                    "enabled": True,
                    "command": "python src/services/monitoring/metrics_service.py",
                    "host": "0.0.0.0",
                    "port": 9090,
                    "env": {},
                    "health_check": "http://localhost:9090/health",
                    "dependencies": [],
                    "startup_timeout": 30
                }
            },
            "startup_order": [
                "monitoring-service",
                "data-service",
                "prediction-service",
                "training-service",
                "rest-api",
                "grpc-server",
                "graphql-server"
            ]
        }
        
        # Load from file if exists
        if Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config:
                    default_config.update(loaded_config)
        
        self.services = default_config["services"]
        self.startup_order = default_config.get(
            "startup_order", 
            list(self.services.keys())
        )
    
    def check_port_available(self, port: int) -> bool:
        """
        Check if a port is available
        
        Args:
            port: Port number to check
            
        Returns:
            True if port is available
        """
        for conn in psutil.net_connections():
            if conn.laddr.port == port and conn.status == 'LISTEN':
                return False
        return True
    
    def start_service(self, service_name: str) -> Tuple[bool, str]:
        """
        Start a single service
        
        Args:
            service_name: Name of the service to start
            
        Returns:
            Tuple of (success, message)
        """
        try:
            service_config = self.services[service_name]
            
            if not service_config.get("enabled", True):
                return True, f"{service_name} is disabled"
            
            # Check if port is available
            port = service_config.get("port")
            if port and not self.check_port_available(port):
                return False, f"Port {port} is already in use"
            
            # Prepare environment variables
            env = os.environ.copy()
            env.update(service_config.get("env", {}))
            env["SERVICE_NAME"] = service_name
            env["SERVICE_HOST"] = service_config.get("host", "0.0.0.0")
            env["SERVICE_PORT"] = str(port) if port else ""
            
            # Start the service process
            command = service_config["command"]
            logger.info(f"Starting {service_name}: {command}")
            
            process = subprocess.Popen(
                command.split(),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(PROJECT_ROOT)
            )
            
            self.processes[service_name] = process
            
            # Register with service discovery
            if port:
                self.discovery.register_service(ServiceConfig(
                    name=service_name,
                    host=service_config.get("host", "localhost"),
                    port=port,
                    protocol=self._get_protocol(service_config),
                    health_check_interval=30
                ))
            
            return True, f"{service_name} started successfully"
            
        except Exception as e:
            logger.error(f"Failed to start {service_name}: {e}")
            return False, str(e)
    
    def _get_protocol(self, service_config: Dict[str, Any]) -> str:
        """Extract protocol from health check URL"""
        health_check = service_config.get("health_check", "")
        if health_check.startswith("grpc://"):
            return "grpc"
        elif health_check.startswith("http://"):
            return "http"
        elif health_check.startswith("https://"):
            return "https"
        return "http"
    
    def check_service_health(self, service_name: str) -> bool:
        """
        Check if a service is healthy
        
        Args:
            service_name: Name of the service
            
        Returns:
            True if service is healthy
        """
        service_config = self.services.get(service_name, {})
        health_check = service_config.get("health_check")
        
        if not health_check:
            # If no health check defined, check process status
            process = self.processes.get(service_name)
            return process and process.poll() is None
        
        try:
            if health_check.startswith("http"):
                response = requests.get(health_check, timeout=5)
                return response.status_code == 200
            elif health_check.startswith("grpc"):
                # Parse gRPC health check URL
                url = health_check.replace("grpc://", "")
                channel = grpc.insecure_channel(url)
                try:
                    grpc.channel_ready_future(channel).result(timeout=5)
                    return True
                finally:
                    channel.close()
            return False
        except Exception as e:
            logger.debug(f"Health check failed for {service_name}: {e}")
            return False
    
    def wait_for_service(self, service_name: str, timeout: int = 60) -> bool:
        """
        Wait for a service to become healthy
        
        Args:
            service_name: Name of the service
            timeout: Maximum wait time in seconds
            
        Returns:
            True if service became healthy
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.check_service_health(service_name):
                return True
            time.sleep(2)
        
        return False
    
    async def start_all_services(self, parallel: bool = False) -> bool:
        """
        Start all services
        
        Args:
            parallel: Whether to start services in parallel
            
        Returns:
            True if all services started successfully
        """
        console.print("\n[bold blue]Starting API Services[/bold blue]\n")
        
        # Create status table
        table = Table(title="Service Startup Status")
        table.add_column("Service", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Port", style="yellow")
        table.add_column("Health", style="magenta")
        
        all_success = True
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            if parallel:
                # Start services in parallel (respecting dependencies)
                await self._start_parallel(progress, table)
            else:
                # Start services sequentially
                for service_name in self.startup_order:
                    if service_name not in self.services:
                        continue
                    
                    task = progress.add_task(
                        f"Starting {service_name}...", 
                        total=1
                    )
                    
                    # Check dependencies
                    deps = self.services[service_name].get("dependencies", [])
                    for dep in deps:
                        if not self.check_service_health(dep):
                            console.print(
                                f"[red]Dependency {dep} not healthy for {service_name}[/red]"
                            )
                            all_success = False
                            continue
                    
                    # Start the service
                    success, message = self.start_service(service_name)
                    
                    if success:
                        # Wait for health check
                        timeout = self.services[service_name].get(
                            "startup_timeout", 30
                        )
                        healthy = self.wait_for_service(service_name, timeout)
                        
                        port = self.services[service_name].get("port", "N/A")
                        health_status = "✓" if healthy else "✗"
                        
                        table.add_row(
                            service_name,
                            "Started" if healthy else "Started (unhealthy)",
                            str(port),
                            health_status
                        )
                        
                        if not healthy:
                            console.print(
                                f"[yellow]Warning: {service_name} started but not healthy[/yellow]"
                            )
                    else:
                        table.add_row(
                            service_name,
                            f"Failed: {message}",
                            "-",
                            "✗"
                        )
                        all_success = False
                    
                    progress.update(task, completed=1)
        
        console.print(table)
        return all_success
    
    async def _start_parallel(self, progress, table):
        """Start services in parallel with dependency resolution"""
        # Group services by dependency level
        levels = self._calculate_dependency_levels()
        
        for level, services in sorted(levels.items()):
            tasks = []
            with ThreadPoolExecutor(max_workers=len(services)) as executor:
                futures = {}
                
                for service_name in services:
                    task = progress.add_task(
                        f"Starting {service_name}...", 
                        total=1
                    )
                    future = executor.submit(self.start_service, service_name)
                    futures[future] = (service_name, task)
                
                for future in as_completed(futures):
                    service_name, task = futures[future]
                    success, message = future.result()
                    
                    if success:
                        timeout = self.services[service_name].get(
                            "startup_timeout", 30
                        )
                        healthy = self.wait_for_service(service_name, timeout)
                        
                        port = self.services[service_name].get("port", "N/A")
                        health_status = "✓" if healthy else "✗"
                        
                        table.add_row(
                            service_name,
                            "Started" if healthy else "Started (unhealthy)",
                            str(port),
                            health_status
                        )
                    else:
                        table.add_row(
                            service_name,
                            f"Failed: {message}",
                            "-",
                            "✗"
                        )
                    
                    progress.update(task, completed=1)
    
    def _calculate_dependency_levels(self) -> Dict[int, List[str]]:
        """Calculate dependency levels for parallel startup"""
        levels = {}
        visited = set()
        
        def get_level(service_name: str, current_level: int = 0):
            if service_name in visited:
                return current_level
            
            visited.add(service_name)
            deps = self.services.get(service_name, {}).get("dependencies", [])
            
            max_dep_level = current_level
            for dep in deps:
                dep_level = get_level(dep, current_level + 1)
                max_dep_level = max(max_dep_level, dep_level)
            
            if max_dep_level not in levels:
                levels[max_dep_level] = []
            levels[max_dep_level].append(service_name)
            
            return max_dep_level
        
        for service_name in self.services:
            if service_name not in visited:
                get_level(service_name)
        
        return levels
    
    def stop_service(self, service_name: str):
        """Stop a single service"""
        process = self.processes.get(service_name)
        if process:
            logger.info(f"Stopping {service_name}...")
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing {service_name}")
                process.kill()
            del self.processes[service_name]
            
            # Deregister from service discovery
            service_config = self.services.get(service_name, {})
            service_id = f"{service_name}-{service_config.get('host', 'localhost')}-{service_config.get('port', '')}"
            self.discovery.deregister_service(service_id)
    
    def stop_all_services(self):
        """Stop all running services"""
        console.print("\n[bold red]Stopping all services...[/bold red]\n")
        
        # Stop in reverse order
        for service_name in reversed(self.startup_order):
            if service_name in self.processes:
                self.stop_service(service_name)
                console.print(f"[green]✓[/green] Stopped {service_name}")
    
    def monitor_services(self):
        """Monitor service health continuously"""
        console.print("\n[bold cyan]Monitoring Services[/bold cyan]\n")
        
        while not self.shutdown_event.is_set():
            table = Table(title=f"Service Status - {datetime.now().strftime('%H:%M:%S')}")
            table.add_column("Service", style="cyan")
            table.add_column("Process", style="yellow")
            table.add_column("Health", style="green")
            table.add_column("Memory (MB)", style="magenta")
            table.add_column("CPU %", style="blue")
            
            for service_name, process in self.processes.items():
                # Check process status
                process_status = "Running" if process.poll() is None else "Stopped"
                
                # Check health
                health = "✓" if self.check_service_health(service_name) else "✗"
                
                # Get resource usage
                try:
                    proc = psutil.Process(process.pid)
                    memory_mb = proc.memory_info().rss / 1024 / 1024
                    cpu_percent = proc.cpu_percent(interval=0.1)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    memory_mb = 0
                    cpu_percent = 0
                
                table.add_row(
                    service_name,
                    process_status,
                    health,
                    f"{memory_mb:.1f}",
                    f"{cpu_percent:.1f}"
                )
            
            console.clear()
            console.print(table)
            
            # Check for failed services and restart if needed
            for service_name, process in list(self.processes.items()):
                if process.poll() is not None:
                    console.print(
                        f"[yellow]Service {service_name} crashed, restarting...[/yellow]"
                    )
                    self.start_service(service_name)
            
            time.sleep(5)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Start all API services for AG News Text Classification"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to services configuration file"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Start services in parallel"
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Enable continuous monitoring"
    )
    parser.add_argument(
        "--services",
        nargs="+",
        help="Specific services to start"
    )
    
    args = parser.parse_args()
    
    # Initialize service manager
    manager = ServiceManager(args.config)
    
    # Filter services if specified
    if args.services:
        manager.services = {
            k: v for k, v in manager.services.items() 
            if k in args.services
        }
        manager.startup_order = [
            s for s in manager.startup_order 
            if s in args.services
        ]
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        console.print("\n[yellow]Received shutdown signal[/yellow]")
        manager.shutdown_event.set()
        manager.stop_all_services()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start all services
        success = await manager.start_all_services(parallel=args.parallel)
        
        if success:
            console.print(
                "\n[bold green]All services started successfully![/bold green]"
            )
            
            if args.monitor:
                # Start monitoring
                manager.monitor_services()
            else:
                console.print(
                    "\n[cyan]Services are running. Press Ctrl+C to stop.[/cyan]"
                )
                # Keep running until interrupted
                await manager.shutdown_event.wait()
        else:
            console.print(
                "\n[bold red]Some services failed to start[/bold red]"
            )
            manager.stop_all_services()
            sys.exit(1)
            
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        manager.stop_all_services()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
