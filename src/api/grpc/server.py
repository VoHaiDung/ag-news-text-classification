"""
gRPC Server Implementation
================================================================================
This module implements the main gRPC server with service registration,
interceptor chain, and graceful shutdown handling.

Implements production-grade features including:
- Service multiplexing
- Interceptor chain for cross-cutting concerns
- Health checking service
- Reflection for debugging
- Graceful shutdown

References:
    - gRPC Python Documentation
    - Google API Design Guide
    - Martin, R. C. (2008). Clean Code

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
import signal
import sys
import logging
from concurrent import futures
from typing import List, Optional, Any
import grpc
from grpc_reflection.v1alpha import reflection
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from .services.classification_service import ClassificationService
from .services.training_service import TrainingService
from .services.model_service import ModelManagementService
from .services.data_service import DataService
from .interceptors.auth_interceptor import AuthInterceptor
from .interceptors.logging_interceptor import LoggingInterceptor
from .interceptors.metrics_interceptor import MetricsInterceptor
from .interceptors.error_interceptor import ErrorInterceptor
from . import DEFAULT_CONFIG, SERVICES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GRPCServer:
    """
    Main gRPC server implementation with service management.
    
    Attributes:
        config: Server configuration
        server: gRPC server instance
        health_servicer: Health checking service
        services: Registered services
        interceptors: Server interceptors
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize gRPC server.
        
        Args:
            config: Server configuration dictionary
        """
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.server = None
        self.health_servicer = None
        self.services = {}
        self.interceptors = []
        self._shutdown_event = asyncio.Event()
        
    def _setup_interceptors(self) -> List[grpc.ServerInterceptor]:
        """
        Set up server interceptors.
        
        Returns:
            List[grpc.ServerInterceptor]: Configured interceptors
        """
        interceptors = [
            ErrorInterceptor(),
            MetricsInterceptor(),
            LoggingInterceptor(),
            AuthInterceptor()
        ]
        
        logger.info(f"Configured {len(interceptors)} interceptors")
        return interceptors
    
    def _register_services(self, server: grpc.Server) -> None:
        """
        Register all gRPC services.
        
        Args:
            server: gRPC server instance
        """
        # Register classification service
        classification_service = ClassificationService()
        self.services['classification'] = classification_service
        classification_service.register(server)
        logger.info("Registered ClassificationService")
        
        # Register training service
        training_service = TrainingService()
        self.services['training'] = training_service
        training_service.register(server)
        logger.info("Registered TrainingService")
        
        # Register model management service
        model_service = ModelManagementService()
        self.services['model'] = model_service
        model_service.register(server)
        logger.info("Registered ModelManagementService")
        
        # Register data service
        data_service = DataService()
        self.services['data'] = data_service
        data_service.register(server)
        logger.info("Registered DataService")
        
    def _setup_health_service(self, server: grpc.Server) -> None:
        """
        Set up health checking service.
        
        Args:
            server: gRPC server instance
        """
        self.health_servicer = health.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(
            self.health_servicer,
            server
        )
        
        # Set initial health status for all services
        for service_name in SERVICES.values():
            self.health_servicer.set(
                service_name,
                health_pb2.HealthCheckResponse.SERVING
            )
        
        # Set overall server health
        self.health_servicer.set(
            "",
            health_pb2.HealthCheckResponse.SERVING
        )
        
        logger.info("Health service configured")
    
    def _setup_reflection(self, server: grpc.Server) -> None:
        """
        Enable gRPC reflection for debugging.
        
        Args:
            server: gRPC server instance
        """
        service_names = [
            "ag_news.ClassificationService",
            "ag_news.TrainingService",
            "ag_news.ModelManagementService",
            "ag_news.DataService",
            health_pb2.DESCRIPTOR.services_by_name['Health'].full_name,
            reflection.SERVICE_NAME
        ]
        
        reflection.enable_server_reflection(service_names, server)
        logger.info("Reflection enabled for debugging")
    
    def _create_server(self) -> grpc.Server:
        """
        Create and configure gRPC server.
        
        Returns:
            grpc.Server: Configured server instance
        """
        # Set up thread pool
        executor = futures.ThreadPoolExecutor(
            max_workers=self.config['max_workers']
        )
        
        # Create server with interceptors
        self.interceptors = self._setup_interceptors()
        server = grpc.server(
            executor,
            interceptors=self.interceptors,
            options=self.config['options']
        )
        
        # Register services
        self._register_services(server)
        
        # Set up health service
        self._setup_health_service(server)
        
        # Enable reflection in development
        if self.config.get('enable_reflection', True):
            self._setup_reflection(server)
        
        return server
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def start(self) -> None:
        """
        Start the gRPC server.
        
        Raises:
            RuntimeError: If server fails to start
        """
        try:
            # Create server
            self.server = self._create_server()
            
            # Add insecure port (use secure port in production)
            address = f"{self.config['host']}:{self.config['port']}"
            port = self.server.add_insecure_port(address)
            
            if port == 0:
                raise RuntimeError(f"Failed to bind to {address}")
            
            # Start server
            await self.server.start()
            logger.info(f"gRPC server started on {address}")
            
            # Set up signal handlers
            self._setup_signal_handlers()
            
            # Wait for shutdown
            await self._shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise
    
    async def shutdown(self, grace_period: float = 5.0) -> None:
        """
        Gracefully shutdown the server.
        
        Args:
            grace_period: Grace period in seconds
        """
        logger.info("Initiating graceful shutdown...")
        
        if self.server:
            # Update health status to NOT_SERVING
            if self.health_servicer:
                for service_name in SERVICES.values():
                    self.health_servicer.set(
                        service_name,
                        health_pb2.HealthCheckResponse.NOT_SERVING
                    )
                self.health_servicer.set(
                    "",
                    health_pb2.HealthCheckResponse.NOT_SERVING
                )
            
            # Stop accepting new requests
            self.server.stop(grace_period)
            
            # Wait for pending requests to complete
            await asyncio.sleep(grace_period)
            
            # Clean up services
            for service in self.services.values():
                if hasattr(service, 'cleanup'):
                    await service.cleanup()
            
            logger.info("Server shutdown complete")
        
        # Signal shutdown completion
        self._shutdown_event.set()
    
    def get_service(self, name: str) -> Any:
        """
        Get registered service by name.
        
        Args:
            name: Service name
            
        Returns:
            Any: Service instance or None
        """
        return self.services.get(name)
    
    def get_metrics(self) -> dict:
        """
        Get server metrics.
        
        Returns:
            dict: Server metrics
        """
        metrics = {
            "server_status": "running" if self.server else "stopped",
            "registered_services": len(self.services),
            "interceptors": len(self.interceptors),
            "config": self.config
        }
        
        # Collect metrics from interceptors
        for interceptor in self.interceptors:
            if hasattr(interceptor, 'get_metrics'):
                metrics.update(interceptor.get_metrics())
        
        return metrics

async def run_server(config: Optional[dict] = None) -> None:
    """
    Run the gRPC server.
    
    Args:
        config: Optional server configuration
    """
    server = GRPCServer(config)
    
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
    finally:
        await server.shutdown()

def main():
    """Main entry point for gRPC server."""
    asyncio.run(run_server())

if __name__ == "__main__":
    main()
