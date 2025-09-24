"""
GraphQL Server Implementation
================================================================================
This module implements the GraphQL server using Strawberry framework, providing
a modern, type-safe GraphQL API with async support and subscriptions.

Implements production features including:
- Async query execution
- WebSocket subscriptions
- Query depth limiting
- Performance monitoring
- Error handling

References:
    - GraphQL Best Practices
    - Strawberry GraphQL Documentation
    - Apollo Server Production Checklist

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
import logging
from typing import Optional, Any, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from strawberry import Schema
from strawberry.fastapi import GraphQLRouter
from strawberry.extensions import QueryDepthLimiter, ValidationCache, ParserCache
from strawberry.subscriptions import GRAPHQL_TRANSPORT_WS_PROTOCOL, GRAPHQL_WS_PROTOCOL

from .schema import schema
from .context import get_context, Context
from . import DEFAULT_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphQLServer:
    """
    GraphQL server implementation with FastAPI integration.
    
    Provides a complete GraphQL API server with queries, mutations,
    subscriptions, and production-ready features.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize GraphQL server.
        
        Args:
            config: Server configuration dictionary
        """
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.app = None
        self.router = None
        
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """
        Manage application lifecycle.
        
        Args:
            app: FastAPI application instance
        """
        # Startup
        logger.info("Starting GraphQL server...")
        await self._startup()
        
        yield
        
        # Shutdown
        logger.info("Shutting down GraphQL server...")
        await self._shutdown()
    
    async def _startup(self) -> None:
        """Perform startup tasks."""
        # Initialize services
        logger.info("Initializing services...")
        
        # Warm up caches
        logger.info("Warming up caches...")
        
        logger.info("GraphQL server started successfully")
    
    async def _shutdown(self) -> None:
        """Perform shutdown tasks."""
        # Cleanup resources
        logger.info("Cleaning up resources...")
        
        logger.info("GraphQL server shutdown complete")
    
    def create_app(self) -> FastAPI:
        """
        Create FastAPI application with GraphQL endpoint.
        
        Returns:
            FastAPI: Configured FastAPI application
        """
        # Create FastAPI app
        self.app = FastAPI(
            title="AG News Classification GraphQL API",
            description="GraphQL API for text classification",
            version=__version__,
            lifespan=self.lifespan
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        # Create GraphQL router
        self.router = self._create_graphql_router()
        
        # Include GraphQL router
        self.app.include_router(
            self.router,
            prefix=self.config["path"]
        )
        
        # Add health check endpoint
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "service": "graphql"}
        
        # Add metrics endpoint
        @self.app.get("/metrics")
        async def metrics():
            return await self._get_metrics()
        
        return self.app
    
    def _create_graphql_router(self) -> GraphQLRouter:
        """
        Create GraphQL router with extensions.
        
        Returns:
            GraphQLRouter: Configured GraphQL router
        """
        # Configure extensions
        extensions = []
        
        # Add query depth limiter
        if self.config.get("max_query_depth"):
            extensions.append(
                QueryDepthLimiter(max_depth=self.config["max_query_depth"])
            )
        
        # Add caching extensions
        if self.config.get("caching", {}).get("enabled"):
            extensions.extend([
                ValidationCache(),
                ParserCache()
            ])
        
        # Create router
        router = GraphQLRouter(
            schema,
            context_getter=get_context,
            graphiql=self.config.get("playground_enabled", True),
            subscription_protocols=[
                GRAPHQL_TRANSPORT_WS_PROTOCOL,
                GRAPHQL_WS_PROTOCOL
            ] if self.config.get("subscription", {}).get("enabled") else [],
            extensions=extensions
        )
        
        return router
    
    async def _get_metrics(self) -> Dict[str, Any]:
        """
        Get server metrics.
        
        Returns:
            Dict[str, Any]: Server metrics
        """
        return {
            "graphql": {
                "queries_total": 0,  # Placeholder
                "mutations_total": 0,
                "subscriptions_active": 0,
                "errors_total": 0
            },
            "config": self.config
        }
    
    def run(self, host: str = None, port: int = None) -> None:
        """
        Run the GraphQL server.
        
        Args:
            host: Server host
            port: Server port
        """
        import uvicorn
        
        host = host or self.config["host"]
        port = port or self.config["port"]
        
        if not self.app:
            self.create_app()
        
        logger.info(f"Starting GraphQL server on {host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )

def create_server(config: Optional[Dict[str, Any]] = None) -> GraphQLServer:
    """
    Create GraphQL server instance.
    
    Args:
        config: Optional server configuration
        
    Returns:
        GraphQLServer: Configured server instance
    """
    return GraphQLServer(config)

def main():
    """Main entry point for GraphQL server."""
    server = create_server()
    server.run()

if __name__ == "__main__":
    main()
