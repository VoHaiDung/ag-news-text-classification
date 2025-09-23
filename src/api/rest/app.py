"""
FastAPI Application for AG News Text Classification REST API
================================================================================
Main application module implementing the REST API server with comprehensive
features including authentication, rate limiting, monitoring, and service orchestration.

This module serves as the entry point for the API application, configuring
all middleware, routers, and services following microservices architecture patterns.

References:
    - Ramírez, S. (2021). FastAPI Documentation
    - Fielding, R. T. (2000). Architectural Styles and the Design of Network-based Software
    - Richardson, C. (2018). Microservices Patterns

Author: Võ Hải Dũng
License: MIT
"""

import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.base.auth import AuthenticationManager
from src.api.base.cors_handler import CORSConfig, CORSHandler
from src.api.rate_limiter import RateLimitManager, RateLimitConfig, RateLimitStrategy
from src.api.rest.middleware.logging_middleware import LoggingMiddleware
from src.api.rest.middleware.metrics_middleware import MetricsMiddleware, MetricsCollector
from src.api.rest.middleware.security_middleware import SecurityMiddleware
from src.api.rest.routers import (
    classification_router,
    health_router,
    models_router,
    training_router,
    data_router,
    metrics_router
)
from src.configs.config_loader import ConfigLoader
from src.core.exceptions import AGNewsException
from src.services.service_registry import ServiceRegistry
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    
    Handles initialization and cleanup of services, connections,
    and resources following graceful shutdown patterns.
    
    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Starting AG News Classification API")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        config_loader = ConfigLoader()
        api_config = config_loader.load_config("api/rest_config.yaml")
        app.state.config = api_config
        logger.info("Configuration loaded successfully")
        
        # Initialize service registry
        service_registry = ServiceRegistry()
        app.state.services = service_registry
        logger.info("Service registry initialized")
        
        # Initialize services
        await _initialize_services(service_registry, api_config)
        logger.info("All services initialized")
        
        # Initialize authentication manager
        auth_config = config_loader.load_config("api/auth_config.yaml")
        app.state.auth_manager = AuthenticationManager(auth_config)
        logger.info("Authentication manager initialized")
        
        # Initialize rate limiter
        rate_limit_config = _create_rate_limit_config(api_config)
        app.state.rate_limiter = RateLimitManager(rate_limit_config)
        logger.info("Rate limiter initialized")
        
        # Initialize metrics collector
        app.state.metrics_collector = MetricsCollector()
        logger.info("Metrics collector initialized")
        
        # Initialize CORS handler
        cors_config = _create_cors_config(api_config)
        app.state.cors_handler = CORSHandler(cors_config)
        logger.info("CORS handler initialized")
        
        logger.info("=" * 60)
        logger.info("API initialization complete")
        logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'production')}")
        logger.info(f"Version: {api_config.get('version', '1.0.0')}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {str(e)}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("=" * 60)
    logger.info("Shutting down AG News Classification API")
    logger.info("=" * 60)
    
    try:
        # Shutdown services
        if hasattr(app.state, "services"):
            await app.state.services.shutdown()
            logger.info("Services shutdown complete")
        
        # Close rate limiter connections
        if hasattr(app.state, "rate_limiter"):
            if hasattr(app.state.rate_limiter.default_limiter, "close"):
                await app.state.rate_limiter.default_limiter.close()
            logger.info("Rate limiter connections closed")
        
        logger.info("API shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}", exc_info=True)


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Configures all middleware, routers, exception handlers, and
    application settings following best practices.
    
    Returns:
        Configured FastAPI application instance
    """
    # Determine environment
    environment = os.getenv("ENVIRONMENT", "production")
    is_development = environment in ["development", "dev"]
    
    # Create FastAPI application
    app = FastAPI(
        title="AG News Text Classification API",
        description="""
        Production-grade REST API for AG News text classification.
        
        ## Features
        - High-accuracy text classification using state-of-the-art models
        - Support for single and batch processing
        - Multiple model options including ensemble methods
        - Real-time and asynchronous processing
        - Comprehensive monitoring and metrics
        - Robust error handling and rate limiting
        
        ## Models
        - DeBERTa-v3-xlarge: Highest accuracy single model
        - RoBERTa-large: Fast and accurate
        - XLNet-large: Bidirectional context understanding
        - Ensemble: Combined predictions for best accuracy
        
        ## Authentication
        Supports JWT tokens and API keys for secure access.
        
        ## Rate Limiting
        Default limits: 60 requests/minute, 1000 requests/hour
        """,
        version="1.0.0",
        docs_url="/docs" if is_development else None,
        redoc_url="/redoc" if is_development else None,
        openapi_url="/openapi.json" if is_development else None,
        lifespan=lifespan
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Total-Count", "X-RateLimit-Limit", "X-RateLimit-Remaining"]
    )
    
    # Add compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add trusted host validation in production
    if not is_development:
        allowed_hosts = os.getenv("ALLOWED_HOSTS", "*").split(",")
        if "*" not in allowed_hosts:
            app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=allowed_hosts
            )
    
    # Add custom middleware
    app.add_middleware(SecurityMiddleware, enable_csrf=False)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(LoggingMiddleware, log_request_body=is_development)
    
    # Include routers
    app.include_router(health_router, prefix="/health", tags=["Health"])
    app.include_router(
        classification_router,
        prefix="/api/v1",
        tags=["Classification"]
    )
    app.include_router(
        models_router,
        prefix="/api/v1",
        tags=["Models"]
    )
    app.include_router(
        training_router,
        prefix="/api/v1",
        tags=["Training"]
    )
    app.include_router(
        data_router,
        prefix="/api/v1",
        tags=["Data"]
    )
    app.include_router(
        metrics_router,
        prefix="/api/v1",
        tags=["Metrics"]
    )
    
    # Mount static files if available
    static_dir = "static"
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    # Configure exception handlers
    _configure_exception_handlers(app)
    
    # Add root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint providing API information."""
        return {
            "name": "AG News Text Classification API",
            "version": "1.0.0",
            "status": "operational",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "documentation": "/docs" if is_development else "Available at /api/v1/docs",
            "health": "/health",
            "endpoints": {
                "classification": "/api/v1/classify",
                "models": "/api/v1/models",
                "training": "/api/v1/training",
                "data": "/api/v1/data",
                "metrics": "/api/v1/metrics"
            }
        }
    
    # API information endpoint
    @app.get("/api/v1/info", tags=["Info"])
    async def api_info(request: Request):
        """Get detailed API information and capabilities."""
        return {
            "name": "AG News Text Classification API",
            "version": "1.0.0",
            "environment": environment,
            "capabilities": {
                "models": [
                    "deberta-v3-xlarge",
                    "roberta-large",
                    "xlnet-large",
                    "electra-large",
                    "longformer-large",
                    "ensemble"
                ],
                "features": [
                    "single_classification",
                    "batch_classification",
                    "streaming_classification",
                    "model_management",
                    "training",
                    "data_upload",
                    "metrics_monitoring"
                ],
                "max_text_length": 10000,
                "max_batch_size": 100,
                "supported_languages": ["en"]
            },
            "rate_limits": {
                "default": {
                    "requests_per_minute": 60,
                    "requests_per_hour": 1000,
                    "burst_size": 10
                }
            },
            "authentication": {
                "methods": ["JWT", "API_KEY"],
                "required_for": ["training", "model_management", "data_upload"]
            },
            "api_version": "v1",
            "base_url": str(request.base_url),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    return app


def _configure_exception_handlers(app: FastAPI):
    """
    Configure global exception handlers.
    
    Args:
        app: FastAPI application instance
    """
    
    @app.exception_handler(AGNewsException)
    async def agnews_exception_handler(request: Request, exc: AGNewsException):
        """Handle custom AG News exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error_code or "AG_NEWS_ERROR",
                "message": str(exc),
                "details": exc.details,
                "request_id": getattr(request.state, "request_id", str(uuid.uuid4())),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with consistent format."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP_ERROR",
                "message": exc.detail,
                "status_code": exc.status_code,
                "request_id": getattr(request.state, "request_id", str(uuid.uuid4())),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle value errors."""
        return JSONResponse(
            status_code=400,
            content={
                "error": "VALIDATION_ERROR",
                "message": str(exc),
                "request_id": getattr(request.state, "request_id", str(uuid.uuid4())),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        logger.error(
            f"Unhandled exception: {str(exc)}",
            exc_info=True,
            extra={"request_id": getattr(request.state, "request_id", "unknown")}
        )
        
        # Determine if we should show details
        environment = os.getenv("ENVIRONMENT", "production")
        is_development = environment in ["development", "dev"]
        
        if is_development:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "INTERNAL_ERROR",
                    "message": str(exc),
                    "type": type(exc).__name__,
                    "request_id": getattr(request.state, "request_id", str(uuid.uuid4())),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "INTERNAL_ERROR",
                    "message": "An internal error occurred. Please try again later.",
                    "request_id": getattr(request.state, "request_id", str(uuid.uuid4())),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )


async def _initialize_services(service_registry: ServiceRegistry, config: Dict[str, Any]):
    """
    Initialize all required services.
    
    Args:
        service_registry: Service registry instance
        config: API configuration
    """
    # Import and initialize services
    from src.services.core.prediction_service import PredictionService
    from src.services.core.training_service import TrainingService
    from src.services.core.data_service import DataService
    from src.services.core.model_management_service import ModelManagementService
    
    # Create service configurations
    from src.services.base_service import ServiceConfig
    
    # Initialize prediction service
    prediction_config = ServiceConfig(
        name="prediction_service",
        version="1.0.0",
        dependencies=["model_management_service"]
    )
    prediction_service = PredictionService(prediction_config)
    await service_registry.register_service(prediction_service)
    
    # Initialize model management service
    model_config = ServiceConfig(
        name="model_management_service",
        version="1.0.0"
    )
    model_service = ModelManagementService(model_config)
    await service_registry.register_service(model_service)
    
    # Initialize training service
    training_config = ServiceConfig(
        name="training_service",
        version="1.0.0",
        dependencies=["data_service", "model_management_service"]
    )
    training_service = TrainingService(training_config)
    await service_registry.register_service(training_service)
    
    # Initialize data service
    data_config = ServiceConfig(
        name="data_service",
        version="1.0.0"
    )
    data_service = DataService(data_config)
    await service_registry.register_service(data_service)


def _create_rate_limit_config(api_config: Dict[str, Any]) -> RateLimitConfig:
    """
    Create rate limit configuration from API config.
    
    Args:
        api_config: API configuration dictionary
        
    Returns:
        RateLimitConfig instance
    """
    rate_limit = api_config.get("rate_limit", {})
    
    # Determine strategy
    strategy_name = rate_limit.get("strategy", "token_bucket")
    strategy_map = {
        "token_bucket": RateLimitStrategy.TOKEN_BUCKET,
        "sliding_window": RateLimitStrategy.SLIDING_WINDOW,
        "distributed": RateLimitStrategy.DISTRIBUTED,
        "adaptive": RateLimitStrategy.ADAPTIVE
    }
    strategy = strategy_map.get(strategy_name, RateLimitStrategy.TOKEN_BUCKET)
    
    return RateLimitConfig(
        requests_per_second=rate_limit.get("requests_per_second"),
        requests_per_minute=rate_limit.get("requests_per_minute", 60),
        requests_per_hour=rate_limit.get("requests_per_hour", 1000),
        requests_per_day=rate_limit.get("requests_per_day", 10000),
        burst_size=rate_limit.get("burst_size", 10),
        strategy=strategy,
        redis_url=rate_limit.get("redis_url"),
        enable_adaptive=rate_limit.get("enable_adaptive", False)
    )


def _create_cors_config(api_config: Dict[str, Any]) -> CORSConfig:
    """
    Create CORS configuration from API config.
    
    Args:
        api_config: API configuration dictionary
        
    Returns:
        CORSConfig instance
    """
    cors = api_config.get("cors", {})
    
    return CORSConfig(
        allowed_origins=cors.get("allowed_origins", ["*"]),
        allowed_methods=cors.get("allowed_methods", ["GET", "POST", "PUT", "DELETE", "OPTIONS"]),
        allowed_headers=cors.get("allowed_headers", ["*"]),
        exposed_headers=cors.get("exposed_headers", ["X-Total-Count"]),
        allow_credentials=cors.get("allow_credentials", True),
        max_age=cors.get("max_age", 3600)
    )


# Create application instance
app = create_app()


if __name__ == "__main__":
    """Run the application using uvicorn."""
    # Configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("API_WORKERS", "4"))
    environment = os.getenv("ENVIRONMENT", "production")
    reload = environment in ["development", "dev"]
    
    # Configure logging
    log_level = "debug" if reload else "info"
    
    # Run server
    uvicorn.run(
        "src.api.rest.app:app" if reload else app,
        host=host,
        port=port,
        workers=1 if reload else workers,
        reload=reload,
        log_level=log_level,
        access_log=True,
        use_colors=True
    )
