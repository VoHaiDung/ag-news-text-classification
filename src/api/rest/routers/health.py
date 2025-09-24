"""
Health Check Router
================================================================================
This module implements health check endpoints for service monitoring and
readiness probes following cloud-native patterns.

Implements comprehensive health checks including:
- Liveness probes for container orchestration
- Readiness checks for load balancing
- Dependency health aggregation
- Performance metrics collection

References:
    - Nygard, M. (2018). Release It!: Design and Deploy Production-Ready Software
    - Google SRE Book (2016). Chapter 5: Eliminating Toil

Author: Võ Hải Dũng
License: MIT
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import psutil
import asyncio
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from ...base.base_handler import BaseHandler
from ..dependencies import get_current_user, get_service_status
from ..schemas.response_schemas import HealthResponse, ComponentHealth

# Initialize router
router = APIRouter()

class HealthChecker(BaseHandler):
    """
    Health checking service for API and dependencies.
    
    Attributes:
        start_time: Service start timestamp
        checks_performed: Total health checks counter
        last_check_time: Last health check timestamp
    """
    
    def __init__(self):
        """Initialize health checker."""
        super().__init__(logger_name="health_checker")
        self.start_time = datetime.utcnow()
        self.checks_performed = 0
        self.last_check_time = None
        
    async def check_database(self) -> ComponentHealth:
        """
        Check database connectivity and performance.
        
        Returns:
            ComponentHealth: Database health status
        """
        try:
            # Simulate database ping
            start = datetime.utcnow()
            await asyncio.sleep(0.01)  # Simulate query
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            
            return ComponentHealth(
                name="database",
                status="healthy",
                latency_ms=latency,
                details={"connections": 5, "pool_size": 10}
            )
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return ComponentHealth(
                name="database",
                status="unhealthy",
                error=str(e)
            )
    
    async def check_cache(self) -> ComponentHealth:
        """
        Check cache service health.
        
        Returns:
            ComponentHealth: Cache health status
        """
        try:
            start = datetime.utcnow()
            # Simulate cache check
            await asyncio.sleep(0.005)
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            
            return ComponentHealth(
                name="cache",
                status="healthy",
                latency_ms=latency,
                details={"hit_rate": 0.95, "memory_used": "256MB"}
            )
        except Exception as e:
            return ComponentHealth(
                name="cache",
                status="degraded",
                error=str(e)
            )
    
    async def check_model_service(self) -> ComponentHealth:
        """
        Check model service availability.
        
        Returns:
            ComponentHealth: Model service health status
        """
        try:
            # Check if models are loaded
            models_loaded = True  # Placeholder
            
            if models_loaded:
                return ComponentHealth(
                    name="model_service",
                    status="healthy",
                    details={"models_loaded": 3, "gpu_available": True}
                )
            else:
                return ComponentHealth(
                    name="model_service",
                    status="degraded",
                    details={"models_loaded": 0}
                )
        except Exception as e:
            return ComponentHealth(
                name="model_service",
                status="unhealthy",
                error=str(e)
            )
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Collect system resource metrics.
        
        Returns:
            Dict[str, Any]: System metrics
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent
                },
                "disk": {
                    "total": disk.total,
                    "free": disk.free,
                    "percent": disk.percent
                },
                "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds()
            }
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return {}

# Initialize health checker
health_checker = HealthChecker()

@router.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Basic health check endpoint.
    
    Returns:
        HealthResponse: Service health status
    """
    health_checker.checks_performed += 1
    health_checker.last_check_time = datetime.utcnow()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        uptime_seconds=(datetime.utcnow() - health_checker.start_time).total_seconds()
    )

@router.get("/live")
async def liveness_probe() -> JSONResponse:
    """
    Kubernetes liveness probe endpoint.
    
    Returns:
        JSONResponse: Liveness status
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "alive", "timestamp": datetime.utcnow().isoformat()}
    )

@router.get("/ready")
async def readiness_probe() -> JSONResponse:
    """
    Kubernetes readiness probe endpoint.
    
    Returns:
        JSONResponse: Readiness status
        
    Raises:
        HTTPException: If service not ready
    """
    try:
        # Check critical dependencies
        db_health = await health_checker.check_database()
        model_health = await health_checker.check_model_service()
        
        if db_health.status == "unhealthy" or model_health.status == "unhealthy":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready"
            )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "ready", "timestamp": datetime.utcnow().isoformat()}
        )
    except HTTPException:
        raise
    except Exception as e:
        health_checker.logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Readiness check failed: {str(e)}"
        )

@router.get("/detailed", response_model=Dict[str, Any])
async def detailed_health_check(
    include_metrics: bool = True,
    check_dependencies: bool = True
) -> Dict[str, Any]:
    """
    Detailed health check with component statuses.
    
    Args:
        include_metrics: Include system metrics
        check_dependencies: Check external dependencies
        
    Returns:
        Dict[str, Any]: Detailed health information
    """
    health_info = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": {
            "name": "ag-news-classification-api",
            "version": "1.0.0",
            "uptime_seconds": (datetime.utcnow() - health_checker.start_time).total_seconds(),
            "checks_performed": health_checker.checks_performed
        }
    }
    
    # Check dependencies
    if check_dependencies:
        components = []
        
        # Parallel health checks
        tasks = [
            health_checker.check_database(),
            health_checker.check_cache(),
            health_checker.check_model_service()
        ]
        
        component_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in component_results:
            if isinstance(result, Exception):
                components.append(ComponentHealth(
                    name="unknown",
                    status="error",
                    error=str(result)
                ))
            else:
                components.append(result)
        
        # Determine overall health
        unhealthy_count = sum(1 for c in components if c.status == "unhealthy")
        degraded_count = sum(1 for c in components if c.status == "degraded")
        
        if unhealthy_count > 0:
            health_info["status"] = "unhealthy"
        elif degraded_count > 0:
            health_info["status"] = "degraded"
        
        health_info["components"] = [c.dict() for c in components]
    
    # Include system metrics
    if include_metrics:
        health_info["metrics"] = health_checker.get_system_metrics()
    
    return health_info

@router.get("/metrics/system")
async def system_metrics() -> Dict[str, Any]:
    """
    Get current system resource metrics.
    
    Returns:
        Dict[str, Any]: System metrics
    """
    return health_checker.get_system_metrics()

@router.post("/maintenance/enable")
async def enable_maintenance_mode(
    reason: str,
    estimated_duration_minutes: int = 30,
    current_user: Dict = Depends(get_current_user)
) -> JSONResponse:
    """
    Enable maintenance mode.
    
    Args:
        reason: Maintenance reason
        estimated_duration_minutes: Estimated duration
        current_user: Authenticated user
        
    Returns:
        JSONResponse: Maintenance status
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    # Set maintenance mode (placeholder implementation)
    maintenance_info = {
        "enabled": True,
        "reason": reason,
        "started_at": datetime.utcnow().isoformat(),
        "estimated_end": (datetime.utcnow() + timedelta(minutes=estimated_duration_minutes)).isoformat(),
        "initiated_by": current_user.get("username")
    }
    
    health_checker.logger.info(f"Maintenance mode enabled: {maintenance_info}")
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=maintenance_info
    )
