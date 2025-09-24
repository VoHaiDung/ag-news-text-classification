"""
Admin Router
================================================================================
This module implements administrative endpoints for system management,
configuration updates, and operational tasks following security best practices.

Provides administrative capabilities including:
- System configuration management
- User management and access control
- Service orchestration
- Monitoring and debugging tools

References:
    - OWASP Top 10 (2021). Security Risks for Web Applications
    - Beyer, B., et al. (2016). Site Reliability Engineering

Author: Võ Hải Dũng
License: MIT
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import os
import json
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
import psutil

from ...base.base_handler import BaseHandler
from ...base.auth import verify_admin_token
from ..dependencies import get_current_admin, get_config_manager
from ..schemas.request_schemas import (
    ConfigUpdateRequest,
    UserManagementRequest,
    ServiceControlRequest
)
from ..schemas.response_schemas import (
    AdminActionResponse,
    SystemInfoResponse,
    ConfigResponse
)

# Initialize router
router = APIRouter()

class AdminController(BaseHandler):
    """
    Administrative controller for system management.
    
    Attributes:
        config_manager: Configuration management service
        audit_log: Audit logging for admin actions
    """
    
    def __init__(self):
        """Initialize admin controller."""
        super().__init__(logger_name="admin_controller")
        self.audit_log = []
        self.config_cache = {}
        
    def log_admin_action(
        self,
        action: str,
        user: str,
        details: Dict[str, Any]
    ) -> None:
        """
        Log administrative action for audit trail.
        
        Args:
            action: Action performed
            user: Admin username
            details: Action details
        """
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "user": user,
            "details": details
        }
        self.audit_log.append(audit_entry)
        self.logger.info(f"Admin action: {json.dumps(audit_entry)}")
        
    async def reload_configuration(self) -> bool:
        """
        Reload system configuration from files.
        
        Returns:
            bool: Success status
        """
        try:
            # Clear configuration cache
            self.config_cache.clear()
            
            # Reload configurations (placeholder)
            self.logger.info("Configuration reloaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Configuration reload failed: {e}")
            return False
    
    async def collect_system_info(self) -> Dict[str, Any]:
        """
        Collect comprehensive system information.
        
        Returns:
            Dict[str, Any]: System information
        """
        try:
            # System info
            cpu_info = {
                "cores": psutil.cpu_count(),
                "usage_percent": psutil.cpu_percent(interval=1),
                "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            }
            
            memory_info = psutil.virtual_memory()._asdict()
            disk_info = psutil.disk_usage('/')._asdict()
            
            # Process info
            process = psutil.Process()
            process_info = {
                "pid": process.pid,
                "memory_percent": process.memory_percent(),
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
                "create_time": datetime.fromtimestamp(process.create_time()).isoformat()
            }
            
            # Network info
            network_stats = psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "system": {
                    "platform": os.name,
                    "python_version": os.sys.version,
                    "cpu": cpu_info,
                    "memory": memory_info,
                    "disk": disk_info,
                    "network": network_stats
                },
                "process": process_info,
                "environment": {
                    "variables": len(os.environ),
                    "working_directory": os.getcwd()
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to collect system info: {e}")
            raise

# Initialize controller
admin_controller = AdminController()

@router.get("/info", response_model=SystemInfoResponse)
async def get_system_info(
    current_admin: Dict = Depends(get_current_admin)
) -> SystemInfoResponse:
    """
    Get comprehensive system information.
    
    Args:
        current_admin: Authenticated admin user
        
    Returns:
        SystemInfoResponse: System information
    """
    try:
        info = await admin_controller.collect_system_info()
        
        admin_controller.log_admin_action(
            action="system_info_viewed",
            user=current_admin.get("username"),
            details={"timestamp": datetime.utcnow().isoformat()}
        )
        
        return SystemInfoResponse(**info)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve system info: {str(e)}"
        )

@router.post("/config/update", response_model=AdminActionResponse)
async def update_configuration(
    request: ConfigUpdateRequest,
    background_tasks: BackgroundTasks,
    current_admin: Dict = Depends(get_current_admin)
) -> AdminActionResponse:
    """
    Update system configuration.
    
    Args:
        request: Configuration update request
        background_tasks: Background task manager
        current_admin: Authenticated admin user
        
    Returns:
        AdminActionResponse: Update result
    """
    try:
        # Validate configuration changes
        if request.config_section not in ["api", "models", "training", "data"]:
            raise ValueError(f"Invalid config section: {request.config_section}")
        
        # Apply configuration (placeholder)
        admin_controller.config_cache[request.config_section] = request.config_values
        
        # Schedule configuration reload in background
        background_tasks.add_task(admin_controller.reload_configuration)
        
        # Log action
        admin_controller.log_admin_action(
            action="config_updated",
            user=current_admin.get("username"),
            details={
                "section": request.config_section,
                "changes": request.config_values
            }
        )
        
        return AdminActionResponse(
            success=True,
            message=f"Configuration updated for section: {request.config_section}",
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        admin_controller.logger.error(f"Configuration update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/config/reload", response_model=AdminActionResponse)
async def reload_configuration(
    current_admin: Dict = Depends(get_current_admin)
) -> AdminActionResponse:
    """
    Reload all configurations from files.
    
    Args:
        current_admin: Authenticated admin user
        
    Returns:
        AdminActionResponse: Reload result
    """
    success = await admin_controller.reload_configuration()
    
    admin_controller.log_admin_action(
        action="config_reloaded",
        user=current_admin.get("username"),
        details={"success": success}
    )
    
    if success:
        return AdminActionResponse(
            success=True,
            message="Configuration reloaded successfully",
            timestamp=datetime.utcnow()
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Configuration reload failed"
        )

@router.get("/config/{section}", response_model=ConfigResponse)
async def get_configuration(
    section: str,
    current_admin: Dict = Depends(get_current_admin)
) -> ConfigResponse:
    """
    Get current configuration for a section.
    
    Args:
        section: Configuration section name
        current_admin: Authenticated admin user
        
    Returns:
        ConfigResponse: Configuration values
    """
    if section not in ["api", "models", "training", "data"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration section not found: {section}"
        )
    
    config_values = admin_controller.config_cache.get(section, {})
    
    return ConfigResponse(
        section=section,
        values=config_values,
        last_updated=datetime.utcnow()
    )

@router.post("/cache/clear", response_model=AdminActionResponse)
async def clear_cache(
    cache_type: str = "all",
    current_admin: Dict = Depends(get_current_admin)
) -> AdminActionResponse:
    """
    Clear system caches.
    
    Args:
        cache_type: Type of cache to clear
        current_admin: Authenticated admin user
        
    Returns:
        AdminActionResponse: Clear result
    """
    valid_cache_types = ["all", "model", "data", "api", "config"]
    
    if cache_type not in valid_cache_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid cache type. Must be one of: {valid_cache_types}"
        )
    
    # Clear cache (placeholder implementation)
    if cache_type == "all" or cache_type == "config":
        admin_controller.config_cache.clear()
    
    admin_controller.log_admin_action(
        action="cache_cleared",
        user=current_admin.get("username"),
        details={"cache_type": cache_type}
    )
    
    return AdminActionResponse(
        success=True,
        message=f"Cache cleared: {cache_type}",
        timestamp=datetime.utcnow()
    )

@router.get("/audit/logs")
async def get_audit_logs(
    limit: int = 100,
    offset: int = 0,
    current_admin: Dict = Depends(get_current_admin)
) -> List[Dict[str, Any]]:
    """
    Retrieve audit logs.
    
    Args:
        limit: Maximum number of logs to return
        offset: Offset for pagination
        current_admin: Authenticated admin user
        
    Returns:
        List[Dict[str, Any]]: Audit log entries
    """
    logs = admin_controller.audit_log[offset:offset + limit]
    
    admin_controller.log_admin_action(
        action="audit_logs_viewed",
        user=current_admin.get("username"),
        details={"limit": limit, "offset": offset}
    )
    
    return logs

@router.post("/service/restart", response_model=AdminActionResponse)
async def restart_service(
    request: ServiceControlRequest,
    background_tasks: BackgroundTasks,
    current_admin: Dict = Depends(get_current_admin)
) -> AdminActionResponse:
    """
    Restart a service component.
    
    Args:
        request: Service control request
        background_tasks: Background task manager
        current_admin: Authenticated admin user
        
    Returns:
        AdminActionResponse: Restart result
    """
    valid_services = ["api", "model_service", "data_service", "cache_service"]
    
    if request.service_name not in valid_services:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid service name. Must be one of: {valid_services}"
        )
    
    # Schedule service restart (placeholder)
    admin_controller.log_admin_action(
        action="service_restarted",
        user=current_admin.get("username"),
        details={
            "service": request.service_name,
            "graceful": request.graceful
        }
    )
    
    return AdminActionResponse(
        success=True,
        message=f"Service restart initiated: {request.service_name}",
        timestamp=datetime.utcnow()
    )

@router.get("/metrics/summary")
async def get_metrics_summary(
    time_range: str = "1h",
    current_admin: Dict = Depends(get_current_admin)
) -> Dict[str, Any]:
    """
    Get summary of system metrics.
    
    Args:
        time_range: Time range for metrics
        current_admin: Authenticated admin user
        
    Returns:
        Dict[str, Any]: Metrics summary
    """
    # Calculate time range
    time_ranges = {
        "1h": timedelta(hours=1),
        "24h": timedelta(hours=24),
        "7d": timedelta(days=7),
        "30d": timedelta(days=30)
    }
    
    if time_range not in time_ranges:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid time range. Must be one of: {list(time_ranges.keys())}"
        )
    
    # Collect metrics (placeholder)
    metrics = {
        "time_range": time_range,
        "period_start": (datetime.utcnow() - time_ranges[time_range]).isoformat(),
        "period_end": datetime.utcnow().isoformat(),
        "api": {
            "total_requests": 10000,
            "error_rate": 0.01,
            "avg_latency_ms": 45.2
        },
        "models": {
            "predictions_made": 8500,
            "avg_inference_time_ms": 25.3,
            "cache_hit_rate": 0.75
        },
        "system": await admin_controller.collect_system_info()
    }
    
    return metrics
