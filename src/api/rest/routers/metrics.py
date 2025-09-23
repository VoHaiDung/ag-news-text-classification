"""
Metrics Router for REST API
================================================================================
Implements endpoints for metrics collection, monitoring, and performance
analysis following observability best practices.

This module provides comprehensive metrics and monitoring capabilities
for system health, performance tracking, and usage analytics.

References:
    - Google SRE Book: Chapter 6 - Monitoring Distributed Systems
    - Prometheus Documentation: Metric Types and Best Practices
    - OpenTelemetry Observability Framework

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse

from src.api.base.auth import AuthToken, Role
from src.api.rest.dependencies import (
    get_current_user,
    require_role
)
from src.api.rest.schemas.request_schemas import MetricsRequest
from src.api.rest.schemas.response_schemas import MetricsResponse
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/metrics",
    tags=["Metrics"],
    responses={
        403: {"description": "Forbidden - Metrics access restricted"},
        500: {"description": "Internal server error"}
    }
)


@router.get(
    "/",
    summary="Get system metrics",
    description="Get current system metrics and performance indicators"
)
async def get_metrics(
    format: str = Query("json", description="Response format (json or prometheus)"),
    current_user: Optional[AuthToken] = Depends(get_current_user)
):
    """
    Get system metrics.
    
    Args:
        format: Response format
        current_user: Authenticated user
        
    Returns:
        Metrics in requested format
    """
    try:
        # Collect metrics
        metrics = await _collect_system_metrics()
        
        if format == "prometheus":
            # Format as Prometheus metrics
            return PlainTextResponse(
                content=_format_prometheus_metrics(metrics),
                media_type="text/plain; version=0.0.4"
            )
        else:
            # Return as JSON
            return JSONResponse(content=metrics)
            
    except Exception as e:
        logger.error(f"Failed to collect metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to collect metrics"
        )


@router.post(
    "/query",
    response_model=MetricsResponse,
    summary="Query metrics",
    description="Query historical metrics with filtering and aggregation"
)
async def query_metrics(
    request: MetricsRequest,
    current_user: Optional[AuthToken] = Depends(get_current_user)
) -> MetricsResponse:
    """
    Query historical metrics.
    
    Args:
        request: Metrics query request
        current_user: Authenticated user
        
    Returns:
        MetricsResponse with queried data
    """
    try:
        # Parse time range
        time_range = _parse_time_range(request.time_range)
        
        # Query metrics based on type
        if request.metric_type == "performance":
            data = await _query_performance_metrics(
                time_range,
                request.aggregation,
                request.filters
            )
        elif request.metric_type == "accuracy":
            data = await _query_accuracy_metrics(
                time_range,
                request.aggregation,
                request.filters
            )
        elif request.metric_type == "latency":
            data = await _query_latency_metrics(
                time_range,
                request.aggregation,
                request.filters
            )
        elif request.metric_type == "throughput":
            data = await _query_throughput_metrics(
                time_range,
                request.aggregation,
                request.filters
            )
        elif request.metric_type == "errors":
            data = await _query_error_metrics(
                time_range,
                request.aggregation,
                request.filters
            )
        else:
            raise ValueError(f"Unknown metric type: {request.metric_type}")
        
        # Calculate summary statistics
        summary = _calculate_summary(data, request.aggregation)
        
        return MetricsResponse(
            request_id=str(uuid.uuid4()),
            metric_type=request.metric_type,
            time_range=request.time_range,
            aggregation=request.aggregation,
            data=data,
            summary=summary
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to query metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to query metrics"
        )


@router.get(
    "/health-score",
    summary="Get system health score",
    description="Calculate overall system health score based on multiple metrics"
)
async def get_health_score(
    current_user: Optional[AuthToken] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Calculate system health score.
    
    Args:
        current_user: Authenticated user
        
    Returns:
        Health score and components
    """
    try:
        # Collect various metrics
        metrics = await _collect_system_metrics()
        
        # Calculate component scores
        scores = {
            "api_health": _calculate_api_health_score(metrics),
            "model_health": _calculate_model_health_score(metrics),
            "resource_health": _calculate_resource_health_score(metrics),
            "error_rate": _calculate_error_score(metrics)
        }
        
        # Calculate overall score (weighted average)
        weights = {
            "api_health": 0.3,
            "model_health": 0.3,
            "resource_health": 0.2,
            "error_rate": 0.2
        }
        
        overall_score = sum(
            scores[component] * weight
            for component, weight in weights.items()
        )
        
        # Determine health status
        if overall_score >= 90:
            status = "excellent"
        elif overall_score >= 75:
            status = "good"
        elif overall_score >= 50:
            status = "fair"
        else:
            status = "poor"
        
        return {
            "overall_score": round(overall_score, 2),
            "status": status,
            "component_scores": scores,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recommendations": _get_health_recommendations(scores)
        }
        
    except Exception as e:
        logger.error(f"Failed to calculate health score: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to calculate health score"
        )


@router.get(
    "/usage",
    summary="Get usage statistics",
    description="Get API usage statistics and trends",
    dependencies=[Depends(require_role([Role.ADMIN]))]
)
async def get_usage_statistics(
    time_range: str = Query("24h", description="Time range"),
    group_by: str = Query("endpoint", description="Group by (endpoint, user, model)"),
    current_user: AuthToken = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get usage statistics.
    
    Args:
        time_range: Time range for statistics
        group_by: Grouping dimension
        current_user: Authenticated user
        
    Returns:
        Usage statistics
    """
    try:
        # Parse time range
        start_time, end_time = _parse_time_range(time_range)
        
        # Collect usage data
        usage_data = await _collect_usage_data(start_time, end_time, group_by)
        
        # Calculate trends
        trends = _calculate_trends(usage_data)
        
        return {
            "time_range": time_range,
            "group_by": group_by,
            "total_requests": sum(item["count"] for item in usage_data),
            "unique_users": len(set(item.get("user_id") for item in usage_data if item.get("user_id"))),
            "data": usage_data,
            "trends": trends,
            "top_endpoints": _get_top_endpoints(usage_data) if group_by == "endpoint" else None,
            "top_users": _get_top_users(usage_data) if group_by == "user" else None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get usage statistics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve usage statistics"
        )


@router.get(
    "/performance-report",
    summary="Get performance report",
    description="Generate comprehensive performance report",
    dependencies=[Depends(require_role([Role.ADMIN]))]
)
async def get_performance_report(
    time_range: str = Query("7d", description="Time range for report"),
    include_details: bool = Query(True, description="Include detailed metrics"),
    current_user: AuthToken = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Generate performance report.
    
    Args:
        time_range: Time range for report
        include_details: Whether to include details
        current_user: Authenticated user
        
    Returns:
        Performance report
    """
    try:
        # Parse time range
        start_time, end_time = _parse_time_range(time_range)
        
        # Collect performance data
        report = {
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "summary": await _get_performance_summary(start_time, end_time),
            "availability": await _calculate_availability(start_time, end_time),
            "response_times": await _get_response_time_statistics(start_time, end_time),
            "error_rates": await _get_error_rate_statistics(start_time, end_time),
            "throughput": await _get_throughput_statistics(start_time, end_time)
        }
        
        if include_details:
            report["details"] = {
                "slowest_endpoints": await _get_slowest_endpoints(start_time, end_time),
                "error_breakdown": await _get_error_breakdown(start_time, end_time),
                "peak_usage": await _get_peak_usage_times(start_time, end_time)
            }
        
        report["generated_at"] = datetime.now(timezone.utc).isoformat()
        
        return report
        
    except Exception as e:
        logger.error(f"Failed to generate performance report: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate performance report"
        )


# Helper functions

async def _collect_system_metrics() -> Dict[str, Any]:
    """Collect current system metrics."""
    import psutil
    
    # CPU metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_count = psutil.cpu_count()
    
    # Memory metrics
    memory = psutil.virtual_memory()
    
    # Disk metrics
    disk = psutil.disk_usage("/")
    
    # Process metrics
    process = psutil.Process()
    
    return {
        "system": {
            "cpu_percent": cpu_percent,
            "cpu_count": cpu_count,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3)
        },
        "process": {
            "memory_mb": process.memory_info().rss / (1024**2),
            "cpu_percent": process.cpu_percent(),
            "num_threads": process.num_threads(),
            "uptime_seconds": time.time() - process.create_time()
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


def _format_prometheus_metrics(metrics: Dict[str, Any]) -> str:
    """Format metrics in Prometheus format."""
    lines = []
    
    # System metrics
    system = metrics.get("system", {})
    lines.append(f"# HELP agnews_cpu_usage CPU usage percentage")
    lines.append(f"# TYPE agnews_cpu_usage gauge")
    lines.append(f"agnews_cpu_usage {system.get('cpu_percent', 0)}")
    
    lines.append(f"# HELP agnews_memory_usage Memory usage percentage")
    lines.append(f"# TYPE agnews_memory_usage gauge")
    lines.append(f"agnews_memory_usage {system.get('memory_percent', 0)}")
    
    # Process metrics
    process = metrics.get("process", {})
    lines.append(f"# HELP agnews_process_memory_mb Process memory usage in MB")
    lines.append(f"# TYPE agnews_process_memory_mb gauge")
    lines.append(f"agnews_process_memory_mb {process.get('memory_mb', 0)}")
    
    lines.append(f"# HELP agnews_uptime_seconds Process uptime in seconds")
    lines.append(f"# TYPE agnews_uptime_seconds counter")
    lines.append(f"agnews_uptime_seconds {process.get('uptime_seconds', 0)}")
    
    return "\n".join(lines)


def _parse_time_range(time_range: str) -> tuple[datetime, datetime]:
    """Parse time range string to datetime objects."""
    now = datetime.now(timezone.utc)
    
    # Parse format: 1h, 24h, 7d, 30d, etc.
    import re
    match = re.match(r"(\d+)([hdwmM])", time_range)
    
    if not match:
        raise ValueError(f"Invalid time range format: {time_range}")
    
    value = int(match.group(1))
    unit = match.group(2)
    
    if unit == "h":
        delta = timedelta(hours=value)
    elif unit == "d":
        delta = timedelta(days=value)
    elif unit == "w":
        delta = timedelta(weeks=value)
    elif unit == "m" or unit == "M":
        delta = timedelta(days=value * 30)  # Approximate
    else:
        raise ValueError(f"Invalid time unit: {unit}")
    
    return now - delta, now


def _calculate_summary(data: List[Dict], aggregation: str) -> Dict[str, Any]:
    """Calculate summary statistics."""
    if not data:
        return {}
    
    values = [item.get("value", 0) for item in data]
    
    return {
        "count": len(values),
        "sum": sum(values),
        "avg": sum(values) / len(values) if values else 0,
        "min": min(values) if values else 0,
        "max": max(values) if values else 0
    }


async def _query_performance_metrics(
    time_range: tuple[datetime, datetime],
    aggregation: str,
    filters: Optional[Dict]
) -> List[Dict[str, Any]]:
    """Query performance metrics."""
    # Mock implementation - replace with actual database query
    return [
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "value": 95.5,
            "metric": "accuracy"
        }
    ]


async def _query_latency_metrics(
    time_range: tuple[datetime, datetime],
    aggregation: str,
    filters: Optional[Dict]
) -> List[Dict[str, Any]]:
    """Query latency metrics."""
    # Mock implementation
    return [
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "value": 125,
            "metric": "p50_latency_ms"
        }
    ]


def _calculate_api_health_score(metrics: Dict) -> float:
    """Calculate API health score."""
    # Simple calculation based on CPU and memory
    cpu = metrics.get("system", {}).get("cpu_percent", 0)
    memory = metrics.get("system", {}).get("memory_percent", 0)
    
    # Inverse relationship - lower usage is better
    cpu_score = max(0, 100 - cpu)
    memory_score = max(0, 100 - memory)
    
    return (cpu_score + memory_score) / 2


def _get_health_recommendations(scores: Dict[str, float]) -> List[str]:
    """Get health improvement recommendations."""
    recommendations = []
    
    if scores.get("api_health", 100) < 70:
        recommendations.append("Consider scaling API instances")
    
    if scores.get("resource_health", 100) < 70:
        recommendations.append("System resources running low, consider upgrading")
    
    if scores.get("error_rate", 100) < 80:
        recommendations.append("High error rate detected, investigate recent changes")
    
    return recommendations
