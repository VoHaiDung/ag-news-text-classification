"""
Training Router for REST API
================================================================================
Implements endpoints for model training, fine-tuning, and training job management
following MLOps best practices and asynchronous processing patterns.

This module provides comprehensive training capabilities including distributed
training, hyperparameter optimization, and experiment tracking.

References:
    - Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization
    - Dean, J., et al. (2012). Large Scale Distributed Deep Networks
    - MLflow Documentation: Tracking ML Experiments

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, WebSocket
from fastapi.responses import JSONResponse

from src.api.base.auth import AuthToken, Role
from src.api.rest.dependencies import (
    get_current_user,
    get_training_service,
    get_data_service,
    require_role,
    PaginationParams
)
from src.api.rest.schemas.request_schemas import TrainingRequest
from src.api.rest.schemas.response_schemas import (
    BaseResponse,
    TrainingResponse,
    TrainingStatus
)
from src.services.core.training_service import TrainingService, TrainingJob
from src.services.core.data_service import DataService
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/training",
    tags=["Training"],
    responses={
        403: {"description": "Forbidden - Training requires authentication"},
        404: {"description": "Training job or dataset not found"},
        409: {"description": "Conflict - Training already in progress"},
        500: {"description": "Internal server error"}
    }
)


@router.post(
    "/start",
    response_model=TrainingResponse,
    summary="Start model training",
    description="Initiate a new model training job",
    dependencies=[Depends(require_role([Role.ADMIN, Role.USER]))]
)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: AuthToken = Depends(get_current_user),
    training_service: TrainingService = Depends(get_training_service),
    data_service: DataService = Depends(get_data_service)
) -> TrainingResponse:
    """
    Start a new model training job.
    
    Args:
        request: Training configuration request
        background_tasks: FastAPI background tasks
        current_user: Authenticated user
        training_service: Training service instance
        data_service: Data service instance
        
    Returns:
        TrainingResponse with job information
        
    Raises:
        HTTPException: For various error conditions
    """
    try:
        # Validate dataset exists
        if not await data_service.dataset_exists(request.dataset_name):
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{request.dataset_name}' not found"
            )
        
        # Check if model is already being trained
        active_jobs = await training_service.get_active_jobs()
        for job in active_jobs:
            if (job.model_name == request.model_name.value and 
                job.dataset_name == request.dataset_name):
                raise HTTPException(
                    status_code=409,
                    detail=f"Training already in progress for {request.model_name} on {request.dataset_name}"
                )
        
        # Create training job
        job_id = str(uuid.uuid4())
        
        logger.info(
            f"Starting training job {job_id} for model {request.model_name} "
            f"requested by user {current_user.user_id}"
        )
        
        # Prepare training configuration
        training_config = {
            "model_name": request.model_name.value,
            "dataset_name": request.dataset_name,
            "validation_split": request.validation_split,
            "epochs": request.epochs,
            "batch_size": request.batch_size,
            "learning_rate": request.learning_rate,
            "early_stopping": request.early_stopping,
            "checkpoint_interval": request.checkpoint_interval,
            "user_id": current_user.user_id,
            **(request.training_config or {})
        }
        
        # Create training job
        job = await training_service.create_job(
            job_id=job_id,
            model_name=request.model_name.value,
            dataset_name=request.dataset_name,
            config=training_config
        )
        
        # Start training in background
        background_tasks.add_task(
            training_service.run_training,
            job_id,
            training_config
        )
        
        # Estimate duration based on dataset size and model
        estimated_duration = await _estimate_training_duration(
            request.model_name.value,
            request.dataset_name,
            request.epochs,
            data_service
        )
        
        return TrainingResponse(
            request_id=str(uuid.uuid4()),
            job_id=job_id,
            model_name=request.model_name.value,
            dataset_name=request.dataset_name,
            status="initiated",
            estimated_duration=estimated_duration
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start training: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start training: {str(e)}"
        )


@router.get(
    "/jobs",
    summary="List training jobs",
    description="Get list of all training jobs with optional filtering"
)
async def list_training_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    model_name: Optional[str] = Query(None, description="Filter by model"),
    pagination: PaginationParams = Depends(),
    current_user: AuthToken = Depends(get_current_user),
    training_service: TrainingService = Depends(get_training_service)
) -> Dict[str, Any]:
    """
    List training jobs with optional filtering.
    
    Args:
        status: Optional status filter
        model_name: Optional model name filter
        pagination: Pagination parameters
        current_user: Authenticated user
        training_service: Training service instance
        
    Returns:
        List of training jobs
    """
    try:
        # Get all jobs
        jobs = await training_service.list_jobs(
            user_id=current_user.user_id if current_user.has_role(Role.USER) else None,
            status=status,
            model_name=model_name
        )
        
        # Apply pagination
        total = len(jobs)
        start_idx = pagination.offset
        end_idx = start_idx + pagination.limit
        paginated_jobs = jobs[start_idx:end_idx]
        
        # Format response
        formatted_jobs = []
        for job in paginated_jobs:
            formatted_jobs.append({
                "job_id": job.job_id,
                "model_name": job.model_name,
                "dataset_name": job.dataset_name,
                "status": job.status,
                "progress": job.progress,
                "created_at": job.created_at.isoformat(),
                "updated_at": job.updated_at.isoformat(),
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "metrics": job.metrics
            })
        
        return {
            "jobs": formatted_jobs,
            "total": total,
            "page": pagination.page,
            "page_size": pagination.page_size,
            "total_pages": (total + pagination.page_size - 1) // pagination.page_size
        }
        
    except Exception as e:
        logger.error(f"Failed to list training jobs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve training jobs"
        )


@router.get(
    "/jobs/{job_id}",
    response_model=TrainingStatus,
    summary="Get training job status",
    description="Get detailed status of a specific training job"
)
async def get_training_status(
    job_id: str,
    current_user: AuthToken = Depends(get_current_user),
    training_service: TrainingService = Depends(get_training_service)
) -> TrainingStatus:
    """
    Get status of a specific training job.
    
    Args:
        job_id: Training job ID
        current_user: Authenticated user
        training_service: Training service instance
        
    Returns:
        TrainingStatus with job details
        
    Raises:
        HTTPException: If job not found or access denied
    """
    try:
        # Get job details
        job = await training_service.get_job(job_id)
        
        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Training job '{job_id}' not found"
            )
        
        # Check access permissions
        if (current_user.has_role(Role.USER) and 
            job.config.get("user_id") != current_user.user_id):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this training job"
            )
        
        # Calculate estimated time remaining
        estimated_remaining = None
        if job.status == "running" and job.progress > 0:
            elapsed = (datetime.now(timezone.utc) - job.updated_at).total_seconds()
            if job.progress > 0:
                total_estimate = elapsed / (job.progress / 100)
                estimated_remaining = int(total_estimate - elapsed)
        
        return TrainingStatus(
            job_id=job.job_id,
            status=job.status,
            progress=job.progress,
            current_epoch=job.current_epoch,
            total_epochs=job.config.get("epochs", 10),
            metrics=job.metrics,
            estimated_time_remaining=estimated_remaining
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve training status"
        )


@router.post(
    "/jobs/{job_id}/stop",
    response_model=BaseResponse,
    summary="Stop training job",
    description="Stop a running training job",
    dependencies=[Depends(require_role([Role.ADMIN, Role.USER]))]
)
async def stop_training(
    job_id: str,
    save_checkpoint: bool = Query(True, description="Save checkpoint before stopping"),
    current_user: AuthToken = Depends(get_current_user),
    training_service: TrainingService = Depends(get_training_service)
) -> BaseResponse:
    """
    Stop a running training job.
    
    Args:
        job_id: Training job ID
        save_checkpoint: Whether to save checkpoint
        current_user: Authenticated user
        training_service: Training service instance
        
    Returns:
        BaseResponse indicating success
        
    Raises:
        HTTPException: If job not found or cannot be stopped
    """
    try:
        # Get job details
        job = await training_service.get_job(job_id)
        
        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Training job '{job_id}' not found"
            )
        
        # Check access permissions
        if (current_user.has_role(Role.USER) and 
            job.config.get("user_id") != current_user.user_id):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this training job"
            )
        
        # Check if job can be stopped
        if job.status not in ["running", "pending"]:
            raise HTTPException(
                status_code=409,
                detail=f"Job is not running (status: {job.status})"
            )
        
        # Stop the job
        await training_service.stop_job(job_id, save_checkpoint)
        
        logger.info(
            f"Training job {job_id} stopped by user {current_user.user_id}"
        )
        
        return BaseResponse(
            request_id=str(uuid.uuid4()),
            status="success",
            message=f"Training job {job_id} stopped successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop training: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stop training: {str(e)}"
        )


@router.post(
    "/jobs/{job_id}/resume",
    response_model=BaseResponse,
    summary="Resume training job",
    description="Resume a stopped or failed training job",
    dependencies=[Depends(require_role([Role.ADMIN, Role.USER]))]
)
async def resume_training(
    job_id: str,
    background_tasks: BackgroundTasks,
    from_checkpoint: Optional[str] = Query(None, description="Checkpoint to resume from"),
    current_user: AuthToken = Depends(get_current_user),
    training_service: TrainingService = Depends(get_training_service)
) -> BaseResponse:
    """
    Resume a stopped or failed training job.
    
    Args:
        job_id: Training job ID
        background_tasks: FastAPI background tasks
        from_checkpoint: Optional checkpoint to resume from
        current_user: Authenticated user
        training_service: Training service instance
        
    Returns:
        BaseResponse indicating success
    """
    try:
        # Get job details
        job = await training_service.get_job(job_id)
        
        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Training job '{job_id}' not found"
            )
        
        # Check access permissions
        if (current_user.has_role(Role.USER) and 
            job.config.get("user_id") != current_user.user_id):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this training job"
            )
        
        # Check if job can be resumed
        if job.status not in ["stopped", "failed"]:
            raise HTTPException(
                status_code=409,
                detail=f"Job cannot be resumed (status: {job.status})"
            )
        
        # Resume training in background
        background_tasks.add_task(
            training_service.resume_training,
            job_id,
            from_checkpoint
        )
        
        logger.info(
            f"Training job {job_id} resumed by user {current_user.user_id}"
        )
        
        return BaseResponse(
            request_id=str(uuid.uuid4()),
            status="success",
            message=f"Training job {job_id} resumed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resume training: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to resume training: {str(e)}"
        )


@router.delete(
    "/jobs/{job_id}",
    response_model=BaseResponse,
    summary="Delete training job",
    description="Delete a training job and its artifacts",
    dependencies=[Depends(require_role([Role.ADMIN, Role.USER]))]
)
async def delete_training_job(
    job_id: str,
    delete_artifacts: bool = Query(False, description="Delete associated artifacts"),
    current_user: AuthToken = Depends(get_current_user),
    training_service: TrainingService = Depends(get_training_service)
) -> BaseResponse:
    """
    Delete a training job.
    
    Args:
        job_id: Training job ID
        delete_artifacts: Whether to delete artifacts
        current_user: Authenticated user
        training_service: Training service instance
        
    Returns:
        BaseResponse indicating success
    """
    try:
        # Get job details
        job = await training_service.get_job(job_id)
        
        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Training job '{job_id}' not found"
            )
        
        # Check access permissions
        if (current_user.has_role(Role.USER) and 
            job.config.get("user_id") != current_user.user_id):
            raise HTTPException(
                status_code=403,
                detail="Access denied to this training job"
            )
        
        # Delete the job
        await training_service.delete_job(job_id, delete_artifacts)
        
        logger.info(
            f"Training job {job_id} deleted by user {current_user.user_id}"
        )
        
        return BaseResponse(
            request_id=str(uuid.uuid4()),
            status="success",
            message=f"Training job {job_id} deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete training job: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete training job: {str(e)}"
        )


@router.get(
    "/jobs/{job_id}/logs",
    summary="Get training logs",
    description="Stream training logs for a job"
)
async def get_training_logs(
    job_id: str,
    lines: int = Query(100, description="Number of log lines"),
    follow: bool = Query(False, description="Follow log output"),
    current_user: AuthToken = Depends(get_current_user),
    training_service: TrainingService = Depends(get_training_service)
):
    """
    Get training logs for a job.
    
    Args:
        job_id: Training job ID
        lines: Number of log lines to return
        follow: Whether to follow log output
        current_user: Authenticated user
        training_service: Training service instance
        
    Returns:
        Log lines or streaming response
    """
    try:
        # Get job details
        job = await training_service.get_job(job_id)
        
        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Training job '{job_id}' not found"
            )
        
        # Get logs
        logs = await training_service.get_logs(job_id, lines)
        
        if not follow:
            return {
                "job_id": job_id,
                "logs": logs,
                "lines": len(logs)
            }
        else:
            # Stream logs using Server-Sent Events
            from fastapi.responses import StreamingResponse
            
            async def log_generator():
                async for log_line in training_service.stream_logs(job_id):
                    yield f"data: {log_line}\n\n"
            
            return StreamingResponse(
                log_generator(),
                media_type="text/event-stream"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training logs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve training logs"
        )


@router.websocket("/jobs/{job_id}/ws")
async def training_websocket(
    websocket: WebSocket,
    job_id: str,
    training_service: TrainingService = Depends(get_training_service)
):
    """
    WebSocket endpoint for real-time training updates.
    
    Args:
        websocket: WebSocket connection
        job_id: Training job ID
        training_service: Training service instance
    """
    await websocket.accept()
    
    try:
        # Verify job exists
        job = await training_service.get_job(job_id)
        if not job:
            await websocket.send_json({
                "type": "error",
                "message": f"Job {job_id} not found"
            })
            await websocket.close()
            return
        
        # Send initial status
        await websocket.send_json({
            "type": "status",
            "data": {
                "job_id": job_id,
                "status": job.status,
                "progress": job.progress,
                "metrics": job.metrics
            }
        })
        
        # Stream updates
        while True:
            # Get updated job status
            job = await training_service.get_job(job_id)
            
            if job:
                await websocket.send_json({
                    "type": "update",
                    "data": {
                        "status": job.status,
                        "progress": job.progress,
                        "current_epoch": job.current_epoch,
                        "metrics": job.metrics
                    }
                })
                
                # Check if job is complete
                if job.status in ["completed", "failed", "stopped"]:
                    await websocket.send_json({
                        "type": "complete",
                        "data": {"status": job.status}
                    })
                    break
            
            # Wait before next update
            await asyncio.sleep(5)
            
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        await websocket.close()


async def _estimate_training_duration(
    model_name: str,
    dataset_name: str,
    epochs: int,
    data_service: DataService
) -> int:
    """
    Estimate training duration based on model and dataset.
    
    Args:
        model_name: Model name
        dataset_name: Dataset name
        epochs: Number of epochs
        data_service: Data service instance
        
    Returns:
        Estimated duration in seconds
    """
    try:
        # Get dataset size
        dataset_info = await data_service.get_dataset_info(dataset_name)
        num_samples = dataset_info.get("total_samples", 100000)
        
        # Estimate based on model and dataset size
        # These are rough estimates
        samples_per_second = {
            "deberta-v3-large": 10,
            "roberta-large": 15,
            "xlnet-large": 12,
            "electra-large": 18,
            "longformer-large": 8,
            "ensemble": 5
        }
        
        sps = samples_per_second.get(model_name, 10)
        duration = (num_samples * epochs) / sps
        
        return int(duration)
        
    except Exception:
        # Return default estimate
        return epochs * 600  # 10 minutes per epoch
