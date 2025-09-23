"""
Model Management Router for REST API
================================================================================
Implements endpoints for model management including loading, unloading,
versioning, and configuration of classification models.

This module provides comprehensive model lifecycle management capabilities
following MLOps best practices.

References:
    - Google Cloud ML Engine: Model Versioning
    - MLflow Model Registry Documentation
    - Kubeflow Model Management

Author: Võ Hải Dũng
License: MIT
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse

from src.api.base.auth import AuthToken, Role
from src.api.rest.dependencies import (
    get_current_user,
    get_model_service,
    require_role,
    PaginationParams
)
from src.api.rest.schemas.request_schemas import ModelManagementRequest
from src.api.rest.schemas.response_schemas import (
    ModelInfo,
    ModelListResponse,
    BaseResponse
)
from src.services.core.model_management_service import ModelManagementService
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/models",
    tags=["Models"],
    responses={
        403: {"description": "Forbidden"},
        404: {"description": "Model not found"},
        500: {"description": "Internal server error"}
    }
)


@router.get(
    "/",
    response_model=ModelListResponse,
    summary="List available models",
    description="Get list of all available models with their status and metadata"
)
async def list_models(
    pagination: PaginationParams = Depends(),
    include_unloaded: bool = Query(False, description="Include unloaded models"),
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    current_user: Optional[AuthToken] = Depends(get_current_user),
    model_service: ModelManagementService = Depends(get_model_service)
) -> ModelListResponse:
    """
    List all available models.
    
    Args:
        pagination: Pagination parameters
        include_unloaded: Whether to include unloaded models
        model_type: Optional model type filter
        current_user: Current authenticated user
        model_service: Model management service
        
    Returns:
        ModelListResponse with model information
    """
    try:
        # Get all models
        models = await model_service.list_models(
            include_unloaded=include_unloaded,
            model_type=model_type
        )
        
        # Apply pagination
        start_idx = pagination.offset
        end_idx = start_idx + pagination.limit
        paginated_models = models[start_idx:end_idx]
        
        # Convert to response format
        model_infos = []
        for model in paginated_models:
            model_info = ModelInfo(
                name=model["name"],
                version=model.get("version", "1.0.0"),
                status=model.get("status", "available"),
                loaded=model.get("loaded", False),
                accuracy=model.get("accuracy"),
                parameters=model.get("parameters"),
                size_mb=model.get("size_mb"),
                last_updated=model.get("last_updated")
            )
            model_infos.append(model_info)
        
        return ModelListResponse(
            request_id=str(uuid.uuid4()),
            models=model_infos,
            total=len(models),
            default_model=await model_service.get_default_model()
        )
        
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve models")


@router.get(
    "/{model_name}",
    response_model=ModelInfo,
    summary="Get model details",
    description="Get detailed information about a specific model"
)
async def get_model_details(
    model_name: str,
    version: Optional[str] = Query(None, description="Model version"),
    current_user: Optional[AuthToken] = Depends(get_current_user),
    model_service: ModelManagementService = Depends(get_model_service)
) -> ModelInfo:
    """
    Get detailed information about a specific model.
    
    Args:
        model_name: Name of the model
        version: Optional model version
        current_user: Current authenticated user
        model_service: Model management service
        
    Returns:
        ModelInfo with detailed model information
    """
    try:
        model_details = await model_service.get_model_details(
            model_name=model_name,
            version=version
        )
        
        if not model_details:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found"
            )
        
        return ModelInfo(
            name=model_details["name"],
            version=model_details.get("version", "1.0.0"),
            status=model_details.get("status", "available"),
            loaded=model_details.get("loaded", False),
            accuracy=model_details.get("accuracy"),
            parameters=model_details.get("parameters"),
            size_mb=model_details.get("size_mb"),
            last_updated=model_details.get("last_updated")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model details: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve model details"
        )


@router.post(
    "/{model_name}/load",
    response_model=BaseResponse,
    summary="Load a model",
    description="Load a model into memory for inference",
    dependencies=[Depends(require_role([Role.ADMIN, Role.SERVICE]))]
)
async def load_model(
    model_name: str,
    version: Optional[str] = Query(None, description="Model version"),
    background_tasks: BackgroundTasks = None,
    current_user: AuthToken = Depends(get_current_user),
    model_service: ModelManagementService = Depends(get_model_service)
) -> BaseResponse:
    """
    Load a model into memory.
    
    Args:
        model_name: Name of the model to load
        version: Optional model version
        background_tasks: Background tasks for async loading
        current_user: Current authenticated user
        model_service: Model management service
        
    Returns:
        BaseResponse indicating success or failure
    """
    try:
        logger.info(
            f"Loading model '{model_name}' requested by user {current_user.user_id}"
        )
        
        # Check if model exists
        if not await model_service.model_exists(model_name, version):
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found"
            )
        
        # Check if already loaded
        if await model_service.is_model_loaded(model_name, version):
            return BaseResponse(
                request_id=str(uuid.uuid4()),
                status="success",
                message=f"Model '{model_name}' is already loaded"
            )
        
        # Load model (can be async in background)
        if background_tasks:
            background_tasks.add_task(
                model_service.load_model,
                model_name,
                version
            )
            message = f"Model '{model_name}' loading initiated"
        else:
            await model_service.load_model(model_name, version)
            message = f"Model '{model_name}' loaded successfully"
        
        return BaseResponse(
            request_id=str(uuid.uuid4()),
            status="success",
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )


@router.post(
    "/{model_name}/unload",
    response_model=BaseResponse,
    summary="Unload a model",
    description="Unload a model from memory",
    dependencies=[Depends(require_role([Role.ADMIN, Role.SERVICE]))]
)
async def unload_model(
    model_name: str,
    version: Optional[str] = Query(None, description="Model version"),
    current_user: AuthToken = Depends(get_current_user),
    model_service: ModelManagementService = Depends(get_model_service)
) -> BaseResponse:
    """
    Unload a model from memory.
    
    Args:
        model_name: Name of the model to unload
        version: Optional model version
        current_user: Current authenticated user
        model_service: Model management service
        
    Returns:
        BaseResponse indicating success or failure
    """
    try:
        logger.info(
            f"Unloading model '{model_name}' requested by user {current_user.user_id}"
        )
        
        # Check if model is loaded
        if not await model_service.is_model_loaded(model_name, version):
            return BaseResponse(
                request_id=str(uuid.uuid4()),
                status="success",
                message=f"Model '{model_name}' is not loaded"
            )
        
        # Unload model
        await model_service.unload_model(model_name, version)
        
        return BaseResponse(
            request_id=str(uuid.uuid4()),
            status="success",
            message=f"Model '{model_name}' unloaded successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to unload model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to unload model: {str(e)}"
        )


@router.put(
    "/{model_name}/update",
    response_model=BaseResponse,
    summary="Update model configuration",
    description="Update model configuration or metadata",
    dependencies=[Depends(require_role([Role.ADMIN]))]
)
async def update_model(
    model_name: str,
    update_request: ModelManagementRequest,
    current_user: AuthToken = Depends(get_current_user),
    model_service: ModelManagementService = Depends(get_model_service)
) -> BaseResponse:
    """
    Update model configuration or metadata.
    
    Args:
        model_name: Name of the model to update
        update_request: Update request with new configuration
        current_user: Current authenticated user
        model_service: Model management service
        
    Returns:
        BaseResponse indicating success or failure
    """
    try:
        logger.info(
            f"Updating model '{model_name}' requested by user {current_user.user_id}"
        )
        
        # Validate model exists
        if not await model_service.model_exists(model_name):
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found"
            )
        
        # Update model
        await model_service.update_model(
            model_name=model_name,
            config=update_request.metadata
        )
        
        return BaseResponse(
            request_id=str(uuid.uuid4()),
            status="success",
            message=f"Model '{model_name}' updated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update model: {str(e)}"
        )


@router.delete(
    "/{model_name}",
    response_model=BaseResponse,
    summary="Delete a model",
    description="Delete a model from the system",
    dependencies=[Depends(require_role([Role.ADMIN]))]
)
async def delete_model(
    model_name: str,
    version: Optional[str] = Query(None, description="Model version"),
    force: bool = Query(False, description="Force deletion even if loaded"),
    current_user: AuthToken = Depends(get_current_user),
    model_service: ModelManagementService = Depends(get_model_service)
) -> BaseResponse:
    """
    Delete a model from the system.
    
    Args:
        model_name: Name of the model to delete
        version: Optional model version
        force: Force deletion even if model is loaded
        current_user: Current authenticated user
        model_service: Model management service
        
    Returns:
        BaseResponse indicating success or failure
    """
    try:
        logger.warning(
            f"Deleting model '{model_name}' requested by user {current_user.user_id}"
        )
        
        # Check if model exists
        if not await model_service.model_exists(model_name, version):
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found"
            )
        
        # Check if model is loaded
        if await model_service.is_model_loaded(model_name, version) and not force:
            raise HTTPException(
                status_code=409,
                detail=f"Model '{model_name}' is currently loaded. Use force=true to delete"
            )
        
        # Delete model
        await model_service.delete_model(model_name, version, force)
        
        return BaseResponse(
            request_id=str(uuid.uuid4()),
            status="success",
            message=f"Model '{model_name}' deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete model: {str(e)}"
        )


@router.post(
    "/{model_name}/set-default",
    response_model=BaseResponse,
    summary="Set default model",
    description="Set a model as the default for classification",
    dependencies=[Depends(require_role([Role.ADMIN]))]
)
async def set_default_model(
    model_name: str,
    current_user: AuthToken = Depends(get_current_user),
    model_service: ModelManagementService = Depends(get_model_service)
) -> BaseResponse:
    """
    Set a model as the default.
    
    Args:
        model_name: Name of the model to set as default
        current_user: Current authenticated user
        model_service: Model management service
        
    Returns:
        BaseResponse indicating success
    """
    try:
        logger.info(
            f"Setting default model to '{model_name}' by user {current_user.user_id}"
        )
        
        # Validate model exists
        if not await model_service.model_exists(model_name):
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found"
            )
        
        # Set as default
        await model_service.set_default_model(model_name)
        
        return BaseResponse(
            request_id=str(uuid.uuid4()),
            status="success",
            message=f"Model '{model_name}' set as default"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set default model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to set default model: {str(e)}"
        )
