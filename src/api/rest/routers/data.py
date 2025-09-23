"""
Data Management Router for REST API
================================================================================
Implements endpoints for data upload, management, preprocessing, and
augmentation following data engineering best practices.

This module provides comprehensive data management capabilities including
dataset versioning, quality validation, and data transformation pipelines.

References:
    - Polyzotis, N., et al. (2017). Data Management Challenges in Production Machine Learning
    - Schelter, S., et al. (2018). Automating Large-Scale Data Quality Verification
    - DataVersion Control (DVC) Documentation

Author: Võ Hải Dũng
License: MIT
"""

import io
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from src.api.base.auth import AuthToken, Role
from src.api.rest.dependencies import (
    get_current_user,
    get_data_service,
    require_role,
    PaginationParams,
    FilterParams
)
from src.api.rest.schemas.request_schemas import DataUploadRequest
from src.api.rest.schemas.response_schemas import BaseResponse
from src.services.core.data_service import DataService
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/data",
    tags=["Data"],
    responses={
        403: {"description": "Forbidden - Data management requires authentication"},
        404: {"description": "Dataset not found"},
        413: {"description": "File too large"},
        415: {"description": "Unsupported file format"},
        500: {"description": "Internal server error"}
    }
)


@router.post(
    "/upload",
    response_model=BaseResponse,
    summary="Upload dataset",
    description="Upload a new dataset for training or evaluation",
    dependencies=[Depends(require_role([Role.ADMIN, Role.USER]))]
)
async def upload_dataset(
    file: UploadFile = File(..., description="Dataset file"),
    dataset_name: str = Form(..., description="Dataset name"),
    description: Optional[str] = Form(None, description="Dataset description"),
    format: str = Form("csv", description="File format (csv, json, jsonl)"),
    validation: bool = Form(True, description="Validate data"),
    current_user: AuthToken = Depends(get_current_user),
    data_service: DataService = Depends(get_data_service)
) -> BaseResponse:
    """
    Upload a new dataset.
    
    Args:
        file: Uploaded file
        dataset_name: Name for the dataset
        description: Dataset description
        format: File format
        validation: Whether to validate data
        current_user: Authenticated user
        data_service: Data service instance
        
    Returns:
        BaseResponse with upload status
        
    Raises:
        HTTPException: For various error conditions
    """
    try:
        # Validate file size (max 500MB)
        max_size = 500 * 1024 * 1024
        if file.size and file.size > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum of {max_size / (1024*1024):.0f}MB"
            )
        
        # Validate file format
        allowed_formats = ["csv", "json", "jsonl", "tsv", "txt"]
        if format not in allowed_formats:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported format. Allowed: {allowed_formats}"
            )
        
        # Check if dataset name already exists
        if await data_service.dataset_exists(dataset_name):
            raise HTTPException(
                status_code=409,
                detail=f"Dataset '{dataset_name}' already exists"
            )
        
        logger.info(
            f"Uploading dataset '{dataset_name}' by user {current_user.user_id}, "
            f"format: {format}, size: {file.size}"
        )
        
        # Read file content
        content = await file.read()
        
        # Process and store dataset
        dataset_id = await data_service.upload_dataset(
            name=dataset_name,
            content=content,
            format=format,
            description=description,
            user_id=current_user.user_id,
            validate=validation
        )
        
        return BaseResponse(
            request_id=str(uuid.uuid4()),
            status="success",
            message=f"Dataset '{dataset_name}' uploaded successfully",
            metadata={"dataset_id": dataset_id}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload dataset: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload dataset: {str(e)}"
        )


@router.get(
    "/datasets",
    summary="List datasets",
    description="Get list of available datasets with metadata"
)
async def list_datasets(
    pagination: PaginationParams = Depends(),
    filters: FilterParams = Depends(),
    include_system: bool = Query(False, description="Include system datasets"),
    current_user: Optional[AuthToken] = Depends(get_current_user),
    data_service: DataService = Depends(get_data_service)
) -> Dict[str, Any]:
    """
    List available datasets.
    
    Args:
        pagination: Pagination parameters
        filters: Filter parameters
        include_system: Whether to include system datasets
        current_user: Authenticated user
        data_service: Data service instance
        
    Returns:
        List of datasets with metadata
    """
    try:
        # Get datasets based on user permissions
        if current_user and current_user.has_role(Role.ADMIN):
            datasets = await data_service.list_datasets(
                include_system=include_system,
                filters=filters.to_dict()
            )
        else:
            # Regular users only see their own and public datasets
            datasets = await data_service.list_datasets(
                user_id=current_user.user_id if current_user else None,
                include_system=False,
                filters=filters.to_dict()
            )
        
        # Apply pagination
        total = len(datasets)
        start_idx = pagination.offset
        end_idx = start_idx + pagination.limit
        paginated_datasets = datasets[start_idx:end_idx]
        
        return {
            "datasets": paginated_datasets,
            "total": total,
            "page": pagination.page,
            "page_size": pagination.page_size,
            "total_pages": (total + pagination.page_size - 1) // pagination.page_size
        }
        
    except Exception as e:
        logger.error(f"Failed to list datasets: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve datasets"
        )


@router.get(
    "/datasets/{dataset_name}",
    summary="Get dataset details",
    description="Get detailed information about a specific dataset"
)
async def get_dataset_info(
    dataset_name: str,
    current_user: Optional[AuthToken] = Depends(get_current_user),
    data_service: DataService = Depends(get_data_service)
) -> Dict[str, Any]:
    """
    Get dataset information.
    
    Args:
        dataset_name: Dataset name
        current_user: Authenticated user
        data_service: Data service instance
        
    Returns:
        Dataset information
        
    Raises:
        HTTPException: If dataset not found
    """
    try:
        # Check if dataset exists
        if not await data_service.dataset_exists(dataset_name):
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{dataset_name}' not found"
            )
        
        # Get dataset info
        info = await data_service.get_dataset_info(dataset_name)
        
        # Check access permissions
        if info.get("private") and current_user:
            if info.get("user_id") != current_user.user_id and not current_user.has_role(Role.ADMIN):
                raise HTTPException(
                    status_code=403,
                    detail="Access denied to private dataset"
                )
        
        return info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dataset info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve dataset information"
        )


@router.get(
    "/datasets/{dataset_name}/download",
    summary="Download dataset",
    description="Download a dataset in specified format"
)
async def download_dataset(
    dataset_name: str,
    format: str = Query("csv", description="Download format"),
    sample_size: Optional[int] = Query(None, description="Number of samples"),
    current_user: Optional[AuthToken] = Depends(get_current_user),
    data_service: DataService = Depends(get_data_service)
):
    """
    Download a dataset.
    
    Args:
        dataset_name: Dataset name
        format: Download format
        sample_size: Optional sample size
        current_user: Authenticated user
        data_service: Data service instance
        
    Returns:
        Streaming response with dataset
    """
    try:
        # Check if dataset exists
        if not await data_service.dataset_exists(dataset_name):
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{dataset_name}' not found"
            )
        
        # Get dataset content
        content = await data_service.download_dataset(
            dataset_name,
            format=format,
            sample_size=sample_size
        )
        
        # Determine content type
        content_types = {
            "csv": "text/csv",
            "json": "application/json",
            "jsonl": "application/x-jsonlines",
            "tsv": "text/tab-separated-values"
        }
        content_type = content_types.get(format, "application/octet-stream")
        
        # Create streaming response
        return StreamingResponse(
            io.BytesIO(content),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename={dataset_name}.{format}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download dataset: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to download dataset"
        )


@router.post(
    "/datasets/{dataset_name}/preprocess",
    response_model=BaseResponse,
    summary="Preprocess dataset",
    description="Apply preprocessing to a dataset",
    dependencies=[Depends(require_role([Role.ADMIN, Role.USER]))]
)
async def preprocess_dataset(
    dataset_name: str,
    preprocessing_config: Dict[str, Any],
    save_as: Optional[str] = Query(None, description="Save preprocessed data as new dataset"),
    current_user: AuthToken = Depends(get_current_user),
    data_service: DataService = Depends(get_data_service)
) -> BaseResponse:
    """
    Preprocess a dataset.
    
    Args:
        dataset_name: Dataset name
        preprocessing_config: Preprocessing configuration
        save_as: Optional name for preprocessed dataset
        current_user: Authenticated user
        data_service: Data service instance
        
    Returns:
        BaseResponse with preprocessing status
    """
    try:
        # Check if dataset exists
        if not await data_service.dataset_exists(dataset_name):
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{dataset_name}' not found"
            )
        
        # Apply preprocessing
        result = await data_service.preprocess_dataset(
            dataset_name,
            preprocessing_config,
            save_as=save_as,
            user_id=current_user.user_id
        )
        
        return BaseResponse(
            request_id=str(uuid.uuid4()),
            status="success",
            message=f"Dataset preprocessed successfully",
            metadata=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to preprocess dataset: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to preprocess dataset: {str(e)}"
        )


@router.post(
    "/datasets/{dataset_name}/augment",
    response_model=BaseResponse,
    summary="Augment dataset",
    description="Apply data augmentation techniques",
    dependencies=[Depends(require_role([Role.ADMIN, Role.USER]))]
)
async def augment_dataset(
    dataset_name: str,
    augmentation_config: Dict[str, Any],
    augmentation_factor: float = Query(2.0, description="Augmentation factor"),
    save_as: Optional[str] = Query(None, description="Save augmented data as new dataset"),
    current_user: AuthToken = Depends(get_current_user),
    data_service: DataService = Depends(get_data_service)
) -> BaseResponse:
    """
    Augment a dataset.
    
    Args:
        dataset_name: Dataset name
        augmentation_config: Augmentation configuration
        augmentation_factor: How much to augment
        save_as: Optional name for augmented dataset
        current_user: Authenticated user
        data_service: Data service instance
        
    Returns:
        BaseResponse with augmentation status
    """
    try:
        # Check if dataset exists
        if not await data_service.dataset_exists(dataset_name):
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{dataset_name}' not found"
            )
        
        # Apply augmentation
        result = await data_service.augment_dataset(
            dataset_name,
            augmentation_config,
            factor=augmentation_factor,
            save_as=save_as,
            user_id=current_user.user_id
        )
        
        return BaseResponse(
            request_id=str(uuid.uuid4()),
            status="success",
            message=f"Dataset augmented successfully",
            metadata=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to augment dataset: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to augment dataset: {str(e)}"
        )


@router.delete(
    "/datasets/{dataset_name}",
    response_model=BaseResponse,
    summary="Delete dataset",
    description="Delete a dataset and its associated files",
    dependencies=[Depends(require_role([Role.ADMIN, Role.USER]))]
)
async def delete_dataset(
    dataset_name: str,
    force: bool = Query(False, description="Force deletion even if used in training"),
    current_user: AuthToken = Depends(get_current_user),
    data_service: DataService = Depends(get_data_service)
) -> BaseResponse:
    """
    Delete a dataset.
    
    Args:
        dataset_name: Dataset name
        force: Force deletion
        current_user: Authenticated user
        data_service: Data service instance
        
    Returns:
        BaseResponse with deletion status
    """
    try:
        # Check if dataset exists
        if not await data_service.dataset_exists(dataset_name):
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{dataset_name}' not found"
            )
        
        # Check ownership
        info = await data_service.get_dataset_info(dataset_name)
        if info.get("user_id") != current_user.user_id and not current_user.has_role(Role.ADMIN):
            raise HTTPException(
                status_code=403,
                detail="Access denied to delete this dataset"
            )
        
        # Delete dataset
        await data_service.delete_dataset(dataset_name, force)
        
        logger.info(f"Dataset '{dataset_name}' deleted by user {current_user.user_id}")
        
        return BaseResponse(
            request_id=str(uuid.uuid4()),
            status="success",
            message=f"Dataset '{dataset_name}' deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete dataset: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete dataset: {str(e)}"
        )


@router.get(
    "/datasets/{dataset_name}/statistics",
    summary="Get dataset statistics",
    description="Get statistical analysis of a dataset"
)
async def get_dataset_statistics(
    dataset_name: str,
    current_user: Optional[AuthToken] = Depends(get_current_user),
    data_service: DataService = Depends(get_data_service)
) -> Dict[str, Any]:
    """
    Get dataset statistics.
    
    Args:
        dataset_name: Dataset name
        current_user: Authenticated user
        data_service: Data service instance
        
    Returns:
        Dataset statistics
    """
    try:
        # Check if dataset exists
        if not await data_service.dataset_exists(dataset_name):
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{dataset_name}' not found"
            )
        
        # Get statistics
        stats = await data_service.get_dataset_statistics(dataset_name)
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dataset statistics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve dataset statistics"
        )


@router.post(
    "/datasets/{dataset_name}/validate",
    response_model=BaseResponse,
    summary="Validate dataset",
    description="Validate dataset quality and format"
)
async def validate_dataset(
    dataset_name: str,
    validation_rules: Optional[Dict[str, Any]] = None,
    current_user: Optional[AuthToken] = Depends(get_current_user),
    data_service: DataService = Depends(get_data_service)
) -> BaseResponse:
    """
    Validate a dataset.
    
    Args:
        dataset_name: Dataset name
        validation_rules: Custom validation rules
        current_user: Authenticated user
        data_service: Data service instance
        
    Returns:
        Validation results
    """
    try:
        # Check if dataset exists
        if not await data_service.dataset_exists(dataset_name):
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{dataset_name}' not found"
            )
        
        # Validate dataset
        validation_result = await data_service.validate_dataset(
            dataset_name,
            validation_rules
        )
        
        return BaseResponse(
            request_id=str(uuid.uuid4()),
            status="success" if validation_result["is_valid"] else "error",
            message="Dataset validation complete",
            metadata=validation_result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate dataset: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to validate dataset"
        )
