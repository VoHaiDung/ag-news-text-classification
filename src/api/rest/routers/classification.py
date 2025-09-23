"""
Classification Router for REST API
================================================================================
Implements text classification endpoints for single and batch processing
with support for multiple models and configurations.

This module provides the main classification functionality of the API
following RESTful principles and async processing patterns.

References:
    - FastAPI Documentation on Path Operations
    - Pydantic Documentation on Data Validation
    - AsyncIO Documentation for Asynchronous Programming

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query, Body, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator

from src.api.base.auth import AuthToken, Role
from src.api.base.base_handler import APIContext, APIResponse, ResponseStatus
from src.api.rest.dependencies import (
    get_current_user,
    get_prediction_service,
    get_rate_limiter,
    verify_api_key
)
from src.api.rest.schemas.request_schemas import (
    ClassificationRequest,
    BatchClassificationRequest,
    StreamingClassificationRequest
)
from src.api.rest.schemas.response_schemas import (
    ClassificationResponse,
    BatchClassificationResponse,
    PredictionResult
)
from src.services.core.prediction_service import PredictionService
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/classify",
    tags=["Classification"],
    responses={
        400: {"description": "Bad request"},
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"}
    }
)


@router.post(
    "/",
    response_model=ClassificationResponse,
    summary="Classify single text",
    description="Classify a single text into one of the AG News categories"
)
async def classify_text(
    request: ClassificationRequest,
    background_tasks: BackgroundTasks,
    current_user: AuthToken = Depends(get_current_user),
    prediction_service: PredictionService = Depends(get_prediction_service),
    _: None = Depends(get_rate_limiter)
) -> ClassificationResponse:
    """
    Classify a single text document.
    
    Args:
        request: Classification request with text and options
        background_tasks: FastAPI background tasks
        current_user: Authenticated user token
        prediction_service: Prediction service instance
        
    Returns:
        ClassificationResponse with prediction results
        
    Raises:
        HTTPException: For various error conditions
    """
    # Create API context
    context = APIContext(
        request_id=str(uuid.uuid4()),
        user_id=current_user.user_id,
        source="rest_api"
    )
    
    try:
        # Log request
        logger.info(
            f"Classification request from user {current_user.user_id}",
            extra={"request_id": context.request_id}
        )
        
        # Perform prediction
        result = await prediction_service.predict(
            text=request.text,
            model_name=request.model_name,
            return_probabilities=request.return_probabilities,
            return_explanations=request.return_explanations,
            context=context
        )
        
        # Format response
        response = ClassificationResponse(
            request_id=context.request_id,
            text=request.text[:100] + "..." if len(request.text) > 100 else request.text,
            category=result.category,
            confidence=result.confidence,
            probabilities=result.probabilities if request.return_probabilities else None,
            explanations=result.explanations if request.return_explanations else None,
            model_used=result.model_name,
            processing_time=result.processing_time,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Schedule background logging
        background_tasks.add_task(
            log_prediction,
            context.request_id,
            current_user.user_id,
            result
        )
        
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}", extra={"request_id": context.request_id})
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            f"Classification error: {str(e)}",
            extra={"request_id": context.request_id},
            exc_info=True
        )
        raise HTTPException(status_code=500, detail="Classification failed")


@router.post(
    "/batch",
    response_model=BatchClassificationResponse,
    summary="Classify multiple texts",
    description="Classify multiple texts in a single request"
)
async def classify_batch(
    request: BatchClassificationRequest,
    background_tasks: BackgroundTasks,
    current_user: AuthToken = Depends(get_current_user),
    prediction_service: PredictionService = Depends(get_prediction_service),
    _: None = Depends(get_rate_limiter)
) -> BatchClassificationResponse:
    """
    Classify multiple texts in batch.
    
    Args:
        request: Batch classification request
        background_tasks: FastAPI background tasks
        current_user: Authenticated user token
        prediction_service: Prediction service instance
        
    Returns:
        BatchClassificationResponse with all results
        
    Raises:
        HTTPException: For various error conditions
    """
    # Create API context
    context = APIContext(
        request_id=str(uuid.uuid4()),
        user_id=current_user.user_id,
        source="rest_api"
    )
    
    # Validate batch size
    if len(request.texts) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size exceeds maximum of 100 texts"
        )
    
    try:
        logger.info(
            f"Batch classification request for {len(request.texts)} texts",
            extra={"request_id": context.request_id}
        )
        
        # Process texts in parallel
        tasks = [
            prediction_service.predict(
                text=text,
                model_name=request.model_name,
                return_probabilities=request.return_probabilities,
                context=context
            )
            for text in request.texts
        ]
        
        # Await all predictions
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        predictions = []
        errors = []
        
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append({
                    "index": idx,
                    "error": str(result)
                })
            else:
                predictions.append(
                    PredictionResult(
                        index=idx,
                        text=request.texts[idx][:100] + "..." 
                             if len(request.texts[idx]) > 100 
                             else request.texts[idx],
                        category=result.category,
                        confidence=result.confidence,
                        probabilities=result.probabilities 
                                    if request.return_probabilities 
                                    else None
                    )
                )
        
        # Determine status
        if errors and not predictions:
            status = "error"
        elif errors and predictions:
            status = "partial"
        else:
            status = "success"
        
        response = BatchClassificationResponse(
            request_id=context.request_id,
            total_texts=len(request.texts),
            processed=len(predictions),
            failed=len(errors),
            status=status,
            predictions=predictions,
            errors=errors if errors else None,
            model_used=request.model_name or "default",
            timestamp=datetime.now(timezone.utc)
        )
        
        # Schedule background logging
        background_tasks.add_task(
            log_batch_prediction,
            context.request_id,
            current_user.user_id,
            len(predictions),
            len(errors)
        )
        
        return response
        
    except Exception as e:
        logger.error(
            f"Batch classification error: {str(e)}",
            extra={"request_id": context.request_id},
            exc_info=True
        )
        raise HTTPException(status_code=500, detail="Batch classification failed")


@router.post(
    "/stream",
    summary="Stream classification results",
    description="Stream classification results for multiple texts"
)
async def classify_stream(
    request: StreamingClassificationRequest,
    current_user: AuthToken = Depends(get_current_user),
    prediction_service: PredictionService = Depends(get_prediction_service),
    _: None = Depends(get_rate_limiter)
):
    """
    Stream classification results using Server-Sent Events.
    
    Args:
        request: Streaming classification request
        current_user: Authenticated user token
        prediction_service: Prediction service instance
        
    Returns:
        StreamingResponse with SSE events
    """
    # Create API context
    context = APIContext(
        request_id=str(uuid.uuid4()),
        user_id=current_user.user_id,
        source="rest_api"
    )
    
    async def generate():
        """Generate SSE events for streaming."""
        try:
            for idx, text in enumerate(request.texts):
                # Perform prediction
                result = await prediction_service.predict(
                    text=text,
                    model_name=request.model_name,
                    context=context
                )
                
                # Format SSE event
                event_data = {
                    "index": idx,
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "category": result.category,
                    "confidence": result.confidence,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                yield f"data: {json.dumps(event_data)}\n\n"
                
                # Small delay to prevent overwhelming client
                await asyncio.sleep(0.1)
            
            # Send completion event
            yield f"data: {json.dumps({'event': 'complete'})}\n\n"
            
        except Exception as e:
            # Send error event
            error_data = {
                "event": "error",
                "error": str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


# Background task functions
async def log_prediction(request_id: str, user_id: str, result: Any):
    """Log prediction for analytics."""
    logger.info(
        f"Prediction logged",
        extra={
            "request_id": request_id,
            "user_id": user_id,
            "category": result.category,
            "confidence": result.confidence
        }
    )


async def log_batch_prediction(request_id: str, user_id: str, 
                              processed: int, failed: int):
    """Log batch prediction for analytics."""
    logger.info(
        f"Batch prediction logged",
        extra={
            "request_id": request_id,
            "user_id": user_id,
            "processed": processed,
            "failed": failed
        }
    )
