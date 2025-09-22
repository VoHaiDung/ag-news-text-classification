"""
REST API Endpoints
==================

Implements RESTful API endpoints following REST architectural constraints from:
- Fielding (2000): "Architectural Styles and the Design of Network-based Software Architectures"
- Richardson (2007): "RESTful Web Services"
- Masse (2011): "REST API Design Rulebook"

Author: Team SOTA AGNews
License: MIT
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio
from pathlib import Path
import hashlib
import json

from fastapi import (
    FastAPI, 
    HTTPException, 
    Depends, 
    File, 
    UploadFile,
    Query,
    Body,
    Header,
    status,
    BackgroundTasks
)
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
import uvicorn
import torch
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging
from src.services.data_service import get_data_service
from src.core.exceptions import DataError, ModelError
from src.api import API_VERSION, API_PREFIX, API_CONFIG
from src.api.rest.validators import (
    validate_text_input,
    validate_batch_input,
    validate_model_name
)
from src.api.rest.middleware import (
    RateLimitMiddleware,
    AuthenticationMiddleware,
    LoggingMiddleware,
    MetricsMiddleware
)
from configs.constants import AG_NEWS_CLASSES

logger = setup_logging(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="AG News Classification API",
    description="State-of-the-art text classification API implementing transformer-based models",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{API_PREFIX}/openapi.json"
)

# Add middleware
if API_CONFIG["security"]["enable_auth"]:
    app.add_middleware(AuthenticationMiddleware)
if API_CONFIG["security"]["rate_limit"]:
    app.add_middleware(RateLimitMiddleware, max_requests=API_CONFIG["security"]["rate_limit"])
if API_CONFIG["monitoring"]["enable_logging"]:
    app.add_middleware(LoggingMiddleware)
if API_CONFIG["monitoring"]["enable_metrics"]:
    app.add_middleware(MetricsMiddleware)

# Request/Response Models following OpenAPI Specification 3.0
class TextInput(BaseModel):
    """Single text input model."""
    
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Input text for classification"
    )
    model_name: Optional[str] = Field(
        default="deberta-v3",
        description="Model to use for prediction"
    )
    return_probabilities: bool = Field(
        default=False,
        description="Whether to return class probabilities"
    )
    
    @validator("text")
    def validate_text(cls, v):
        """Validate text input."""
        return validate_text_input(v)

class BatchTextInput(BaseModel):
    """Batch text input model."""
    
    texts: List[str] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description="List of texts for batch classification"
    )
    model_name: Optional[str] = Field(
        default="deberta-v3",
        description="Model to use for predictions"
    )
    return_probabilities: bool = Field(
        default=False,
        description="Whether to return class probabilities"
    )
    parallel: bool = Field(
        default=True,
        description="Whether to process in parallel"
    )
    
    @validator("texts")
    def validate_texts(cls, v):
        """Validate batch text input."""
        return validate_batch_input(v)

class PredictionResponse(BaseModel):
    """Prediction response model."""
    
    prediction_id: str = Field(..., description="Unique prediction ID")
    text: str = Field(..., description="Input text")
    predicted_class: str = Field(..., description="Predicted class name")
    predicted_label: int = Field(..., description="Predicted class label")
    confidence: float = Field(..., description="Prediction confidence score")
    probabilities: Optional[Dict[str, float]] = Field(
        None,
        description="Class probabilities"
    )
    model_name: str = Field(..., description="Model used for prediction")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Prediction timestamp")

class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""
    
    batch_id: str = Field(..., description="Unique batch ID")
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_count: int = Field(..., description="Total number of predictions")
    successful_count: int = Field(..., description="Number of successful predictions")
    failed_count: int = Field(..., description="Number of failed predictions")
    total_processing_time: float = Field(..., description="Total processing time")
    timestamp: str = Field(..., description="Batch processing timestamp")

class ModelInfo(BaseModel):
    """Model information response."""
    
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type (transformer/ensemble)")
    version: str = Field(..., description="Model version")
    parameters: int = Field(..., description="Number of parameters")
    accuracy: float = Field(..., description="Model accuracy on test set")
    f1_score: float = Field(..., description="Model F1 score")
    classes: List[str] = Field(..., description="Supported classes")
    max_length: int = Field(..., description="Maximum input length")
    status: str = Field(..., description="Model status (available/loading/error)")

class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")
    models_loaded: int = Field(..., description="Number of loaded models")
    uptime: float = Field(..., description="Service uptime in seconds")

# Global state
class APIState:
    """API state management."""
    
    def __init__(self):
        self.models = {}
        self.prediction_service = None
        self.data_service = get_data_service()
        self.start_time = datetime.now()
        self.prediction_cache = {}
        self.stats = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }

api_state = APIState()

# Dependency injection following Fowler's Inversion of Control pattern
async def get_prediction_service():
    """Get prediction service instance."""
    if api_state.prediction_service is None:
        # Initialize prediction service (placeholder)
        logger.warning("Prediction service not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service not available"
        )
    return api_state.prediction_service

# Health and monitoring endpoints
@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Service health status
    """
    uptime = (datetime.now() - api_state.start_time).total_seconds()
    
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        timestamp=datetime.now().isoformat(),
        models_loaded=len(api_state.models),
        uptime=uptime
    )

@app.get("/metrics", tags=["monitoring"])
async def get_metrics():
    """
    Get API metrics.
    
    Returns:
        API metrics and statistics
    """
    return {
        "statistics": api_state.stats,
        "data_service": api_state.data_service.get_statistics(),
        "cache_size": len(api_state.prediction_cache),
        "timestamp": datetime.now().isoformat()
    }

# Model management endpoints
@app.get(f"{API_PREFIX}/models", response_model=List[ModelInfo], tags=["models"])
async def list_models():
    """
    List available models.
    
    Returns:
        List of available model information
    """
    models = []
    
    # Placeholder model information
    available_models = [
        {
            "name": "deberta-v3",
            "type": "transformer",
            "version": "1.0.0",
            "parameters": 434000000,
            "accuracy": 0.956,
            "f1_score": 0.954
        },
        {
            "name": "roberta-large",
            "type": "transformer",
            "version": "1.0.0",
            "parameters": 355000000,
            "accuracy": 0.948,
            "f1_score": 0.946
        },
        {
            "name": "ensemble",
            "type": "ensemble",
            "version": "1.0.0",
            "parameters": 1200000000,
            "accuracy": 0.962,
            "f1_score": 0.961
        }
    ]
    
    for model_info in available_models:
        models.append(ModelInfo(
            model_name=model_info["name"],
            model_type=model_info["type"],
            version=model_info["version"],
            parameters=model_info["parameters"],
            accuracy=model_info["accuracy"],
            f1_score=model_info["f1_score"],
            classes=AG_NEWS_CLASSES,
            max_length=512,
            status="available"
        ))
    
    return models

@app.get(f"{API_PREFIX}/models/{{model_name}}", response_model=ModelInfo, tags=["models"])
async def get_model_info(model_name: str):
    """
    Get specific model information.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model information
    """
    validate_model_name(model_name)
    
    # Placeholder implementation
    return ModelInfo(
        model_name=model_name,
        model_type="transformer",
        version="1.0.0",
        parameters=434000000,
        accuracy=0.956,
        f1_score=0.954,
        classes=AG_NEWS_CLASSES,
        max_length=512,
        status="available"
    )

# Prediction endpoints
@app.post(f"{API_PREFIX}/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict_single(
    input_data: TextInput,
    api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Single text prediction endpoint.
    
    Args:
        input_data: Text input for prediction
        api_key: Optional API key for authentication
        
    Returns:
        Prediction response
    """
    start_time = datetime.now()
    
    # Generate prediction ID
    prediction_id = hashlib.md5(
        f"{input_data.text}{start_time.isoformat()}".encode()
    ).hexdigest()[:12]
    
    # Check cache
    cache_key = hashlib.md5(
        f"{input_data.text}{input_data.model_name}".encode()
    ).hexdigest()
    
    if cache_key in api_state.prediction_cache:
        api_state.stats["cache_hits"] += 1
        cached_result = api_state.prediction_cache[cache_key]
        cached_result["prediction_id"] = prediction_id
        cached_result["timestamp"] = datetime.now().isoformat()
        return PredictionResponse(**cached_result)
    
    api_state.stats["cache_misses"] += 1
    
    try:
        # Prepare data
        prepared_data = api_state.data_service.prepare_data(
            texts=[input_data.text],
            clean=True,
            augment=False
        )
        
        # Mock prediction (replace with actual prediction service)
        predicted_label = np.random.randint(0, len(AG_NEWS_CLASSES))
        confidence = np.random.uniform(0.85, 0.99)
        
        if input_data.return_probabilities:
            probabilities = np.random.dirichlet(np.ones(len(AG_NEWS_CLASSES)))
            probabilities_dict = {
                AG_NEWS_CLASSES[i]: float(probabilities[i])
                for i in range(len(AG_NEWS_CLASSES))
            }
        else:
            probabilities_dict = None
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            "prediction_id": prediction_id,
            "text": input_data.text,
            "predicted_class": AG_NEWS_CLASSES[predicted_label],
            "predicted_label": predicted_label,
            "confidence": float(confidence),
            "probabilities": probabilities_dict,
            "model_name": input_data.model_name,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache result
        api_state.prediction_cache[cache_key] = result
        
        # Update statistics
        api_state.stats["total_predictions"] += 1
        api_state.stats["successful_predictions"] += 1
        
        return PredictionResponse(**result)
        
    except Exception as e:
        api_state.stats["total_predictions"] += 1
        api_state.stats["failed_predictions"] += 1
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post(f"{API_PREFIX}/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
async def predict_batch(
    input_data: BatchTextInput,
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Batch text prediction endpoint.
    
    Args:
        input_data: Batch text input for prediction
        background_tasks: FastAPI background tasks
        api_key: Optional API key
        
    Returns:
        Batch prediction response
    """
    start_time = datetime.now()
    
    # Generate batch ID
    batch_id = hashlib.md5(
        f"{len(input_data.texts)}{start_time.isoformat()}".encode()
    ).hexdigest()[:12]
    
    predictions = []
    successful_count = 0
    failed_count = 0
    
    # Process each text
    for text in input_data.texts:
        try:
            # Reuse single prediction logic
            single_input = TextInput(
                text=text,
                model_name=input_data.model_name,
                return_probabilities=input_data.return_probabilities
            )
            
            # Mock prediction
            predicted_label = np.random.randint(0, len(AG_NEWS_CLASSES))
            confidence = np.random.uniform(0.85, 0.99)
            
            prediction = PredictionResponse(
                prediction_id=hashlib.md5(f"{text}{datetime.now()}".encode()).hexdigest()[:12],
                text=text,
                predicted_class=AG_NEWS_CLASSES[predicted_label],
                predicted_label=predicted_label,
                confidence=float(confidence),
                probabilities=None,
                model_name=input_data.model_name,
                processing_time=0.1,
                timestamp=datetime.now().isoformat()
            )
            
            predictions.append(prediction)
            successful_count += 1
            
        except Exception as e:
            logger.error(f"Failed to process text: {str(e)}")
            failed_count += 1
    
    total_processing_time = (datetime.now() - start_time).total_seconds()
    
    # Update statistics
    api_state.stats["total_predictions"] += len(input_data.texts)
    api_state.stats["successful_predictions"] += successful_count
    api_state.stats["failed_predictions"] += failed_count
    
    return BatchPredictionResponse(
        batch_id=batch_id,
        predictions=predictions,
        total_count=len(input_data.texts),
        successful_count=successful_count,
        failed_count=failed_count,
        total_processing_time=total_processing_time,
        timestamp=datetime.now().isoformat()
    )

@app.post(f"{API_PREFIX}/predict/file", tags=["prediction"])
async def predict_file(
    file: UploadFile = File(...),
    model_name: str = Query(default="deberta-v3"),
    api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    File-based batch prediction.
    
    Args:
        file: Text file with one text per line
        model_name: Model to use
        api_key: Optional API key
        
    Returns:
        Batch prediction response
    """
    if not file.filename.endswith(('.txt', '.csv', '.json')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .txt, .csv, and .json files are supported"
        )
    
    try:
        contents = await file.read()
        texts = contents.decode('utf-8').strip().split('\n')
        
        # Create batch input
        batch_input = BatchTextInput(
            texts=texts[:100],  # Limit to 100 for demo
            model_name=model_name,
            return_probabilities=False
        )
        
        # Process batch
        return await predict_batch(batch_input, BackgroundTasks(), api_key)
        
    except Exception as e:
        logger.error(f"File processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File processing failed: {str(e)}"
        )

# Async streaming endpoint for real-time predictions
@app.post(f"{API_PREFIX}/predict/stream", tags=["prediction"])
async def predict_stream(
    input_data: TextInput,
    api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Streaming prediction endpoint for real-time processing.
    
    Args:
        input_data: Text input
        api_key: Optional API key
        
    Returns:
        Server-sent event stream
    """
    async def generate():
        # Initial response
        yield f"data: {json.dumps({'status': 'processing', 'progress': 0})}\n\n"
        await asyncio.sleep(0.1)
        
        # Processing stages
        stages = [
            {"stage": "preprocessing", "progress": 25},
            {"stage": "tokenization", "progress": 50},
            {"stage": "inference", "progress": 75},
            {"stage": "postprocessing", "progress": 100}
        ]
        
        for stage in stages:
            yield f"data: {json.dumps({'status': 'processing', **stage})}\n\n"
            await asyncio.sleep(0.5)
        
        # Final prediction
        predicted_label = np.random.randint(0, len(AG_NEWS_CLASSES))
        confidence = np.random.uniform(0.85, 0.99)
        
        result = {
            "status": "completed",
            "predicted_class": AG_NEWS_CLASSES[predicted_label],
            "confidence": float(confidence)
        }
        
        yield f"data: {json.dumps(result)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

# Error handlers
@app.exception_handler(DataError)
async def data_error_handler(request, exc: DataError):
    """Handle data errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)}
    )

@app.exception_handler(ModelError)
async def model_error_handler(request, exc: ModelError):
    """Handle model errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)}
    )

# Application lifecycle
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting AG News API service")
    
    # Initialize services
    api_state.data_service = get_data_service()
    
    # Load models (placeholder)
    logger.info("API service started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API service")
    
    # Clear caches
    api_state.prediction_cache.clear()
    if api_state.data_service:
        api_state.data_service.clear_cache()
    
    logger.info("API service shut down successfully")

def run_api_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 4,
    reload: bool = False
):
    """
    Run API server.
    
    Args:
        host: Server host
        port: Server port
        workers: Number of workers
        reload: Enable auto-reload
    """
    uvicorn.run(
        "src.api.rest.endpoints:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    run_api_server()
