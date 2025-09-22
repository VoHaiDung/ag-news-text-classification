"""
gRPC Services Module
====================

Implements gRPC service layer following patterns from:
- Google (2015): "gRPC: A high-performance, open-source universal RPC framework"
- Hohpe & Woolf (2003): "Enterprise Integration Patterns"
- Tanenbaum & Van Steen (2007): "Distributed Systems: Principles and Paradigms"

This module provides gRPC-based API for high-performance communication
between distributed components of the AG News classification system.

Author: Võ Hải Dũng
License: MIT
"""

import logging
import asyncio
import time
import hashlib
from typing import Dict, List, Optional, Any, AsyncIterator
from datetime import datetime
from pathlib import Path
from concurrent import futures
import json

import grpc
from grpc_reflection.v1alpha import reflection
import numpy as np
import torch
from google.protobuf import empty_pb2, timestamp_pb2
from google.protobuf.json_format import MessageToDict, ParseDict

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Import proto-generated files
from src.api.grpc.protos import (
    agnews_pb2,
    agnews_pb2_grpc,
    model_pb2,
    model_pb2_grpc,
    training_pb2,
    training_pb2_grpc
)

from src.utils.logging_config import setup_logging
from src.services.prediction_service import (
    get_prediction_service,
    PredictionConfig,
    PredictionResult
)
from src.services.training_service import (
    get_training_service,
    TrainingConfig,
    TrainingStatus
)
from src.services.model_management import (
    get_model_management_service,
    ModelStatus,
    DeploymentEnvironment
)
from src.services.data_service import get_data_service
from src.core.exceptions import (
    PredictionError,
    TrainingError,
    ModelError,
    DataError
)
from configs.constants import AG_NEWS_CLASSES

logger = setup_logging(__name__)

class AGNewsServicer(agnews_pb2_grpc.AGNewsServiceServicer):
    """
    Main gRPC service for AG News classification.
    
    Implements RPC patterns from:
    - Birrell & Nelson (1984): "Implementing Remote Procedure Calls"
    - Vinoski (2008): "RPC Under Fire"
    """
    
    def __init__(self):
        """Initialize AG News gRPC service."""
        self.prediction_service = get_prediction_service()
        self.training_service = get_training_service()
        self.model_service = get_model_management_service()
        self.data_service = get_data_service()
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0
        }
        
        logger.info("AG News gRPC service initialized")
    
    def Predict(
        self,
        request: agnews_pb2.PredictRequest,
        context: grpc.ServicerContext
    ) -> agnews_pb2.PredictResponse:
        """
        Single text prediction RPC.
        
        Args:
            request: Prediction request
            context: gRPC context
            
        Returns:
            Prediction response
        """
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        try:
            # Validate request
            if not request.text:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Text cannot be empty")
                self.stats["failed_requests"] += 1
                return agnews_pb2.PredictResponse()
            
            # Perform prediction
            result = self.prediction_service.predict(
                text=request.text,
                model_name=request.model_name or "deberta-v3",
                return_probabilities=request.return_probabilities
            )
            
            # Build response
            response = agnews_pb2.PredictResponse(
                prediction_id=hashlib.md5(
                    f"{request.text}{datetime.now()}".encode()
                ).hexdigest()[:12],
                predicted_class=result.predicted_class,
                predicted_label=result.predicted_label,
                confidence=result.confidence,
                model_name=result.model_name,
                processing_time=time.time() - start_time
            )
            
            # Add probabilities if requested
            if request.return_probabilities and result.probabilities:
                for class_name, prob in result.probabilities.items():
                    response.probabilities[class_name] = prob
            
            self.stats["successful_requests"] += 1
            self.stats["total_processing_time"] += response.processing_time
            
            return response
            
        except PredictionError as e:
            logger.error(f"Prediction failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            self.stats["failed_requests"] += 1
            return agnews_pb2.PredictResponse()
        
        except Exception as e:
            logger.error(f"Unexpected error in Predict: {str(e)}")
            context.set_code(grpc.StatusCode.UNKNOWN)
            context.set_details("Internal server error")
            self.stats["failed_requests"] += 1
            return agnews_pb2.PredictResponse()
    
    def PredictBatch(
        self,
        request: agnews_pb2.PredictBatchRequest,
        context: grpc.ServicerContext
    ) -> agnews_pb2.PredictBatchResponse:
        """
        Batch prediction RPC.
        
        Implements batching strategies from:
        - Dean & Barroso (2013): "The Tail at Scale"
        
        Args:
            request: Batch prediction request
            context: gRPC context
            
        Returns:
            Batch prediction response
        """
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        try:
            # Validate request
            if not request.texts:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Text list cannot be empty")
                self.stats["failed_requests"] += 1
                return agnews_pb2.PredictBatchResponse()
            
            # Perform batch prediction
            results = self.prediction_service.predict_batch(
                texts=list(request.texts),
                model_name=request.model_name or "deberta-v3",
                return_probabilities=request.return_probabilities,
                parallel=request.parallel
            )
            
            # Build response
            response = agnews_pb2.PredictBatchResponse(
                batch_id=hashlib.md5(
                    f"{len(request.texts)}{datetime.now()}".encode()
                ).hexdigest()[:12],
                total_processing_time=time.time() - start_time
            )
            
            # Add predictions
            for result in results:
                prediction = agnews_pb2.Prediction(
                    text=result.text,
                    predicted_class=result.predicted_class,
                    predicted_label=result.predicted_label,
                    confidence=result.confidence
                )
                
                if request.return_probabilities and result.probabilities:
                    for class_name, prob in result.probabilities.items():
                        prediction.probabilities[class_name] = prob
                
                response.predictions.append(prediction)
            
            response.successful_count = len([
                r for r in results if r.predicted_label >= 0
            ])
            response.failed_count = len(results) - response.successful_count
            
            self.stats["successful_requests"] += 1
            self.stats["total_processing_time"] += response.total_processing_time
            
            return response
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            self.stats["failed_requests"] += 1
            return agnews_pb2.PredictBatchResponse()
    
    def PredictStream(
        self,
        request_iterator: AsyncIterator[agnews_pb2.PredictRequest],
        context: grpc.ServicerContext
    ) -> AsyncIterator[agnews_pb2.PredictResponse]:
        """
        Streaming prediction RPC for real-time processing.
        
        Implements streaming patterns from:
        - Akidau et al. (2015): "The Dataflow Model"
        
        Args:
            request_iterator: Stream of prediction requests
            context: gRPC context
            
        Yields:
            Stream of prediction responses
        """
        async for request in request_iterator:
            try:
                # Process each request
                result = self.prediction_service.predict(
                    text=request.text,
                    model_name=request.model_name or "deberta-v3",
                    return_probabilities=request.return_probabilities
                )
                
                # Build response
                response = agnews_pb2.PredictResponse(
                    prediction_id=hashlib.md5(
                        f"{request.text}{datetime.now()}".encode()
                    ).hexdigest()[:12],
                    predicted_class=result.predicted_class,
                    predicted_label=result.predicted_label,
                    confidence=result.confidence,
                    model_name=result.model_name,
                    processing_time=result.processing_time
                )
                
                if request.return_probabilities and result.probabilities:
                    for class_name, prob in result.probabilities.items():
                        response.probabilities[class_name] = prob
                
                yield response
                
            except Exception as e:
                logger.error(f"Stream prediction error: {str(e)}")
                # Continue processing stream despite errors
                continue
    
    def GetModelInfo(
        self,
        request: agnews_pb2.ModelInfoRequest,
        context: grpc.ServicerContext
    ) -> agnews_pb2.ModelInfoResponse:
        """
        Get model information RPC.
        
        Args:
            request: Model info request
            context: gRPC context
            
        Returns:
            Model information response
        """
        try:
            # Get model metadata from catalog
            model_catalog = self.model_service.catalog
            
            # Find model by name
            model_metadata = None
            for metadata in model_catalog.values():
                if metadata.name == request.model_name:
                    model_metadata = metadata
                    break
            
            if not model_metadata:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Model {request.model_name} not found")
                return agnews_pb2.ModelInfoResponse()
            
            # Build response
            response = agnews_pb2.ModelInfoResponse(
                model_name=model_metadata.name,
                model_type=model_metadata.model_type,
                version=model_metadata.version,
                status=model_metadata.status.value,
                num_parameters=model_metadata.num_parameters,
                model_size_mb=model_metadata.model_size_mb
            )
            
            # Add metrics
            for metric_name, metric_value in model_metadata.metrics.items():
                response.metrics[metric_name] = metric_value
            
            # Add supported classes
            response.classes.extend(AG_NEWS_CLASSES)
            
            return response
            
        except Exception as e:
            logger.error(f"GetModelInfo failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return agnews_pb2.ModelInfoResponse()
    
    def ListModels(
        self,
        request: empty_pb2.Empty,
        context: grpc.ServicerContext
    ) -> agnews_pb2.ModelListResponse:
        """
        List available models RPC.
        
        Args:
            request: Empty request
            context: gRPC context
            
        Returns:
            List of available models
        """
        try:
            response = agnews_pb2.ModelListResponse()
            
            # Get models from catalog
            for model_metadata in self.model_service.catalog.values():
                model_info = agnews_pb2.ModelInfo(
                    model_name=model_metadata.name,
                    model_type=model_metadata.model_type,
                    version=model_metadata.version,
                    status=model_metadata.status.value
                )
                
                # Add metrics
                for metric_name, metric_value in model_metadata.metrics.items():
                    model_info.metrics[metric_name] = metric_value
                
                response.models.append(model_info)
            
            return response
            
        except Exception as e:
            logger.error(f"ListModels failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return agnews_pb2.ModelListResponse()
    
    def GetStatistics(
        self,
        request: empty_pb2.Empty,
        context: grpc.ServicerContext
    ) -> agnews_pb2.StatisticsResponse:
        """
        Get service statistics RPC.
        
        Args:
            request: Empty request
            context: gRPC context
            
        Returns:
            Service statistics
        """
        try:
            # Gather statistics from all services
            prediction_stats = self.prediction_service.get_statistics()
            training_stats = self.training_service.get_statistics()
            model_stats = self.model_service.get_statistics()
            data_stats = self.data_service.get_statistics()
            
            # Build response
            response = agnews_pb2.StatisticsResponse()
            
            # Add gRPC service stats
            response.service_stats["total_requests"] = self.stats["total_requests"]
            response.service_stats["successful_requests"] = self.stats["successful_requests"]
            response.service_stats["failed_requests"] = self.stats["failed_requests"]
            response.service_stats["avg_processing_time"] = (
                self.stats["total_processing_time"] / max(self.stats["successful_requests"], 1)
            )
            
            # Add prediction stats
            response.prediction_stats["total_predictions"] = prediction_stats["total_predictions"]
            response.prediction_stats["cache_hit_rate"] = prediction_stats["cache_stats"]["hit_rate"]
            response.prediction_stats["avg_processing_time"] = prediction_stats["avg_processing_time"]
            
            # Add training stats
            response.training_stats["total_experiments"] = training_stats["total_experiments"]
            response.training_stats["successful_experiments"] = training_stats["successful_experiments"]
            response.training_stats["avg_training_time"] = training_stats["avg_training_time"]
            
            # Add model stats
            response.model_stats["total_models"] = model_stats["total_models"]
            response.model_stats["production_models"] = model_stats["production_models"]
            response.model_stats["total_deployments"] = model_stats["total_deployments"]
            
            return response
            
        except Exception as e:
            logger.error(f"GetStatistics failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return agnews_pb2.StatisticsResponse()

class ModelServicer(model_pb2_grpc.ModelServiceServicer):
    """
    gRPC service for model management operations.
    
    Implements model lifecycle patterns from:
    - Sculley et al. (2015): "Hidden Technical Debt in Machine Learning Systems"
    """
    
    def __init__(self):
        """Initialize model gRPC service."""
        self.model_service = get_model_management_service()
        logger.info("Model gRPC service initialized")
    
    def RegisterModel(
        self,
        request: model_pb2.RegisterModelRequest,
        context: grpc.ServicerContext
    ) -> model_pb2.RegisterModelResponse:
        """
        Register new model RPC.
        
        Args:
            request: Model registration request
            context: gRPC context
            
        Returns:
            Registration response
        """
        try:
            # Load model from path
            model_path = Path(request.model_path)
            if not model_path.exists():
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Model path {model_path} not found")
                return model_pb2.RegisterModelResponse()
            
            # Register model (placeholder - needs actual model loading)
            # In production, this would load the model and tokenizer
            
            response = model_pb2.RegisterModelResponse(
                model_id=hashlib.md5(
                    f"{request.model_name}{datetime.now()}".encode()
                ).hexdigest()[:12],
                status="registered",
                message=f"Model {request.model_name} registered successfully"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"RegisterModel failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return model_pb2.RegisterModelResponse()
    
    def PromoteModel(
        self,
        request: model_pb2.PromoteModelRequest,
        context: grpc.ServicerContext
    ) -> model_pb2.PromoteModelResponse:
        """
        Promote model to new status RPC.
        
        Args:
            request: Model promotion request
            context: gRPC context
            
        Returns:
            Promotion response
        """
        try:
            # Convert string to enum
            target_status = ModelStatus(request.target_status)
            
            # Promote model
            model_metadata = self.model_service.promote_model(
                request.model_id,
                target_status
            )
            
            response = model_pb2.PromoteModelResponse(
                success=True,
                new_status=model_metadata.status.value,
                message=f"Model promoted to {model_metadata.status.value}"
            )
            
            return response
            
        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return model_pb2.PromoteModelResponse(success=False)
        
        except Exception as e:
            logger.error(f"PromoteModel failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return model_pb2.PromoteModelResponse(success=False)
    
    def DeployModel(
        self,
        request: model_pb2.DeployModelRequest,
        context: grpc.ServicerContext
    ) -> model_pb2.DeployModelResponse:
        """
        Deploy model RPC.
        
        Args:
            request: Deployment request
            context: gRPC context
            
        Returns:
            Deployment response
        """
        try:
            # Convert string to enum
            environment = DeploymentEnvironment(request.environment)
            
            # Deploy model
            deployment = self.model_service.deploy_model(
                request.model_id,
                environment,
                request.endpoint if request.endpoint else None
            )
            
            response = model_pb2.DeployModelResponse(
                deployment_id=deployment["deployment_id"],
                status="deployed",
                endpoint=deployment.get("endpoint", ""),
                message=f"Model deployed to {environment.value}"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"DeployModel failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return model_pb2.DeployModelResponse()
    
    def CompareModels(
        self,
        request: model_pb2.CompareModelsRequest,
        context: grpc.ServicerContext
    ) -> model_pb2.CompareModelsResponse:
        """
        Compare multiple models RPC.
        
        Args:
            request: Comparison request
            context: gRPC context
            
        Returns:
            Comparison results
        """
        try:
            # Compare models
            comparison = self.model_service.compare_models(
                list(request.model_ids),
                list(request.metrics) if request.metrics else None
            )
            
            response = model_pb2.CompareModelsResponse(
                best_model_id=comparison.get("best_model", "")
            )
            
            # Add model comparisons
            for model_info in comparison["models"]:
                model_comparison = model_pb2.ModelComparison(
                    model_id=model_info["model_id"],
                    model_name=model_info["name"],
                    version=model_info["version"]
                )
                
                # Add metrics
                for metric_name, metric_value in model_info["metrics"].items():
                    model_comparison.metrics[metric_name] = metric_value
                
                response.models.append(model_comparison)
            
            return response
            
        except Exception as e:
            logger.error(f"CompareModels failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return model_pb2.CompareModelsResponse()

class TrainingServicer(training_pb2_grpc.TrainingServiceServicer):
    """
    gRPC service for training operations.
    
    Implements training patterns from:
    - Dean et al. (2012): "Large Scale Distributed Deep Networks"
    """
    
    def __init__(self):
        """Initialize training gRPC service."""
        self.training_service = get_training_service()
        logger.info("Training gRPC service initialized")
    
    def StartTraining(
        self,
        request: training_pb2.StartTrainingRequest,
        context: grpc.ServicerContext
    ) -> training_pb2.StartTrainingResponse:
        """
        Start training job RPC.
        
        Args:
            request: Training request
            context: gRPC context
            
        Returns:
            Training job response
        """
        try:
            # Parse configuration
            config_dict = MessageToDict(request.config)
            config = TrainingConfig(**config_dict)
            
            # Start training (async in production)
            result = self.training_service.train_model(config)
            
            response = training_pb2.StartTrainingResponse(
                job_id=result.experiment_id,
                status=result.status.value,
                message=f"Training job {result.experiment_id} started"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"StartTraining failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return training_pb2.StartTrainingResponse()
    
    def GetTrainingStatus(
        self,
        request: training_pb2.TrainingStatusRequest,
        context: grpc.ServicerContext
    ) -> training_pb2.TrainingStatusResponse:
        """
        Get training job status RPC.
        
        Args:
            request: Status request
            context: gRPC context
            
        Returns:
            Training status response
        """
        try:
            # Get experiment info
            experiment = self.training_service.experiment_manager.get_experiment(
                request.job_id
            )
            
            response = training_pb2.TrainingStatusResponse(
                job_id=request.job_id,
                status=experiment["status"],
                progress=0.0,  # Placeholder
                current_epoch=0,  # Placeholder
                total_epochs=0  # Placeholder
            )
            
            # Add metrics
            for metric_name, metric_value in experiment["metrics"].items():
                response.metrics[metric_name] = metric_value
            
            return response
            
        except ValueError as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return training_pb2.TrainingStatusResponse()
        
        except Exception as e:
            logger.error(f"GetTrainingStatus failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return training_pb2.TrainingStatusResponse()
    
    def StopTraining(
        self,
        request: training_pb2.StopTrainingRequest,
        context: grpc.ServicerContext
    ) -> training_pb2.StopTrainingResponse:
        """
        Stop training job RPC.
        
        Args:
            request: Stop request
            context: gRPC context
            
        Returns:
            Stop response
        """
        try:
            # Update experiment status
            self.training_service.experiment_manager.update_experiment(
                request.job_id,
                status=TrainingStatus.CANCELLED
            )
            
            response = training_pb2.StopTrainingResponse(
                success=True,
                message=f"Training job {request.job_id} stopped"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"StopTraining failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return training_pb2.StopTrainingResponse(success=False)

def create_grpc_server(
    port: int = 50051,
    max_workers: int = 10,
    reflection_enabled: bool = True
) -> grpc.Server:
    """
    Create and configure gRPC server.
    
    Args:
        port: Server port
        max_workers: Maximum worker threads
        reflection_enabled: Enable server reflection
        
    Returns:
        Configured gRPC server
    """
    # Create server with thread pool
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            ('grpc.keepalive_time_ms', 10000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.http2.min_time_between_pings_ms', 10000),
        ]
    )
    
    # Add services
    agnews_pb2_grpc.add_AGNewsServiceServicer_to_server(
        AGNewsServicer(), server
    )
    model_pb2_grpc.add_ModelServiceServicer_to_server(
        ModelServicer(), server
    )
    training_pb2_grpc.add_TrainingServiceServicer_to_server(
        TrainingServicer(), server
    )
    
    # Enable reflection for debugging
    if reflection_enabled:
        SERVICE_NAMES = (
            agnews_pb2.DESCRIPTOR.services_by_name['AGNewsService'].full_name,
            model_pb2.DESCRIPTOR.services_by_name['ModelService'].full_name,
            training_pb2.DESCRIPTOR.services_by_name['TrainingService'].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(SERVICE_NAMES, server)
    
    # Bind to port
    server.add_insecure_port(f'[::]:{port}')
    
    logger.info(f"gRPC server created on port {port}")
    
    return server

def serve(port: int = 50051):
    """
    Start gRPC server.
    
    Args:
        port: Server port
    """
    server = create_grpc_server(port)
    server.start()
    
    logger.info(f"gRPC server started on port {port}")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server...")
        server.stop(grace_period=10)

if __name__ == "__main__":
    serve()
