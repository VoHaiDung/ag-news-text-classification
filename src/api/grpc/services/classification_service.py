"""
Classification gRPC Service
================================================================================
This module implements the gRPC service for text classification, providing
high-performance prediction capabilities with streaming support.

Implements service methods for:
- Single text classification
- Batch classification with streaming
- Model selection and configuration
- Confidence thresholds and filtering

References:
    - gRPC Python Documentation
    - Protocol Buffers Language Guide
    - Vaswani, A., et al. (2017). Attention is All You Need

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
import logging
from typing import Iterator, List, Dict, Any
import grpc
import time
import numpy as np

from . import BaseGRPCService
from ..protos import classification_pb2, classification_pb2_grpc
from ..protos.common import types_pb2, status_pb2
from ....services.core.prediction_service import PredictionService
from ....core.exceptions import ModelNotFoundError, PredictionError

logger = logging.getLogger(__name__)

class ClassificationService(
    BaseGRPCService,
    classification_pb2_grpc.ClassificationServiceServicer
):
    """
    gRPC service implementation for text classification.
    
    Provides high-performance classification with support for:
    - Multiple model types
    - Streaming predictions
    - Explanation generation
    - Confidence calibration
    """
    
    def __init__(self):
        """Initialize classification service."""
        super().__init__("ClassificationService")
        self.prediction_service = PredictionService()
        self.model_cache = {}
        
    def register(self, server: grpc.Server) -> None:
        """
        Register service with gRPC server.
        
        Args:
            server: gRPC server instance
        """
        classification_pb2_grpc.add_ClassificationServiceServicer_to_server(
            self,
            server
        )
        logger.info(f"Registered {self.service_name}")
    
    def Classify(
        self,
        request: classification_pb2.ClassifyRequest,
        context: grpc.ServicerContext
    ) -> classification_pb2.ClassifyResponse:
        """
        Classify single text.
        
        Args:
            request: Classification request
            context: gRPC context
            
        Returns:
            classification_pb2.ClassifyResponse: Classification result
        """
        try:
            # Validate request
            if not self.validate_request(request, ['text']):
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Text field is required"
                )
            
            # Start timing
            start_time = time.time()
            
            # Get model type
            model_type = request.model_type or "ensemble"
            
            # Perform prediction
            result = self.prediction_service.predict(
                text=request.text,
                model_type=model_type,
                return_probabilities=request.return_probabilities,
                return_explanations=request.return_explanations
            )
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Build response
            response = classification_pb2.ClassifyResponse(
                prediction=classification_pb2.Prediction(
                    label=result['label'],
                    confidence=result['confidence'],
                    model_type=model_type,
                    processing_time_ms=processing_time_ms
                ),
                status=status_pb2.Status(
                    code=status_pb2.StatusCode.OK,
                    message="Classification successful"
                )
            )
            
            # Add probabilities if requested
            if request.return_probabilities and 'probabilities' in result:
                for label, prob in result['probabilities'].items():
                    response.prediction.probabilities.append(
                        classification_pb2.ClassProbability(
                            label=label,
                            probability=prob
                        )
                    )
            
            # Add explanations if requested
            if request.return_explanations and 'explanations' in result:
                response.prediction.explanation.CopyFrom(
                    self._build_explanation(result['explanations'])
                )
            
            self.increment_success_metric()
            return response
            
        except ModelNotFoundError as e:
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.NOT_FOUND, str(e))
        except PredictionError as e:
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.INTERNAL, str(e))
        except Exception as e:
            logger.error(f"Classification error: {e}")
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.INTERNAL, "Classification failed")
    
    def ClassifyBatch(
        self,
        request: classification_pb2.ClassifyBatchRequest,
        context: grpc.ServicerContext
    ) -> classification_pb2.ClassifyBatchResponse:
        """
        Classify batch of texts.
        
        Args:
            request: Batch classification request
            context: gRPC context
            
        Returns:
            classification_pb2.ClassifyBatchResponse: Batch results
        """
        try:
            # Validate request
            if not request.texts:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Texts list cannot be empty"
                )
            
            start_time = time.time()
            predictions = []
            failed_indices = []
            
            # Process batch
            for idx, text in enumerate(request.texts):
                try:
                    result = self.prediction_service.predict(
                        text=text,
                        model_type=request.model_type or "ensemble",
                        return_probabilities=request.return_probabilities
                    )
                    
                    prediction = classification_pb2.Prediction(
                        label=result['label'],
                        confidence=result['confidence'],
                        model_type=request.model_type or "ensemble"
                    )
                    
                    if request.return_probabilities and 'probabilities' in result:
                        for label, prob in result['probabilities'].items():
                            prediction.probabilities.append(
                                classification_pb2.ClassProbability(
                                    label=label,
                                    probability=prob
                                )
                            )
                    
                    predictions.append(prediction)
                    
                except Exception as e:
                    logger.error(f"Failed to classify text at index {idx}: {e}")
                    failed_indices.append(idx)
                    predictions.append(classification_pb2.Prediction())
            
            # Calculate statistics
            total_time_ms = (time.time() - start_time) * 1000
            avg_confidence = np.mean([p.confidence for p in predictions if p.confidence > 0])
            
            # Build response
            response = classification_pb2.ClassifyBatchResponse(
                predictions=predictions,
                total_processed=len(request.texts),
                successful=len(request.texts) - len(failed_indices),
                failed=len(failed_indices),
                average_confidence=avg_confidence,
                total_time_ms=total_time_ms,
                status=status_pb2.Status(
                    code=status_pb2.StatusCode.OK if not failed_indices else status_pb2.StatusCode.PARTIAL_SUCCESS,
                    message=f"Processed {len(request.texts)} texts"
                )
            )
            
            if failed_indices:
                response.failed_indices.extend(failed_indices)
            
            self.increment_success_metric()
            return response
            
        except Exception as e:
            logger.error(f"Batch classification error: {e}")
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.INTERNAL, "Batch classification failed")
    
    def ClassifyStream(
        self,
        request_iterator: Iterator[classification_pb2.ClassifyRequest],
        context: grpc.ServicerContext
    ) -> Iterator[classification_pb2.ClassifyResponse]:
        """
        Stream classification for multiple texts.
        
        Args:
            request_iterator: Stream of classification requests
            context: gRPC context
            
        Yields:
            classification_pb2.ClassifyResponse: Classification results
        """
        try:
            for request in request_iterator:
                # Check if client cancelled
                if context.is_active():
                    response = self.Classify(request, context)
                    yield response
                else:
                    break
                    
        except Exception as e:
            logger.error(f"Stream classification error: {e}")
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.INTERNAL, "Stream classification failed")
    
    def GetModelInfo(
        self,
        request: classification_pb2.ModelInfoRequest,
        context: grpc.ServicerContext
    ) -> classification_pb2.ModelInfoResponse:
        """
        Get information about available models.
        
        Args:
            request: Model info request
            context: gRPC context
            
        Returns:
            classification_pb2.ModelInfoResponse: Model information
        """
        try:
            models = self.prediction_service.get_available_models()
            
            response = classification_pb2.ModelInfoResponse(
                status=status_pb2.Status(
                    code=status_pb2.StatusCode.OK,
                    message="Model info retrieved"
                )
            )
            
            for model_name, model_info in models.items():
                model = classification_pb2.ModelInfo(
                    name=model_name,
                    type=model_info.get('type', 'unknown'),
                    version=model_info.get('version', '1.0.0'),
                    is_loaded=model_info.get('is_loaded', False)
                )
                
                # Add metrics if available
                if 'metrics' in model_info:
                    for metric_name, value in model_info['metrics'].items():
                        model.metrics[metric_name] = value
                
                response.models.append(model)
            
            self.increment_success_metric()
            return response
            
        except Exception as e:
            logger.error(f"Get model info error: {e}")
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to get model info")
    
    def _build_explanation(self, explanation_data: Dict[str, Any]) -> classification_pb2.Explanation:
        """
        Build explanation protobuf message.
        
        Args:
            explanation_data: Explanation data dictionary
            
        Returns:
            classification_pb2.Explanation: Explanation message
        """
        explanation = classification_pb2.Explanation()
        
        # Add attention weights if available
        if 'attention_weights' in explanation_data:
            for token, weight in explanation_data['attention_weights'].items():
                explanation.attention_weights[token] = weight
        
        # Add feature importance
        if 'feature_importance' in explanation_data:
            for feature, importance in explanation_data['feature_importance'].items():
                explanation.feature_importance[feature] = importance
        
        # Add text snippets
        if 'important_phrases' in explanation_data:
            explanation.important_phrases.extend(explanation_data['important_phrases'])
        
        return explanation
    
    async def cleanup(self) -> None:
        """Cleanup service resources."""
        # Clear model cache
        self.model_cache.clear()
        
        # Cleanup prediction service
        if hasattr(self.prediction_service, 'cleanup'):
            await self.prediction_service.cleanup()
        
        logger.info(f"{self.service_name} cleaned up")
