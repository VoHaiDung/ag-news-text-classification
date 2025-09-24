"""
GraphQL Subscription Definitions
================================================================================
This module defines GraphQL subscriptions for real-time updates in the AG News
classification API, implementing WebSocket-based push notifications.

Subscription operations include:
- Training progress monitoring
- Real-time classification results
- Model deployment status
- System metrics streaming

References:
    - GraphQL Subscriptions Specification
    - WebSocket Protocol RFC 6455
    - Server-Sent Events (SSE) W3C Specification

Author: Võ Hải Dũng
License: MIT
"""

import strawberry
from typing import AsyncGenerator, Optional
import asyncio
import logging
from datetime import datetime
from uuid import uuid4

from .types import (
    Classification,
    Training,
    Model,
    Metrics,
    TrainingStatus,
    ModelType
)
from .context import Info

logger = logging.getLogger(__name__)

@strawberry.type
class Subscription:
    """
    Root subscription type for GraphQL schema.
    
    Provides real-time updates via WebSocket connections
    for monitoring and live data streaming.
    """
    
    @strawberry.subscription
    async def training_progress(
        self,
        info: Info,
        training_id: strawberry.ID
    ) -> AsyncGenerator[Training, None]:
        """
        Subscribe to training job progress updates.
        
        Args:
            info: GraphQL context info
            training_id: Training job ID to monitor
            
        Yields:
            Training: Training progress updates
        """
        try:
            training_service = info.context.training_service
            
            # Initial status
            job_data = await training_service.get_training_job(str(training_id))
            
            if not job_data:
                logger.error(f"Training job not found: {training_id}")
                return
            
            # Send updates every 2 seconds while training
            while True:
                # Get current job status
                job_data = await training_service.get_training_job(str(training_id))
                
                if not job_data:
                    break
                
                # Build training object
                training = Training(
                    id=training_id,
                    model_type=ModelType[job_data['model_type'].upper()],
                    status=TrainingStatus[job_data['status'].upper()],
                    started_at=job_data['started_at'],
                    completed_at=job_data.get('completed_at'),
                    epochs=job_data['epochs'],
                    batch_size=job_data['batch_size'],
                    learning_rate=job_data['learning_rate'],
                    dataset_id=strawberry.ID(job_data['dataset_id']),
                    current_epoch=job_data.get('current_epoch', 0),
                    current_loss=job_data.get('current_loss'),
                    best_validation_score=job_data.get('best_validation_score'),
                    final_metrics=job_data.get('final_metrics'),
                    model_id=strawberry.ID(job_data['model_id']) if job_data.get('model_id') else None,
                    error_message=job_data.get('error_message')
                )
                
                yield training
                
                # Stop if training is complete
                if training.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED]:
                    break
                
                # Wait before next update
                await asyncio.sleep(2)
                
        except asyncio.CancelledError:
            logger.info(f"Training progress subscription cancelled: {training_id}")
            raise
        except Exception as e:
            logger.error(f"Training progress subscription error: {e}")
            raise
    
    @strawberry.subscription
    async def live_classifications(
        self,
        info: Info,
        model_type: Optional[ModelType] = ModelType.ENSEMBLE
    ) -> AsyncGenerator[Classification, None]:
        """
        Subscribe to live classification results.
        
        Args:
            info: GraphQL context info
            model_type: Filter by model type
            
        Yields:
            Classification: Live classification results
        """
        try:
            # Get event stream from prediction service
            prediction_service = info.context.prediction_service
            
            # Subscribe to classification events
            async for event in prediction_service.get_classification_stream(model_type.value):
                # Build classification object
                classification = Classification(
                    id=strawberry.ID(event.get('id', str(uuid4()))),
                    text=event['text'],
                    label=event['label'],
                    confidence=event['confidence'],
                    model_type=model_type,
                    processing_time_ms=event.get('processing_time_ms', 0),
                    timestamp=datetime.utcnow(),
                    probabilities=event.get('probabilities')
                )
                
                yield classification
                
        except asyncio.CancelledError:
            logger.info("Live classifications subscription cancelled")
            raise
        except Exception as e:
            logger.error(f"Live classifications subscription error: {e}")
            raise
    
    @strawberry.subscription
    async def model_deployments(
        self,
        info: Info
    ) -> AsyncGenerator[Model, None]:
        """
        Subscribe to model deployment status changes.
        
        Args:
            info: GraphQL context info
            
        Yields:
            Model: Model deployment updates
        """
        try:
            model_service = info.context.model_service
            
            # Subscribe to deployment events
            async for event in model_service.get_deployment_stream():
                # Build model object
                model = Model(
                    id=strawberry.ID(event['id']),
                    name=event['name'],
                    type=ModelType[event['type'].upper()],
                    version=event['version'],
                    created_at=event['created_at'],
                    updated_at=datetime.utcnow(),
                    is_deployed=event.get('is_deployed', False),
                    is_default=event.get('is_default', False),
                    accuracy=event.get('accuracy'),
                    f1_score=event.get('f1_score'),
                    precision=event.get('precision'),
                    recall=event.get('recall')
                )
                
                yield model
                
        except asyncio.CancelledError:
            logger.info("Model deployments subscription cancelled")
            raise
        except Exception as e:
            logger.error(f"Model deployments subscription error: {e}")
            raise
    
    @strawberry.subscription
    async def system_metrics(
        self,
        info: Info,
        interval_seconds: int = 5
    ) -> AsyncGenerator[Metrics, None]:
        """
        Subscribe to system metrics updates.
        
        Args:
            info: GraphQL context info
            interval_seconds: Update interval in seconds
            
        Yields:
            Metrics: System metrics updates
        """
        try:
            # Validate interval
            interval_seconds = max(1, min(60, interval_seconds))
            
            while True:
                # Get current metrics
                metrics_data = await info.context.get_metrics("real-time")
                
                # Build metrics object
                metrics = Metrics(
                    timestamp=datetime.utcnow(),
                    model_accuracy=metrics_data.get('model_accuracy'),
                    model_latency_ms=metrics_data.get('model_latency_ms'),
                    model_throughput=metrics_data.get('model_throughput'),
                    api_requests_total=metrics_data.get('api_requests_total'),
                    api_errors_total=metrics_data.get('api_errors_total'),
                    api_latency_p50=metrics_data.get('api_latency_p50'),
                    api_latency_p95=metrics_data.get('api_latency_p95'),
                    api_latency_p99=metrics_data.get('api_latency_p99'),
                    cpu_usage_percent=metrics_data.get('cpu_usage_percent'),
                    memory_usage_mb=metrics_data.get('memory_usage_mb'),
                    gpu_usage_percent=metrics_data.get('gpu_usage_percent')
                )
                
                yield metrics
                
                # Wait for next interval
                await asyncio.sleep(interval_seconds)
                
        except asyncio.CancelledError:
            logger.info("System metrics subscription cancelled")
            raise
        except Exception as e:
            logger.error(f"System metrics subscription error: {e}")
            raise
    
    @strawberry.subscription
    async def classification_queue(
        self,
        info: Info,
        batch_size: int = 10,
        timeout_seconds: int = 30
    ) -> AsyncGenerator[List[Classification], None]:
        """
        Subscribe to batched classification queue.
        
        Args:
            info: GraphQL context info
            batch_size: Number of classifications per batch
            timeout_seconds: Timeout for batch collection
            
        Yields:
            List[Classification]: Batched classification results
        """
        try:
            prediction_service = info.context.prediction_service
            batch = []
            last_yield_time = asyncio.get_event_loop().time()
            
            async for event in prediction_service.get_classification_stream():
                # Build classification
                classification = Classification(
                    id=strawberry.ID(event.get('id', str(uuid4()))),
                    text=event['text'],
                    label=event['label'],
                    confidence=event['confidence'],
                    model_type=ModelType[event.get('model_type', 'ENSEMBLE')],
                    processing_time_ms=event.get('processing_time_ms', 0),
                    timestamp=datetime.utcnow(),
                    probabilities=event.get('probabilities')
                )
                
                batch.append(classification)
                
                # Check if batch is ready to send
                current_time = asyncio.get_event_loop().time()
                time_elapsed = current_time - last_yield_time
                
                if len(batch) >= batch_size or time_elapsed >= timeout_seconds:
                    if batch:
                        yield batch
                        batch = []
                        last_yield_time = current_time
                        
        except asyncio.CancelledError:
            # Yield remaining batch if any
            if batch:
                yield batch
            logger.info("Classification queue subscription cancelled")
            raise
        except Exception as e:
            logger.error(f"Classification queue subscription error: {e}")
            raise
