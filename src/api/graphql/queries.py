"""
GraphQL Query Definitions
================================================================================
This module defines GraphQL queries for the AG News classification API,
implementing read operations with efficient data fetching and caching.

Query operations include:
- Text classification queries
- Model information queries
- Dataset queries
- Metrics and monitoring queries

References:
    - GraphQL Query Language Specification
    - DataLoader Pattern for N+1 Query Prevention
    - GraphQL Best Practices for Query Design

Author: Võ Hải Dũng
License: MIT
"""

import strawberry
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from .types import (
    Classification,
    Model,
    Dataset,
    Training,
    Metrics,
    Error,
    ModelType,
    DatasetSplit,
    TrainingStatus
)
from .context import Info
from ...services.core.prediction_service import PredictionService
from ...services.core.model_management_service import ModelManagementService
from ...services.core.data_service import DataService

logger = logging.getLogger(__name__)

@strawberry.type
class Query:
    """
    Root query type for GraphQL schema.
    
    Provides read-only operations for fetching data
    from the classification system.
    """
    
    @strawberry.field
    async def classify(
        self,
        info: Info,
        text: str,
        model_type: Optional[ModelType] = ModelType.ENSEMBLE,
        return_probabilities: bool = True,
        return_explanation: bool = False
    ) -> Classification:
        """
        Classify a single text document.
        
        Args:
            info: GraphQL context info
            text: Text to classify
            model_type: Model to use for classification
            return_probabilities: Include probability scores
            return_explanation: Include explanation
            
        Returns:
            Classification: Classification result
        """
        try:
            # Get prediction service from context
            prediction_service = info.context.prediction_service
            
            # Perform classification
            result = await prediction_service.predict_async(
                text=text,
                model_type=model_type.value,
                return_probabilities=return_probabilities,
                return_explanation=return_explanation
            )
            
            # Build response
            classification = Classification(
                id=strawberry.ID(str(result.get('id', 'temp'))),
                text=text,
                label=result['label'],
                confidence=result['confidence'],
                model_type=model_type,
                processing_time_ms=result.get('processing_time_ms', 0),
                timestamp=datetime.utcnow()
            )
            
            if return_probabilities:
                classification.probabilities = result.get('probabilities')
            
            if return_explanation:
                classification.explanation = result.get('explanation')
                classification.attention_weights = result.get('attention_weights')
            
            return classification
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            raise Exception(f"Classification failed: {str(e)}")
    
    @strawberry.field
    async def classify_batch(
        self,
        info: Info,
        texts: List[str],
        model_type: Optional[ModelType] = ModelType.ENSEMBLE,
        return_probabilities: bool = True
    ) -> List[Classification]:
        """
        Classify multiple text documents in batch.
        
        Args:
            info: GraphQL context info
            texts: List of texts to classify
            model_type: Model to use for classification
            return_probabilities: Include probability scores
            
        Returns:
            List[Classification]: Classification results
        """
        try:
            prediction_service = info.context.prediction_service
            
            # Perform batch classification
            results = await prediction_service.predict_batch_async(
                texts=texts,
                model_type=model_type.value,
                return_probabilities=return_probabilities
            )
            
            # Build response list
            classifications = []
            for i, result in enumerate(results):
                classification = Classification(
                    id=strawberry.ID(f"batch_{i}"),
                    text=texts[i],
                    label=result['label'],
                    confidence=result['confidence'],
                    model_type=model_type,
                    processing_time_ms=result.get('processing_time_ms', 0),
                    timestamp=datetime.utcnow()
                )
                
                if return_probabilities:
                    classification.probabilities = result.get('probabilities')
                
                classifications.append(classification)
            
            return classifications
            
        except Exception as e:
            logger.error(f"Batch classification error: {e}")
            raise Exception(f"Batch classification failed: {str(e)}")
    
    @strawberry.field
    async def model(
        self,
        info: Info,
        id: strawberry.ID
    ) -> Optional[Model]:
        """
        Get model by ID.
        
        Args:
            info: GraphQL context info
            id: Model ID
            
        Returns:
            Optional[Model]: Model if found
        """
        try:
            model_service = info.context.model_service
            
            model_data = await model_service.get_model(str(id))
            
            if not model_data:
                return None
            
            return Model(
                id=strawberry.ID(model_data['id']),
                name=model_data['name'],
                type=ModelType[model_data['type'].upper()],
                version=model_data['version'],
                created_at=model_data['created_at'],
                updated_at=model_data.get('updated_at'),
                is_deployed=model_data.get('is_deployed', False),
                is_default=model_data.get('is_default', False),
                accuracy=model_data.get('accuracy'),
                f1_score=model_data.get('f1_score'),
                precision=model_data.get('precision'),
                recall=model_data.get('recall'),
                parameters=model_data.get('parameters'),
                training_config=model_data.get('training_config'),
                size_mb=model_data.get('size_mb')
            )
            
        except Exception as e:
            logger.error(f"Error fetching model: {e}")
            return None
    
    @strawberry.field
    async def models(
        self,
        info: Info,
        model_type: Optional[ModelType] = None,
        is_deployed: Optional[bool] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[Model]:
        """
        List available models with filtering.
        
        Args:
            info: GraphQL context info
            model_type: Filter by model type
            is_deployed: Filter by deployment status
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            List[Model]: List of models
        """
        try:
            model_service = info.context.model_service
            
            # Build filters
            filters = {}
            if model_type:
                filters['type'] = model_type.value
            if is_deployed is not None:
                filters['is_deployed'] = is_deployed
            
            # Fetch models
            models_data = await model_service.list_models(
                filters=filters,
                limit=limit,
                offset=offset
            )
            
            # Convert to GraphQL types
            models = []
            for data in models_data:
                models.append(Model(
                    id=strawberry.ID(data['id']),
                    name=data['name'],
                    type=ModelType[data['type'].upper()],
                    version=data['version'],
                    created_at=data['created_at'],
                    updated_at=data.get('updated_at'),
                    is_deployed=data.get('is_deployed', False),
                    is_default=data.get('is_default', False),
                    accuracy=data.get('accuracy'),
                    f1_score=data.get('f1_score'),
                    precision=data.get('precision'),
                    recall=data.get('recall')
                ))
            
            return models
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    @strawberry.field
    async def dataset(
        self,
        info: Info,
        id: strawberry.ID
    ) -> Optional[Dataset]:
        """
        Get dataset by ID.
        
        Args:
            info: GraphQL context info
            id: Dataset ID
            
        Returns:
            Optional[Dataset]: Dataset if found
        """
        try:
            data_service = info.context.data_service
            
            dataset_data = await data_service.get_dataset(str(id))
            
            if not dataset_data:
                return None
            
            return Dataset(
                id=strawberry.ID(dataset_data['id']),
                name=dataset_data['name'],
                description=dataset_data.get('description'),
                split=DatasetSplit[dataset_data['split'].upper()],
                size=dataset_data['size'],
                created_at=dataset_data['created_at'],
                label_distribution=dataset_data.get('label_distribution'),
                avg_text_length=dataset_data.get('avg_text_length'),
                vocabulary_size=dataset_data.get('vocabulary_size')
            )
            
        except Exception as e:
            logger.error(f"Error fetching dataset: {e}")
            return None
    
    @strawberry.field
    async def datasets(
        self,
        info: Info,
        split: Optional[DatasetSplit] = None,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dataset]:
        """
        List available datasets.
        
        Args:
            info: GraphQL context info
            split: Filter by split type
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            List[Dataset]: List of datasets
        """
        try:
            data_service = info.context.data_service
            
            # Build filters
            filters = {}
            if split:
                filters['split'] = split.value
            
            # Fetch datasets
            datasets_data = await data_service.list_datasets(
                filters=filters,
                limit=limit,
                offset=offset
            )
            
            # Convert to GraphQL types
            datasets = []
            for data in datasets_data:
                datasets.append(Dataset(
                    id=strawberry.ID(data['id']),
                    name=data['name'],
                    description=data.get('description'),
                    split=DatasetSplit[data['split'].upper()],
                    size=data['size'],
                    created_at=data['created_at']
                ))
            
            return datasets
            
        except Exception as e:
            logger.error(f"Error listing datasets: {e}")
            return []
    
    @strawberry.field
    async def training_job(
        self,
        info: Info,
        id: strawberry.ID
    ) -> Optional[Training]:
        """
        Get training job by ID.
        
        Args:
            info: GraphQL context info
            id: Training job ID
            
        Returns:
            Optional[Training]: Training job if found
        """
        try:
            training_service = info.context.training_service
            
            job_data = await training_service.get_training_job(str(id))
            
            if not job_data:
                return None
            
            return Training(
                id=strawberry.ID(job_data['id']),
                model_type=ModelType[job_data['model_type'].upper()],
                status=TrainingStatus[job_data['status'].upper()],
                started_at=job_data['started_at'],
                completed_at=job_data.get('completed_at'),
                epochs=job_data['epochs'],
                batch_size=job_data['batch_size'],
                learning_rate=job_data['learning_rate'],
                dataset_id=strawberry.ID(job_data['dataset_id']),
                current_epoch=job_data.get('current_epoch'),
                current_loss=job_data.get('current_loss'),
                best_validation_score=job_data.get('best_validation_score'),
                final_metrics=job_data.get('final_metrics'),
                model_id=strawberry.ID(job_data['model_id']) if job_data.get('model_id') else None,
                error_message=job_data.get('error_message')
            )
            
        except Exception as e:
            logger.error(f"Error fetching training job: {e}")
            return None
    
    @strawberry.field
    async def metrics(
        self,
        info: Info,
        time_range: str = "1h"
    ) -> Metrics:
        """
        Get system metrics.
        
        Args:
            info: GraphQL context info
            time_range: Time range for metrics (1h, 24h, 7d)
            
        Returns:
            Metrics: System metrics
        """
        try:
            # Get metrics from context services
            metrics_data = await info.context.get_metrics(time_range)
            
            return Metrics(
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
            
        except Exception as e:
            logger.error(f"Error fetching metrics: {e}")
            return Metrics(timestamp=datetime.utcnow())
    
    @strawberry.field
    async def health(self, info: Info) -> Dict[str, Any]:
        """
        Health check query.
        
        Args:
            info: GraphQL context info
            
        Returns:
            Dict[str, Any]: Health status
        """
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
