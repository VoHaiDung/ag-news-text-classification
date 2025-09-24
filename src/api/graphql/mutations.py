"""
GraphQL Mutation Definitions
================================================================================
This module defines GraphQL mutations for the AG News classification API,
implementing write operations for data modification and system control.

Mutation operations include:
- Model training and deployment
- Dataset management
- Configuration updates
- Batch operations

References:
    - GraphQL Mutation Best Practices
    - Command Query Responsibility Segregation (CQRS)
    - Domain-Driven Design (Evans, 2003)

Author: Võ Hải Dũng
License: MIT
"""

import strawberry
from typing import List, Optional, Union
from datetime import datetime
import logging
import asyncio
from uuid import uuid4

from .types import (
    Classification,
    Model,
    Dataset,
    Training,
    Error,
    ClassificationInput,
    BatchClassificationInput,
    TrainingInput,
    ModelDeployInput,
    DatasetUploadInput,
    ModelType,
    TrainingStatus,
    DatasetSplit
)
from .context import Info
from ...services.core.training_service import TrainingService
from ...services.core.model_management_service import ModelManagementService
from ...services.core.data_service import DataService
from ...core.exceptions import (
    ModelNotFoundError,
    DataValidationError,
    ResourceExhaustedError
)

logger = logging.getLogger(__name__)

@strawberry.type
class Mutation:
    """
    Root mutation type for GraphQL schema.
    
    Provides write operations for modifying system state
    and triggering asynchronous operations.
    """
    
    @strawberry.mutation
    async def classify_text(
        self,
        info: Info,
        input: ClassificationInput
    ) -> Union[Classification, Error]:
        """
        Classify a single text with specific options.
        
        Args:
            info: GraphQL context info
            input: Classification input parameters
            
        Returns:
            Union[Classification, Error]: Classification result or error
        """
        try:
            # Validate input
            if not input.text or len(input.text.strip()) == 0:
                return Error(
                    code="INVALID_INPUT",
                    message="Text cannot be empty",
                    field="text"
                )
            
            # Get prediction service
            prediction_service = info.context.prediction_service
            
            # Perform classification
            result = await prediction_service.predict_async(
                text=input.text,
                model_type=input.model_type.value,
                return_probabilities=input.return_probabilities,
                return_explanation=input.return_explanation
            )
            
            # Build response
            classification = Classification(
                id=strawberry.ID(str(uuid4())),
                text=input.text,
                label=result['label'],
                confidence=result['confidence'],
                model_type=input.model_type,
                processing_time_ms=result.get('processing_time_ms', 0),
                timestamp=datetime.utcnow()
            )
            
            if input.return_probabilities:
                classification.probabilities = result.get('probabilities')
            
            if input.return_explanation:
                classification.explanation = result.get('explanation')
                classification.attention_weights = result.get('attention_weights')
            
            # Log successful classification
            logger.info(f"Text classified: {classification.label} (confidence: {classification.confidence:.2f})")
            
            return classification
            
        except Exception as e:
            logger.error(f"Classification mutation error: {e}")
            return Error(
                code="CLASSIFICATION_ERROR",
                message=str(e),
                details={"input": input.text[:100]}
            )
    
    @strawberry.mutation
    async def classify_batch(
        self,
        info: Info,
        input: BatchClassificationInput
    ) -> List[Union[Classification, Error]]:
        """
        Classify multiple texts in batch.
        
        Args:
            info: GraphQL context info
            input: Batch classification input
            
        Returns:
            List[Union[Classification, Error]]: Classification results
        """
        try:
            # Validate input
            if not input.texts or len(input.texts) == 0:
                return [Error(
                    code="INVALID_INPUT",
                    message="Texts list cannot be empty",
                    field="texts"
                )]
            
            if len(input.texts) > 100:
                return [Error(
                    code="BATCH_SIZE_EXCEEDED",
                    message="Batch size cannot exceed 100 texts",
                    field="texts"
                )]
            
            # Get prediction service
            prediction_service = info.context.prediction_service
            
            # Perform batch classification
            results = await prediction_service.predict_batch_async(
                texts=input.texts,
                model_type=input.model_type.value,
                return_probabilities=input.return_probabilities
            )
            
            # Build response list
            classifications = []
            for i, result in enumerate(results):
                if 'error' in result:
                    classifications.append(Error(
                        code="CLASSIFICATION_ERROR",
                        message=result['error'],
                        details={"index": i}
                    ))
                else:
                    classification = Classification(
                        id=strawberry.ID(f"batch_{uuid4()}_{i}"),
                        text=input.texts[i],
                        label=result['label'],
                        confidence=result['confidence'],
                        model_type=input.model_type,
                        processing_time_ms=result.get('processing_time_ms', 0),
                        timestamp=datetime.utcnow()
                    )
                    
                    if input.return_probabilities:
                        classification.probabilities = result.get('probabilities')
                    
                    classifications.append(classification)
            
            logger.info(f"Batch classified: {len(classifications)} texts")
            return classifications
            
        except Exception as e:
            logger.error(f"Batch classification error: {e}")
            return [Error(
                code="BATCH_ERROR",
                message=str(e),
                details={"batch_size": len(input.texts)}
            )]
    
    @strawberry.mutation
    async def start_training(
        self,
        info: Info,
        input: TrainingInput
    ) -> Union[Training, Error]:
        """
        Start a new model training job.
        
        Args:
            info: GraphQL context info
            input: Training configuration
            
        Returns:
            Union[Training, Error]: Training job or error
        """
        try:
            # Check authorization
            if not info.context.user or info.context.user.role != "admin":
                return Error(
                    code="UNAUTHORIZED",
                    message="Admin privileges required for training"
                )
            
            # Validate input
            if input.epochs < 1 or input.epochs > 100:
                return Error(
                    code="INVALID_INPUT",
                    message="Epochs must be between 1 and 100",
                    field="epochs"
                )
            
            # Get training service
            training_service = info.context.training_service
            
            # Start training job
            job_id = str(uuid4())
            training_config = {
                "model_type": input.model_type.value,
                "dataset_id": str(input.dataset_id),
                "epochs": input.epochs,
                "batch_size": input.batch_size,
                "learning_rate": input.learning_rate,
                "validation_split": input.validation_split
            }
            
            # Start training asynchronously
            await training_service.start_training(job_id, training_config)
            
            # Create training job object
            training = Training(
                id=strawberry.ID(job_id),
                model_type=input.model_type,
                status=TrainingStatus.PENDING,
                started_at=datetime.utcnow(),
                completed_at=None,
                epochs=input.epochs,
                batch_size=input.batch_size,
                learning_rate=input.learning_rate,
                dataset_id=input.dataset_id,
                current_epoch=0,
                current_loss=None,
                best_validation_score=None
            )
            
            logger.info(f"Training job started: {job_id}")
            return training
            
        except ResourceExhaustedError as e:
            return Error(
                code="RESOURCE_EXHAUSTED",
                message="Insufficient resources to start training",
                details={"error": str(e)}
            )
        except Exception as e:
            logger.error(f"Training start error: {e}")
            return Error(
                code="TRAINING_ERROR",
                message=str(e)
            )
    
    @strawberry.mutation
    async def stop_training(
        self,
        info: Info,
        training_id: strawberry.ID
    ) -> Union[Training, Error]:
        """
        Stop a running training job.
        
        Args:
            info: GraphQL context info
            training_id: Training job ID to stop
            
        Returns:
            Union[Training, Error]: Updated training job or error
        """
        try:
            # Check authorization
            if not info.context.user or info.context.user.role != "admin":
                return Error(
                    code="UNAUTHORIZED",
                    message="Admin privileges required"
                )
            
            # Get training service
            training_service = info.context.training_service
            
            # Stop training
            job_data = await training_service.stop_training(str(training_id))
            
            if not job_data:
                return Error(
                    code="NOT_FOUND",
                    message=f"Training job not found: {training_id}"
                )
            
            # Build response
            training = Training(
                id=training_id,
                model_type=ModelType[job_data['model_type'].upper()],
                status=TrainingStatus.CANCELLED,
                started_at=job_data['started_at'],
                completed_at=datetime.utcnow(),
                epochs=job_data['epochs'],
                batch_size=job_data['batch_size'],
                learning_rate=job_data['learning_rate'],
                dataset_id=strawberry.ID(job_data['dataset_id']),
                current_epoch=job_data.get('current_epoch'),
                error_message="Training stopped by user"
            )
            
            logger.info(f"Training job stopped: {training_id}")
            return training
            
        except Exception as e:
            logger.error(f"Stop training error: {e}")
            return Error(
                code="STOP_ERROR",
                message=str(e)
            )
    
    @strawberry.mutation
    async def deploy_model(
        self,
        info: Info,
        input: ModelDeployInput
    ) -> Union[Model, Error]:
        """
        Deploy a trained model for serving.
        
        Args:
            info: GraphQL context info
            input: Deployment configuration
            
        Returns:
            Union[Model, Error]: Deployed model or error
        """
        try:
            # Check authorization
            if not info.context.user or info.context.user.role not in ["admin", "deployer"]:
                return Error(
                    code="UNAUTHORIZED",
                    message="Deployment privileges required"
                )
            
            # Get model service
            model_service = info.context.model_service
            
            # Deploy model
            deployment_config = {
                "model_id": str(input.model_id),
                "make_default": input.make_default,
                "replicas": input.replicas
            }
            
            model_data = await model_service.deploy_model(deployment_config)
            
            if not model_data:
                return Error(
                    code="DEPLOYMENT_FAILED",
                    message=f"Failed to deploy model: {input.model_id}"
                )
            
            # Build response
            model = Model(
                id=input.model_id,
                name=model_data['name'],
                type=ModelType[model_data['type'].upper()],
                version=model_data['version'],
                created_at=model_data['created_at'],
                updated_at=datetime.utcnow(),
                is_deployed=True,
                is_default=input.make_default,
                accuracy=model_data.get('accuracy'),
                f1_score=model_data.get('f1_score')
            )
            
            logger.info(f"Model deployed: {input.model_id}")
            return model
            
        except ModelNotFoundError:
            return Error(
                code="MODEL_NOT_FOUND",
                message=f"Model not found: {input.model_id}"
            )
        except Exception as e:
            logger.error(f"Model deployment error: {e}")
            return Error(
                code="DEPLOYMENT_ERROR",
                message=str(e)
            )
    
    @strawberry.mutation
    async def undeploy_model(
        self,
        info: Info,
        model_id: strawberry.ID
    ) -> Union[Model, Error]:
        """
        Undeploy a model from serving.
        
        Args:
            info: GraphQL context info
            model_id: Model ID to undeploy
            
        Returns:
            Union[Model, Error]: Updated model or error
        """
        try:
            # Check authorization
            if not info.context.user or info.context.user.role not in ["admin", "deployer"]:
                return Error(
                    code="UNAUTHORIZED",
                    message="Deployment privileges required"
                )
            
            # Get model service
            model_service = info.context.model_service
            
            # Undeploy model
            model_data = await model_service.undeploy_model(str(model_id))
            
            if not model_data:
                return Error(
                    code="UNDEPLOY_FAILED",
                    message=f"Failed to undeploy model: {model_id}"
                )
            
            # Build response
            model = Model(
                id=model_id,
                name=model_data['name'],
                type=ModelType[model_data['type'].upper()],
                version=model_data['version'],
                created_at=model_data['created_at'],
                updated_at=datetime.utcnow(),
                is_deployed=False,
                is_default=False
            )
            
            logger.info(f"Model undeployed: {model_id}")
            return model
            
        except Exception as e:
            logger.error(f"Model undeploy error: {e}")
            return Error(
                code="UNDEPLOY_ERROR",
                message=str(e)
            )
    
    @strawberry.mutation
    async def upload_dataset(
        self,
        info: Info,
        input: DatasetUploadInput,
        file_content: str
    ) -> Union[Dataset, Error]:
        """
        Upload a new dataset for training.
        
        Args:
            info: GraphQL context info
            input: Dataset metadata
            file_content: Base64 encoded file content
            
        Returns:
            Union[Dataset, Error]: Created dataset or error
        """
        try:
            # Check authorization
            if not info.context.user:
                return Error(
                    code="UNAUTHORIZED",
                    message="Authentication required"
                )
            
            # Get data service
            data_service = info.context.data_service
            
            # Create dataset
            dataset_id = str(uuid4())
            dataset_config = {
                "id": dataset_id,
                "name": input.name,
                "description": input.description,
                "split": input.split.value,
                "format": input.format,
                "content": file_content
            }
            
            dataset_data = await data_service.create_dataset(dataset_config)
            
            if not dataset_data:
                return Error(
                    code="DATASET_CREATION_FAILED",
                    message="Failed to create dataset"
                )
            
            # Build response
            dataset = Dataset(
                id=strawberry.ID(dataset_id),
                name=input.name,
                description=input.description,
                split=input.split,
                size=dataset_data.get('size', 0),
                created_at=datetime.utcnow(),
                label_distribution=dataset_data.get('label_distribution'),
                avg_text_length=dataset_data.get('avg_text_length'),
                vocabulary_size=dataset_data.get('vocabulary_size')
            )
            
            logger.info(f"Dataset uploaded: {dataset_id}")
            return dataset
            
        except DataValidationError as e:
            return Error(
                code="INVALID_DATASET",
                message=f"Dataset validation failed: {str(e)}",
                field="file_content"
            )
        except Exception as e:
            logger.error(f"Dataset upload error: {e}")
            return Error(
                code="UPLOAD_ERROR",
                message=str(e)
            )
    
    @strawberry.mutation
    async def delete_dataset(
        self,
        info: Info,
        dataset_id: strawberry.ID
    ) -> Union[bool, Error]:
        """
        Delete a dataset.
        
        Args:
            info: GraphQL context info
            dataset_id: Dataset ID to delete
            
        Returns:
            Union[bool, Error]: Success status or error
        """
        try:
            # Check authorization
            if not info.context.user or info.context.user.role != "admin":
                return Error(
                    code="UNAUTHORIZED",
                    message="Admin privileges required"
                )
            
            # Get data service
            data_service = info.context.data_service
            
            # Delete dataset
            success = await data_service.delete_dataset(str(dataset_id))
            
            if not success:
                return Error(
                    code="DELETE_FAILED",
                    message=f"Failed to delete dataset: {dataset_id}"
                )
            
            logger.info(f"Dataset deleted: {dataset_id}")
            return True
            
        except Exception as e:
            logger.error(f"Dataset deletion error: {e}")
            return Error(
                code="DELETE_ERROR",
                message=str(e)
            )
    
    @strawberry.mutation
    async def clear_cache(
        self,
        info: Info,
        cache_type: Optional[str] = "all"
    ) -> Union[bool, Error]:
        """
        Clear system caches.
        
        Args:
            info: GraphQL context info
            cache_type: Type of cache to clear
            
        Returns:
            Union[bool, Error]: Success status or error
        """
        try:
            # Check authorization
            if not info.context.user or info.context.user.role != "admin":
                return Error(
                    code="UNAUTHORIZED",
                    message="Admin privileges required"
                )
            
            # Clear cache based on type
            if cache_type == "all":
                await info.context.clear_all_caches()
            elif cache_type == "model":
                await info.context.clear_model_cache()
            elif cache_type == "data":
                await info.context.clear_data_cache()
            else:
                return Error(
                    code="INVALID_CACHE_TYPE",
                    message=f"Invalid cache type: {cache_type}",
                    field="cache_type"
                )
            
            logger.info(f"Cache cleared: {cache_type}")
            return True
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return Error(
                code="CACHE_ERROR",
                message=str(e)
            )
