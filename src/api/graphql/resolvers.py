"""
GraphQL Resolvers for AG News Text Classification API
================================================================================
This module implements GraphQL resolvers for queries, mutations, and subscriptions.
Resolvers handle the business logic for GraphQL operations, interfacing with
the service layer to perform actual operations.

The implementation follows GraphQL best practices for resolver design,
error handling, and performance optimization through dataloaders.

References:
    - GraphQL Specification: https://spec.graphql.org/
    - Principled GraphQL: https://principledgraphql.com/
    - GraphQL Best Practices: https://graphql.org/learn/best-practices/

Author: Võ Hải Dũng
License: MIT
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import asyncio
import logging
from dataclasses import dataclass

from graphql import GraphQLError
from graphene import ObjectType, String, Int, Float, List as GrapheneList, Field, Boolean

from ...services.core.prediction_service import PredictionService
from ...services.core.training_service import TrainingService
from ...services.core.data_service import DataService
from ...services.core.model_management_service import ModelManagementService
from ...core.exceptions import ValidationError, ServiceError
from .context import GraphQLContext

# Configure logging
logger = logging.getLogger(__name__)


class QueryResolvers:
    """
    Resolvers for GraphQL queries
    
    Implements read operations for classification, models, and data.
    Uses dataloaders for efficient batching and caching of database queries.
    """
    
    @staticmethod
    async def resolve_classify(root, info: GraphQLContext, text: str, 
                              model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Resolve single text classification query
        
        Args:
            root: Parent resolver result
            info: GraphQL execution context
            text: Input text to classify
            model_id: Optional model identifier
            
        Returns:
            Classification result dictionary
            
        Raises:
            GraphQLError: If classification fails
        """
        try:
            # Get prediction service from context
            prediction_service = info.context.get_service('prediction')
            
            # Validate input
            if not text or len(text.strip()) == 0:
                raise GraphQLError("Text input cannot be empty")
            
            if len(text) > 10000:
                raise GraphQLError("Text exceeds maximum length of 10000 characters")
            
            # Perform classification
            result = await prediction_service.classify_async(
                text=text,
                model_id=model_id,
                user_id=info.context.user_id
            )
            
            # Log successful classification
            logger.info(f"Classification completed for user {info.context.user_id}")
            
            return {
                'predictions': result['predictions'],
                'model_id': result['model_id'],
                'processing_time': result['processing_time_ms'],
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except ValidationError as e:
            logger.warning(f"Validation error in classify: {e}")
            raise GraphQLError(f"Validation error: {str(e)}")
        except ServiceError as e:
            logger.error(f"Service error in classify: {e}")
            raise GraphQLError(f"Service error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in classify: {e}")
            raise GraphQLError("Internal server error during classification")
    
    @staticmethod
    async def resolve_classify_batch(root, info: GraphQLContext, 
                                    texts: List[str],
                                    model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Resolve batch text classification query
        
        Args:
            root: Parent resolver result
            info: GraphQL execution context
            texts: List of texts to classify
            model_id: Optional model identifier
            
        Returns:
            Batch classification results
        """
        try:
            # Validate batch size
            if len(texts) > 100:
                raise GraphQLError("Batch size exceeds maximum of 100 texts")
            
            # Get prediction service
            prediction_service = info.context.get_service('prediction')
            
            # Use dataloader for efficient batch processing
            dataloader = info.context.dataloaders.get('classification_batch')
            if dataloader:
                results = await dataloader.load_many(
                    [(text, model_id) for text in texts]
                )
            else:
                # Fallback to direct service call
                results = await prediction_service.classify_batch_async(
                    texts=texts,
                    model_id=model_id,
                    user_id=info.context.user_id
                )
            
            return {
                'results': results,
                'total': len(results),
                'model_id': model_id,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in batch classification: {e}")
            raise GraphQLError(f"Batch classification failed: {str(e)}")
    
    @staticmethod
    async def resolve_get_model(root, info: GraphQLContext, 
                               model_id: str) -> Dict[str, Any]:
        """
        Resolve model information query
        
        Args:
            root: Parent resolver result
            info: GraphQL execution context
            model_id: Model identifier
            
        Returns:
            Model information dictionary
        """
        try:
            # Get model management service
            model_service = info.context.get_service('model_management')
            
            # Use dataloader for caching
            dataloader = info.context.dataloaders.get('model')
            if dataloader:
                model_info = await dataloader.load(model_id)
            else:
                model_info = await model_service.get_model_info_async(model_id)
            
            if not model_info:
                raise GraphQLError(f"Model {model_id} not found")
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error retrieving model {model_id}: {e}")
            raise GraphQLError(f"Failed to retrieve model: {str(e)}")
    
    @staticmethod
    async def resolve_list_models(root, info: GraphQLContext,
                                 limit: int = 10,
                                 offset: int = 0,
                                 filter_active: Optional[bool] = None) -> Dict[str, Any]:
        """
        Resolve list of available models
        
        Args:
            root: Parent resolver result
            info: GraphQL execution context
            limit: Maximum number of models to return
            offset: Pagination offset
            filter_active: Optional filter for active models
            
        Returns:
            Dictionary with models list and metadata
        """
        try:
            model_service = info.context.get_service('model_management')
            
            # Apply filters
            filters = {}
            if filter_active is not None:
                filters['active'] = filter_active
            
            # Get models with pagination
            models = await model_service.list_models_async(
                limit=limit,
                offset=offset,
                filters=filters
            )
            
            return {
                'models': models['items'],
                'total': models['total'],
                'limit': limit,
                'offset': offset
            }
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise GraphQLError(f"Failed to list models: {str(e)}")


class MutationResolvers:
    """
    Resolvers for GraphQL mutations
    
    Implements write operations for training, model management, and data updates.
    Handles transactional operations and ensures data consistency.
    """
    
    @staticmethod
    async def resolve_start_training(root, info: GraphQLContext,
                                    config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve training initiation mutation
        
        Args:
            root: Parent resolver result
            info: GraphQL execution context
            config: Training configuration
            
        Returns:
            Training job information
        """
        try:
            # Check user permissions
            if not info.context.has_permission('training:create'):
                raise GraphQLError("Insufficient permissions to start training")
            
            # Get training service
            training_service = info.context.get_service('training')
            
            # Validate configuration
            validated_config = await training_service.validate_config_async(config)
            
            # Start training job
            job_info = await training_service.start_training_async(
                config=validated_config,
                user_id=info.context.user_id
            )
            
            logger.info(f"Training job {job_info['job_id']} started by user {info.context.user_id}")
            
            return {
                'job_id': job_info['job_id'],
                'status': 'STARTED',
                'config': validated_config,
                'created_at': datetime.utcnow().isoformat()
            }
            
        except ValidationError as e:
            logger.warning(f"Invalid training config: {e}")
            raise GraphQLError(f"Invalid configuration: {str(e)}")
        except Exception as e:
            logger.error(f"Error starting training: {e}")
            raise GraphQLError(f"Failed to start training: {str(e)}")
    
    @staticmethod
    async def resolve_stop_training(root, info: GraphQLContext,
                                   job_id: str) -> Dict[str, Any]:
        """
        Resolve training termination mutation
        
        Args:
            root: Parent resolver result
            info: GraphQL execution context
            job_id: Training job identifier
            
        Returns:
            Updated job status
        """
        try:
            # Check permissions
            if not info.context.has_permission('training:stop'):
                raise GraphQLError("Insufficient permissions to stop training")
            
            training_service = info.context.get_service('training')
            
            # Stop training job
            result = await training_service.stop_training_async(
                job_id=job_id,
                user_id=info.context.user_id
            )
            
            return {
                'job_id': job_id,
                'status': 'STOPPED',
                'stopped_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error stopping training job {job_id}: {e}")
            raise GraphQLError(f"Failed to stop training: {str(e)}")
    
    @staticmethod
    async def resolve_deploy_model(root, info: GraphQLContext,
                                  model_id: str,
                                  environment: str = 'staging') -> Dict[str, Any]:
        """
        Resolve model deployment mutation
        
        Args:
            root: Parent resolver result
            info: GraphQL execution context
            model_id: Model identifier
            environment: Deployment environment
            
        Returns:
            Deployment information
        """
        try:
            # Check deployment permissions
            if not info.context.has_permission(f'model:deploy:{environment}'):
                raise GraphQLError(f"Insufficient permissions to deploy to {environment}")
            
            model_service = info.context.get_service('model_management')
            
            # Deploy model
            deployment_info = await model_service.deploy_model_async(
                model_id=model_id,
                environment=environment,
                user_id=info.context.user_id
            )
            
            logger.info(f"Model {model_id} deployed to {environment} by user {info.context.user_id}")
            
            return {
                'model_id': model_id,
                'environment': environment,
                'status': 'DEPLOYED',
                'endpoint': deployment_info['endpoint'],
                'deployed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error deploying model {model_id}: {e}")
            raise GraphQLError(f"Failed to deploy model: {str(e)}")
    
    @staticmethod
    async def resolve_upload_data(root, info: GraphQLContext,
                                 dataset_name: str,
                                 data: str,
                                 format: str = 'json') -> Dict[str, Any]:
        """
        Resolve data upload mutation
        
        Args:
            root: Parent resolver result
            info: GraphQL execution context
            dataset_name: Name for the dataset
            data: Data content (base64 encoded for binary)
            format: Data format (json, csv, etc.)
            
        Returns:
            Upload status and dataset information
        """
        try:
            # Check data upload permissions
            if not info.context.has_permission('data:upload'):
                raise GraphQLError("Insufficient permissions to upload data")
            
            data_service = info.context.get_service('data')
            
            # Process and store data
            dataset_info = await data_service.upload_dataset_async(
                name=dataset_name,
                data=data,
                format=format,
                user_id=info.context.user_id
            )
            
            return {
                'dataset_id': dataset_info['id'],
                'name': dataset_name,
                'size': dataset_info['size'],
                'format': format,
                'status': 'UPLOADED',
                'created_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error uploading data: {e}")
            raise GraphQLError(f"Failed to upload data: {str(e)}")


class SubscriptionResolvers:
    """
    Resolvers for GraphQL subscriptions
    
    Implements real-time updates for training progress, model updates,
    and classification results using WebSocket connections.
    """
    
    @staticmethod
    async def resolve_training_progress(root, info: GraphQLContext,
                                       job_id: str):
        """
        Resolve training progress subscription
        
        Args:
            root: Parent resolver result
            info: GraphQL execution context
            job_id: Training job identifier
            
        Yields:
            Training progress updates
        """
        try:
            training_service = info.context.get_service('training')
            
            # Subscribe to training updates
            async for update in training_service.subscribe_to_job_async(job_id):
                yield {
                    'job_id': job_id,
                    'epoch': update.get('epoch'),
                    'batch': update.get('batch'),
                    'loss': update.get('loss'),
                    'metrics': update.get('metrics'),
                    'status': update.get('status'),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error in training progress subscription: {e}")
            yield {
                'error': str(e),
                'job_id': job_id,
                'status': 'ERROR'
            }
    
    @staticmethod
    async def resolve_model_updates(root, info: GraphQLContext):
        """
        Resolve model updates subscription
        
        Args:
            root: Parent resolver result
            info: GraphQL execution context
            
        Yields:
            Model update notifications
        """
        try:
            model_service = info.context.get_service('model_management')
            
            # Subscribe to model updates
            async for update in model_service.subscribe_to_updates_async():
                yield {
                    'model_id': update.get('model_id'),
                    'event_type': update.get('event_type'),
                    'details': update.get('details'),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error in model updates subscription: {e}")
            yield {
                'error': str(e),
                'event_type': 'ERROR'
            }
    
    @staticmethod
    async def resolve_classification_stream(root, info: GraphQLContext,
                                           stream_id: str):
        """
        Resolve streaming classification subscription
        
        Args:
            root: Parent resolver result
            info: GraphQL execution context
            stream_id: Stream identifier
            
        Yields:
            Classification results stream
        """
        try:
            prediction_service = info.context.get_service('prediction')
            
            # Subscribe to classification stream
            async for result in prediction_service.subscribe_to_stream_async(stream_id):
                yield {
                    'stream_id': stream_id,
                    'text': result.get('text'),
                    'predictions': result.get('predictions'),
                    'model_id': result.get('model_id'),
                    'sequence_number': result.get('sequence_number'),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error in classification stream subscription: {e}")
            yield {
                'error': str(e),
                'stream_id': stream_id
            }


# Export resolver classes
__all__ = [
    'QueryResolvers',
    'MutationResolvers',
    'SubscriptionResolvers'
]
