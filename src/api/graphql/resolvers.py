"""
GraphQL Resolvers Implementation
================================================================================
This module implements GraphQL resolvers for the AG News classification system,
providing field resolution logic for queries, mutations, and subscriptions
following GraphQL best practices and DataLoader patterns.

The implementation follows the GraphQL specification and implements efficient
data fetching strategies to avoid N+1 query problems.

References:
    - GraphQL Specification (2021). https://spec.graphql.org/
    - Buna, S. (2018). Learning GraphQL: Declarative Data Fetching
    - DataLoader Pattern: https://github.com/graphql/dataloader

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
import logging
from functools import wraps

from graphql import GraphQLResolveInfo
from src.services.core.prediction_service import PredictionService
from src.services.core.training_service import TrainingService
from src.services.core.data_service import DataService
from src.services.core.model_management_service import ModelManagementService
from src.core.exceptions import (
    ValidationError, NotFoundError, AuthenticationError,
    AuthorizationError, ServiceError
)

logger = logging.getLogger(__name__)


class BaseResolver:
    """
    Base class for GraphQL resolvers implementing common patterns.
    
    This class provides shared functionality for all resolvers including
    error handling, authentication checks, and logging.
    """
    
    def __init__(self):
        """Initialize base resolver with service instances."""
        self.prediction_service = PredictionService()
        self.training_service = TrainingService()
        self.data_service = DataService()
        self.model_service = ModelManagementService()
        
    def require_auth(self, func: Callable) -> Callable:
        """
        Decorator to require authentication for resolver.
        
        Args:
            func: Resolver function to wrap
            
        Returns:
            Wrapped function with authentication check
        """
        @wraps(func)
        async def wrapper(parent, info: GraphQLResolveInfo, **kwargs):
            context = info.context
            if not context.get('user'):
                raise AuthenticationError("Authentication required")
            return await func(parent, info, **kwargs)
        return wrapper
        
    def require_role(self, role: str) -> Callable:
        """
        Decorator to require specific role for resolver.
        
        Args:
            role: Required role name
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(parent, info: GraphQLResolveInfo, **kwargs):
                context = info.context
                user = context.get('user')
                if not user:
                    raise AuthenticationError("Authentication required")
                if role not in user.get('roles', []):
                    raise AuthorizationError(f"Role '{role}' required")
                return await func(parent, info, **kwargs)
            return wrapper
        return decorator
        
    async def handle_error(self, error: Exception) -> Dict[str, Any]:
        """
        Handle and format errors for GraphQL response.
        
        Args:
            error: Exception to handle
            
        Returns:
            Formatted error response
        """
        logger.error(f"Resolver error: {str(error)}", exc_info=True)
        
        if isinstance(error, ValidationError):
            return {
                'success': False,
                'error': {
                    'code': 'VALIDATION_ERROR',
                    'message': str(error),
                    'details': error.details if hasattr(error, 'details') else None
                }
            }
        elif isinstance(error, NotFoundError):
            return {
                'success': False,
                'error': {
                    'code': 'NOT_FOUND',
                    'message': str(error)
                }
            }
        elif isinstance(error, (AuthenticationError, AuthorizationError)):
            return {
                'success': False,
                'error': {
                    'code': 'AUTH_ERROR',
                    'message': str(error)
                }
            }
        else:
            return {
                'success': False,
                'error': {
                    'code': 'INTERNAL_ERROR',
                    'message': 'An internal error occurred'
                }
            }


class QueryResolvers(BaseResolver):
    """
    Implementation of GraphQL query resolvers.
    
    This class handles all read operations for the GraphQL API,
    implementing efficient data fetching with DataLoader pattern.
    """
    
    async def resolve_classify_text(
        self,
        parent: Any,
        info: GraphQLResolveInfo,
        text: str,
        model_id: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Resolve text classification query.
        
        Args:
            parent: Parent resolver result
            info: GraphQL resolve info
            text: Text to classify
            model_id: Optional model identifier
            options: Classification options
            
        Returns:
            Classification result
        """
        try:
            # Use DataLoader if available in context
            dataloader = info.context.get('classification_loader')
            if dataloader:
                result = await dataloader.load((text, model_id, options))
            else:
                result = await self.prediction_service.predict(
                    text=text,
                    model_id=model_id,
                    options=options or {}
                )
                
            return {
                'success': True,
                'result': result
            }
        except Exception as e:
            return await self.handle_error(e)
            
    async def resolve_batch_classify(
        self,
        parent: Any,
        info: GraphQLResolveInfo,
        texts: List[str],
        model_id: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Resolve batch text classification query.
        
        Args:
            parent: Parent resolver result
            info: GraphQL resolve info
            texts: List of texts to classify
            model_id: Optional model identifier
            options: Classification options
            
        Returns:
            Batch classification results
        """
        try:
            results = await self.prediction_service.predict_batch(
                texts=texts,
                model_id=model_id,
                options=options or {}
            )
            
            return {
                'success': True,
                'results': results
            }
        except Exception as e:
            return await self.handle_error(e)
            
    @BaseResolver.require_auth
    async def resolve_model(
        self,
        parent: Any,
        info: GraphQLResolveInfo,
        model_id: str
    ) -> Dict[str, Any]:
        """
        Resolve model information query.
        
        Args:
            parent: Parent resolver result
            info: GraphQL resolve info
            model_id: Model identifier
            
        Returns:
            Model information
        """
        try:
            model_info = await self.model_service.get_model_info(model_id)
            return {
                'success': True,
                'model': model_info
            }
        except Exception as e:
            return await self.handle_error(e)
            
    @BaseResolver.require_auth
    async def resolve_models(
        self,
        parent: Any,
        info: GraphQLResolveInfo,
        filter: Optional[Dict[str, Any]] = None,
        pagination: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Resolve models list query with filtering and pagination.
        
        Args:
            parent: Parent resolver result
            info: GraphQL resolve info
            filter: Filter criteria
            pagination: Pagination parameters
            
        Returns:
            List of models with pagination info
        """
        try:
            models = await self.model_service.list_models(
                filter=filter or {},
                pagination=pagination or {'limit': 10, 'offset': 0}
            )
            
            return {
                'success': True,
                'models': models['items'],
                'pagination': models['pagination']
            }
        except Exception as e:
            return await self.handle_error(e)
            
    @BaseResolver.require_auth
    async def resolve_training_job(
        self,
        parent: Any,
        info: GraphQLResolveInfo,
        job_id: str
    ) -> Dict[str, Any]:
        """
        Resolve training job information query.
        
        Args:
            parent: Parent resolver result
            info: GraphQL resolve info
            job_id: Training job identifier
            
        Returns:
            Training job information
        """
        try:
            job_info = await self.training_service.get_job_status(job_id)
            return {
                'success': True,
                'job': job_info
            }
        except Exception as e:
            return await self.handle_error(e)
            
    @BaseResolver.require_auth
    async def resolve_dataset(
        self,
        parent: Any,
        info: GraphQLResolveInfo,
        dataset_id: str
    ) -> Dict[str, Any]:
        """
        Resolve dataset information query.
        
        Args:
            parent: Parent resolver result
            info: GraphQL resolve info
            dataset_id: Dataset identifier
            
        Returns:
            Dataset information
        """
        try:
            dataset_info = await self.data_service.get_dataset_info(dataset_id)
            return {
                'success': True,
                'dataset': dataset_info
            }
        except Exception as e:
            return await self.handle_error(e)


class MutationResolvers(BaseResolver):
    """
    Implementation of GraphQL mutation resolvers.
    
    This class handles all write operations for the GraphQL API,
    implementing transactional patterns and optimistic updates.
    """
    
    @BaseResolver.require_auth
    async def resolve_start_training(
        self,
        parent: Any,
        info: GraphQLResolveInfo,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve start training mutation.
        
        Args:
            parent: Parent resolver result
            info: GraphQL resolve info
            config: Training configuration
            
        Returns:
            Training job information
        """
        try:
            job_id = await self.training_service.start_training(
                config=config,
                user_id=info.context['user']['id']
            )
            
            return {
                'success': True,
                'job_id': job_id,
                'message': 'Training job started successfully'
            }
        except Exception as e:
            return await self.handle_error(e)
            
    @BaseResolver.require_auth
    async def resolve_stop_training(
        self,
        parent: Any,
        info: GraphQLResolveInfo,
        job_id: str
    ) -> Dict[str, Any]:
        """
        Resolve stop training mutation.
        
        Args:
            parent: Parent resolver result
            info: GraphQL resolve info
            job_id: Training job identifier
            
        Returns:
            Operation result
        """
        try:
            await self.training_service.stop_training(job_id)
            
            return {
                'success': True,
                'message': 'Training job stopped successfully'
            }
        except Exception as e:
            return await self.handle_error(e)
            
    @BaseResolver.require_role('admin')
    async def resolve_deploy_model(
        self,
        parent: Any,
        info: GraphQLResolveInfo,
        model_id: str,
        deployment_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve model deployment mutation.
        
        Args:
            parent: Parent resolver result
            info: GraphQL resolve info
            model_id: Model identifier
            deployment_config: Deployment configuration
            
        Returns:
            Deployment information
        """
        try:
            deployment_id = await self.model_service.deploy_model(
                model_id=model_id,
                config=deployment_config
            )
            
            return {
                'success': True,
                'deployment_id': deployment_id,
                'message': 'Model deployed successfully'
            }
        except Exception as e:
            return await self.handle_error(e)
            
    @BaseResolver.require_role('admin')
    async def resolve_delete_model(
        self,
        parent: Any,
        info: GraphQLResolveInfo,
        model_id: str
    ) -> Dict[str, Any]:
        """
        Resolve model deletion mutation.
        
        Args:
            parent: Parent resolver result
            info: GraphQL resolve info
            model_id: Model identifier
            
        Returns:
            Operation result
        """
        try:
            await self.model_service.delete_model(model_id)
            
            return {
                'success': True,
                'message': 'Model deleted successfully'
            }
        except Exception as e:
            return await self.handle_error(e)
            
    @BaseResolver.require_auth
    async def resolve_upload_dataset(
        self,
        parent: Any,
        info: GraphQLResolveInfo,
        dataset: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve dataset upload mutation.
        
        Args:
            parent: Parent resolver result
            info: GraphQL resolve info
            dataset: Dataset information and data
            
        Returns:
            Upload result
        """
        try:
            dataset_id = await self.data_service.upload_dataset(
                dataset=dataset,
                user_id=info.context['user']['id']
            )
            
            return {
                'success': True,
                'dataset_id': dataset_id,
                'message': 'Dataset uploaded successfully'
            }
        except Exception as e:
            return await self.handle_error(e)


class SubscriptionResolvers(BaseResolver):
    """
    Implementation of GraphQL subscription resolvers.
    
    This class handles real-time updates for the GraphQL API,
    implementing WebSocket-based subscriptions for live data.
    """
    
    @BaseResolver.require_auth
    async def resolve_training_progress(
        self,
        parent: Any,
        info: GraphQLResolveInfo,
        job_id: str
    ):
        """
        Resolve training progress subscription.
        
        Args:
            parent: Parent resolver result
            info: GraphQL resolve info
            job_id: Training job identifier
            
        Yields:
            Training progress updates
        """
        try:
            async for update in self.training_service.stream_progress(job_id):
                yield {
                    'job_id': job_id,
                    'progress': update['progress'],
                    'metrics': update.get('metrics', {}),
                    'status': update['status'],
                    'timestamp': datetime.utcnow().isoformat()
                }
        except Exception as e:
            yield {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
            
    @BaseResolver.require_auth
    async def resolve_model_metrics(
        self,
        parent: Any,
        info: GraphQLResolveInfo,
        model_id: str
    ):
        """
        Resolve model metrics subscription.
        
        Args:
            parent: Parent resolver result
            info: GraphQL resolve info
            model_id: Model identifier
            
        Yields:
            Model metrics updates
        """
        try:
            async for metrics in self.model_service.stream_metrics(model_id):
                yield {
                    'model_id': model_id,
                    'metrics': metrics,
                    'timestamp': datetime.utcnow().isoformat()
                }
        except Exception as e:
            yield {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
            
    async def resolve_system_status(
        self,
        parent: Any,
        info: GraphQLResolveInfo
    ):
        """
        Resolve system status subscription.
        
        Args:
            parent: Parent resolver result
            info: GraphQL resolve info
            
        Yields:
            System status updates
        """
        try:
            while True:
                status = await self._get_system_status()
                yield {
                    'status': status,
                    'timestamp': datetime.utcnow().isoformat()
                }
                await asyncio.sleep(5)  # Update every 5 seconds
        except Exception as e:
            yield {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
            
    async def _get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status.
        
        Returns:
            System status information
        """
        return {
            'api': 'healthy',
            'services': await self._check_services_health(),
            'models': await self.model_service.get_active_models_count(),
            'jobs': await self.training_service.get_active_jobs_count()
        }
        
    async def _check_services_health(self) -> Dict[str, str]:
        """
        Check health status of all services.
        
        Returns:
            Services health status
        """
        services = {
            'prediction': self.prediction_service,
            'training': self.training_service,
            'data': self.data_service,
            'model': self.model_service
        }
        
        health_status = {}
        for name, service in services.items():
            try:
                await service.health_check()
                health_status[name] = 'healthy'
            except Exception:
                health_status[name] = 'unhealthy'
                
        return health_status


# Initialize resolver instances
query_resolvers = QueryResolvers()
mutation_resolvers = MutationResolvers()
subscription_resolvers = SubscriptionResolvers()

# Export resolver mappings
resolvers = {
    'Query': {
        'classifyText': query_resolvers.resolve_classify_text,
        'batchClassify': query_resolvers.resolve_batch_classify,
        'model': query_resolvers.resolve_model,
        'models': query_resolvers.resolve_models,
        'trainingJob': query_resolvers.resolve_training_job,
        'dataset': query_resolvers.resolve_dataset
    },
    'Mutation': {
        'startTraining': mutation_resolvers.resolve_start_training,
        'stopTraining': mutation_resolvers.resolve_stop_training,
        'deployModel': mutation_resolvers.resolve_deploy_model,
        'deleteModel': mutation_resolvers.resolve_delete_model,
        'uploadDataset': mutation_resolvers.resolve_upload_dataset
    },
    'Subscription': {
        'trainingProgress': subscription_resolvers.resolve_training_progress,
        'modelMetrics': subscription_resolvers.resolve_model_metrics,
        'systemStatus': subscription_resolvers.resolve_system_status
    }
}
