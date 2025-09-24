"""
Model Management gRPC Service
================================================================================
This module implements the gRPC service for model management, providing
model lifecycle operations and deployment control.

Implements service methods for:
- Model registration and versioning
- Model deployment and serving
- Model evaluation and comparison
- Model export and import

References:
    - gRPC Python Documentation
    - MLOps Best Practices
    - Sculley, D., et al. (2015). Hidden Technical Debt in Machine Learning Systems

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Iterator, Dict, Any
import grpc
import time
from datetime import datetime

from . import BaseGRPCService
from ..protos import model_management_pb2, model_management_pb2_grpc
from ..protos.common import types_pb2, status_pb2
from ....services.core.model_management_service import ModelManagementService as CoreModelService
from ....core.exceptions import (
    ModelNotFoundError,
    ModelDeploymentError,
    ModelVersionError
)

logger = logging.getLogger(__name__)

class ModelManagementService(
    BaseGRPCService,
    model_management_pb2_grpc.ModelManagementServiceServicer
):
    """
    gRPC service implementation for model management.
    
    Provides model lifecycle management with support for:
    - Model versioning and registry
    - Deployment orchestration
    - A/B testing and canary deployments
    - Model monitoring and governance
    """
    
    def __init__(self):
        """Initialize model management service."""
        super().__init__("ModelManagementService")
        self.core_service = CoreModelService()
        self.deployment_cache = {}
        
    def register(self, server: grpc.Server) -> None:
        """
        Register service with gRPC server.
        
        Args:
            server: gRPC server instance
        """
        model_management_pb2_grpc.add_ModelManagementServiceServicer_to_server(
            self,
            server
        )
        logger.info(f"Registered {self.service_name}")
    
    def RegisterModel(
        self,
        request: model_management_pb2.RegisterModelRequest,
        context: grpc.ServicerContext
    ) -> model_management_pb2.RegisterModelResponse:
        """
        Register a new model in the registry.
        
        Args:
            request: Model registration request
            context: gRPC context
            
        Returns:
            model_management_pb2.RegisterModelResponse: Registration result
        """
        try:
            # Validate request
            if not self.validate_request(request, ['name', 'type', 'path']):
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Model name, type, and path are required"
                )
            
            # Build model metadata
            metadata = {
                'name': request.name,
                'type': request.type,
                'path': request.path,
                'version': request.version or '1.0.0',
                'description': request.description,
                'framework': request.framework,
                'framework_version': request.framework_version,
                'training_job_id': request.training_job_id,
                'tags': list(request.tags)
            }
            
            # Add metrics if provided
            if request.HasField('metrics'):
                metadata['metrics'] = {
                    'accuracy': request.metrics.accuracy,
                    'f1_score': request.metrics.f1_score,
                    'precision': request.metrics.precision,
                    'recall': request.metrics.recall
                }
            
            # Register model
            model_info = self.core_service.register_model(metadata)
            
            # Build response
            response = model_management_pb2.RegisterModelResponse(
                model=model_management_pb2.Model(
                    model_id=model_info['model_id'],
                    name=model_info['name'],
                    type=model_info['type'],
                    version=model_info['version'],
                    status=model_management_pb2.ModelStatus.REGISTERED,
                    created_at=int(time.time())
                ),
                status=status_pb2.Status(
                    code=status_pb2.StatusCode.OK,
                    message="Model registered successfully"
                )
            )
            
            self.increment_success_metric()
            logger.info(f"Model registered: {model_info['model_id']}")
            return response
            
        except Exception as e:
            logger.error(f"Model registration error: {e}")
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to register model")
    
    def DeployModel(
        self,
        request: model_management_pb2.DeployModelRequest,
        context: grpc.ServicerContext
    ) -> model_management_pb2.DeployModelResponse:
        """
        Deploy a model for serving.
        
        Args:
            request: Deployment request
            context: gRPC context
            
        Returns:
            model_management_pb2.DeployModelResponse: Deployment result
        """
        try:
            # Validate request
            if not request.model_id:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Model ID is required"
                )
            
            # Build deployment configuration
            deployment_config = {
                'model_id': request.model_id,
                'deployment_name': request.deployment_name or f"deployment-{request.model_id}",
                'replicas': request.replicas or 1,
                'cpu_limit': request.cpu_limit or "1",
                'memory_limit': request.memory_limit or "2Gi",
                'gpu_required': request.gpu_required,
                'autoscaling_enabled': request.autoscaling_enabled,
                'min_replicas': request.min_replicas or 1,
                'max_replicas': request.max_replicas or 10,
                'target_cpu_utilization': request.target_cpu_utilization or 80
            }
            
            # Deploy model
            deployment_info = self.core_service.deploy_model(deployment_config)
            
            # Cache deployment info
            self.deployment_cache[request.model_id] = deployment_info
            
            # Build response
            response = model_management_pb2.DeployModelResponse(
                deployment=model_management_pb2.Deployment(
                    deployment_id=deployment_info['deployment_id'],
                    model_id=request.model_id,
                    deployment_name=deployment_info['deployment_name'],
                    status=model_management_pb2.DeploymentStatus.DEPLOYING,
                    endpoint=deployment_info.get('endpoint', ''),
                    created_at=int(time.time())
                ),
                status=status_pb2.Status(
                    code=status_pb2.StatusCode.OK,
                    message="Model deployment initiated"
                )
            )
            
            self.increment_success_metric()
            logger.info(f"Model deployment started: {deployment_info['deployment_id']}")
            return response
            
        except ModelNotFoundError as e:
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.NOT_FOUND, str(e))
        except ModelDeploymentError as e:
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
        except Exception as e:
            logger.error(f"Model deployment error: {e}")
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to deploy model")
    
    def UndeployModel(
        self,
        request: model_management_pb2.UndeployModelRequest,
        context: grpc.ServicerContext
    ) -> model_management_pb2.UndeployModelResponse:
        """
        Undeploy a model from serving.
        
        Args:
            request: Undeploy request
            context: gRPC context
            
        Returns:
            model_management_pb2.UndeployModelResponse: Undeploy result
        """
        try:
            # Validate request
            if not request.deployment_id:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Deployment ID is required"
                )
            
            # Undeploy model
            success = self.core_service.undeploy_model(
                request.deployment_id,
                force=request.force
            )
            
            if success:
                # Remove from cache
                for model_id, deployment in list(self.deployment_cache.items()):
                    if deployment.get('deployment_id') == request.deployment_id:
                        del self.deployment_cache[model_id]
                        break
                
                response = model_management_pb2.UndeployModelResponse(
                    success=True,
                    message="Model undeployed successfully",
                    status=status_pb2.Status(
                        code=status_pb2.StatusCode.OK,
                        message="Undeployed"
                    )
                )
            else:
                response = model_management_pb2.UndeployModelResponse(
                    success=False,
                    message="Failed to undeploy model",
                    status=status_pb2.Status(
                        code=status_pb2.StatusCode.INTERNAL,
                        message="Undeploy failed"
                    )
                )
            
            self.increment_success_metric()
            return response
            
        except Exception as e:
            logger.error(f"Model undeploy error: {e}")
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to undeploy model")
    
    def GetModel(
        self,
        request: model_management_pb2.GetModelRequest,
        context: grpc.ServicerContext
    ) -> model_management_pb2.GetModelResponse:
        """
        Get model information.
        
        Args:
            request: Get model request
            context: gRPC context
            
        Returns:
            model_management_pb2.GetModelResponse: Model information
        """
        try:
            # Validate request
            if not request.model_id:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Model ID is required"
                )
            
            # Get model info
            model_info = self.core_service.get_model(request.model_id)
            
            if not model_info:
                context.abort(
                    grpc.StatusCode.NOT_FOUND,
                    f"Model not found: {request.model_id}"
                )
            
            # Build response
            model = model_management_pb2.Model(
                model_id=model_info['model_id'],
                name=model_info['name'],
                type=model_info['type'],
                version=model_info['version'],
                status=self._map_model_status(model_info['status']),
                created_at=int(model_info['created_at']),
                path=model_info.get('path', ''),
                framework=model_info.get('framework', ''),
                framework_version=model_info.get('framework_version', '')
            )
            
            # Add metrics if available
            if 'metrics' in model_info:
                model.metrics.CopyFrom(model_management_pb2.ModelMetrics(
                    accuracy=model_info['metrics'].get('accuracy', 0.0),
                    f1_score=model_info['metrics'].get('f1_score', 0.0),
                    precision=model_info['metrics'].get('precision', 0.0),
                    recall=model_info['metrics'].get('recall', 0.0)
                ))
            
            # Add deployment info if deployed
            if request.model_id in self.deployment_cache:
                deployment_info = self.deployment_cache[request.model_id]
                model.deployment.CopyFrom(model_management_pb2.Deployment(
                    deployment_id=deployment_info['deployment_id'],
                    model_id=request.model_id,
                    deployment_name=deployment_info['deployment_name'],
                    status=model_management_pb2.DeploymentStatus.DEPLOYED,
                    endpoint=deployment_info.get('endpoint', '')
                ))
            
            response = model_management_pb2.GetModelResponse(
                model=model,
                status=status_pb2.Status(
                    code=status_pb2.StatusCode.OK,
                    message="Model retrieved"
                )
            )
            
            self.increment_success_metric()
            return response
            
        except Exception as e:
            logger.error(f"Get model error: {e}")
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to get model")
    
    def ListModels(
        self,
        request: model_management_pb2.ListModelsRequest,
        context: grpc.ServicerContext
    ) -> model_management_pb2.ListModelsResponse:
        """
        List models with filtering.
        
        Args:
            request: List request
            context: gRPC context
            
        Returns:
            model_management_pb2.ListModelsResponse: Model list
        """
        try:
            # Build filters
            filters = {}
            if request.type:
                filters['type'] = request.type
            if request.status:
                filters['status'] = request.status
            if request.deployed_only:
                filters['deployed'] = True
            
            # Get models
            models = self.core_service.list_models(
                filters=filters,
                limit=request.limit or 10,
                offset=request.offset or 0
            )
            
            # Build response
            response = model_management_pb2.ListModelsResponse(
                total_count=len(models),
                status=status_pb2.Status(
                    code=status_pb2.StatusCode.OK,
                    message="Models retrieved"
                )
            )
            
            for model_info in models:
                model = model_management_pb2.Model(
                    model_id=model_info['model_id'],
                    name=model_info['name'],
                    type=model_info['type'],
                    version=model_info['version'],
                    status=self._map_model_status(model_info['status']),
                    created_at=int(model_info['created_at'])
                )
                response.models.append(model)
            
            self.increment_success_metric()
            return response
            
        except Exception as e:
            logger.error(f"List models error: {e}")
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to list models")
    
    def CompareModels(
        self,
        request: model_management_pb2.CompareModelsRequest,
        context: grpc.ServicerContext
    ) -> model_management_pb2.CompareModelsResponse:
        """
        Compare multiple models.
        
        Args:
            request: Compare request
            context: gRPC context
            
        Returns:
            model_management_pb2.CompareModelsResponse: Comparison results
        """
        try:
            # Validate request
            if len(request.model_ids) < 2:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "At least 2 models required for comparison"
                )
            
            # Compare models
            comparison = self.core_service.compare_models(
                list(request.model_ids),
                metrics=list(request.metrics) if request.metrics else None
            )
            
            # Build response
            response = model_management_pb2.CompareModelsResponse(
                status=status_pb2.Status(
                    code=status_pb2.StatusCode.OK,
                    message="Comparison complete"
                )
            )
            
            for model_id, metrics in comparison.items():
                comparison_result = model_management_pb2.ModelComparison(
                    model_id=model_id
                )
                for metric_name, value in metrics.items():
                    comparison_result.metrics[metric_name] = value
                
                response.comparisons.append(comparison_result)
            
            # Add winner if clear
            if comparison:
                best_model = max(
                    comparison.items(),
                    key=lambda x: x[1].get('accuracy', 0)
                )
                response.best_model_id = best_model[0]
            
            self.increment_success_metric()
            return response
            
        except Exception as e:
            logger.error(f"Compare models error: {e}")
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to compare models")
    
    def _map_model_status(self, status: str) -> model_management_pb2.ModelStatus:
        """
        Map internal status to protobuf enum.
        
        Args:
            status: Internal status string
            
        Returns:
            model_management_pb2.ModelStatus: Protobuf status enum
        """
        status_map = {
            'REGISTERED': model_management_pb2.ModelStatus.REGISTERED,
            'VALIDATED': model_management_pb2.ModelStatus.VALIDATED,
            'DEPLOYED': model_management_pb2.ModelStatus.DEPLOYED,
            'DEPRECATED': model_management_pb2.ModelStatus.DEPRECATED,
            'ARCHIVED': model_management_pb2.ModelStatus.ARCHIVED
        }
        return status_map.get(status, model_management_pb2.ModelStatus.UNKNOWN)
    
    async def cleanup(self) -> None:
        """Cleanup service resources."""
        # Clear deployment cache
        self.deployment_cache.clear()
        
        # Cleanup core service
        if hasattr(self.core_service, 'cleanup'):
            await self.core_service.cleanup()
        
        logger.info(f"{self.service_name} cleaned up")
