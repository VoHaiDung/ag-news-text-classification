"""
Training gRPC Service
================================================================================
This module implements the gRPC service for model training management,
providing training job control and monitoring capabilities.

Implements service methods for:
- Starting and stopping training jobs
- Monitoring training progress
- Managing training configurations
- Hyperparameter optimization

References:
    - gRPC Python Documentation
    - Distributed Training Best Practices
    - Goodfellow, I., et al. (2016). Deep Learning

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
import logging
from typing import Iterator, Dict, Any
import grpc
import time
from uuid import uuid4
from datetime import datetime

from . import BaseGRPCService
from ..protos import training_pb2, training_pb2_grpc
from ..protos.common import types_pb2, status_pb2
from ....services.core.training_service import TrainingService as CoreTrainingService
from ....core.exceptions import (
    ResourceExhaustedError,
    TrainingError,
    DataValidationError
)

logger = logging.getLogger(__name__)

class TrainingService(
    BaseGRPCService,
    training_pb2_grpc.TrainingServiceServicer
):
    """
    gRPC service implementation for model training.
    
    Provides training job management with support for:
    - Distributed training coordination
    - Real-time progress monitoring
    - Hyperparameter optimization
    - Checkpoint management
    """
    
    def __init__(self):
        """Initialize training service."""
        super().__init__("TrainingService")
        self.core_service = CoreTrainingService()
        self.active_jobs = {}
        
    def register(self, server: grpc.Server) -> None:
        """
        Register service with gRPC server.
        
        Args:
            server: gRPC server instance
        """
        training_pb2_grpc.add_TrainingServiceServicer_to_server(
            self,
            server
        )
        logger.info(f"Registered {self.service_name}")
    
    def StartTraining(
        self,
        request: training_pb2.StartTrainingRequest,
        context: grpc.ServicerContext
    ) -> training_pb2.StartTrainingResponse:
        """
        Start a new training job.
        
        Args:
            request: Training configuration request
            context: gRPC context
            
        Returns:
            training_pb2.StartTrainingResponse: Training job information
        """
        try:
            # Validate request
            if not self.validate_request(request, ['model_type', 'dataset_id']):
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Model type and dataset ID are required"
                )
            
            # Generate job ID
            job_id = str(uuid4())
            
            # Build training configuration
            config = {
                'model_type': request.model_type,
                'dataset_id': request.dataset_id,
                'epochs': request.epochs or 10,
                'batch_size': request.batch_size or 32,
                'learning_rate': request.learning_rate or 2e-5,
                'validation_split': request.validation_split or 0.2,
                'early_stopping': request.early_stopping,
                'checkpoint_interval': request.checkpoint_interval or 1
            }
            
            # Add optimizer configuration
            if request.HasField('optimizer_config'):
                config['optimizer'] = {
                    'type': request.optimizer_config.type,
                    'parameters': dict(request.optimizer_config.parameters)
                }
            
            # Add scheduler configuration
            if request.HasField('scheduler_config'):
                config['scheduler'] = {
                    'type': request.scheduler_config.type,
                    'parameters': dict(request.scheduler_config.parameters)
                }
            
            # Start training job
            job_info = self.core_service.start_training(job_id, config)
            
            # Track active job
            self.active_jobs[job_id] = {
                'status': 'RUNNING',
                'started_at': time.time(),
                'config': config
            }
            
            # Build response
            response = training_pb2.StartTrainingResponse(
                job=training_pb2.TrainingJob(
                    job_id=job_id,
                    status=training_pb2.TrainingStatus.RUNNING,
                    model_type=request.model_type,
                    dataset_id=request.dataset_id,
                    epochs=config['epochs'],
                    current_epoch=0,
                    started_at=int(time.time())
                ),
                status=status_pb2.Status(
                    code=status_pb2.StatusCode.OK,
                    message="Training job started successfully"
                )
            )
            
            self.increment_success_metric()
            logger.info(f"Training job started: {job_id}")
            return response
            
        except ResourceExhaustedError as e:
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(e))
        except Exception as e:
            logger.error(f"Start training error: {e}")
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to start training")
    
    def StopTraining(
        self,
        request: training_pb2.StopTrainingRequest,
        context: grpc.ServicerContext
    ) -> training_pb2.StopTrainingResponse:
        """
        Stop a running training job.
        
        Args:
            request: Stop training request
            context: gRPC context
            
        Returns:
            training_pb2.StopTrainingResponse: Stop result
        """
        try:
            # Validate request
            if not request.job_id:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Job ID is required"
                )
            
            # Check if job exists
            if request.job_id not in self.active_jobs:
                context.abort(
                    grpc.StatusCode.NOT_FOUND,
                    f"Training job not found: {request.job_id}"
                )
            
            # Stop training
            success = self.core_service.stop_training(
                request.job_id,
                save_checkpoint=request.save_checkpoint
            )
            
            if success:
                # Update job status
                self.active_jobs[request.job_id]['status'] = 'STOPPED'
                
                response = training_pb2.StopTrainingResponse(
                    success=True,
                    message="Training job stopped successfully",
                    status=status_pb2.Status(
                        code=status_pb2.StatusCode.OK,
                        message="Job stopped"
                    )
                )
            else:
                response = training_pb2.StopTrainingResponse(
                    success=False,
                    message="Failed to stop training job",
                    status=status_pb2.Status(
                        code=status_pb2.StatusCode.INTERNAL,
                        message="Stop failed"
                    )
                )
            
            self.increment_success_metric()
            return response
            
        except Exception as e:
            logger.error(f"Stop training error: {e}")
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to stop training")
    
    def GetTrainingStatus(
        self,
        request: training_pb2.GetTrainingStatusRequest,
        context: grpc.ServicerContext
    ) -> training_pb2.GetTrainingStatusResponse:
        """
        Get training job status.
        
        Args:
            request: Status request
            context: gRPC context
            
        Returns:
            training_pb2.GetTrainingStatusResponse: Job status
        """
        try:
            # Validate request
            if not request.job_id:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Job ID is required"
                )
            
            # Get job status
            job_info = self.core_service.get_training_status(request.job_id)
            
            if not job_info:
                context.abort(
                    grpc.StatusCode.NOT_FOUND,
                    f"Training job not found: {request.job_id}"
                )
            
            # Build response
            job = training_pb2.TrainingJob(
                job_id=request.job_id,
                status=self._map_training_status(job_info['status']),
                model_type=job_info['model_type'],
                dataset_id=job_info['dataset_id'],
                epochs=job_info['epochs'],
                current_epoch=job_info.get('current_epoch', 0),
                started_at=int(job_info['started_at']),
                current_loss=job_info.get('current_loss', 0.0),
                best_validation_score=job_info.get('best_validation_score', 0.0)
            )
            
            # Add metrics if available
            if 'metrics' in job_info:
                for key, value in job_info['metrics'].items():
                    job.metrics[key] = value
            
            # Add completed time if finished
            if job_info.get('completed_at'):
                job.completed_at = int(job_info['completed_at'])
            
            response = training_pb2.GetTrainingStatusResponse(
                job=job,
                status=status_pb2.Status(
                    code=status_pb2.StatusCode.OK,
                    message="Status retrieved"
                )
            )
            
            self.increment_success_metric()
            return response
            
        except Exception as e:
            logger.error(f"Get training status error: {e}")
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to get status")
    
    def StreamTrainingProgress(
        self,
        request: training_pb2.StreamProgressRequest,
        context: grpc.ServicerContext
    ) -> Iterator[training_pb2.TrainingProgress]:
        """
        Stream training progress updates.
        
        Args:
            request: Stream request
            context: gRPC context
            
        Yields:
            training_pb2.TrainingProgress: Progress updates
        """
        try:
            # Validate request
            if not request.job_id:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Job ID is required"
                )
            
            # Stream progress updates
            update_interval = request.update_interval or 1.0
            
            while context.is_active():
                # Get current progress
                progress = self.core_service.get_training_progress(request.job_id)
                
                if not progress:
                    break
                
                # Build progress message
                progress_msg = training_pb2.TrainingProgress(
                    job_id=request.job_id,
                    current_epoch=progress['current_epoch'],
                    total_epochs=progress['total_epochs'],
                    current_step=progress.get('current_step', 0),
                    total_steps=progress.get('total_steps', 0),
                    current_loss=progress.get('current_loss', 0.0),
                    learning_rate=progress.get('learning_rate', 0.0),
                    timestamp=int(time.time())
                )
                
                # Add metrics
                if 'metrics' in progress:
                    for key, value in progress['metrics'].items():
                        progress_msg.metrics[key] = value
                
                # Add ETA if available
                if 'eta_seconds' in progress:
                    progress_msg.eta_seconds = progress['eta_seconds']
                
                yield progress_msg
                
                # Check if training completed
                if progress.get('status') in ['COMPLETED', 'FAILED', 'STOPPED']:
                    break
                
                # Wait for next update
                time.sleep(update_interval)
                
        except Exception as e:
            logger.error(f"Stream progress error: {e}")
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.INTERNAL, "Streaming failed")
    
    def ListTrainingJobs(
        self,
        request: training_pb2.ListTrainingJobsRequest,
        context: grpc.ServicerContext
    ) -> training_pb2.ListTrainingJobsResponse:
        """
        List training jobs with filtering.
        
        Args:
            request: List request
            context: gRPC context
            
        Returns:
            training_pb2.ListTrainingJobsResponse: Job list
        """
        try:
            # Build filters
            filters = {}
            if request.status:
                filters['status'] = request.status
            if request.model_type:
                filters['model_type'] = request.model_type
            
            # Get jobs
            jobs = self.core_service.list_training_jobs(
                filters=filters,
                limit=request.limit or 10,
                offset=request.offset or 0
            )
            
            # Build response
            response = training_pb2.ListTrainingJobsResponse(
                total_count=len(jobs),
                status=status_pb2.Status(
                    code=status_pb2.StatusCode.OK,
                    message="Jobs retrieved"
                )
            )
            
            for job_info in jobs:
                job = training_pb2.TrainingJob(
                    job_id=job_info['job_id'],
                    status=self._map_training_status(job_info['status']),
                    model_type=job_info['model_type'],
                    dataset_id=job_info['dataset_id'],
                    epochs=job_info['epochs'],
                    current_epoch=job_info.get('current_epoch', 0),
                    started_at=int(job_info['started_at'])
                )
                response.jobs.append(job)
            
            self.increment_success_metric()
            return response
            
        except Exception as e:
            logger.error(f"List training jobs error: {e}")
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to list jobs")
    
    def GetTrainingLogs(
        self,
        request: training_pb2.GetTrainingLogsRequest,
        context: grpc.ServicerContext
    ) -> training_pb2.GetTrainingLogsResponse:
        """
        Get training job logs.
        
        Args:
            request: Logs request
            context: gRPC context
            
        Returns:
            training_pb2.GetTrainingLogsResponse: Training logs
        """
        try:
            # Validate request
            if not request.job_id:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Job ID is required"
                )
            
            # Get logs
            logs = self.core_service.get_training_logs(
                request.job_id,
                lines=request.lines or 100,
                level=request.log_level or "INFO"
            )
            
            # Build response
            response = training_pb2.GetTrainingLogsResponse(
                status=status_pb2.Status(
                    code=status_pb2.StatusCode.OK,
                    message="Logs retrieved"
                )
            )
            
            for log_entry in logs:
                response.logs.append(training_pb2.LogEntry(
                    timestamp=int(log_entry['timestamp']),
                    level=log_entry['level'],
                    message=log_entry['message']
                ))
            
            self.increment_success_metric()
            return response
            
        except Exception as e:
            logger.error(f"Get training logs error: {e}")
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to get logs")
    
    def _map_training_status(self, status: str) -> training_pb2.TrainingStatus:
        """
        Map internal status to protobuf enum.
        
        Args:
            status: Internal status string
            
        Returns:
            training_pb2.TrainingStatus: Protobuf status enum
        """
        status_map = {
            'PENDING': training_pb2.TrainingStatus.PENDING,
            'RUNNING': training_pb2.TrainingStatus.RUNNING,
            'COMPLETED': training_pb2.TrainingStatus.COMPLETED,
            'FAILED': training_pb2.TrainingStatus.FAILED,
            'STOPPED': training_pb2.TrainingStatus.STOPPED
        }
        return status_map.get(status, training_pb2.TrainingStatus.UNKNOWN)
    
    async def cleanup(self) -> None:
        """Cleanup service resources."""
        # Stop all active jobs
        for job_id in list(self.active_jobs.keys()):
            try:
                self.core_service.stop_training(job_id, save_checkpoint=True)
            except Exception as e:
                logger.error(f"Error stopping job {job_id}: {e}")
        
        self.active_jobs.clear()
        
        # Cleanup core service
        if hasattr(self.core_service, 'cleanup'):
            await self.core_service.cleanup()
        
        logger.info(f"{self.service_name} cleaned up")
