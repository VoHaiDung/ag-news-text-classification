"""
Data Management gRPC Service
================================================================================
This module implements the gRPC service for data management, providing
dataset operations, data preprocessing, and augmentation capabilities.

Implements service methods for:
- Dataset upload and management
- Data preprocessing pipelines
- Data augmentation strategies
- Data quality validation

References:
    - gRPC Python Documentation
    - Data-Centric AI Principles
    - Ng, A. (2021). MLOps: From Model-centric to Data-centric AI

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Iterator, Dict, Any, List
import grpc
import time
import json
from datetime import datetime
import base64

from . import BaseGRPCService
from ..protos import data_service_pb2, data_service_pb2_grpc
from ..protos.common import types_pb2, status_pb2
from ....services.core.data_service import DataService as CoreDataService
from ....core.exceptions import (
    DataValidationError,
    DatasetNotFoundError,
    StorageError
)

logger = logging.getLogger(__name__)

class DataService(
    BaseGRPCService,
    data_service_pb2_grpc.DataServiceServicer
):
    """
    gRPC service implementation for data management.
    
    Provides comprehensive data operations with support for:
    - Multi-format data ingestion
    - Data versioning and lineage
    - Quality validation and profiling
    - Augmentation and preprocessing
    """
    
    def __init__(self):
        """Initialize data service."""
        super().__init__("DataService")
        self.core_service = CoreDataService()
        self.upload_sessions = {}
        
    def register(self, server: grpc.Server) -> None:
        """
        Register service with gRPC server.
        
        Args:
            server: gRPC server instance
        """
        data_service_pb2_grpc.add_DataServiceServicer_to_server(
            self,
            server
        )
        logger.info(f"Registered {self.service_name}")
    
    def UploadDataset(
        self,
        request_iterator: Iterator[data_service_pb2.UploadDatasetRequest],
        context: grpc.ServicerContext
    ) -> data_service_pb2.UploadDatasetResponse:
        """
        Upload dataset using streaming.
        
        Args:
            request_iterator: Stream of upload chunks
            context: gRPC context
            
        Returns:
            data_service_pb2.UploadDatasetResponse: Upload result
        """
        try:
            dataset_metadata = None
            data_chunks = []
            total_bytes = 0
            
            # Process streaming chunks
            for request in request_iterator:
                if request.HasField('metadata'):
                    # First chunk contains metadata
                    dataset_metadata = {
                        'name': request.metadata.name,
                        'description': request.metadata.description,
                        'format': request.metadata.format,
                        'split': request.metadata.split,
                        'labels': list(request.metadata.labels),
                        'source': request.metadata.source
                    }
                
                if request.chunk_data:
                    # Data chunk
                    data_chunks.append(request.chunk_data)
                    total_bytes += len(request.chunk_data)
            
            # Validate metadata
            if not dataset_metadata:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Dataset metadata is required"
                )
            
            # Combine chunks
            full_data = b''.join(data_chunks)
            
            # Create dataset
            dataset_info = self.core_service.create_dataset(
                metadata=dataset_metadata,
                data=full_data
            )
            
            # Build response
            response = data_service_pb2.UploadDatasetResponse(
                dataset=data_service_pb2.Dataset(
                    dataset_id=dataset_info['dataset_id'],
                    name=dataset_info['name'],
                    size=dataset_info['size'],
                    num_samples=dataset_info['num_samples'],
                    created_at=int(time.time()),
                    status=data_service_pb2.DatasetStatus.READY
                ),
                bytes_received=total_bytes,
                status=status_pb2.Status(
                    code=status_pb2.StatusCode.OK,
                    message="Dataset uploaded successfully"
                )
            )
            
            self.increment_success_metric()
            logger.info(f"Dataset uploaded: {dataset_info['dataset_id']}")
            return response
            
        except DataValidationError as e:
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except StorageError as e:
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(e))
        except Exception as e:
            logger.error(f"Dataset upload error: {e}")
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to upload dataset")
    
    def GetDataset(
        self,
        request: data_service_pb2.GetDatasetRequest,
        context: grpc.ServicerContext
    ) -> data_service_pb2.GetDatasetResponse:
        """
        Get dataset information.
        
        Args:
            request: Get dataset request
            context: gRPC context
            
        Returns:
            data_service_pb2.GetDatasetResponse: Dataset information
        """
        try:
            # Validate request
            if not request.dataset_id:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Dataset ID is required"
                )
            
            # Get dataset info
            dataset_info = self.core_service.get_dataset(request.dataset_id)
            
            if not dataset_info:
                context.abort(
                    grpc.StatusCode.NOT_FOUND,
                    f"Dataset not found: {request.dataset_id}"
                )
            
            # Build response
            dataset = data_service_pb2.Dataset(
                dataset_id=dataset_info['dataset_id'],
                name=dataset_info['name'],
                description=dataset_info.get('description', ''),
                size=dataset_info['size'],
                num_samples=dataset_info['num_samples'],
                created_at=int(dataset_info['created_at']),
                updated_at=int(dataset_info.get('updated_at', dataset_info['created_at'])),
                status=self._map_dataset_status(dataset_info['status']),
                format=dataset_info.get('format', ''),
                split=dataset_info.get('split', '')
            )
            
            # Add statistics if requested
            if request.include_statistics:
                stats = self.core_service.get_dataset_statistics(request.dataset_id)
                if stats:
                    dataset.statistics.CopyFrom(self._build_statistics(stats))
            
            response = data_service_pb2.GetDatasetResponse(
                dataset=dataset,
                status=status_pb2.Status(
                    code=status_pb2.StatusCode.OK,
                    message="Dataset retrieved"
                )
            )
            
            self.increment_success_metric()
            return response
            
        except Exception as e:
            logger.error(f"Get dataset error: {e}")
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to get dataset")
    
    def ListDatasets(
        self,
        request: data_service_pb2.ListDatasetsRequest,
        context: grpc.ServicerContext
    ) -> data_service_pb2.ListDatasetsResponse:
        """
        List datasets with filtering.
        
        Args:
            request: List request
            context: gRPC context
            
        Returns:
            data_service_pb2.ListDatasetsResponse: Dataset list
        """
        try:
            # Build filters
            filters = {}
            if request.split:
                filters['split'] = request.split
            if request.format:
                filters['format'] = request.format
            if request.min_samples:
                filters['min_samples'] = request.min_samples
            
            # Get datasets
            datasets = self.core_service.list_datasets(
                filters=filters,
                limit=request.limit or 10,
                offset=request.offset or 0
            )
            
            # Build response
            response = data_service_pb2.ListDatasetsResponse(
                total_count=len(datasets),
                status=status_pb2.Status(
                    code=status_pb2.StatusCode.OK,
                    message="Datasets retrieved"
                )
            )
            
            for dataset_info in datasets:
                dataset = data_service_pb2.Dataset(
                    dataset_id=dataset_info['dataset_id'],
                    name=dataset_info['name'],
                    size=dataset_info['size'],
                    num_samples=dataset_info['num_samples'],
                    created_at=int(dataset_info['created_at']),
                    status=self._map_dataset_status(dataset_info['status'])
                )
                response.datasets.append(dataset)
            
            self.increment_success_metric()
            return response
            
        except Exception as e:
            logger.error(f"List datasets error: {e}")
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to list datasets")
    
    def DeleteDataset(
        self,
        request: data_service_pb2.DeleteDatasetRequest,
        context: grpc.ServicerContext
    ) -> data_service_pb2.DeleteDatasetResponse:
        """
        Delete a dataset.
        
        Args:
            request: Delete request
            context: gRPC context
            
        Returns:
            data_service_pb2.DeleteDatasetResponse: Delete result
        """
        try:
            # Validate request
            if not request.dataset_id:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Dataset ID is required"
                )
            
            # Delete dataset
            success = self.core_service.delete_dataset(
                request.dataset_id,
                force=request.force
            )
            
            response = data_service_pb2.DeleteDatasetResponse(
                success=success,
                message="Dataset deleted" if success else "Failed to delete dataset",
                status=status_pb2.Status(
                    code=status_pb2.StatusCode.OK if success else status_pb2.StatusCode.INTERNAL,
                    message="Deleted" if success else "Delete failed"
                )
            )
            
            self.increment_success_metric()
            return response
            
        except DatasetNotFoundError as e:
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.NOT_FOUND, str(e))
        except Exception as e:
            logger.error(f"Delete dataset error: {e}")
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to delete dataset")
    
    def PreprocessData(
        self,
        request: data_service_pb2.PreprocessDataRequest,
        context: grpc.ServicerContext
    ) -> data_service_pb2.PreprocessDataResponse:
        """
        Preprocess dataset with specified pipeline.
        
        Args:
            request: Preprocessing request
            context: gRPC context
            
        Returns:
            data_service_pb2.PreprocessDataResponse: Preprocessing result
        """
        try:
            # Validate request
            if not request.dataset_id:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Dataset ID is required"
                )
            
            # Build preprocessing config
            config = {
                'lowercase': request.lowercase,
                'remove_punctuation': request.remove_punctuation,
                'remove_stopwords': request.remove_stopwords,
                'lemmatize': request.lemmatize,
                'max_length': request.max_length or 512,
                'min_length': request.min_length or 10
            }
            
            # Add custom steps if provided
            if request.custom_steps:
                config['custom_steps'] = list(request.custom_steps)
            
            # Run preprocessing
            result = self.core_service.preprocess_dataset(
                request.dataset_id,
                config,
                create_new=request.create_new_dataset
            )
            
            response = data_service_pb2.PreprocessDataResponse(
                dataset_id=result['dataset_id'],
                samples_processed=result['samples_processed'],
                samples_filtered=result.get('samples_filtered', 0),
                processing_time_seconds=result.get('processing_time', 0),
                status=status_pb2.Status(
                    code=status_pb2.StatusCode.OK,
                    message="Preprocessing completed"
                )
            )
            
            self.increment_success_metric()
            logger.info(f"Dataset preprocessed: {result['dataset_id']}")
            return response
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to preprocess data")
    
    def AugmentData(
        self,
        request: data_service_pb2.AugmentDataRequest,
        context: grpc.ServicerContext
    ) -> data_service_pb2.AugmentDataResponse:
        """
        Augment dataset with specified techniques.
        
        Args:
            request: Augmentation request
            context: gRPC context
            
        Returns:
            data_service_pb2.AugmentDataResponse: Augmentation result
        """
        try:
            # Validate request
            if not request.dataset_id:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Dataset ID is required"
                )
            
            # Build augmentation config
            techniques = []
            
            if request.back_translation:
                techniques.append({
                    'type': 'back_translation',
                    'languages': list(request.translation_languages) or ['fr', 'de']
                })
            
            if request.paraphrase:
                techniques.append({
                    'type': 'paraphrase',
                    'num_paraphrases': request.num_paraphrases or 2
                })
            
            if request.synonym_replacement:
                techniques.append({
                    'type': 'synonym_replacement',
                    'replacement_ratio': request.replacement_ratio or 0.2
                })
            
            if request.noise_injection:
                techniques.append({
                    'type': 'noise_injection',
                    'noise_level': request.noise_level or 0.1
                })
            
            # Run augmentation
            result = self.core_service.augment_dataset(
                request.dataset_id,
                techniques,
                augmentation_factor=request.augmentation_factor or 2
            )
            
            response = data_service_pb2.AugmentDataResponse(
                augmented_dataset_id=result['augmented_dataset_id'],
                original_samples=result['original_samples'],
                augmented_samples=result['augmented_samples'],
                total_samples=result['total_samples'],
                status=status_pb2.Status(
                    code=status_pb2.StatusCode.OK,
                    message="Augmentation completed"
                )
            )
            
            self.increment_success_metric()
            logger.info(f"Dataset augmented: {result['augmented_dataset_id']}")
            return response
            
        except Exception as e:
            logger.error(f"Augmentation error: {e}")
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to augment data")
    
    def ValidateData(
        self,
        request: data_service_pb2.ValidateDataRequest,
        context: grpc.ServicerContext
    ) -> data_service_pb2.ValidateDataResponse:
        """
        Validate dataset quality and consistency.
        
        Args:
            request: Validation request
            context: gRPC context
            
        Returns:
            data_service_pb2.ValidateDataResponse: Validation results
        """
        try:
            # Validate request
            if not request.dataset_id:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Dataset ID is required"
                )
            
            # Run validation
            validation_result = self.core_service.validate_dataset(
                request.dataset_id,
                checks=list(request.validation_checks) if request.validation_checks else None
            )
            
            # Build response
            response = data_service_pb2.ValidateDataResponse(
                is_valid=validation_result['is_valid'],
                status=status_pb2.Status(
                    code=status_pb2.StatusCode.OK,
                    message="Validation completed"
                )
            )
            
            # Add validation issues
            for issue in validation_result.get('issues', []):
                response.issues.append(data_service_pb2.ValidationIssue(
                    severity=issue['severity'],
                    category=issue['category'],
                    message=issue['message'],
                    affected_samples=issue.get('affected_samples', 0)
                ))
            
            # Add quality metrics
            if 'quality_metrics' in validation_result:
                metrics = validation_result['quality_metrics']
                response.quality_metrics.CopyFrom(data_service_pb2.QualityMetrics(
                    completeness=metrics.get('completeness', 0.0),
                    consistency=metrics.get('consistency', 0.0),
                    accuracy=metrics.get('accuracy', 0.0),
                    uniqueness=metrics.get('uniqueness', 0.0)
                ))
            
            self.increment_success_metric()
            return response
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to validate data")
    
    def StreamDataSamples(
        self,
        request: data_service_pb2.StreamDataSamplesRequest,
        context: grpc.ServicerContext
    ) -> Iterator[data_service_pb2.DataSample]:
        """
        Stream dataset samples.
        
        Args:
            request: Stream request
            context: gRPC context
            
        Yields:
            data_service_pb2.DataSample: Data samples
        """
        try:
            # Validate request
            if not request.dataset_id:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Dataset ID is required"
                )
            
            # Stream samples
            batch_size = request.batch_size or 10
            offset = 0
            
            while context.is_active():
                # Get batch of samples
                samples = self.core_service.get_samples(
                    request.dataset_id,
                    offset=offset,
                    limit=batch_size
                )
                
                if not samples:
                    break
                
                for sample in samples:
                    yield data_service_pb2.DataSample(
                        sample_id=sample['id'],
                        text=sample['text'],
                        label=sample.get('label', ''),
                        metadata=json.dumps(sample.get('metadata', {}))
                    )
                
                offset += batch_size
                
                # Check if we've reached the limit
                if request.max_samples and offset >= request.max_samples:
                    break
                    
        except Exception as e:
            logger.error(f"Stream samples error: {e}")
            self.handle_error(context, e)
            context.abort(grpc.StatusCode.INTERNAL, "Failed to stream samples")
    
    def _map_dataset_status(self, status: str) -> data_service_pb2.DatasetStatus:
        """
        Map internal status to protobuf enum.
        
        Args:
            status: Internal status string
            
        Returns:
            data_service_pb2.DatasetStatus: Protobuf status enum
        """
        status_map = {
            'UPLOADING': data_service_pb2.DatasetStatus.UPLOADING,
            'PROCESSING': data_service_pb2.DatasetStatus.PROCESSING,
            'READY': data_service_pb2.DatasetStatus.READY,
            'ERROR': data_service_pb2.DatasetStatus.ERROR,
            'ARCHIVED': data_service_pb2.DatasetStatus.ARCHIVED
        }
        return status_map.get(status, data_service_pb2.DatasetStatus.UNKNOWN)
    
    def _build_statistics(self, stats: Dict[str, Any]) -> data_service_pb2.DatasetStatistics:
        """
        Build dataset statistics message.
        
        Args:
            stats: Statistics dictionary
            
        Returns:
            data_service_pb2.DatasetStatistics: Statistics message
        """
        statistics = data_service_pb2.DatasetStatistics(
            total_samples=stats.get('total_samples', 0),
            avg_text_length=stats.get('avg_text_length', 0.0),
            min_text_length=stats.get('min_text_length', 0),
            max_text_length=stats.get('max_text_length', 0),
            vocabulary_size=stats.get('vocabulary_size', 0)
        )
        
        # Add label distribution
        if 'label_distribution' in stats:
            for label, count in stats['label_distribution'].items():
                statistics.label_distribution[label] = count
        
        return statistics
    
    async def cleanup(self) -> None:
        """Cleanup service resources."""
        # Clear upload sessions
        self.upload_sessions.clear()
        
        # Cleanup core service
        if hasattr(self.core_service, 'cleanup'):
            await self.core_service.cleanup()
        
        logger.info(f"{self.service_name} cleaned up")
