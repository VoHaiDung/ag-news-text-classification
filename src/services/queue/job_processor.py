"""
Job Processor Implementation for AG News Text Classification
================================================================================
This module implements job processing strategies for handling various types of
computational tasks with different execution patterns.

The job processor provides:
- Multiple processing strategies (batch, stream, parallel)
- Job result aggregation
- Progress tracking
- Error recovery

References:
    - Dean, J., & Ghemawat, S. (2008). MapReduce: Simplified Data Processing
    - Akidau, T., et al. (2015). The Dataflow Model

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable, Union, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod

from src.services.base_service import BaseService, ServiceConfig
from src.core.exceptions import ServiceException
from src.utils.logging_config import get_logger


class ProcessingStrategy(Enum):
    """
    Job processing strategies.
    
    Strategies:
        SEQUENTIAL: Process jobs one by one
        PARALLEL: Process multiple jobs concurrently
        BATCH: Process jobs in batches
        STREAMING: Process jobs as stream
        MAP_REDUCE: Map-reduce processing pattern
    """
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    BATCH = "batch"
    STREAMING = "streaming"
    MAP_REDUCE = "map_reduce"


@dataclass
class ProcessingResult:
    """
    Result of job processing.
    
    Attributes:
        job_id: Job identifier
        success: Whether processing succeeded
        result: Processing result
        error: Error message if failed
        duration: Processing duration in seconds
        metadata: Additional metadata
    """
    job_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProcessingPipeline(ABC):
    """
    Abstract base class for processing pipelines.
    """
    
    @abstractmethod
    async def process(self, data: Any) -> Any:
        """Process data through pipeline."""
        pass


class JobProcessor(BaseService):
    """
    Service for processing jobs with various strategies.
    
    This service provides flexible job processing with support for
    different execution patterns and result aggregation.
    """
    
    def __init__(
        self,
        config: Optional[ServiceConfig] = None,
        max_concurrent: int = 10,
        batch_size: int = 100
    ):
        """
        Initialize job processor.
        
        Args:
            config: Service configuration
            max_concurrent: Maximum concurrent jobs
            batch_size: Default batch size
        """
        if config is None:
            config = ServiceConfig(name="job_processor")
        super().__init__(config)
        
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        
        # Processing pipelines
        self._pipelines: Dict[str, ProcessingPipeline] = {}
        
        # Job tracking
        self._active_jobs: Dict[str, Any] = {}
        self._completed_jobs: Dict[str, ProcessingResult] = {}
        
        # Progress tracking
        self._progress: Dict[str, Dict[str, Any]] = {}
        
        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
        self.logger = get_logger("service.job_processor")
    
    async def _initialize(self) -> None:
        """Initialize job processor."""
        self.logger.info(f"Initializing job processor with max_concurrent={self.max_concurrent}")
    
    async def _start(self) -> None:
        """Start job processor service."""
        self.logger.info("Job processor started")
    
    async def _stop(self) -> None:
        """Stop job processor service."""
        # Wait for active jobs to complete
        if self._active_jobs:
            self.logger.info(f"Waiting for {len(self._active_jobs)} active jobs to complete")
            await asyncio.gather(
                *self._active_jobs.values(),
                return_exceptions=True
            )
        
        self.logger.info("Job processor stopped")
    
    async def _cleanup(self) -> None:
        """Cleanup processor resources."""
        self._active_jobs.clear()
        self._pipelines.clear()
    
    async def _check_health(self) -> bool:
        """Check processor health."""
        return len(self._active_jobs) <= self.max_concurrent
    
    def register_pipeline(self, name: str, pipeline: ProcessingPipeline) -> None:
        """
        Register a processing pipeline.
        
        Args:
            name: Pipeline name
            pipeline: Pipeline instance
        """
        self._pipelines[name] = pipeline
        self.logger.info(f"Registered pipeline: {name}")
    
    async def process_job(
        self,
        job_id: str,
        data: Any,
        processor: Callable,
        strategy: ProcessingStrategy = ProcessingStrategy.SEQUENTIAL,
        **kwargs
    ) -> ProcessingResult:
        """
        Process a job with specified strategy.
        
        Args:
            job_id: Job identifier
            data: Job data
            processor: Processing function
            strategy: Processing strategy
            **kwargs: Additional strategy parameters
            
        Returns:
            ProcessingResult: Processing result
        """
        start_time = datetime.now()
        
        try:
            # Track active job
            self._active_jobs[job_id] = asyncio.current_task()
            
            # Initialize progress
            self._progress[job_id] = {
                "status": "processing",
                "progress": 0,
                "total": self._estimate_total(data, strategy)
            }
            
            # Process based on strategy
            if strategy == ProcessingStrategy.SEQUENTIAL:
                result = await self._process_sequential(job_id, data, processor)
            elif strategy == ProcessingStrategy.PARALLEL:
                result = await self._process_parallel(job_id, data, processor, **kwargs)
            elif strategy == ProcessingStrategy.BATCH:
                result = await self._process_batch(job_id, data, processor, **kwargs)
            elif strategy == ProcessingStrategy.STREAMING:
                result = await self._process_streaming(job_id, data, processor, **kwargs)
            elif strategy == ProcessingStrategy.MAP_REDUCE:
                result = await self._process_map_reduce(job_id, data, processor, **kwargs)
            else:
                raise ServiceException(f"Unknown processing strategy: {strategy}")
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            # Create result
            processing_result = ProcessingResult(
                job_id=job_id,
                success=True,
                result=result,
                duration=duration,
                metadata={"strategy": strategy.value}
            )
            
            # Update progress
            self._progress[job_id]["status"] = "completed"
            self._progress[job_id]["progress"] = self._progress[job_id]["total"]
            
            self.logger.info(f"Job {job_id} completed in {duration:.2f}s")
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            
            processing_result = ProcessingResult(
                job_id=job_id,
                success=False,
                error=str(e),
                duration=duration,
                metadata={"strategy": strategy.value}
            )
            
            self._progress[job_id]["status"] = "failed"
            
            self.logger.error(f"Job {job_id} failed: {e}")
        
        finally:
            # Clean up
            self._active_jobs.pop(job_id, None)
            self._completed_jobs[job_id] = processing_result
        
        return processing_result
    
    async def _process_sequential(
        self,
        job_id: str,
        data: List[Any],
        processor: Callable
    ) -> List[Any]:
        """
        Process data sequentially.
        
        Args:
            job_id: Job identifier
            data: Data items to process
            processor: Processing function
            
        Returns:
            List[Any]: Processing results
        """
        results = []
        
        for i, item in enumerate(data):
            # Process item
            if asyncio.iscoroutinefunction(processor):
                result = await processor(item)
            else:
                result = processor(item)
            
            results.append(result)
            
            # Update progress
            self._update_progress(job_id, i + 1)
        
        return results
    
    async def _process_parallel(
        self,
        job_id: str,
        data: List[Any],
        processor: Callable,
        max_workers: Optional[int] = None
    ) -> List[Any]:
        """
        Process data in parallel.
        
        Args:
            job_id: Job identifier
            data: Data items to process
            processor: Processing function
            max_workers: Maximum parallel workers
            
        Returns:
            List[Any]: Processing results
        """
        max_workers = max_workers or self.max_concurrent
        semaphore = asyncio.Semaphore(max_workers)
        
        async def process_with_semaphore(item, index):
            async with semaphore:
                if asyncio.iscoroutinefunction(processor):
                    result = await processor(item)
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, processor, item
                    )
                
                # Update progress
                self._update_progress(job_id, index + 1)
                return result
        
        # Process all items in parallel
        tasks = [
            process_with_semaphore(item, i)
            for i, item in enumerate(data)
        ]
        
        results = await asyncio.gather(*tasks)
        return results
    
    async def _process_batch(
        self,
        job_id: str,
        data: List[Any],
        processor: Callable,
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """
        Process data in batches.
        
        Args:
            job_id: Job identifier
            data: Data items to process
            processor: Processing function
            batch_size: Batch size
            
        Returns:
            List[Any]: Processing results
        """
        batch_size = batch_size or self.batch_size
        results = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            # Process batch
            if asyncio.iscoroutinefunction(processor):
                batch_results = await processor(batch)
            else:
                batch_results = processor(batch)
            
            # Handle batch results
            if isinstance(batch_results, list):
                results.extend(batch_results)
            else:
                results.append(batch_results)
            
            # Update progress
            self._update_progress(job_id, min(i + batch_size, len(data)))
        
        return results
    
    async def _process_streaming(
        self,
        job_id: str,
        data: Union[AsyncIterator, List],
        processor: Callable,
        buffer_size: int = 100
    ) -> List[Any]:
        """
        Process data as stream.
        
        Args:
            job_id: Job identifier
            data: Data stream or list
            processor: Processing function
            buffer_size: Stream buffer size
            
        Returns:
            List[Any]: Processing results
        """
        results = []
        buffer = []
        processed = 0
        
        # Convert list to async iterator if needed
        if isinstance(data, list):
            async def list_iterator():
                for item in data:
                    yield item
            
            data = list_iterator()
        
        async for item in data:
            buffer.append(item)
            
            # Process when buffer is full
            if len(buffer) >= buffer_size:
                if asyncio.iscoroutinefunction(processor):
                    batch_results = await processor(buffer)
                else:
                    batch_results = processor(buffer)
                
                results.extend(batch_results if isinstance(batch_results, list) else [batch_results])
                processed += len(buffer)
                self._update_progress(job_id, processed)
                
                buffer.clear()
        
        # Process remaining items
        if buffer:
            if asyncio.iscoroutinefunction(processor):
                batch_results = await processor(buffer)
            else:
                batch_results = processor(buffer)
            
            results.extend(batch_results if isinstance(batch_results, list) else [batch_results])
            processed += len(buffer)
            self._update_progress(job_id, processed)
        
        return results
    
    async def _process_map_reduce(
        self,
        job_id: str,
        data: List[Any],
        processor: Dict[str, Callable],
        num_mappers: int = 4,
        num_reducers: int = 2
    ) -> Any:
        """
        Process data using map-reduce pattern.
        
        Args:
            job_id: Job identifier
            data: Input data
            processor: Dict with 'map' and 'reduce' functions
            num_mappers: Number of map workers
            num_reducers: Number of reduce workers
            
        Returns:
            Any: Reduced result
        """
        if "map" not in processor or "reduce" not in processor:
            raise ServiceException("Map-reduce requires 'map' and 'reduce' functions")
        
        map_fn = processor["map"]
        reduce_fn = processor["reduce"]
        
        # Map phase
        chunk_size = len(data) // num_mappers + 1
        map_tasks = []
        
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            
            async def map_chunk(c):
                results = []
                for item in c:
                    if asyncio.iscoroutinefunction(map_fn):
                        result = await map_fn(item)
                    else:
                        result = map_fn(item)
                    results.append(result)
                return results
            
            map_tasks.append(map_chunk(chunk))
        
        # Execute map tasks
        map_results = await asyncio.gather(*map_tasks)
        
        # Flatten results
        intermediate = []
        for chunk_results in map_results:
            intermediate.extend(chunk_results)
        
        self._update_progress(job_id, len(data) // 2)  # Map phase complete
        
        # Reduce phase
        while len(intermediate) > 1:
            reduce_tasks = []
            
            for i in range(0, len(intermediate), 2):
                if i + 1 < len(intermediate):
                    pair = (intermediate[i], intermediate[i + 1])
                else:
                    pair = (intermediate[i], None)
                
                async def reduce_pair(p):
                    if p[1] is None:
                        return p[0]
                    
                    if asyncio.iscoroutinefunction(reduce_fn):
                        return await reduce_fn(p[0], p[1])
                    else:
                        return reduce_fn(p[0], p[1])
                
                reduce_tasks.append(reduce_pair(pair))
            
            intermediate = await asyncio.gather(*reduce_tasks)
        
        self._update_progress(job_id, len(data))  # Reduce phase complete
        
        return intermediate[0] if intermediate else None
    
    def _estimate_total(self, data: Any, strategy: ProcessingStrategy) -> int:
        """
        Estimate total work units for progress tracking.
        
        Args:
            data: Input data
            strategy: Processing strategy
            
        Returns:
            int: Estimated total units
        """
        if isinstance(data, list):
            return len(data)
        elif hasattr(data, "__len__"):
            return len(data)
        else:
            return 100  # Default estimate
    
    def _update_progress(self, job_id: str, completed: int) -> None:
        """
        Update job progress.
        
        Args:
            job_id: Job identifier
            completed: Number of completed units
        """
        if job_id in self._progress:
            self._progress[job_id]["progress"] = completed
            
            # Calculate percentage
            total = self._progress[job_id]["total"]
            if total > 0:
                percentage = (completed / total) * 100
                self.logger.debug(f"Job {job_id} progress: {percentage:.1f}%")
    
    def get_job_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job progress.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Optional[Dict[str, Any]]: Progress information
        """
        return self._progress.get(job_id)
    
    def get_job_result(self, job_id: str) -> Optional[ProcessingResult]:
        """
        Get job result.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Optional[ProcessingResult]: Job result
        """
        return self._completed_jobs.get(job_id)
    
    def get_active_jobs(self) -> List[str]:
        """
        Get list of active job IDs.
        
        Returns:
            List[str]: Active job IDs
        """
        return list(self._active_jobs.keys())
