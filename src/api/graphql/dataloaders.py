"""
GraphQL DataLoaders
================================================================================
This module implements DataLoader pattern for efficient batch loading and
caching of data, preventing N+1 query problems in GraphQL resolution.

DataLoaders provide:
- Batch loading of related data
- Request-scoped caching
- Deduplication of requests
- Optimized database queries

References:
    - DataLoader Pattern (Facebook)
    - GraphQL DataLoader Specification
    - Fowler, M. (2002). Patterns of Enterprise Application Architecture

Author: Võ Hải Dũng
License: MIT
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import asyncio
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Base DataLoader implementation.
    
    Provides batching and caching for efficient data loading
    following the Facebook DataLoader pattern.
    """
    
    def __init__(self, batch_load_fn, cache: bool = True, max_batch_size: int = 100):
        """
        Initialize DataLoader.
        
        Args:
            batch_load_fn: Async function to batch load data
            cache: Enable caching
            max_batch_size: Maximum batch size
        """
        self.batch_load_fn = batch_load_fn
        self.cache_enabled = cache
        self.max_batch_size = max_batch_size
        
        # Caches
        self._cache = {}
        self._queue = []
        self._batch_promise = None
        
    async def load(self, key: Any) -> Any:
        """
        Load single item by key.
        
        Args:
            key: Item key
            
        Returns:
            Any: Loaded item
        """
        # Check cache
        if self.cache_enabled and key in self._cache:
            return self._cache[key]
        
        # Add to batch queue
        future = asyncio.Future()
        self._queue.append((key, future))
        
        # Schedule batch if needed
        if self._batch_promise is None:
            self._batch_promise = asyncio.create_task(self._dispatch_batch())
        
        # Wait for result
        result = await future
        
        # Cache result
        if self.cache_enabled:
            self._cache[key] = result
        
        return result
    
    async def load_many(self, keys: List[Any]) -> List[Any]:
        """
        Load multiple items by keys.
        
        Args:
            keys: List of keys
            
        Returns:
            List[Any]: Loaded items
        """
        return await asyncio.gather(*[self.load(key) for key in keys])
    
    async def _dispatch_batch(self):
        """Dispatch queued batch load."""
        await asyncio.sleep(0)  # Yield to event loop
        
        # Get current queue
        batch = self._queue[:self.max_batch_size]
        self._queue = self._queue[self.max_batch_size:]
        
        if not batch:
            self._batch_promise = None
            return
        
        # Extract keys
        keys = [item[0] for item in batch]
        futures = [item[1] for item in batch]
        
        try:
            # Batch load
            results = await self.batch_load_fn(keys)
            
            # Map results to futures
            result_map = {key: result for key, result in zip(keys, results)}
            
            for key, future in zip(keys, futures):
                if not future.done():
                    future.set_result(result_map.get(key))
                    
        except Exception as e:
            # Set exception for all futures
            for future in futures:
                if not future.done():
                    future.set_exception(e)
        
        # Continue with remaining queue
        if self._queue:
            self._batch_promise = asyncio.create_task(self._dispatch_batch())
        else:
            self._batch_promise = None
    
    def clear(self, key: Optional[Any] = None):
        """
        Clear cache.
        
        Args:
            key: Specific key to clear, or None for all
        """
        if key is None:
            self._cache.clear()
        else:
            self._cache.pop(key, None)
    
    def prime(self, key: Any, value: Any):
        """
        Prime cache with value.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if self.cache_enabled:
            self._cache[key] = value

class ModelLoader(DataLoader):
    """DataLoader for model data."""
    
    def __init__(self, model_service):
        """
        Initialize model loader.
        
        Args:
            model_service: Model management service
        """
        self.model_service = model_service
        super().__init__(self._batch_load_models)
    
    async def _batch_load_models(self, model_ids: List[str]) -> List[Optional[Dict]]:
        """
        Batch load models by IDs.
        
        Args:
            model_ids: List of model IDs
            
        Returns:
            List[Optional[Dict]]: Model data
        """
        try:
            # Fetch models in batch
            models = await self.model_service.get_models_batch(model_ids)
            
            # Create result map
            model_map = {model['id']: model for model in models}
            
            # Return in requested order
            return [model_map.get(model_id) for model_id in model_ids]
            
        except Exception as e:
            logger.error(f"Batch load models error: {e}")
            return [None] * len(model_ids)

class DatasetLoader(DataLoader):
    """DataLoader for dataset data."""
    
    def __init__(self, data_service):
        """
        Initialize dataset loader.
        
        Args:
            data_service: Data service
        """
        self.data_service = data_service
        super().__init__(self._batch_load_datasets)
    
    async def _batch_load_datasets(self, dataset_ids: List[str]) -> List[Optional[Dict]]:
        """
        Batch load datasets by IDs.
        
        Args:
            dataset_ids: List of dataset IDs
            
        Returns:
            List[Optional[Dict]]: Dataset data
        """
        try:
            # Fetch datasets in batch
            datasets = await self.data_service.get_datasets_batch(dataset_ids)
            
            # Create result map
            dataset_map = {dataset['id']: dataset for dataset in datasets}
            
            # Return in requested order
            return [dataset_map.get(dataset_id) for dataset_id in dataset_ids]
            
        except Exception as e:
            logger.error(f"Batch load datasets error: {e}")
            return [None] * len(dataset_ids)

class TrainingJobLoader(DataLoader):
    """DataLoader for training job data."""
    
    def __init__(self, training_service):
        """
        Initialize training job loader.
        
        Args:
            training_service: Training service
        """
        self.training_service = training_service
        super().__init__(self._batch_load_jobs)
    
    async def _batch_load_jobs(self, job_ids: List[str]) -> List[Optional[Dict]]:
        """
        Batch load training jobs by IDs.
        
        Args:
            job_ids: List of job IDs
            
        Returns:
            List[Optional[Dict]]: Job data
        """
        try:
            # Fetch jobs in batch
            jobs = await self.training_service.get_jobs_batch(job_ids)
            
            # Create result map
            job_map = {job['id']: job for job in jobs}
            
            # Return in requested order
            return [job_map.get(job_id) for job_id in job_ids]
            
        except Exception as e:
            logger.error(f"Batch load training jobs error: {e}")
            return [None] * len(job_ids)

class UserLoader(DataLoader):
    """DataLoader for user data."""
    
    def __init__(self):
        """Initialize user loader."""
        super().__init__(self._batch_load_users)
    
    async def _batch_load_users(self, user_ids: List[str]) -> List[Optional[Dict]]:
        """
        Batch load users by IDs.
        
        Args:
            user_ids: List of user IDs
            
        Returns:
            List[Optional[Dict]]: User data
        """
        try:
            # Placeholder implementation
            # In production, fetch from user service or database
            users = []
            
            for user_id in user_ids:
                users.append({
                    'id': user_id,
                    'username': f'user_{user_id}',
                    'role': 'user'
                })
            
            return users
            
        except Exception as e:
            logger.error(f"Batch load users error: {e}")
            return [None] * len(user_ids)

class MetricsLoader(DataLoader):
    """DataLoader for metrics data."""
    
    def __init__(self):
        """Initialize metrics loader."""
        super().__init__(self._batch_load_metrics, cache=False)
    
    async def _batch_load_metrics(self, metric_keys: List[str]) -> List[Optional[float]]:
        """
        Batch load metrics by keys.
        
        Args:
            metric_keys: List of metric keys
            
        Returns:
            List[Optional[float]]: Metric values
        """
        try:
            # Placeholder implementation
            # In production, fetch from metrics service
            import random
            
            metrics = []
            for key in metric_keys:
                if 'accuracy' in key:
                    metrics.append(random.uniform(0.8, 0.95))
                elif 'latency' in key:
                    metrics.append(random.uniform(10, 100))
                else:
                    metrics.append(random.uniform(0, 100))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Batch load metrics error: {e}")
            return [None] * len(metric_keys)

# Export DataLoaders
__all__ = [
    "DataLoader",
    "ModelLoader",
    "DatasetLoader",
    "TrainingJobLoader",
    "UserLoader",
    "MetricsLoader"
]
