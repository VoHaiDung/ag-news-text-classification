"""
Base Handler for API Endpoints
================================================================================
Provides abstract base classes and utilities for building consistent API
handlers across different protocols (REST, gRPC, GraphQL).

This module implements common patterns for request handling, validation,
error management, and response formatting following clean architecture
principles.

References:
    - Martin, R. C. (2017). Clean Architecture: A Craftsman's Guide to Software Structure and Design
    - Richardson, C. (2018). Microservices Patterns: With Examples in Java
    - Evans, E. (2003). Domain-Driven Design: Tackling Complexity in the Heart of Software

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
import logging
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union, Callable

from src.core.exceptions import (
    AGNewsException,
    ValidationError,
    NotFoundError,
    UnauthorizedError,
    ForbiddenError
)
from src.core.types import ModelOutput, DataSample
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ResponseStatus(Enum):
    """
    Enumeration of API response statuses.
    
    Based on standard HTTP status semantics and RESTful conventions.
    """
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    PENDING = "pending"
    PROCESSING = "processing"


@dataclass
class APIContext:
    """
    Context object containing request metadata and authentication information.
    
    Implements Context Object pattern for passing request-scoped data
    through the processing pipeline.
    
    Attributes:
        request_id: Unique identifier for request tracing
        user_id: Authenticated user identifier
        timestamp: Request initiation timestamp
        source: Request source identifier (web, mobile, api)
        metadata: Additional context metadata
        correlation_id: Correlation ID for distributed tracing
        span_id: Span ID for distributed tracing
    """
    request_id: str
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "api"
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    span_id: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.metadata is None:
            self.metadata = {}
        
        # Add request timing
        self.metadata["request_start"] = self.timestamp.isoformat()
    
    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add metadata to context.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    def get_elapsed_time(self) -> float:
        """
        Calculate elapsed time since request start.
        
        Returns:
            Elapsed time in seconds
        """
        now = datetime.now(timezone.utc)
        elapsed = (now - self.timestamp).total_seconds()
        return elapsed


@dataclass
class APIResponse:
    """
    Standardized API response structure.
    
    Implements a consistent response format across all API endpoints
    following JSON:API specification principles.
    
    Attributes:
        status: Response status
        data: Response payload
        message: Human-readable message
        errors: List of error details
        metadata: Response metadata
        pagination: Pagination information
        links: HATEOAS links
    """
    status: ResponseStatus
    data: Optional[Any] = None
    message: Optional[str] = None
    errors: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    pagination: Optional[Dict[str, Any]] = None
    links: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert response to dictionary format.
        
        Returns:
            Dictionary representation of response
        """
        response = {
            "status": self.status.value,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if self.data is not None:
            response["data"] = self.data
        
        if self.message:
            response["message"] = self.message
            
        if self.errors:
            response["errors"] = self.errors
            
        if self.metadata:
            response["metadata"] = self.metadata
            
        if self.pagination:
            response["pagination"] = self.pagination
            
        if self.links:
            response["links"] = self.links
            
        return response
    
    def add_error(self, error: Dict[str, Any]) -> None:
        """
        Add error to response.
        
        Args:
            error: Error dictionary
        """
        if self.errors is None:
            self.errors = []
        self.errors.append(error)
    
    def add_link(self, rel: str, href: str) -> None:
        """
        Add HATEOAS link to response.
        
        Args:
            rel: Link relation type
            href: Link URL
        """
        if self.links is None:
            self.links = {}
        self.links[rel] = href


class BaseHandler(ABC):
    """
    Abstract base class for API handlers.
    
    Provides common functionality for handling API requests including
    validation, error handling, middleware processing, and response formatting.
    Implements Template Method pattern for request processing pipeline.
    """
    
    def __init__(self, service_registry=None, config: Dict[str, Any] = None):
        """
        Initialize the base handler.
        
        Args:
            service_registry: Registry containing available services
            config: Handler configuration
        """
        self.service_registry = service_registry
        self.config = config or {}
        self._validators: Dict[str, Callable] = {}
        self._middleware: List[Any] = []
        self._interceptors: List[Any] = []
        
        # Performance monitoring
        self._request_count = 0
        self._error_count = 0
        self._total_latency = 0.0
        
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def handle(self, request: Any, context: APIContext) -> APIResponse:
        """
        Handle the API request.
        
        Template method defining the request processing pipeline.
        
        Args:
            request: The incoming request object
            context: Request context with metadata
            
        Returns:
            APIResponse object with results
        """
        pass
    
    async def process_request(self, request: Any, context: APIContext) -> APIResponse:
        """
        Process request through complete pipeline.
        
        Implements the main request processing flow with error handling,
        middleware, and interceptors.
        
        Args:
            request: Request object
            context: Request context
            
        Returns:
            API response
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Pre-processing
            await self.log_request(request, context)
            
            # Apply middleware
            request = await self.apply_middleware(request, context)
            
            # Apply interceptors
            for interceptor in self._interceptors:
                request = await interceptor.pre_process(request, context)
            
            # Main handler logic
            response = await self.handle(request, context)
            
            # Post-processing
            for interceptor in reversed(self._interceptors):
                response = await interceptor.post_process(response, context)
            
            # Update metrics
            self._update_metrics(start_time, success=True)
            
            # Log response
            await self.log_response(response, context)
            
            return response
            
        except Exception as e:
            # Update error metrics
            self._update_metrics(start_time, success=False)
            
            # Format error response
            response = self.format_error(e, context)
            
            # Log error
            await self.log_error(e, context)
            
            return response
    
    def validate_request(self, request: Any, schema: Type = None) -> None:
        """
        Validate the incoming request against a schema.
        
        Implements validation using schema definitions or custom validators.
        
        Args:
            request: Request object to validate
            schema: Optional schema class for validation
            
        Raises:
            ValidationError: If validation fails
        """
        if schema is None:
            return
            
        try:
            # Schema-based validation
            if hasattr(schema, 'validate'):
                schema.validate(request)
            else:
                # Type checking
                if not isinstance(request, schema):
                    raise ValidationError(
                        f"Request must be of type {schema.__name__}"
                    )
            
            # Custom validators
            for field_name, validator in self._validators.items():
                if hasattr(request, field_name):
                    field_value = getattr(request, field_name)
                    if not validator(field_value):
                        raise ValidationError(
                            f"Validation failed for field '{field_name}'"
                        )
                        
        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Request validation failed: {str(e)}")
            raise ValidationError(f"Invalid request: {str(e)}")
    
    def format_error(self, error: Exception, context: APIContext) -> APIResponse:
        """
        Format an error into a standardized API response.
        
        Maps exceptions to appropriate HTTP status codes and error formats
        following problem details specification (RFC 7807).
        
        Args:
            error: The exception that occurred
            context: Request context
            
        Returns:
            APIResponse with error details
        """
        error_data = {
            "type": error.__class__.__name__,
            "title": self._get_error_title(error),
            "detail": str(error),
            "instance": f"/requests/{context.request_id}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Add debug information in development
        if self._is_development():
            error_data["traceback"] = traceback.format_exc()
            error_data["context"] = {
                "request_id": context.request_id,
                "user_id": context.user_id,
                "source": context.source
            }
        
        # Map to response status
        if isinstance(error, ValidationError):
            status_message = "Validation failed"
            error_data["status"] = 400
        elif isinstance(error, NotFoundError):
            status_message = "Resource not found"
            error_data["status"] = 404
        elif isinstance(error, UnauthorizedError):
            status_message = "Authentication required"
            error_data["status"] = 401
        elif isinstance(error, ForbiddenError):
            status_message = "Access forbidden"
            error_data["status"] = 403
        elif isinstance(error, AGNewsException):
            status_message = "Processing error"
            error_data["status"] = 422
        else:
            status_message = "Internal server error"
            error_data["status"] = 500
            
        return APIResponse(
            status=ResponseStatus.ERROR,
            message=status_message,
            errors=[error_data],
            metadata={
                "request_id": context.request_id,
                "elapsed_time": context.get_elapsed_time()
            }
        )
    
    def format_success(self, data: Any, message: str = None, 
                      metadata: Dict[str, Any] = None,
                      pagination: Dict[str, Any] = None,
                      links: Dict[str, str] = None) -> APIResponse:
        """
        Format a successful response.
        
        Creates a standardized success response with optional metadata,
        pagination, and HATEOAS links.
        
        Args:
            data: Response data
            message: Optional success message
            metadata: Optional response metadata
            pagination: Optional pagination information
            links: Optional HATEOAS links
            
        Returns:
            APIResponse with success status
        """
        return APIResponse(
            status=ResponseStatus.SUCCESS,
            data=data,
            message=message or "Request processed successfully",
            metadata=metadata,
            pagination=pagination,
            links=links
        )
    
    async def apply_middleware(self, request: Any, context: APIContext) -> Any:
        """
        Apply middleware to the request.
        
        Implements Chain of Responsibility pattern for middleware processing.
        
        Args:
            request: The incoming request
            context: Request context
            
        Returns:
            Modified request after middleware processing
        """
        for middleware in self._middleware:
            try:
                request = await middleware.process(request, context)
            except Exception as e:
                self.logger.error(
                    f"Middleware {middleware.__class__.__name__} failed: {str(e)}"
                )
                raise
                
        return request
    
    def add_middleware(self, middleware: Any) -> None:
        """
        Add middleware to the handler.
        
        Args:
            middleware: Middleware instance to add
        """
        self._middleware.append(middleware)
        self.logger.debug(f"Added middleware: {middleware.__class__.__name__}")
    
    def add_interceptor(self, interceptor: Any) -> None:
        """
        Add interceptor to the handler.
        
        Args:
            interceptor: Interceptor instance to add
        """
        self._interceptors.append(interceptor)
        self.logger.debug(f"Added interceptor: {interceptor.__class__.__name__}")
    
    def add_validator(self, field_name: str, validator: Callable) -> None:
        """
        Add custom field validator.
        
        Args:
            field_name: Field name to validate
            validator: Validation function
        """
        self._validators[field_name] = validator
    
    def get_service(self, service_name: str) -> Any:
        """
        Get a service from the registry.
        
        Args:
            service_name: Name of the service to retrieve
            
        Returns:
            Service instance
            
        Raises:
            NotFoundError: If service not found
        """
        if self.service_registry is None:
            raise NotFoundError("Service registry not initialized")
            
        service = self.service_registry.get_service(service_name)
        if service is None:
            raise NotFoundError(f"Service '{service_name}' not found")
            
        return service
    
    def _is_development(self) -> bool:
        """
        Check if running in development mode.
        
        Returns:
            True if in development mode
        """
        import os
        return os.getenv("ENVIRONMENT", "production") in ["development", "dev"]
    
    def _get_error_title(self, error: Exception) -> str:
        """
        Get human-readable error title.
        
        Args:
            error: Exception instance
            
        Returns:
            Error title string
        """
        error_titles = {
            ValidationError: "Invalid Request",
            NotFoundError: "Not Found",
            UnauthorizedError: "Unauthorized",
            ForbiddenError: "Forbidden",
            AGNewsException: "Processing Error"
        }
        
        for error_type, title in error_titles.items():
            if isinstance(error, error_type):
                return title
                
        return "Internal Error"
    
    def _update_metrics(self, start_time: datetime, success: bool) -> None:
        """
        Update handler metrics.
        
        Args:
            start_time: Request start time
            success: Whether request was successful
        """
        # Calculate latency
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # Update counters
        self._request_count += 1
        if not success:
            self._error_count += 1
        
        # Update latency
        self._total_latency += elapsed
        
        # Calculate average latency
        avg_latency = self._total_latency / self._request_count
        
        # Log metrics periodically
        if self._request_count % 100 == 0:
            self.logger.info(
                f"Handler metrics - Requests: {self._request_count}, "
                f"Errors: {self._error_count}, "
                f"Avg Latency: {avg_latency:.3f}s"
            )
    
    async def log_request(self, request: Any, context: APIContext) -> None:
        """
        Log incoming request details.
        
        Args:
            request: The request object
            context: Request context
        """
        self.logger.info(
            f"Request received",
            extra={
                "request_id": context.request_id,
                "user_id": context.user_id,
                "source": context.source,
                "handler": self.__class__.__name__,
                "correlation_id": context.correlation_id
            }
        )
    
    async def log_response(self, response: APIResponse, context: APIContext) -> None:
        """
        Log response details.
        
        Args:
            response: The response object
            context: Request context
        """
        log_level = logging.INFO if response.status == ResponseStatus.SUCCESS else logging.WARNING
        
        self.logger.log(
            log_level,
            f"Response sent: {response.status.value}",
            extra={
                "request_id": context.request_id,
                "status": response.status.value,
                "has_errors": response.errors is not None,
                "elapsed_time": context.get_elapsed_time()
            }
        )
    
    async def log_error(self, error: Exception, context: APIContext) -> None:
        """
        Log error details.
        
        Args:
            error: The exception that occurred
            context: Request context
        """
        self.logger.error(
            f"Request failed: {str(error)}",
            extra={
                "request_id": context.request_id,
                "error_type": error.__class__.__name__,
                "user_id": context.user_id,
                "elapsed_time": context.get_elapsed_time()
            },
            exc_info=True
        )


class BatchHandler(BaseHandler):
    """
    Base handler for batch processing requests.
    
    Supports processing multiple items in a single request with partial
    failure handling and progress reporting.
    """
    
    def __init__(self, service_registry=None, config: Dict[str, Any] = None):
        """
        Initialize batch handler.
        
        Args:
            service_registry: Registry containing available services
            config: Handler configuration
        """
        super().__init__(service_registry, config)
        self.batch_size = config.get("batch_size", 100) if config else 100
        self.parallel_processing = config.get("parallel", False) if config else False
        self.max_workers = config.get("max_workers", 4) if config else 4
    
    async def handle_batch(self, items: List[Any], context: APIContext,
                          progress_callback: Optional[Callable] = None) -> APIResponse:
        """
        Handle a batch of items with progress reporting.
        
        Args:
            items: List of items to process
            context: Request context
            progress_callback: Optional callback for progress updates
            
        Returns:
            APIResponse with batch results
        """
        if len(items) > self.batch_size:
            raise ValidationError(
                f"Batch size {len(items)} exceeds limit of {self.batch_size}"
            )
        
        results = []
        errors = []
        
        # Process items
        if self.parallel_processing:
            # Parallel processing using asyncio
            tasks = [
                self._process_item_with_error_handling(item, idx, context)
                for idx, item in enumerate(items)
            ]
            
            # Process with semaphore to limit concurrency
            semaphore = asyncio.Semaphore(self.max_workers)
            
            async def process_with_semaphore(task):
                async with semaphore:
                    return await task
            
            batch_results = await asyncio.gather(
                *[process_with_semaphore(task) for task in tasks],
                return_exceptions=False
            )
            
            # Separate results and errors
            for idx, result in enumerate(batch_results):
                if isinstance(result, dict) and "error" in result:
                    errors.append(result)
                else:
                    results.append({"index": idx, "result": result})
                    
                # Report progress
                if progress_callback:
                    progress = (idx + 1) / len(items)
                    await progress_callback(progress, idx + 1, len(items))
        else:
            # Sequential processing
            for idx, item in enumerate(items):
                try:
                    result = await self.process_item(item, context)
                    results.append({"index": idx, "result": result})
                except Exception as e:
                    errors.append({
                        "index": idx,
                        "error": str(e),
                        "error_type": e.__class__.__name__,
                        "item": str(item)[:100]  # Truncate for logging
                    })
                    self.logger.error(f"Failed to process item {idx}: {str(e)}")
                
                # Report progress
                if progress_callback:
                    progress = (idx + 1) / len(items)
                    await progress_callback(progress, idx + 1, len(items))
        
        # Determine response status
        if not errors:
            status = ResponseStatus.SUCCESS
            message = f"All {len(items)} items processed successfully"
        elif not results:
            status = ResponseStatus.ERROR
            message = f"Failed to process all {len(items)} items"
        else:
            status = ResponseStatus.PARTIAL
            message = f"Processed {len(results)} of {len(items)} items successfully"
        
        return APIResponse(
            status=status,
            data={"results": results},
            message=message,
            errors=errors if errors else None,
            metadata={
                "total": len(items),
                "succeeded": len(results),
                "failed": len(errors),
                "batch_size": self.batch_size,
                "processing_mode": "parallel" if self.parallel_processing else "sequential"
            }
        )
    
    async def _process_item_with_error_handling(self, item: Any, index: int,
                                               context: APIContext) -> Dict[str, Any]:
        """
        Process item with error handling for parallel processing.
        
        Args:
            item: Item to process
            index: Item index
            context: Request context
            
        Returns:
            Result dictionary or error dictionary
        """
        try:
            result = await self.process_item(item, context)
            return {"index": index, "result": result}
        except Exception as e:
            return {
                "index": index,
                "error": str(e),
                "error_type": e.__class__.__name__,
                "item": str(item)[:100]
            }
    
    @abstractmethod
    async def process_item(self, item: Any, context: APIContext) -> Any:
        """
        Process a single item from the batch.
        
        Must be implemented by subclasses to define item processing logic.
        
        Args:
            item: Item to process
            context: Request context
            
        Returns:
            Processing result
        """
        pass


class StreamingHandler(BaseHandler):
    """
    Base handler for streaming responses.
    
    Supports server-sent events (SSE) and streaming large responses
    with backpressure handling.
    """
    
    def __init__(self, service_registry=None, config: Dict[str, Any] = None):
        """
        Initialize streaming handler.
        
        Args:
            service_registry: Registry containing available services
            config: Handler configuration
        """
        super().__init__(service_registry, config)
        self.chunk_size = config.get("chunk_size", 1024) if config else 1024
        self.buffer_size = config.get("buffer_size", 10) if config else 10
    
    async def stream_response(self, generator, context: APIContext,
                             content_type: str = "text/event-stream"):
        """
        Stream response data with backpressure handling.
        
        Args:
            generator: Async generator producing response chunks
            context: Request context
            content_type: Response content type
            
        Yields:
            Response chunks
        """
        buffer = asyncio.Queue(maxsize=self.buffer_size)
        
        async def producer():
            """Produce chunks from generator."""
            try:
                async for chunk in generator:
                    await buffer.put(chunk)
            except Exception as e:
                await buffer.put({"error": str(e)})
            finally:
                await buffer.put(None)  # Signal completion
        
        # Start producer task
        producer_task = asyncio.create_task(producer())
        
        try:
            while True:
                # Get chunk with timeout
                try:
                    chunk = await asyncio.wait_for(
                        buffer.get(),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    yield self.format_error_chunk(
                        TimeoutError("Stream timeout"),
                        context
                    )
                    break
                
                # Check for completion
                if chunk is None:
                    break
                
                # Check for error
                if isinstance(chunk, dict) and "error" in chunk:
                    yield self.format_error_chunk(
                        Exception(chunk["error"]),
                        context
                    )
                    break
                
                # Format and yield chunk
                formatted = self.format_chunk(chunk, context, content_type)
                yield formatted
                
        except Exception as e:
            self.logger.error(f"Streaming error: {str(e)}")
            yield self.format_error_chunk(e, context)
        finally:
            # Cancel producer if still running
            if not producer_task.done():
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass
    
    def format_chunk(self, data: Any, context: APIContext,
                    content_type: str = "text/event-stream") -> str:
        """
        Format a single chunk for streaming.
        
        Args:
            data: Chunk data
            context: Request context
            content_type: Content type for formatting
            
        Returns:
            Formatted chunk string
        """
        if content_type == "text/event-stream":
            # Server-Sent Events format
            import json
            return (
                f"id: {context.request_id}\n"
                f"event: data\n"
                f"data: {json.dumps(data)}\n\n"
            )
        elif content_type == "application/x-ndjson":
            # Newline-delimited JSON
            import json
            return json.dumps(data) + "\n"
        else:
            # Plain text
            return str(data) + "\n"
    
    def format_error_chunk(self, error: Exception, context: APIContext) -> str:
        """
        Format an error chunk for streaming.
        
        Args:
            error: The error that occurred
            context: Request context
            
        Returns:
            Formatted error chunk
        """
        import json
        error_data = {
            "error": str(error),
            "type": error.__class__.__name__,
            "request_id": context.request_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return (
            f"id: {context.request_id}\n"
            f"event: error\n"
            f"data: {json.dumps(error_data)}\n\n"
        )
