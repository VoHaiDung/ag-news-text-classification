"""
gRPC Interceptors Module
================================================================================
This module provides interceptor implementations for cross-cutting concerns
in gRPC services, following the Chain of Responsibility pattern.

Interceptors handle:
- Authentication and authorization
- Request/response logging
- Metrics collection
- Error handling and transformation

References:
    - Gamma, E., et al. (1994). Design Patterns: Elements of Reusable Object-Oriented Software
    - gRPC Python Interceptors Documentation
    - Schmidt, D. C. (1996). Reactor: An Object Behavioral Pattern for Concurrent Event Demultiplexing

Author: Võ Hải Dũng
License: MIT
"""

from typing import Callable, Any, Optional, Tuple
import grpc
from abc import ABC, abstractmethod
import time
import logging

logger = logging.getLogger(__name__)

class BaseInterceptor(grpc.ServerInterceptor, ABC):
    """
    Base class for all gRPC interceptors.
    
    Provides common functionality for interceptor implementations
    following the Template Method pattern.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize base interceptor.
        
        Args:
            name: Interceptor name for logging
        """
        self.name = name or self.__class__.__name__
        self.metrics = {
            "requests_intercepted": 0,
            "errors_handled": 0
        }
        
    def intercept_service(
        self,
        continuation: Callable,
        handler_call_details: grpc.HandlerCallDetails
    ) -> grpc.RpcMethodHandler:
        """
        Main interception method.
        
        Args:
            continuation: Next interceptor in chain
            handler_call_details: RPC call details
            
        Returns:
            grpc.RpcMethodHandler: Modified or original handler
        """
        # Get the original handler
        handler = continuation(handler_call_details)
        
        if handler is None:
            return None
        
        # Determine RPC type and wrap accordingly
        if handler.unary_unary:
            return self._wrap_unary_unary(handler, handler_call_details)
        elif handler.unary_stream:
            return self._wrap_unary_stream(handler, handler_call_details)
        elif handler.stream_unary:
            return self._wrap_stream_unary(handler, handler_call_details)
        elif handler.stream_stream:
            return self._wrap_stream_stream(handler, handler_call_details)
        
        return handler
    
    def _wrap_unary_unary(
        self,
        handler: grpc.RpcMethodHandler,
        handler_call_details: grpc.HandlerCallDetails
    ) -> grpc.RpcMethodHandler:
        """
        Wrap unary-unary RPC handler.
        
        Args:
            handler: Original handler
            handler_call_details: Call details
            
        Returns:
            grpc.RpcMethodHandler: Wrapped handler
        """
        def wrapper(request, context):
            return self.intercept_unary_unary(
                request,
                context,
                handler.unary_unary,
                handler_call_details
            )
        
        return grpc.unary_unary_rpc_method_handler(
            wrapper,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer
        )
    
    def _wrap_unary_stream(
        self,
        handler: grpc.RpcMethodHandler,
        handler_call_details: grpc.HandlerCallDetails
    ) -> grpc.RpcMethodHandler:
        """Wrap unary-stream RPC handler."""
        def wrapper(request, context):
            return self.intercept_unary_stream(
                request,
                context,
                handler.unary_stream,
                handler_call_details
            )
        
        return grpc.unary_stream_rpc_method_handler(
            wrapper,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer
        )
    
    def _wrap_stream_unary(
        self,
        handler: grpc.RpcMethodHandler,
        handler_call_details: grpc.HandlerCallDetails
    ) -> grpc.RpcMethodHandler:
        """Wrap stream-unary RPC handler."""
        def wrapper(request_iterator, context):
            return self.intercept_stream_unary(
                request_iterator,
                context,
                handler.stream_unary,
                handler_call_details
            )
        
        return grpc.stream_unary_rpc_method_handler(
            wrapper,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer
        )
    
    def _wrap_stream_stream(
        self,
        handler: grpc.RpcMethodHandler,
        handler_call_details: grpc.HandlerCallDetails
    ) -> grpc.RpcMethodHandler:
        """Wrap stream-stream RPC handler."""
        def wrapper(request_iterator, context):
            return self.intercept_stream_stream(
                request_iterator,
                context,
                handler.stream_stream,
                handler_call_details
            )
        
        return grpc.stream_stream_rpc_method_handler(
            wrapper,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer
        )
    
    @abstractmethod
    def intercept_unary_unary(
        self,
        request: Any,
        context: grpc.ServicerContext,
        method: Callable,
        handler_call_details: grpc.HandlerCallDetails
    ) -> Any:
        """
        Intercept unary-unary RPC.
        
        Args:
            request: Request message
            context: gRPC context
            method: Original method
            handler_call_details: Call details
            
        Returns:
            Any: Response message
        """
        pass
    
    def intercept_unary_stream(
        self,
        request: Any,
        context: grpc.ServicerContext,
        method: Callable,
        handler_call_details: grpc.HandlerCallDetails
    ) -> Any:
        """Intercept unary-stream RPC."""
        return method(request, context)
    
    def intercept_stream_unary(
        self,
        request_iterator: Any,
        context: grpc.ServicerContext,
        method: Callable,
        handler_call_details: grpc.HandlerCallDetails
    ) -> Any:
        """Intercept stream-unary RPC."""
        return method(request_iterator, context)
    
    def intercept_stream_stream(
        self,
        request_iterator: Any,
        context: grpc.ServicerContext,
        method: Callable,
        handler_call_details: grpc.HandlerCallDetails
    ) -> Any:
        """Intercept stream-stream RPC."""
        return method(request_iterator, context)
    
    def get_metrics(self) -> dict:
        """
        Get interceptor metrics.
        
        Returns:
            dict: Metrics dictionary
        """
        return {f"{self.name}_metrics": self.metrics}

# Export base class
__all__ = ["BaseInterceptor"]
