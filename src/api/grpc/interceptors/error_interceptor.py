"""
Error Handling Interceptor
================================================================================
This module implements error handling and transformation interceptor for gRPC
services, providing consistent error responses and recovery mechanisms.

Implements error handling including:
- Exception to gRPC status mapping
- Error message formatting
- Retry logic for transient errors
- Circuit breaker pattern

References:
    - Nygard, M. (2018). Release It!: Design and Deploy Production-Ready Software
    - gRPC Error Handling Best Practices
    - Fowler, M. (2014). Circuit Breaker Pattern

Author: Võ Hải Dũng
License: MIT
"""

import logging
import time
from typing import Callable, Any, Dict, Optional
from enum import Enum
import grpc
import traceback

from . import BaseInterceptor
from ....core.exceptions import (
    ModelNotFoundError,
    PredictionError,
    DataValidationError,
    ResourceExhaustedError
)

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class ErrorInterceptor(BaseInterceptor):
    """
    Error handling interceptor with circuit breaker pattern.
    
    Provides consistent error handling, automatic retries for transient
    failures, and circuit breaking for failing services.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        circuit_threshold: int = 5,
        circuit_timeout: float = 60.0
    ):
        """
        Initialize error interceptor.
        
        Args:
            max_retries: Maximum retry attempts
            circuit_threshold: Failures before opening circuit
            circuit_timeout: Timeout before half-opening circuit
        """
        super().__init__("ErrorInterceptor")
        self.max_retries = max_retries
        
        # Circuit breaker configuration
        self.circuit_threshold = circuit_threshold
        self.circuit_timeout = circuit_timeout
        
        # Circuit breaker state per method
        self.circuits = {}
        
        # Exception to status code mapping
        self.exception_mapping = {
            ValueError: grpc.StatusCode.INVALID_ARGUMENT,
            KeyError: grpc.StatusCode.NOT_FOUND,
            FileNotFoundError: grpc.StatusCode.NOT_FOUND,
            PermissionError: grpc.StatusCode.PERMISSION_DENIED,
            NotImplementedError: grpc.StatusCode.UNIMPLEMENTED,
            TimeoutError: grpc.StatusCode.DEADLINE_EXCEEDED,
            ConnectionError: grpc.StatusCode.UNAVAILABLE,
            ModelNotFoundError: grpc.StatusCode.NOT_FOUND,
            PredictionError: grpc.StatusCode.INTERNAL,
            DataValidationError: grpc.StatusCode.INVALID_ARGUMENT,
            ResourceExhaustedError: grpc.StatusCode.RESOURCE_EXHAUSTED
        }
        
        # Retryable status codes
        self.retryable_codes = {
            grpc.StatusCode.UNAVAILABLE,
            grpc.StatusCode.DEADLINE_EXCEEDED,
            grpc.StatusCode.RESOURCE_EXHAUSTED
        }
    
    def intercept_unary_unary(
        self,
        request: Any,
        context: grpc.ServicerContext,
        method: Callable,
        handler_call_details: grpc.HandlerCallDetails
    ) -> Any:
        """
        Intercept and handle errors for unary-unary RPC.
        
        Args:
            request: Request message
            context: gRPC context
            method: Original method
            handler_call_details: Call details
            
        Returns:
            Any: Response message
        """
        method_name = handler_call_details.method
        
        # Check circuit breaker
        if self._is_circuit_open(method_name):
            context.abort(
                grpc.StatusCode.UNAVAILABLE,
                f"Circuit breaker open for {method_name}"
            )
        
        # Attempt with retries
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Call original method
                response = method(request, context)
                
                # Reset circuit on success
                self._record_success(method_name)
                
                self.metrics["requests_intercepted"] += 1
                return response
                
            except grpc.RpcError as e:
                # Already a gRPC error, check if retryable
                last_error = e
                
                if e.code() not in self.retryable_codes:
                    # Non-retryable error
                    self._record_failure(method_name)
                    raise
                
                # Log retry attempt
                logger.warning(
                    f"Retrying {method_name} (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                
                # Exponential backoff
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt * 0.1)
                    
            except Exception as e:
                # Transform Python exception to gRPC error
                last_error = e
                self._record_failure(method_name)
                
                # Map exception to status code
                status_code = self._map_exception_to_status(e)
                
                # Create detailed error message
                error_message = self._format_error_message(e, method_name)
                
                # Log error
                logger.error(f"Error in {method_name}: {error_message}")
                
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                
                # Abort with appropriate status
                context.abort(status_code, error_message)
        
        # Max retries exceeded
        self._record_failure(method_name)
        self.metrics["errors_handled"] += 1
        
        if isinstance(last_error, grpc.RpcError):
            raise last_error
        else:
            context.abort(
                grpc.StatusCode.UNAVAILABLE,
                f"Max retries exceeded for {method_name}: {str(last_error)}"
            )
    
    def _is_circuit_open(self, method: str) -> bool:
        """
        Check if circuit breaker is open for method.
        
        Args:
            method: Method name
            
        Returns:
            bool: True if circuit is open
        """
        if method not in self.circuits:
            self.circuits[method] = {
                'state': CircuitState.CLOSED,
                'failures': 0,
                'last_failure_time': 0,
                'success_count': 0
            }
            return False
        
        circuit = self.circuits[method]
        
        if circuit['state'] == CircuitState.OPEN:
            # Check if timeout has passed
            if time.time() - circuit['last_failure_time'] > self.circuit_timeout:
                # Move to half-open state
                circuit['state'] = CircuitState.HALF_OPEN
                circuit['success_count'] = 0
                logger.info(f"Circuit half-open for {method}")
                return False
            return True
        
        return False
    
    def _record_success(self, method: str) -> None:
        """
        Record successful method call.
        
        Args:
            method: Method name
        """
        if method not in self.circuits:
            return
        
        circuit = self.circuits[method]
        
        if circuit['state'] == CircuitState.HALF_OPEN:
            circuit['success_count'] += 1
            
            # Close circuit after successful calls
            if circuit['success_count'] >= 3:
                circuit['state'] = CircuitState.CLOSED
                circuit['failures'] = 0
                logger.info(f"Circuit closed for {method}")
        
        elif circuit['state'] == CircuitState.CLOSED:
            # Reset failure count on success
            circuit['failures'] = 0
    
    def _record_failure(self, method: str) -> None:
        """
        Record method failure.
        
        Args:
            method: Method name
        """
        if method not in self.circuits:
            self.circuits[method] = {
                'state': CircuitState.CLOSED,
                'failures': 0,
                'last_failure_time': 0,
                'success_count': 0
            }
        
        circuit = self.circuits[method]
        circuit['failures'] += 1
        circuit['last_failure_time'] = time.time()
        
        if circuit['state'] == CircuitState.HALF_OPEN:
            # Return to open state on failure
            circuit['state'] = CircuitState.OPEN
            logger.warning(f"Circuit reopened for {method}")
        
        elif circuit['failures'] >= self.circuit_threshold:
            # Open circuit after threshold
            circuit['state'] = CircuitState.OPEN
            logger.error(f"Circuit opened for {method} after {circuit['failures']} failures")
    
    def _map_exception_to_status(self, exception: Exception) -> grpc.StatusCode:
        """
        Map Python exception to gRPC status code.
        
        Args:
            exception: Python exception
            
        Returns:
            grpc.StatusCode: Mapped status code
        """
        for exc_type, status_code in self.exception_mapping.items():
            if isinstance(exception, exc_type):
                return status_code
        
        # Default to internal error
        return grpc.StatusCode.INTERNAL
    
    def _format_error_message(self, exception: Exception, method: str) -> str:
        """
        Format error message for client.
        
        Args:
            exception: Exception object
            method: Method name
            
        Returns:
            str: Formatted error message
        """
        # Base error message
        message = f"Error in {method.split('/')[-1]}: {str(exception)}"
        
        # Add additional context for specific exceptions
        if isinstance(exception, ModelNotFoundError):
            message = f"Model not found: {str(exception)}"
        elif isinstance(exception, DataValidationError):
            message = f"Data validation failed: {str(exception)}"
        elif isinstance(exception, ResourceExhaustedError):
            message = f"Resource exhausted: {str(exception)}"
        
        return message
    
    def get_circuit_status(self) -> Dict[str, Any]:
        """
        Get circuit breaker status for all methods.
        
        Returns:
            Dict[str, Any]: Circuit status
        """
        status = {}
        
        for method, circuit in self.circuits.items():
            status[method] = {
                'state': circuit['state'].value,
                'failures': circuit['failures'],
                'time_since_last_failure': time.time() - circuit['last_failure_time'] if circuit['last_failure_time'] > 0 else None
            }
        
        return status
