"""
Logging Interceptor
================================================================================
This module implements comprehensive logging interceptor for gRPC services,
providing detailed request/response logging and audit trails.

Implements logging features including:
- Request/response payload logging
- Performance metrics logging
- Error logging with stack traces
- Structured logging for analysis

References:
    - Google Cloud Logging Best Practices
    - The Twelve-Factor App: XI. Logs
    - Structured Logging Guidelines

Author: Võ Hải Dũng
License: MIT
"""

import logging
import time
import json
import traceback
from typing import Callable, Any, Dict
import grpc
from datetime import datetime

from . import BaseInterceptor

# Configure structured logger
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class LoggingInterceptor(BaseInterceptor):
    """
    Comprehensive logging interceptor for gRPC services.
    
    Logs all RPC calls with configurable detail levels,
    supporting structured logging for analysis and monitoring.
    """
    
    def __init__(self, log_level: str = "INFO", log_payloads: bool = False):
        """
        Initialize logging interceptor.
        
        Args:
            log_level: Logging level
            log_payloads: Whether to log request/response payloads
        """
        super().__init__("LoggingInterceptor")
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.log_payloads = log_payloads
        
    def intercept_unary_unary(
        self,
        request: Any,
        context: grpc.ServicerContext,
        method: Callable,
        handler_call_details: grpc.HandlerCallDetails
    ) -> Any:
        """
        Intercept and log unary-unary RPC.
        
        Args:
            request: Request message
            context: gRPC context
            method: Original method
            handler_call_details: Call details
            
        Returns:
            Any: Response message
        """
        # Start timing
        start_time = time.time()
        
        # Extract metadata
        metadata = dict(context.invocation_metadata())
        peer = context.peer()
        
        # Generate request ID
        request_id = metadata.get('x-request-id', f"{time.time()}")
        
        # Build log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': request_id,
            'method': handler_call_details.method,
            'peer': peer,
            'metadata': self._sanitize_metadata(metadata)
        }
        
        # Log request payload if enabled
        if self.log_payloads:
            log_entry['request'] = self._serialize_message(request)
        
        # Log request
        logger.log(self.log_level, f"gRPC Request: {json.dumps(log_entry)}")
        
        response = None
        error = None
        status_code = grpc.StatusCode.OK
        
        try:
            # Call original method
            response = method(request, context)
            
            # Log successful response
            elapsed_ms = (time.time() - start_time) * 1000
            
            response_log = {
                'timestamp': datetime.utcnow().isoformat(),
                'request_id': request_id,
                'method': handler_call_details.method,
                'status': 'SUCCESS',
                'elapsed_ms': elapsed_ms
            }
            
            if self.log_payloads and response:
                response_log['response'] = self._serialize_message(response)
            
            logger.log(self.log_level, f"gRPC Response: {json.dumps(response_log)}")
            
            self.metrics["requests_intercepted"] += 1
            return response
            
        except grpc.RpcError as e:
            # Log gRPC errors
            error = e
            status_code = e.code()
            raise
            
        except Exception as e:
            # Log unexpected errors
            error = e
            status_code = grpc.StatusCode.INTERNAL
            context.abort(status_code, str(e))
            
        finally:
            if error:
                # Log error
                elapsed_ms = (time.time() - start_time) * 1000
                
                error_log = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'request_id': request_id,
                    'method': handler_call_details.method,
                    'status': 'ERROR',
                    'status_code': status_code.name,
                    'error': str(error),
                    'elapsed_ms': elapsed_ms
                }
                
                if self.log_level == logging.DEBUG:
                    error_log['traceback'] = traceback.format_exc()
                
                logger.error(f"gRPC Error: {json.dumps(error_log)}")
                self.metrics["errors_handled"] += 1
    
    def intercept_unary_stream(
        self,
        request: Any,
        context: grpc.ServicerContext,
        method: Callable,
        handler_call_details: grpc.HandlerCallDetails
    ) -> Any:
        """Intercept and log unary-stream RPC."""
        start_time = time.time()
        request_id = f"{time.time()}"
        
        logger.log(
            self.log_level,
            f"gRPC Stream Start: method={handler_call_details.method}, "
            f"request_id={request_id}"
        )
        
        try:
            # Call original method
            response_iterator = method(request, context)
            
            # Wrap response iterator to log each item
            return self._log_stream_responses(
                response_iterator,
                handler_call_details.method,
                request_id,
                start_time
            )
            
        except Exception as e:
            logger.error(
                f"gRPC Stream Error: method={handler_call_details.method}, "
                f"error={str(e)}"
            )
            raise
    
    def _log_stream_responses(
        self,
        response_iterator: Any,
        method: str,
        request_id: str,
        start_time: float
    ) -> Any:
        """
        Log streaming responses.
        
        Args:
            response_iterator: Response iterator
            method: Method name
            request_id: Request ID
            start_time: Start timestamp
            
        Yields:
            Any: Response messages
        """
        count = 0
        
        try:
            for response in response_iterator:
                count += 1
                
                if self.log_payloads:
                    logger.debug(
                        f"gRPC Stream Item: method={method}, "
                        f"request_id={request_id}, item={count}"
                    )
                
                yield response
                
        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            
            logger.log(
                self.log_level,
                f"gRPC Stream Complete: method={method}, "
                f"request_id={request_id}, items={count}, "
                f"elapsed_ms={elapsed_ms}"
            )
    
    def _sanitize_metadata(self, metadata: Dict[str, str]) -> Dict[str, str]:
        """
        Sanitize metadata for logging.
        
        Args:
            metadata: Request metadata
            
        Returns:
            Dict[str, str]: Sanitized metadata
        """
        # Remove sensitive headers
        sensitive_headers = ['authorization', 'x-api-key', 'cookie']
        sanitized = {}
        
        for key, value in metadata.items():
            if key.lower() in sensitive_headers:
                sanitized[key] = '***REDACTED***'
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _serialize_message(self, message: Any) -> Dict[str, Any]:
        """
        Serialize protobuf message for logging.
        
        Args:
            message: Protobuf message
            
        Returns:
            Dict[str, Any]: Serialized message
        """
        try:
            # Try to convert to dict
            if hasattr(message, 'DESCRIPTOR'):
                # Protobuf message
                from google.protobuf.json_format import MessageToDict
                return MessageToDict(message, preserving_proto_field_name=True)
            else:
                # Regular object
                return str(message)
        except Exception as e:
            logger.debug(f"Failed to serialize message: {e}")
            return {"_type": type(message).__name__}
