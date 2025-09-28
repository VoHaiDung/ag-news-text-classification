#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API Quick Start Script for AG News Classification
==================================================

This script demonstrates basic API usage patterns for the AG News classification
system, providing examples of REST, gRPC, and GraphQL interactions.

Educational approach following RESTful API design principles from:
- Richardson & Ruby (2013): "RESTful Web APIs"
- Kleppmann (2017): "Designing Data-Intensive Applications"

Author: Võ Hải Dũng
License: MIT
"""

import sys
import argparse
import json
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import requests
import grpc
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

# Import generated gRPC stubs
try:
    from src.api.grpc.compiled import classification_pb2, classification_pb2_grpc
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    print("Warning: gRPC dependencies not installed. Run: pip install grpcio grpcio-tools")

from src.utils.logging_config import setup_logging

# Setup logging
logger = setup_logging(name=__name__)


class RESTAPIClient:
    """
    REST API client for AG News classification.
    
    Implements standard REST patterns following:
    - Fielding (2000): "Architectural Styles and the Design of Network-based Software Architectures"
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        """
        Initialize REST API client.
        
        Parameters
        ----------
        base_url : str
            Base URL of the API endpoint
        api_key : Optional[str]
            API key for authentication
        """
        self.base_url = base_url
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"X-API-Key": api_key})
        
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Make single text prediction.
        
        Parameters
        ----------
        text : str
            Input text to classify
            
        Returns
        -------
        Dict[str, Any]
            Prediction results with label and confidence
        """
        payload = {"text": text}
        response = self.session.post(
            f"{self.base_url}/api/v1/predict",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def predict_batch(self, texts: List[str]) -> Dict[str, Any]:
        """
        Make batch predictions.
        
        Parameters
        ----------
        texts : List[str]
            List of texts to classify
            
        Returns
        -------
        Dict[str, Any]
            Batch prediction results
        """
        payload = {"texts": texts}
        response = self.session.post(
            f"{self.base_url}/api/v1/predict/batch",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_models(self) -> Dict[str, Any]:
        """Get available models."""
        response = self.session.get(f"{self.base_url}/api/v1/models")
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get specific model information."""
        response = self.session.get(f"{self.base_url}/api/v1/models/{model_id}")
        response.raise_for_status()
        return response.json()
    
    def async_predict(self, text: str) -> Dict[str, Any]:
        """
        Submit async prediction job.
        
        Parameters
        ----------
        text : str
            Input text for async processing
            
        Returns
        -------
        Dict[str, Any]
            Job ID and status
        """
        payload = {"text": text, "async": True}
        response = self.session.post(
            f"{self.base_url}/api/v1/predict/async",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Check async job status."""
        response = self.session.get(f"{self.base_url}/api/v1/jobs/{job_id}")
        response.raise_for_status()
        return response.json()


class GRPCClient:
    """
    gRPC client for AG News classification.
    
    Implements gRPC patterns following:
    - Google (2020): "gRPC Documentation and Best Practices"
    """
    
    def __init__(self, host: str = "localhost", port: int = 50051):
        """
        Initialize gRPC client.
        
        Parameters
        ----------
        host : str
            gRPC server host
        port : int
            gRPC server port
        """
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.stub = classification_pb2_grpc.ClassificationServiceStub(self.channel)
    
    def predict(self, text: str) -> Any:
        """Make prediction via gRPC."""
        request = classification_pb2.PredictRequest(text=text)
        response = self.stub.Predict(request)
        return {
            "label": response.label,
            "confidence": response.confidence,
            "probabilities": list(response.probabilities)
        }
    
    def predict_batch(self, texts: List[str]) -> Any:
        """Make batch predictions via gRPC."""
        request = classification_pb2.BatchPredictRequest(texts=texts)
        response = self.stub.BatchPredict(request)
        
        results = []
        for pred in response.predictions:
            results.append({
                "label": pred.label,
                "confidence": pred.confidence
            })
        return results
    
    def close(self):
        """Close gRPC channel."""
        self.channel.close()


class GraphQLClient:
    """
    GraphQL client for AG News classification.
    
    Implements GraphQL patterns following:
    - Facebook (2021): "GraphQL Specification"
    """
    
    def __init__(self, endpoint: str = "http://localhost:8000/graphql"):
        """
        Initialize GraphQL client.
        
        Parameters
        ----------
        endpoint : str
            GraphQL endpoint URL
        """
        transport = RequestsHTTPTransport(url=endpoint)
        self.client = Client(transport=transport, fetch_schema_from_transport=True)
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Make prediction via GraphQL."""
        query = gql("""
            query Predict($text: String!) {
                predict(text: $text) {
                    label
                    confidence
                    probabilities
                    processingTime
                }
            }
        """)
        
        variables = {"text": text}
        result = self.client.execute(query, variable_values=variables)
        return result["predict"]
    
    def get_models(self) -> List[Dict[str, Any]]:
        """Get available models via GraphQL."""
        query = gql("""
            query GetModels {
                models {
                    id
                    name
                    version
                    accuracy
                    status
                }
            }
        """)
        
        result = self.client.execute(query)
        return result["models"]


def demonstrate_rest_api(base_url: str = "http://localhost:8000"):
    """
    Demonstrate REST API usage.
    
    Parameters
    ----------
    base_url : str
        API base URL
    """
    logger.info("=" * 50)
    logger.info("REST API Demonstration")
    logger.info("=" * 50)
    
    client = RESTAPIClient(base_url)
    
    # Health check
    logger.info("\n1. Health Check:")
    try:
        health = client.health_check()
        logger.info(f"   Status: {health.get('status', 'unknown')}")
        logger.info(f"   Version: {health.get('version', 'unknown')}")
    except Exception as e:
        logger.error(f"   Health check failed: {e}")
        return
    
    # Single prediction
    logger.info("\n2. Single Prediction:")
    sample_text = "Apple announces new MacBook Pro with M3 chip"
    try:
        result = client.predict_single(sample_text)
        logger.info(f"   Text: {sample_text[:50]}...")
        logger.info(f"   Predicted Label: {result.get('label')}")
        logger.info(f"   Confidence: {result.get('confidence', 0):.4f}")
    except Exception as e:
        logger.error(f"   Prediction failed: {e}")
    
    # Batch prediction
    logger.info("\n3. Batch Prediction:")
    batch_texts = [
        "Stock market reaches all-time high",
        "New scientific discovery in quantum computing",
        "Local sports team wins championship"
    ]
    try:
        results = client.predict_batch(batch_texts)
        for i, pred in enumerate(results.get("predictions", [])):
            logger.info(f"   Text {i+1}: {batch_texts[i][:30]}...")
            logger.info(f"   -> Label: {pred.get('label')}, Confidence: {pred.get('confidence', 0):.4f}")
    except Exception as e:
        logger.error(f"   Batch prediction failed: {e}")
    
    # Available models
    logger.info("\n4. Available Models:")
    try:
        models = client.get_models()
        for model in models.get("models", []):
            logger.info(f"   - {model.get('name')} (v{model.get('version')})")
    except Exception as e:
        logger.error(f"   Failed to get models: {e}")
    
    # Async prediction
    logger.info("\n5. Async Prediction:")
    try:
        job = client.async_predict("Breaking news: Major event happening now")
        job_id = job.get("job_id")
        logger.info(f"   Job submitted: {job_id}")
        
        # Poll for results
        for _ in range(5):
            time.sleep(1)
            status = client.get_job_status(job_id)
            if status.get("status") == "completed":
                logger.info(f"   Result: {status.get('result')}")
                break
            logger.info(f"   Status: {status.get('status')}")
    except Exception as e:
        logger.error(f"   Async prediction failed: {e}")


def demonstrate_grpc(host: str = "localhost", port: int = 50051):
    """
    Demonstrate gRPC API usage.
    
    Parameters
    ----------
    host : str
        gRPC server host
    port : int
        gRPC server port
    """
    if not GRPC_AVAILABLE:
        logger.warning("gRPC not available. Skipping demonstration.")
        return
    
    logger.info("=" * 50)
    logger.info("gRPC API Demonstration")
    logger.info("=" * 50)
    
    client = GRPCClient(host, port)
    
    try:
        # Single prediction
        logger.info("\n1. Single Prediction:")
        text = "Technology company releases new product"
        result = client.predict(text)
        logger.info(f"   Text: {text}")
        logger.info(f"   Label: {result['label']}")
        logger.info(f"   Confidence: {result['confidence']:.4f}")
        
        # Batch prediction
        logger.info("\n2. Batch Prediction:")
        texts = ["News 1", "News 2", "News 3"]
        results = client.predict_batch(texts)
        for i, pred in enumerate(results):
            logger.info(f"   Text {i+1}: {texts[i]}")
            logger.info(f"   -> Label: {pred['label']}")
    
    except grpc.RpcError as e:
        logger.error(f"gRPC error: {e.code()} - {e.details()}")
    finally:
        client.close()


def demonstrate_graphql(endpoint: str = "http://localhost:8000/graphql"):
    """
    Demonstrate GraphQL API usage.
    
    Parameters
    ----------
    endpoint : str
        GraphQL endpoint
    """
    logger.info("=" * 50)
    logger.info("GraphQL API Demonstration")
    logger.info("=" * 50)
    
    try:
        client = GraphQLClient(endpoint)
        
        # Prediction
        logger.info("\n1. Prediction Query:")
        text = "Breaking news from the tech industry"
        result = client.predict(text)
        logger.info(f"   Text: {text}")
        logger.info(f"   Label: {result['label']}")
        logger.info(f"   Confidence: {result['confidence']:.4f}")
        
        # Get models
        logger.info("\n2. Models Query:")
        models = client.get_models()
        for model in models:
            logger.info(f"   - {model['name']} (Accuracy: {model['accuracy']:.4f})")
    
    except Exception as e:
        logger.error(f"GraphQL error: {e}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="API Quick Start for AG News Classification"
    )
    
    parser.add_argument(
        "--api-type",
        choices=["rest", "grpc", "graphql", "all"],
        default="all",
        help="API type to demonstrate"
    )
    
    parser.add_argument(
        "--rest-url",
        default="http://localhost:8000",
        help="REST API base URL"
    )
    
    parser.add_argument(
        "--grpc-host",
        default="localhost",
        help="gRPC server host"
    )
    
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=50051,
        help="gRPC server port"
    )
    
    parser.add_argument(
        "--graphql-endpoint",
        default="http://localhost:8000/graphql",
        help="GraphQL endpoint"
    )
    
    parser.add_argument(
        "--api-key",
        help="API key for authentication"
    )
    
    args = parser.parse_args()
    
    logger.info("AG News Classification - API Quick Start")
    logger.info("========================================\n")
    
    # Demonstrate selected APIs
    if args.api_type in ["rest", "all"]:
        demonstrate_rest_api(args.rest_url)
    
    if args.api_type in ["grpc", "all"]:
        demonstrate_grpc(args.grpc_host, args.grpc_port)
    
    if args.api_type in ["graphql", "all"]:
        demonstrate_graphql(args.graphql_endpoint)
    
    logger.info("\n" + "=" * 50)
    logger.info("API demonstration completed!")
    logger.info("For more examples, see notebooks/tutorials/07_api_usage.ipynb")


if __name__ == "__main__":
    main()
