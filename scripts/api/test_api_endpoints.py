"""
API Endpoints Testing Script for AG News Text Classification System
================================================================================
This script provides comprehensive testing for all API endpoints including
REST, gRPC, and GraphQL interfaces. It performs functional testing, load
testing, and integration testing to ensure API reliability and performance.

The testing methodology follows best practices for API testing as outlined
in software engineering literature and industry standards.

References:
    - Richardson, L., & Ruby, S. (2013). RESTful Web APIs
    - Martin, R. C. (2017). Clean Architecture
    - Google Cloud API Design Guide

Author: Võ Hải Dũng
License: MIT
"""

import sys
import os
import json
import time
import asyncio
import argparse
import random
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import concurrent.futures
import statistics

import requests
import grpc
import aiohttp
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TimeRemainingColumn
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.grpc.compiled import classification_pb2, classification_pb2_grpc
from src.api.grpc.compiled import health_pb2, health_pb2_grpc

# Configure console
console = Console()


class APITester:
    """
    Comprehensive API testing framework
    
    Provides methods for testing REST, gRPC, and GraphQL endpoints
    with support for functional, load, and integration testing.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize API tester
        
        Args:
            config_path: Path to test configuration
        """
        self.config = self.load_configuration(config_path)
        self.results: Dict[str, List[Dict[str, Any]]] = {
            "rest": [],
            "grpc": [],
            "graphql": []
        }
        self.test_data = self.load_test_data()
    
    def load_configuration(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load test configuration"""
        default_config = {
            "endpoints": {
                "rest": {
                    "base_url": "http://localhost:8000",
                    "version": "v1"
                },
                "grpc": {
                    "host": "localhost",
                    "port": 50051
                },
                "graphql": {
                    "url": "http://localhost:4000/graphql"
                }
            },
            "auth": {
                "enabled": False,
                "token": "",
                "api_key": ""
            },
            "test_settings": {
                "timeout": 30,
                "retries": 3,
                "load_test_duration": 60,
                "concurrent_users": 10
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    default_config.update(loaded)
        
        return default_config
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """Load test data samples"""
        return [
            {
                "text": "The stock market reached new highs today as investors showed confidence.",
                "expected_category": "Business"
            },
            {
                "text": "Scientists discover new species in the Amazon rainforest.",
                "expected_category": "Sci/Tech"
            },
            {
                "text": "The championship game ended with a dramatic overtime victory.",
                "expected_category": "Sports"
            },
            {
                "text": "International summit addresses climate change concerns.",
                "expected_category": "World"
            }
        ]
    
    # REST API Testing
    async def test_rest_endpoints(self) -> Dict[str, Any]:
        """Test all REST API endpoints"""
        console.print("\n[bold blue]Testing REST API Endpoints[/bold blue]\n")
        
        base_url = self.config["endpoints"]["rest"]["base_url"]
        version = self.config["endpoints"]["rest"]["version"]
        
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "endpoints": []
        }
        
        # Test health endpoint
        endpoint_result = await self.test_rest_health(base_url)
        results["endpoints"].append(endpoint_result)
        results["total"] += 1
        if endpoint_result["status"] == "passed":
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        # Test classification endpoint
        endpoint_result = await self.test_rest_classification(base_url, version)
        results["endpoints"].append(endpoint_result)
        results["total"] += 1
        if endpoint_result["status"] == "passed":
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        # Test batch classification
        endpoint_result = await self.test_rest_batch_classification(base_url, version)
        results["endpoints"].append(endpoint_result)
        results["total"] += 1
        if endpoint_result["status"] == "passed":
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        # Test models endpoint
        endpoint_result = await self.test_rest_models(base_url, version)
        results["endpoints"].append(endpoint_result)
        results["total"] += 1
        if endpoint_result["status"] == "passed":
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        return results
    
    async def test_rest_health(self, base_url: str) -> Dict[str, Any]:
        """Test REST health endpoint"""
        endpoint = f"{base_url}/health"
        
        try:
            start_time = time.time()
            response = requests.get(endpoint, timeout=5)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                return {
                    "endpoint": "GET /health",
                    "status": "passed",
                    "response_time_ms": response_time,
                    "status_code": response.status_code
                }
            else:
                return {
                    "endpoint": "GET /health",
                    "status": "failed",
                    "response_time_ms": response_time,
                    "status_code": response.status_code,
                    "error": response.text
                }
        except Exception as e:
            return {
                "endpoint": "GET /health",
                "status": "failed",
                "error": str(e)
            }
    
    async def test_rest_classification(self, base_url: str, version: str) -> Dict[str, Any]:
        """Test REST classification endpoint"""
        endpoint = f"{base_url}/api/{version}/classify"
        
        try:
            test_sample = self.test_data[0]
            payload = {"text": test_sample["text"]}
            
            headers = {}
            if self.config["auth"]["enabled"]:
                headers["Authorization"] = f"Bearer {self.config['auth']['token']}"
            
            start_time = time.time()
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=10
            )
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "endpoint": "POST /api/v1/classify",
                    "status": "passed",
                    "response_time_ms": response_time,
                    "status_code": response.status_code,
                    "response": data
                }
            else:
                return {
                    "endpoint": "POST /api/v1/classify",
                    "status": "failed",
                    "response_time_ms": response_time,
                    "status_code": response.status_code,
                    "error": response.text
                }
        except Exception as e:
            return {
                "endpoint": "POST /api/v1/classify",
                "status": "failed",
                "error": str(e)
            }
    
    async def test_rest_batch_classification(self, base_url: str, version: str) -> Dict[str, Any]:
        """Test REST batch classification endpoint"""
        endpoint = f"{base_url}/api/{version}/classify/batch"
        
        try:
            texts = [sample["text"] for sample in self.test_data]
            payload = {"texts": texts}
            
            headers = {}
            if self.config["auth"]["enabled"]:
                headers["Authorization"] = f"Bearer {self.config['auth']['token']}"
            
            start_time = time.time()
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=30
            )
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "endpoint": "POST /api/v1/classify/batch",
                    "status": "passed",
                    "response_time_ms": response_time,
                    "status_code": response.status_code,
                    "batch_size": len(texts),
                    "response": data
                }
            else:
                return {
                    "endpoint": "POST /api/v1/classify/batch",
                    "status": "failed",
                    "response_time_ms": response_time,
                    "status_code": response.status_code,
                    "error": response.text
                }
        except Exception as e:
            return {
                "endpoint": "POST /api/v1/classify/batch",
                "status": "failed",
                "error": str(e)
            }
    
    async def test_rest_models(self, base_url: str, version: str) -> Dict[str, Any]:
        """Test REST models endpoint"""
        endpoint = f"{base_url}/api/{version}/models"
        
        try:
            headers = {}
            if self.config["auth"]["enabled"]:
                headers["Authorization"] = f"Bearer {self.config['auth']['token']}"
            
            start_time = time.time()
            response = requests.get(
                endpoint,
                headers=headers,
                timeout=10
            )
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "endpoint": "GET /api/v1/models",
                    "status": "passed",
                    "response_time_ms": response_time,
                    "status_code": response.status_code,
                    "models_count": len(data.get("models", []))
                }
            else:
                return {
                    "endpoint": "GET /api/v1/models",
                    "status": "failed",
                    "response_time_ms": response_time,
                    "status_code": response.status_code,
                    "error": response.text
                }
        except Exception as e:
            return {
                "endpoint": "GET /api/v1/models",
                "status": "failed",
                "error": str(e)
            }
    
    # gRPC Testing
    async def test_grpc_endpoints(self) -> Dict[str, Any]:
        """Test all gRPC endpoints"""
        console.print("\n[bold blue]Testing gRPC Endpoints[/bold blue]\n")
        
        host = self.config["endpoints"]["grpc"]["host"]
        port = self.config["endpoints"]["grpc"]["port"]
        
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "endpoints": []
        }
        
        # Test health service
        endpoint_result = await self.test_grpc_health(host, port)
        results["endpoints"].append(endpoint_result)
        results["total"] += 1
        if endpoint_result["status"] == "passed":
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        # Test classification service
        endpoint_result = await self.test_grpc_classification(host, port)
        results["endpoints"].append(endpoint_result)
        results["total"] += 1
        if endpoint_result["status"] == "passed":
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        return results
    
    async def test_grpc_health(self, host: str, port: int) -> Dict[str, Any]:
        """Test gRPC health service"""
        try:
            channel = grpc.insecure_channel(f"{host}:{port}")
            stub = health_pb2_grpc.HealthServiceStub(channel)
            
            request = health_pb2.HealthCheckRequest(service="")
            
            start_time = time.time()
            response = stub.Check(request, timeout=5)
            response_time = (time.time() - start_time) * 1000
            
            return {
                "endpoint": "grpc.health.v1.Health/Check",
                "status": "passed",
                "response_time_ms": response_time,
                "health_status": response.status
            }
        except Exception as e:
            return {
                "endpoint": "grpc.health.v1.Health/Check",
                "status": "failed",
                "error": str(e)
            }
        finally:
            channel.close()
    
    async def test_grpc_classification(self, host: str, port: int) -> Dict[str, Any]:
        """Test gRPC classification service"""
        try:
            channel = grpc.insecure_channel(f"{host}:{port}")
            stub = classification_pb2_grpc.ClassificationServiceStub(channel)
            
            test_sample = self.test_data[0]
            request = classification_pb2.ClassificationRequest(
                text=test_sample["text"]
            )
            
            start_time = time.time()
            response = stub.Classify(request, timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            return {
                "endpoint": "agnews.classification.v1.ClassificationService/Classify",
                "status": "passed",
                "response_time_ms": response_time,
                "predictions": len(response.predictions)
            }
        except Exception as e:
            return {
                "endpoint": "agnews.classification.v1.ClassificationService/Classify",
                "status": "failed",
                "error": str(e)
            }
        finally:
            channel.close()
    
    # GraphQL Testing
    async def test_graphql_endpoints(self) -> Dict[str, Any]:
        """Test GraphQL endpoints"""
        console.print("\n[bold blue]Testing GraphQL Endpoints[/bold blue]\n")
        
        url = self.config["endpoints"]["graphql"]["url"]
        
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "endpoints": []
        }
        
        # Test classification query
        endpoint_result = await self.test_graphql_classification(url)
        results["endpoints"].append(endpoint_result)
        results["total"] += 1
        if endpoint_result["status"] == "passed":
            results["passed"] += 1
        else:
            results["failed"] += 1
        
        return results
    
    async def test_graphql_classification(self, url: str) -> Dict[str, Any]:
        """Test GraphQL classification query"""
        try:
            transport = AIOHTTPTransport(url=url)
            
            async with Client(
                transport=transport,
                fetch_schema_from_transport=True
            ) as session:
                test_sample = self.test_data[0]
                
                query = gql("""
                    query ClassifyText($text: String!) {
                        classify(text: $text) {
                            predictions {
                                label
                                score
                            }
                            modelId
                            processingTime
                        }
                    }
                """)
                
                params = {"text": test_sample["text"]}
                
                start_time = time.time()
                result = await session.execute(query, variable_values=params)
                response_time = (time.time() - start_time) * 1000
                
                return {
                    "endpoint": "Query: classify",
                    "status": "passed",
                    "response_time_ms": response_time,
                    "response": result
                }
        except Exception as e:
            return {
                "endpoint": "Query: classify",
                "status": "failed",
                "error": str(e)
            }
    
    # Load Testing
    async def run_load_test(self, api_type: str = "rest") -> Dict[str, Any]:
        """Run load testing on specified API"""
        console.print(f"\n[bold yellow]Running Load Test on {api_type.upper()} API[/bold yellow]\n")
        
        duration = self.config["test_settings"]["load_test_duration"]
        concurrent_users = self.config["test_settings"]["concurrent_users"]
        
        results = {
            "api_type": api_type,
            "duration_seconds": duration,
            "concurrent_users": concurrent_users,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "errors": []
        }
        
        start_time = time.time()
        tasks = []
        
        with Progress() as progress:
            task = progress.add_task(
                f"Load testing {api_type}...",
                total=duration
            )
            
            while time.time() - start_time < duration:
                # Create concurrent requests
                for _ in range(concurrent_users):
                    if api_type == "rest":
                        tasks.append(self._load_test_rest_request())
                    elif api_type == "grpc":
                        tasks.append(self._load_test_grpc_request())
                    elif api_type == "graphql":
                        tasks.append(self._load_test_graphql_request())
                
                # Wait for batch to complete
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in batch_results:
                    results["total_requests"] += 1
                    if isinstance(result, Exception):
                        results["failed_requests"] += 1
                        results["errors"].append(str(result))
                    elif result["success"]:
                        results["successful_requests"] += 1
                        results["response_times"].append(result["response_time"])
                    else:
                        results["failed_requests"] += 1
                
                tasks.clear()
                progress.update(task, advance=1)
                await asyncio.sleep(1)
        
        # Calculate statistics
        if results["response_times"]:
            results["avg_response_time"] = statistics.mean(results["response_times"])
            results["min_response_time"] = min(results["response_times"])
            results["max_response_time"] = max(results["response_times"])
            results["p95_response_time"] = statistics.quantiles(
                results["response_times"], n=20
            )[18] if len(results["response_times"]) > 20 else max(results["response_times"])
        
        results["success_rate"] = (
            results["successful_requests"] / results["total_requests"] * 100
            if results["total_requests"] > 0 else 0
        )
        
        return results
    
    async def _load_test_rest_request(self) -> Dict[str, Any]:
        """Single REST request for load testing"""
        try:
            base_url = self.config["endpoints"]["rest"]["base_url"]
            version = self.config["endpoints"]["rest"]["version"]
            endpoint = f"{base_url}/api/{version}/classify"
            
            test_sample = random.choice(self.test_data)
            payload = {"text": test_sample["text"]}
            
            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, json=payload) as response:
                    await response.text()
                    response_time = (time.time() - start_time) * 1000
                    
                    return {
                        "success": response.status == 200,
                        "response_time": response_time
                    }
        except Exception as e:
            raise e
    
    async def _load_test_grpc_request(self) -> Dict[str, Any]:
        """Single gRPC request for load testing"""
        try:
            host = self.config["endpoints"]["grpc"]["host"]
            port = self.config["endpoints"]["grpc"]["port"]
            
            channel = grpc.aio.insecure_channel(f"{host}:{port}")
            stub = classification_pb2_grpc.ClassificationServiceStub(channel)
            
            test_sample = random.choice(self.test_data)
            request = classification_pb2.ClassificationRequest(
                text=test_sample["text"]
            )
            
            start_time = time.time()
            response = await stub.Classify(request, timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            await channel.close()
            
            return {
                "success": True,
                "response_time": response_time
            }
        except Exception as e:
            raise e
    
    async def _load_test_graphql_request(self) -> Dict[str, Any]:
        """Single GraphQL request for load testing"""
        try:
            url = self.config["endpoints"]["graphql"]["url"]
            
            test_sample = random.choice(self.test_data)
            
            query = """
                query ClassifyText($text: String!) {
                    classify(text: $text) {
                        predictions {
                            label
                            score
                        }
                    }
                }
            """
            
            payload = {
                "query": query,
                "variables": {"text": test_sample["text"]}
            }
            
            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    await response.text()
                    response_time = (time.time() - start_time) * 1000
                    
                    return {
                        "success": response.status == 200,
                        "response_time": response_time
                    }
        except Exception as e:
            raise e
    
    def generate_report(self, results: Dict[str, Any]):
        """Generate test report"""
        console.print("\n[bold green]API Test Report[/bold green]\n")
        console.print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # REST API Results
        if "rest" in results:
            table = Table(title="REST API Test Results")
            table.add_column("Endpoint", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Response Time (ms)", style="yellow")
            table.add_column("Details", style="magenta")
            
            for endpoint in results["rest"]["endpoints"]:
                status = "✓" if endpoint["status"] == "passed" else "✗"
                response_time = f"{endpoint.get('response_time_ms', 'N/A'):.2f}" if isinstance(endpoint.get('response_time_ms'), (int, float)) else "N/A"
                details = endpoint.get("error", "Success")[:50] if endpoint["status"] == "failed" else "Success"
                
                table.add_row(
                    endpoint["endpoint"],
                    status,
                    response_time,
                    details
                )
            
            console.print(table)
            console.print(f"\nREST API Summary: {results['rest']['passed']}/{results['rest']['total']} passed\n")
        
        # gRPC Results
        if "grpc" in results:
            table = Table(title="gRPC Test Results")
            table.add_column("Service", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Response Time (ms)", style="yellow")
            table.add_column("Details", style="magenta")
            
            for endpoint in results["grpc"]["endpoints"]:
                status = "✓" if endpoint["status"] == "passed" else "✗"
                response_time = f"{endpoint.get('response_time_ms', 'N/A'):.2f}" if isinstance(endpoint.get('response_time_ms'), (int, float)) else "N/A"
                details = endpoint.get("error", "Success")[:50] if endpoint["status"] == "failed" else "Success"
                
                table.add_row(
                    endpoint["endpoint"],
                    status,
                    response_time,
                    details
                )
            
            console.print(table)
            console.print(f"\ngRPC Summary: {results['grpc']['passed']}/{results['grpc']['total']} passed\n")
        
        # GraphQL Results
        if "graphql" in results:
            table = Table(title="GraphQL Test Results")
            table.add_column("Query", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Response Time (ms)", style="yellow")
            table.add_column("Details", style="magenta")
            
            for endpoint in results["graphql"]["endpoints"]:
                status = "✓" if endpoint["status"] == "passed" else "✗"
                response_time = f"{endpoint.get('response_time_ms', 'N/A'):.2f}" if isinstance(endpoint.get('response_time_ms'), (int, float)) else "N/A"
                details = endpoint.get("error", "Success")[:50] if endpoint["status"] == "failed" else "Success"
                
                table.add_row(
                    endpoint["endpoint"],
                    status,
                    response_time,
                    details
                )
            
            console.print(table)
            console.print(f"\nGraphQL Summary: {results['graphql']['passed']}/{results['graphql']['total']} passed\n")
        
        # Load Test Results
        if "load_test" in results:
            table = Table(title="Load Test Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="yellow")
            
            load_results = results["load_test"]
            table.add_row("API Type", load_results["api_type"].upper())
            table.add_row("Duration (seconds)", str(load_results["duration_seconds"]))
            table.add_row("Concurrent Users", str(load_results["concurrent_users"]))
            table.add_row("Total Requests", str(load_results["total_requests"]))
            table.add_row("Successful Requests", str(load_results["successful_requests"]))
            table.add_row("Failed Requests", str(load_results["failed_requests"]))
            table.add_row("Success Rate", f"{load_results.get('success_rate', 0):.2f}%")
            
            if "avg_response_time" in load_results:
                table.add_row("Avg Response Time (ms)", f"{load_results['avg_response_time']:.2f}")
                table.add_row("Min Response Time (ms)", f"{load_results['min_response_time']:.2f}")
                table.add_row("Max Response Time (ms)", f"{load_results['max_response_time']:.2f}")
                table.add_row("P95 Response Time (ms)", f"{load_results['p95_response_time']:.2f}")
            
            console.print(table)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Test API endpoints for AG News Text Classification"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to test configuration file"
    )
    parser.add_argument(
        "--apis",
        nargs="+",
        choices=["rest", "grpc", "graphql"],
        default=["rest", "grpc", "graphql"],
        help="APIs to test"
    )
    parser.add_argument(
        "--load-test",
        action="store_true",
        help="Run load testing"
    )
    parser.add_argument(
        "--load-test-api",
        choices=["rest", "grpc", "graphql"],
        default="rest",
        help="API to load test"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for test results (JSON)"
    )
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = APITester(args.config)
    
    results = {}
    
    try:
        # Run functional tests
        if "rest" in args.apis:
            results["rest"] = await tester.test_rest_endpoints()
        
        if "grpc" in args.apis:
            results["grpc"] = await tester.test_grpc_endpoints()
        
        if "graphql" in args.apis:
            results["graphql"] = await tester.test_graphql_endpoints()
        
        # Run load test if requested
        if args.load_test:
            results["load_test"] = await tester.run_load_test(args.load_test_api)
        
        # Generate report
        tester.generate_report(results)
        
        # Save results if output specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            console.print(f"\n[green]Results saved to {args.output}[/green]")
        
    except Exception as e:
        console.print(f"\n[bold red]Error during testing: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
