"""
Load Testing Script using Locust for AG News Text Classification API
================================================================================
This script implements comprehensive load testing scenarios for the API services
using the Locust framework. It simulates realistic user behavior patterns and
measures system performance under various load conditions.

The load testing methodology follows performance engineering best practices
for distributed systems and microservices architectures.

References:
    - Locust Documentation: https://docs.locust.io/
    - Performance Testing Guidance for Web Applications (Microsoft, 2007)
    - The Art of Application Performance Testing (Molyneaux, 2014)

Author: Võ Hải Dũng
License: MIT
"""

import json
import random
import time
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import grpc

from locust import HttpUser, TaskSet, task, between, events
from locust.contrib.fasthttp import FastHttpUser
from locust.runners import MasterRunner, WorkerRunner
import gevent

# Sample test data
TEST_TEXTS = [
    "Technology company announces groundbreaking artificial intelligence breakthrough",
    "Stock market reaches all-time high amid economic recovery optimism",
    "Championship team wins dramatic overtime victory in finals",
    "International summit addresses urgent climate change concerns",
    "Scientists discover potential cure for rare genetic disease",
    "Major corporation reports record quarterly earnings beating expectations",
    "Athletes prepare for upcoming international sports competition",
    "Global leaders meet to discuss trade agreements and partnerships",
    "New smartphone features revolutionary camera technology improvements",
    "Financial markets react to central bank interest rate decision",
    "Sports team signs star player to record-breaking contract",
    "Researchers publish findings on sustainable energy solutions",
]

# Load test configuration
class LoadTestConfig:
    """Configuration for load testing parameters"""
    
    # API endpoints
    REST_BASE_URL = "http://localhost:8000"
    GRPC_HOST = "localhost"
    GRPC_PORT = 50051
    GRAPHQL_URL = "http://localhost:4000/graphql"
    
    # Test parameters
    MIN_WAIT = 1000  # Minimum wait time between tasks (ms)
    MAX_WAIT = 3000  # Maximum wait time between tasks (ms)
    
    # Test data
    BATCH_SIZES = [1, 5, 10, 20, 50]
    MODEL_IDS = ["deberta-v3-base", "roberta-large", "ensemble-voting"]
    
    # Authentication
    USE_AUTH = False
    AUTH_TOKEN = "test_token"
    API_KEY = "test_api_key"


class APITaskSet(TaskSet):
    """
    Base task set for API testing
    
    Implements common functionality for all API test scenarios.
    """
    
    def on_start(self):
        """Initialize user session"""
        self.user_id = str(uuid.uuid4())
        self.request_count = 0
        self.errors = []
        
        # Authenticate if required
        if LoadTestConfig.USE_AUTH:
            self._authenticate()
    
    def _authenticate(self):
        """Perform authentication and store token"""
        try:
            response = self.client.post(
                "/api/v1/auth/login",
                json={
                    "username": f"user_{self.user_id}",
                    "password": "test_password"
                }
            )
            if response.status_code == 200:
                data = response.json()
                self.auth_token = data.get("token")
                self.client.headers.update({
                    "Authorization": f"Bearer {self.auth_token}"
                })
        except Exception as e:
            print(f"Authentication failed: {e}")
    
    def _get_random_text(self) -> str:
        """Get random test text"""
        return random.choice(TEST_TEXTS)
    
    def _get_random_texts(self, count: int) -> List[str]:
        """Get multiple random test texts"""
        return random.choices(TEST_TEXTS, k=count)


class RESTAPITasks(APITaskSet):
    """
    Task set for REST API load testing
    
    Simulates realistic REST API usage patterns including single
    and batch classification requests.
    """
    
    @task(3)
    def classify_single(self):
        """Test single text classification endpoint"""
        text = self._get_random_text()
        
        with self.client.post(
            "/api/v1/classify",
            json={"text": text},
            catch_response=True,
            name="REST: Single Classification"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "predictions" in data:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)
    def classify_batch(self):
        """Test batch classification endpoint"""
        batch_size = random.choice(LoadTestConfig.BATCH_SIZES[:3])
        texts = self._get_random_texts(batch_size)
        
        with self.client.post(
            "/api/v1/classify/batch",
            json={"texts": texts},
            catch_response=True,
            name=f"REST: Batch Classification (size={batch_size})"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "results" in data and len(data["results"]) == batch_size:
                        response.success()
                    else:
                        response.failure("Invalid batch response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def get_models(self):
        """Test model listing endpoint"""
        with self.client.get(
            "/api/v1/models",
            catch_response=True,
            name="REST: List Models"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "models" in data:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Test health check endpoint"""
        with self.client.get(
            "/health",
            catch_response=True,
            name="REST: Health Check"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")


class GraphQLTasks(APITaskSet):
    """
    Task set for GraphQL API load testing
    
    Simulates GraphQL query patterns including single queries,
    batched queries, and subscriptions.
    """
    
    @task(3)
    def classify_query(self):
        """Test GraphQL classification query"""
        text = self._get_random_text()
        
        query = """
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
        """
        
        with self.client.post(
            LoadTestConfig.GRAPHQL_URL,
            json={
                "query": query,
                "variables": {"text": text}
            },
            catch_response=True,
            name="GraphQL: Classification Query"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "data" in data and "classify" in data["data"]:
                        response.success()
                    else:
                        response.failure("Invalid GraphQL response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)
    def batch_query(self):
        """Test GraphQL batch query"""
        texts = self._get_random_texts(3)
        
        query = """
            query BatchClassify($texts: [String!]!) {
                classifyBatch(texts: $texts) {
                    results {
                        predictions {
                            label
                            score
                        }
                    }
                    totalProcessingTime
                }
            }
        """
        
        with self.client.post(
            LoadTestConfig.GRAPHQL_URL,
            json={
                "query": query,
                "variables": {"texts": texts}
            },
            catch_response=True,
            name="GraphQL: Batch Query"
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "data" in data and "classifyBatch" in data["data"]:
                        response.success()
                    else:
                        response.failure("Invalid GraphQL batch response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def introspection_query(self):
        """Test GraphQL introspection query"""
        query = """
            query {
                __schema {
                    types {
                        name
                        kind
                    }
                }
            }
        """
        
        with self.client.post(
            LoadTestConfig.GRAPHQL_URL,
            json={"query": query},
            catch_response=True,
            name="GraphQL: Introspection"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")


class MixedAPIUser(FastHttpUser):
    """
    User class simulating mixed API usage
    
    Represents users that interact with multiple API types
    during their session.
    """
    
    wait_time = between(
        LoadTestConfig.MIN_WAIT / 1000,
        LoadTestConfig.MAX_WAIT / 1000
    )
    
    tasks = {
        RESTAPITasks: 3,
        GraphQLTasks: 1
    }
    
    host = LoadTestConfig.REST_BASE_URL
    
    def on_start(self):
        """Initialize user session"""
        self.session_id = str(uuid.uuid4())
        self.start_time = time.time()
        print(f"User session started: {self.session_id}")
    
    def on_stop(self):
        """Clean up user session"""
        duration = time.time() - self.start_time
        print(f"User session ended: {self.session_id} (duration: {duration:.2f}s)")


class PowerUser(FastHttpUser):
    """
    Power user simulation for stress testing
    
    Simulates heavy API usage with minimal wait times
    and large batch requests.
    """
    
    wait_time = between(0.1, 0.5)  # Minimal wait time
    
    class tasks(TaskSet):
        @task
        def heavy_batch_classification(self):
            """Test with large batch sizes"""
            batch_size = random.choice([50, 100, 200])
            texts = [f"Test text {i}: {random.choice(TEST_TEXTS)}" 
                    for i in range(batch_size)]
            
            with self.client.post(
                "/api/v1/classify/batch",
                json={"texts": texts},
                catch_response=True,
                timeout=30,
                name=f"Power: Large Batch (size={batch_size})"
            ) as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Large batch failed: {response.status_code}")
        
        @task
        def concurrent_requests(self):
            """Send multiple concurrent requests"""
            jobs = []
            for _ in range(5):
                text = random.choice(TEST_TEXTS)
                job = gevent.spawn(
                    self.client.post,
                    "/api/v1/classify",
                    json={"text": text}
                )
                jobs.append(job)
            
            gevent.joinall(jobs)
    
    host = LoadTestConfig.REST_BASE_URL


# Event handlers for distributed testing
@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Initialize distributed testing environment"""
    if isinstance(environment.runner, MasterRunner):
        print("Running in master mode")
    elif isinstance(environment.runner, WorkerRunner):
        print("Running in worker mode")
    else:
        print("Running in standalone mode")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Actions to perform when test starts"""
    print(f"Load test started at {datetime.now()}")
    print(f"Target host: {environment.host}")
    print(f"Total users: {environment.runner.target_user_count}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Actions to perform when test stops"""
    print(f"Load test stopped at {datetime.now()}")
    
    # Print summary statistics
    stats = environment.stats
    print("\n=== Test Summary ===")
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Total failures: {stats.total.num_failures}")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"Min response time: {stats.total.min_response_time:.2f}ms")
    print(f"Max response time: {stats.total.max_response_time:.2f}ms")
    
    # Calculate percentiles
    if stats.total.num_requests > 0:
        print(f"Median response time: {stats.total.median_response_time:.2f}ms")
        print(f"95th percentile: {stats.total.get_response_time_percentile(0.95):.2f}ms")
        print(f"99th percentile: {stats.total.get_response_time_percentile(0.99):.2f}ms")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, 
               response, context, exception, **kwargs):
    """Custom request event handler for detailed logging"""
    if exception:
        print(f"Request failed: {name} - {exception}")
    elif response_time > 5000:  # Log slow requests (> 5 seconds)
        print(f"Slow request detected: {name} - {response_time}ms")


# Custom load shapes for different test scenarios
class StepLoadShape:
    """
    Step load shape for gradual load increase
    
    Gradually increases load to find system breaking point.
    """
    
    step_time = 60  # Time per step in seconds
    step_users = 10  # Users to add per step
    max_users = 100
    
    def tick(self, run_time):
        """Calculate current user count based on run time"""
        current_step = run_time // self.step_time
        
        if current_step * self.step_users >= self.max_users:
            return self.max_users, self.step_users
        
        return current_step * self.step_users, self.step_users


class SpikeLoadShape:
    """
    Spike load shape for testing sudden load increases
    
    Simulates traffic spikes to test system resilience.
    """
    
    stages = [
        {"duration": 60, "users": 10, "spawn_rate": 1},
        {"duration": 30, "users": 100, "spawn_rate": 10},  # Spike
        {"duration": 60, "users": 10, "spawn_rate": 5},
        {"duration": 30, "users": 200, "spawn_rate": 20},  # Bigger spike
        {"duration": 60, "users": 10, "spawn_rate": 5},
    ]
    
    def tick(self, run_time):
        """Calculate current user count based on stages"""
        total_time = 0
        for stage in self.stages:
            total_time += stage["duration"]
            if run_time <= total_time:
                return stage["users"], stage["spawn_rate"]
        
        return None  # Stop test


# Main execution
if __name__ == "__main__":
    import os
    
    # Set default host if not provided
    if not os.environ.get("LOCUST_HOST"):
        os.environ["LOCUST_HOST"] = LoadTestConfig.REST_BASE_URL
    
    # Run with: locust -f locust_test.py --users 100 --spawn-rate 10
    print("Locust load testing script ready")
    print(f"Target host: {os.environ.get('LOCUST_HOST')}")
    print("Run with: locust -f locust_test.py --users 100 --spawn-rate 10")
