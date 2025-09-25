"""
API Documentation Generation Script for AG News Text Classification System
================================================================================
This script automatically generates comprehensive API documentation from source
code, including REST endpoints, gRPC services, and GraphQL schemas. It produces
both human-readable documentation and machine-readable specifications.

The documentation generation follows OpenAPI 3.0, Protocol Buffers, and GraphQL
introspection standards for comprehensive API documentation.

References:
    - OpenAPI Specification 3.0: https://swagger.io/specification/
    - Protocol Buffers Language Guide: https://developers.google.com/protocol-buffers
    - GraphQL Introspection: https://graphql.org/learn/introspection/

Author: Võ Hải Dũng
License: MIT
"""

import sys
import os
import json
import yaml
import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import argparse
import inspect
from dataclasses import dataclass, asdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
import grpc_tools.protoc
from graphql import build_schema, print_schema
from jinja2 import Template, Environment, FileSystemLoader
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import mistune

# Configure console
console = Console()


@dataclass
class APIEndpoint:
    """API endpoint documentation structure"""
    path: str
    method: str
    summary: str
    description: str
    parameters: List[Dict[str, Any]]
    request_body: Optional[Dict[str, Any]]
    responses: Dict[str, Any]
    tags: List[str]
    authentication: bool = False


@dataclass
class ServiceMethod:
    """Service method documentation structure"""
    name: str
    service: str
    input_type: str
    output_type: str
    description: str
    streaming: str = "none"  # none, client, server, bidirectional


class APIDocumentationGenerator:
    """
    Generates comprehensive API documentation
    
    This class extracts API definitions from source code and generates
    documentation in multiple formats including Markdown, HTML, and OpenAPI.
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize documentation generator
        
        Args:
            output_dir: Output directory for generated documentation
        """
        self.output_dir = Path(output_dir or PROJECT_ROOT / "docs" / "api_reference")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.rest_endpoints: List[APIEndpoint] = []
        self.grpc_services: List[ServiceMethod] = []
        self.graphql_schema: Optional[str] = None
        
        # Template environment
        template_dir = PROJECT_ROOT / "templates" / "api_docs"
        if template_dir.exists():
            self.env = Environment(loader=FileSystemLoader(str(template_dir)))
        else:
            self.env = Environment()
    
    def generate_all_documentation(self):
        """Generate documentation for all API types"""
        console.print("[bold blue]Generating API Documentation[/bold blue]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            # Generate REST API documentation
            task = progress.add_task("Generating REST API docs...", total=1)
            self.generate_rest_docs()
            progress.update(task, completed=1)
            
            # Generate gRPC documentation
            task = progress.add_task("Generating gRPC docs...", total=1)
            self.generate_grpc_docs()
            progress.update(task, completed=1)
            
            # Generate GraphQL documentation
            task = progress.add_task("Generating GraphQL docs...", total=1)
            self.generate_graphql_docs()
            progress.update(task, completed=1)
            
            # Generate combined documentation
            task = progress.add_task("Generating combined docs...", total=1)
            self.generate_combined_docs()
            progress.update(task, completed=1)
            
            # Generate OpenAPI specification
            task = progress.add_task("Generating OpenAPI spec...", total=1)
            self.generate_openapi_spec()
            progress.update(task, completed=1)
        
        console.print(f"\n[green]Documentation generated in {self.output_dir}[/green]")
    
    def generate_rest_docs(self):
        """Generate REST API documentation"""
        try:
            # Import FastAPI app
            from src.api.rest.app import app
            
            # Extract OpenAPI schema
            openapi_schema = get_openapi(
                title="AG News Text Classification API",
                version="1.0.0",
                description="REST API for text classification using state-of-the-art models",
                routes=app.routes,
            )
            
            # Parse endpoints
            for path, path_item in openapi_schema.get("paths", {}).items():
                for method, operation in path_item.items():
                    if method in ["get", "post", "put", "delete", "patch"]:
                        endpoint = APIEndpoint(
                            path=path,
                            method=method.upper(),
                            summary=operation.get("summary", ""),
                            description=operation.get("description", ""),
                            parameters=operation.get("parameters", []),
                            request_body=operation.get("requestBody"),
                            responses=operation.get("responses", {}),
                            tags=operation.get("tags", []),
                            authentication="security" in operation
                        )
                        self.rest_endpoints.append(endpoint)
            
            # Generate Markdown documentation
            self._generate_rest_markdown()
            
        except ImportError as e:
            console.print(f"[yellow]Warning: Could not import REST API: {e}[/yellow]")
    
    def _generate_rest_markdown(self):
        """Generate Markdown documentation for REST API"""
        output_file = self.output_dir / "rest_api.md"
        
        content = [
            "# REST API Documentation",
            "",
            "## Overview",
            "",
            "The AG News Text Classification REST API provides endpoints for text classification,",
            "model management, and system monitoring.",
            "",
            "**Base URL**: `http://localhost:8000/api/v1`",
            "",
            "**Authentication**: Bearer token or API key",
            "",
            "## Endpoints",
            ""
        ]
        
        # Group endpoints by tag
        endpoints_by_tag: Dict[str, List[APIEndpoint]] = {}
        for endpoint in self.rest_endpoints:
            for tag in endpoint.tags or ["General"]:
                if tag not in endpoints_by_tag:
                    endpoints_by_tag[tag] = []
                endpoints_by_tag[tag].append(endpoint)
        
        for tag, endpoints in sorted(endpoints_by_tag.items()):
            content.append(f"### {tag}")
            content.append("")
            
            for endpoint in endpoints:
                content.append(f"#### {endpoint.method} {endpoint.path}")
                content.append("")
                
                if endpoint.summary:
                    content.append(f"**Summary**: {endpoint.summary}")
                    content.append("")
                
                if endpoint.description:
                    content.append(endpoint.description)
                    content.append("")
                
                if endpoint.parameters:
                    content.append("**Parameters**:")
                    content.append("")
                    content.append("| Name | Type | Required | Description |")
                    content.append("|------|------|----------|-------------|")
                    
                    for param in endpoint.parameters:
                        name = param.get("name", "")
                        param_type = param.get("schema", {}).get("type", "string")
                        required = "Yes" if param.get("required", False) else "No"
                        description = param.get("description", "")
                        content.append(f"| {name} | {param_type} | {required} | {description} |")
                    content.append("")
                
                if endpoint.request_body:
                    content.append("**Request Body**:")
                    content.append("")
                    content.append("```json")
                    # Extract schema example
                    schema = endpoint.request_body.get("content", {}).get(
                        "application/json", {}
                    ).get("schema", {})
                    content.append(json.dumps(self._generate_example_from_schema(schema), indent=2))
                    content.append("```")
                    content.append("")
                
                if endpoint.responses:
                    content.append("**Responses**:")
                    content.append("")
                    
                    for status_code, response in endpoint.responses.items():
                        description = response.get("description", "")
                        content.append(f"- **{status_code}**: {description}")
                    content.append("")
                
                content.append("---")
                content.append("")
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(content))
        
        console.print(f"[green]REST API documentation saved to {output_file}[/green]")
    
    def generate_grpc_docs(self):
        """Generate gRPC service documentation"""
        proto_dir = PROJECT_ROOT / "src" / "api" / "grpc" / "protos"
        
        if not proto_dir.exists():
            console.print("[yellow]Warning: gRPC proto directory not found[/yellow]")
            return
        
        # Parse proto files
        for proto_file in proto_dir.glob("*.proto"):
            if proto_file.name.startswith("common"):
                continue
            
            self._parse_proto_file(proto_file)
        
        # Generate Markdown documentation
        self._generate_grpc_markdown()
    
    def _parse_proto_file(self, proto_file: Path):
        """Parse a proto file to extract service definitions"""
        with open(proto_file, 'r') as f:
            content = f.read()
        
        # Regular expressions for parsing proto
        service_pattern = r'service\s+(\w+)\s*\{([^}]*)\}'
        rpc_pattern = r'rpc\s+(\w+)\s*KATEX_INLINE_OPEN([^)]+)KATEX_INLINE_CLOSE\s*returns\s*KATEX_INLINE_OPEN([^)]+)KATEX_INLINE_CLOSE'
        
        # Find all services
        services = re.findall(service_pattern, content, re.DOTALL)
        
        for service_name, service_body in services:
            # Find all RPC methods
            methods = re.findall(rpc_pattern, service_body)
            
            for method_name, input_type, output_type in methods:
                # Determine streaming type
                streaming = "none"
                if "stream" in input_type and "stream" in output_type:
                    streaming = "bidirectional"
                elif "stream" in input_type:
                    streaming = "client"
                elif "stream" in output_type:
                    streaming = "server"
                
                # Clean type names
                input_type = input_type.replace("stream", "").strip()
                output_type = output_type.replace("stream", "").strip()
                
                method = ServiceMethod(
                    name=method_name,
                    service=service_name,
                    input_type=input_type,
                    output_type=output_type,
                    description=f"{method_name} method for {service_name}",
                    streaming=streaming
                )
                self.grpc_services.append(method)
    
    def _generate_grpc_markdown(self):
        """Generate Markdown documentation for gRPC services"""
        output_file = self.output_dir / "grpc_api.md"
        
        content = [
            "# gRPC API Documentation",
            "",
            "## Overview",
            "",
            "The AG News Text Classification gRPC API provides high-performance",
            "RPC methods for text classification and model management.",
            "",
            "**Server Address**: `localhost:50051`",
            "",
            "## Services",
            ""
        ]
        
        # Group methods by service
        services: Dict[str, List[ServiceMethod]] = {}
        for method in self.grpc_services:
            if method.service not in services:
                services[method.service] = []
            services[method.service].append(method)
        
        for service_name, methods in sorted(services.items()):
            content.append(f"### {service_name}")
            content.append("")
            
            for method in methods:
                content.append(f"#### {method.name}")
                content.append("")
                content.append(f"**Description**: {method.description}")
                content.append("")
                content.append(f"**Input**: `{method.input_type}`")
                content.append("")
                content.append(f"**Output**: `{method.output_type}`")
                content.append("")
                
                if method.streaming != "none":
                    content.append(f"**Streaming**: {method.streaming}")
                    content.append("")
                
                content.append("---")
                content.append("")
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(content))
        
        console.print(f"[green]gRPC documentation saved to {output_file}[/green]")
    
    def generate_graphql_docs(self):
        """Generate GraphQL schema documentation"""
        schema_file = PROJECT_ROOT / "src" / "api" / "graphql" / "schema.graphql"
        
        # Default schema if file doesn't exist
        default_schema = """
        type Query {
            classify(text: String!): ClassificationResult!
            classifyBatch(texts: [String!]!): BatchClassificationResult!
            getModel(id: ID!): Model
            listModels(limit: Int = 10, offset: Int = 0): ModelList!
        }
        
        type Mutation {
            startTraining(config: TrainingConfig!): TrainingJob!
            stopTraining(jobId: ID!): TrainingJob!
            deployModel(modelId: ID!, environment: String!): DeploymentResult!
        }
        
        type Subscription {
            trainingProgress(jobId: ID!): TrainingProgress!
            modelUpdates: ModelUpdate!
        }
        
        type ClassificationResult {
            predictions: [Prediction!]!
            modelId: String!
            processingTime: Float!
        }
        
        type Prediction {
            label: String!
            score: Float!
            classId: Int!
        }
        """
        
        if schema_file.exists():
            with open(schema_file, 'r') as f:
                self.graphql_schema = f.read()
        else:
            self.graphql_schema = default_schema
        
        # Generate Markdown documentation
        self._generate_graphql_markdown()
    
    def _generate_graphql_markdown(self):
        """Generate Markdown documentation for GraphQL API"""
        output_file = self.output_dir / "graphql_api.md"
        
        content = [
            "# GraphQL API Documentation",
            "",
            "## Overview",
            "",
            "The AG News Text Classification GraphQL API provides a flexible",
            "query language for accessing classification services.",
            "",
            "**Endpoint**: `http://localhost:4000/graphql`",
            "",
            "**Playground**: `http://localhost:4000/graphql`",
            "",
            "## Schema",
            "",
            "```graphql",
            self.graphql_schema,
            "```",
            "",
            "## Example Queries",
            "",
            "### Classification Query",
            "",
            "```graphql",
            "query ClassifyText {",
            "  classify(text: \"Breaking news about technology\") {",
            "    predictions {",
            "      label",
            "      score",
            "    }",
            "    modelId",
            "    processingTime",
            "  }",
            "}",
            "```",
            "",
            "### Batch Classification",
            "",
            "```graphql",
            "query BatchClassify {",
            "  classifyBatch(texts: [",
            "    \"Tech company announces new product\",",
            "    \"Sports team wins championship\"",
            "  ]) {",
            "    results {",
            "      predictions {",
            "        label",
            "        score",
            "      }",
            "    }",
            "  }",
            "}",
            "```",
            ""
        ]
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(content))
        
        console.print(f"[green]GraphQL documentation saved to {output_file}[/green]")
    
    def generate_combined_docs(self):
        """Generate combined documentation for all APIs"""
        output_file = self.output_dir / "api_documentation.md"
        
        content = [
            "# AG News Text Classification API Documentation",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Table of Contents",
            "",
            "1. [Overview](#overview)",
            "2. [REST API](#rest-api)",
            "3. [gRPC API](#grpc-api)",
            "4. [GraphQL API](#graphql-api)",
            "5. [Authentication](#authentication)",
            "6. [Rate Limiting](#rate-limiting)",
            "7. [Error Handling](#error-handling)",
            "",
            "## Overview",
            "",
            "The AG News Text Classification system provides three API interfaces:",
            "",
            "- **REST API**: Traditional HTTP/JSON interface",
            "- **gRPC API**: High-performance RPC interface",
            "- **GraphQL API**: Flexible query language interface",
            "",
            "All APIs provide access to the same underlying services:",
            "",
            "- Text classification using state-of-the-art models",
            "- Model management and deployment",
            "- Training job management",
            "- System monitoring and health checks",
            "",
            "## REST API",
            "",
            f"Total Endpoints: {len(self.rest_endpoints)}",
            "",
            "See [REST API Documentation](rest_api.md) for details.",
            "",
            "## gRPC API",
            "",
            f"Total Services: {len(set(m.service for m in self.grpc_services))}",
            f"Total Methods: {len(self.grpc_services)}",
            "",
            "See [gRPC API Documentation](grpc_api.md) for details.",
            "",
            "## GraphQL API",
            "",
            "Provides Query, Mutation, and Subscription operations.",
            "",
            "See [GraphQL API Documentation](graphql_api.md) for details.",
            "",
            "## Authentication",
            "",
            "### Bearer Token",
            "",
            "```http",
            "Authorization: Bearer <token>",
            "```",
            "",
            "### API Key",
            "",
            "```http",
            "X-API-Key: <api_key>",
            "```",
            "",
            "## Rate Limiting",
            "",
            "Default limits:",
            "- 1000 requests per minute",
            "- 10000 requests per hour",
            "",
            "Rate limit headers:",
            "- `X-RateLimit-Limit`: Request limit",
            "- `X-RateLimit-Remaining`: Remaining requests",
            "- `X-RateLimit-Reset`: Reset timestamp",
            "",
            "## Error Handling",
            "",
            "All APIs use consistent error response format:",
            "",
            "```json",
            "{",
            "  \"success\": false,",
            "  \"error\": {",
            "    \"code\": 400,",
            "    \"type\": \"BAD_REQUEST\",",
            "    \"message\": \"Invalid input\",",
            "    \"details\": {}",
            "  }",
            "}",
            "```",
            ""
        ]
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(content))
        
        console.print(f"[green]Combined documentation saved to {output_file}[/green]")
    
    def generate_openapi_spec(self):
        """Generate OpenAPI specification"""
        output_file = self.output_dir / "openapi.yaml"
        
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "AG News Text Classification API",
                "version": "1.0.0",
                "description": "API for text classification using state-of-the-art models",
                "contact": {
                    "name": "Võ Hải Dũng",
                    "email": "support@agnews.ai"
                },
                "license": {
                    "name": "MIT",
                    "url": "https://opensource.org/licenses/MIT"
                }
            },
            "servers": [
                {
                    "url": "http://localhost:8000/api/v1",
                    "description": "Development server"
                },
                {
                    "url": "https://api.agnews.ai/v1",
                    "description": "Production server"
                }
            ],
            "paths": {},
            "components": {
                "securitySchemes": {
                    "bearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT"
                    },
                    "apiKey": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-API-Key"
                    }
                }
            }
        }
        
        # Add paths from REST endpoints
        for endpoint in self.rest_endpoints:
            if endpoint.path not in spec["paths"]:
                spec["paths"][endpoint.path] = {}
            
            spec["paths"][endpoint.path][endpoint.method.lower()] = {
                "summary": endpoint.summary,
                "description": endpoint.description,
                "tags": endpoint.tags,
                "parameters": endpoint.parameters,
                "responses": endpoint.responses
            }
            
            if endpoint.request_body:
                spec["paths"][endpoint.path][endpoint.method.lower()]["requestBody"] = endpoint.request_body
            
            if endpoint.authentication:
                spec["paths"][endpoint.path][endpoint.method.lower()]["security"] = [
                    {"bearerAuth": []},
                    {"apiKey": []}
                ]
        
        # Write to file
        with open(output_file, 'w') as f:
            yaml.dump(spec, f, default_flow_style=False, sort_keys=False)
        
        console.print(f"[green]OpenAPI specification saved to {output_file}[/green]")
    
    def _generate_example_from_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate example from JSON schema"""
        if schema.get("type") == "object":
            example = {}
            for prop, prop_schema in schema.get("properties", {}).items():
                example[prop] = self._generate_example_from_schema(prop_schema)
            return example
        elif schema.get("type") == "array":
            item_schema = schema.get("items", {})
            return [self._generate_example_from_schema(item_schema)]
        elif schema.get("type") == "string":
            return schema.get("example", "string")
        elif schema.get("type") == "number":
            return schema.get("example", 0.0)
        elif schema.get("type") == "integer":
            return schema.get("example", 0)
        elif schema.get("type") == "boolean":
            return schema.get("example", True)
        else:
            return None
    
    def generate_html_docs(self):
        """Generate HTML documentation using templates"""
        # Create HTML output directory
        html_dir = self.output_dir / "html"
        html_dir.mkdir(exist_ok=True)
        
        # Generate index.html
        index_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AG News API Documentation</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1, h2, h3 {
                    color: #2c3e50;
                }
                code {
                    background: #f4f4f4;
                    padding: 2px 5px;
                    border-radius: 3px;
                }
                pre {
                    background: #f4f4f4;
                    padding: 15px;
                    border-radius: 5px;
                    overflow-x: auto;
                }
                .endpoint {
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    margin-bottom: 20px;
                }
                .method {
                    display: inline-block;
                    padding: 5px 10px;
                    border-radius: 3px;
                    color: white;
                    font-weight: bold;
                    margin-right: 10px;
                }
                .method-get { background: #61affe; }
                .method-post { background: #49cc90; }
                .method-put { background: #fca130; }
                .method-delete { background: #f93e3e; }
            </style>
        </head>
        <body>
            <h1>AG News Text Classification API Documentation</h1>
            <p>Generated: {datetime}</p>
            
            <h2>Available APIs</h2>
            <ul>
                <li><a href="#rest">REST API</a></li>
                <li><a href="#grpc">gRPC API</a></li>
                <li><a href="#graphql">GraphQL API</a></li>
            </ul>
            
            <div id="rest">
                <h2>REST API</h2>
                <p>Base URL: <code>http://localhost:8000/api/v1</code></p>
                {rest_endpoints}
            </div>
            
            <div id="grpc">
                <h2>gRPC API</h2>
                <p>Server: <code>localhost:50051</code></p>
                {grpc_services}
            </div>
            
            <div id="graphql">
                <h2>GraphQL API</h2>
                <p>Endpoint: <code>http://localhost:4000/graphql</code></p>
                <pre>{graphql_schema}</pre>
            </div>
        </body>
        </html>
        """.format(
            datetime=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            rest_endpoints=self._format_rest_endpoints_html(),
            grpc_services=self._format_grpc_services_html(),
            graphql_schema=self.graphql_schema or "Schema not available"
        )
        
        with open(html_dir / "index.html", 'w') as f:
            f.write(index_content)
        
        console.print(f"[green]HTML documentation saved to {html_dir}[/green]")
    
    def _format_rest_endpoints_html(self) -> str:
        """Format REST endpoints as HTML"""
        html = []
        for endpoint in self.rest_endpoints:
            method_class = f"method-{endpoint.method.lower()}"
            html.append(f'''
                <div class="endpoint">
                    <span class="method {method_class}">{endpoint.method}</span>
                    <code>{endpoint.path}</code>
                    <p>{endpoint.description}</p>
                </div>
            ''')
        return '\n'.join(html)
    
    def _format_grpc_services_html(self) -> str:
        """Format gRPC services as HTML"""
        html = []
        services = {}
        for method in self.grpc_services:
            if method.service not in services:
                services[method.service] = []
            services[method.service].append(method)
        
        for service, methods in services.items():
            html.append(f'<h3>{service}</h3>')
            for method in methods:
                html.append(f'''
                    <div class="endpoint">
                        <strong>{method.name}</strong>
                        <p>Input: <code>{method.input_type}</code></p>
                        <p>Output: <code>{method.output_type}</code></p>
                    </div>
                ''')
        return '\n'.join(html)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate API documentation for AG News Text Classification"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for documentation"
    )
    parser.add_argument(
        "--format",
        nargs="+",
        choices=["markdown", "html", "openapi"],
        default=["markdown", "openapi"],
        help="Output formats"
    )
    parser.add_argument(
        "--apis",
        nargs="+",
        choices=["rest", "grpc", "graphql"],
        default=["rest", "grpc", "graphql"],
        help="APIs to document"
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = APIDocumentationGenerator(args.output)
    
    # Generate documentation
    generator.generate_all_documentation()
    
    # Generate additional formats
    if "html" in args.format:
        generator.generate_html_docs()
    
    console.print("\n[bold green]Documentation generation complete![/bold green]")


if __name__ == "__main__":
    main()
