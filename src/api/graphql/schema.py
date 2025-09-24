"""
GraphQL Schema Definition
================================================================================
This module defines the GraphQL schema for the AG News classification API,
implementing type-safe schema definition with Strawberry framework.

The schema includes:
- Query types for data fetching
- Mutation types for data modification
- Subscription types for real-time updates
- Custom scalar types

References:
    - GraphQL Schema Definition Language
    - Strawberry Schema Documentation
    - GraphQL Schema Design Best Practices

Author: Võ Hải Dũng
License: MIT
"""

import strawberry
from typing import List, Optional
from datetime import datetime

from .queries import Query
from .mutations import Mutation
from .subscriptions import Subscription
from .types import (
    Classification,
    Model,
    Dataset,
    Training,
    Metrics,
    Error,
    ModelType,
    ClassLabel,
    TrainingStatus,
    DatasetSplit
)

# Custom scalar types
@strawberry.scalar
class DateTime:
    """Custom DateTime scalar for GraphQL."""
    
    @staticmethod
    def serialize(value: datetime) -> str:
        """
        Serialize datetime to ISO string.
        
        Args:
            value: Datetime object
            
        Returns:
            str: ISO formatted datetime string
        """
        return value.isoformat()
    
    @staticmethod
    def parse_value(value: str) -> datetime:
        """
        Parse ISO string to datetime.
        
        Args:
            value: ISO formatted datetime string
            
        Returns:
            datetime: Parsed datetime object
        """
        return datetime.fromisoformat(value)
    
    @staticmethod
    def parse_literal(ast) -> datetime:
        """
        Parse GraphQL literal to datetime.
        
        Args:
            ast: GraphQL AST node
            
        Returns:
            datetime: Parsed datetime object
        """
        if hasattr(ast, 'value'):
            return DateTime.parse_value(ast.value)
        return None

@strawberry.scalar
class JSON:
    """Custom JSON scalar for arbitrary JSON data."""
    
    @staticmethod
    def serialize(value: dict) -> dict:
        """Serialize dictionary to JSON."""
        return value
    
    @staticmethod
    def parse_value(value: dict) -> dict:
        """Parse JSON value."""
        return value
    
    @staticmethod
    def parse_literal(ast) -> dict:
        """Parse GraphQL literal to JSON."""
        return ast.value if hasattr(ast, 'value') else {}

# Schema configuration
schema_config = strawberry.Schema.config(
    auto_camel_case=True,  # Convert snake_case to camelCase
    description=(
        "AG News Text Classification GraphQL API\n\n"
        "This API provides access to text classification services for the AG News dataset, "
        "supporting single and batch classification, model management, training job monitoring, "
        "and real-time updates via subscriptions.\n\n"
        "Authentication is required for most operations using JWT tokens or API keys."
    )
)

# Create schema with all components
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription,
    types=[
        # Object types
        Classification,
        Model,
        Dataset,
        Training,
        Metrics,
        Error,
        
        # Enum types
        ModelType,
        ClassLabel,
        TrainingStatus,
        DatasetSplit,
        
        # Scalar types
        DateTime,
        JSON
    ],
    config=schema_config,
    extensions=[]  # Extensions can be added here
)

# Schema introspection utilities
def get_schema_sdl() -> str:
    """
    Get schema definition language (SDL) representation.
    
    Returns:
        str: Schema in SDL format
    """
    return str(schema)

def validate_query(query: str) -> bool:
    """
    Validate a GraphQL query against the schema.
    
    Args:
        query: GraphQL query string
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        from graphql import parse, validate as graphql_validate
        document = parse(query)
        errors = graphql_validate(schema._schema, document)
        return len(errors) == 0
    except Exception:
        return False

# Export schema and utilities
__all__ = [
    "schema",
    "DateTime",
    "JSON",
    "get_schema_sdl",
    "validate_query"
]
