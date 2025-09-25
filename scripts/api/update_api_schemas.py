"""
API Schema Update Script for AG News Text Classification System
================================================================================
This script updates and synchronizes API schemas across different API types
(REST, gRPC, GraphQL) ensuring consistency and compatibility. It validates
schemas, performs migrations, and generates type definitions.

The schema management follows best practices for API versioning and evolution
as outlined in API design literature.

References:
    - Sturgeon, P. (2016). Build APIs You Won't Hate
    - API Evolution: https://cloud.google.com/apis/design/versioning
    - GraphQL Schema Evolution: https://graphql.org/learn/best-practices/#versioning

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
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import argparse
import difflib
from dataclasses import dataclass, field
import hashlib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import BaseModel, Field, create_model
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import semver

# Configure console
console = Console()


@dataclass
class SchemaVersion:
    """Schema version information"""
    version: str
    timestamp: datetime
    changes: List[str]
    breaking_changes: bool = False
    checksum: str = ""


@dataclass
class SchemaField:
    """Schema field definition"""
    name: str
    type: str
    required: bool = False
    default: Any = None
    description: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchemaDefinition:
    """Complete schema definition"""
    name: str
    version: str
    fields: List[SchemaField]
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class SchemaManager:
    """
    Manages API schemas across different protocols
    
    This class handles schema validation, migration, and synchronization
    ensuring consistency across REST, gRPC, and GraphQL APIs.
    """
    
    def __init__(self, schema_dir: Optional[str] = None):
        """
        Initialize schema manager
        
        Args:
            schema_dir: Directory containing schema definitions
        """
        self.schema_dir = Path(schema_dir or PROJECT_ROOT / "src" / "api" / "schemas")
        self.schema_dir.mkdir(parents=True, exist_ok=True)
        
        self.rest_schemas: Dict[str, SchemaDefinition] = {}
        self.grpc_schemas: Dict[str, SchemaDefinition] = {}
        self.graphql_schemas: Dict[str, SchemaDefinition] = {}
        
        self.version_history: List[SchemaVersion] = []
        self.load_schemas()
    
    def load_schemas(self):
        """Load existing schema definitions"""
        # Load REST schemas from Pydantic models
        self._load_rest_schemas()
        
        # Load gRPC schemas from proto files
        self._load_grpc_schemas()
        
        # Load GraphQL schemas
        self._load_graphql_schemas()
        
        # Load version history
        self._load_version_history()
    
    def _load_rest_schemas(self):
        """Load REST API schemas from Pydantic models"""
        schemas_file = PROJECT_ROOT / "src" / "api" / "rest" / "schemas" / "request_schemas.py"
        
        if not schemas_file.exists():
            console.print("[yellow]Warning: REST schemas file not found[/yellow]")
            return
        
        with open(schemas_file, 'r') as f:
            content = f.read()
        
        # Parse Python AST to extract class definitions
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if it's a Pydantic model
                if any(base.id == 'BaseModel' for base in node.bases if hasattr(base, 'id')):
                    schema = self._extract_pydantic_schema(node)
                    if schema:
                        self.rest_schemas[schema.name] = schema
    
    def _extract_pydantic_schema(self, class_node: ast.ClassDef) -> Optional[SchemaDefinition]:
        """Extract schema from Pydantic model AST node"""
        fields = []
        
        for node in class_node.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                field_name = node.target.id
                field_type = ast.unparse(node.annotation) if hasattr(ast, 'unparse') else str(node.annotation)
                
                # Determine if field is required
                required = not (node.value and isinstance(node.value, ast.Call))
                
                field = SchemaField(
                    name=field_name,
                    type=field_type,
                    required=required
                )
                fields.append(field)
        
        if fields:
            return SchemaDefinition(
                name=class_node.name,
                version="1.0.0",
                fields=fields
            )
        return None
    
    def _load_grpc_schemas(self):
        """Load gRPC schemas from proto files"""
        proto_dir = PROJECT_ROOT / "src" / "api" / "grpc" / "protos"
        
        if not proto_dir.exists():
            console.print("[yellow]Warning: gRPC proto directory not found[/yellow]")
            return
        
        for proto_file in proto_dir.glob("*.proto"):
            self._parse_proto_schema(proto_file)
    
    def _parse_proto_schema(self, proto_file: Path):
        """Parse proto file to extract message definitions"""
        with open(proto_file, 'r') as f:
            content = f.read()
        
        # Regular expression for message definitions
        message_pattern = r'message\s+(\w+)\s*\{([^}]*)\}'
        field_pattern = r'(?:optional|required|repeated)?\s*(\w+)\s+(\w+)\s*=\s*(\d+);'
        
        messages = re.findall(message_pattern, content, re.DOTALL)
        
        for message_name, message_body in messages:
            fields = []
            field_matches = re.findall(field_pattern, message_body)
            
            for field_type, field_name, field_number in field_matches:
                field = SchemaField(
                    name=field_name,
                    type=field_type,
                    required=True  # Proto3 fields are optional by default
                )
                fields.append(field)
            
            if fields:
                schema = SchemaDefinition(
                    name=message_name,
                    version="1.0.0",
                    fields=fields
                )
                self.grpc_schemas[message_name] = schema
    
    def _load_graphql_schemas(self):
        """Load GraphQL schemas"""
        schema_file = PROJECT_ROOT / "src" / "api" / "graphql" / "schema.graphql"
        
        if not schema_file.exists():
            console.print("[yellow]Warning: GraphQL schema file not found[/yellow]")
            return
        
        with open(schema_file, 'r') as f:
            content = f.read()
        
        # Parse GraphQL type definitions
        type_pattern = r'type\s+(\w+)\s*\{([^}]*)\}'
        field_pattern = r'(\w+)\s*:\s*([^!\n]+)(!)?'
        
        types = re.findall(type_pattern, content, re.DOTALL)
        
        for type_name, type_body in types:
            fields = []
            field_matches = re.findall(field_pattern, type_body)
            
            for field_name, field_type, required in field_matches:
                field = SchemaField(
                    name=field_name,
                    type=field_type.strip(),
                    required=bool(required)
                )
                fields.append(field)
            
            if fields:
                schema = SchemaDefinition(
                    name=type_name,
                    version="1.0.0",
                    fields=fields
                )
                self.graphql_schemas[type_name] = schema
    
    def _load_version_history(self):
        """Load schema version history"""
        history_file = self.schema_dir / "version_history.json"
        
        if history_file.exists():
            with open(history_file, 'r') as f:
                data = json.load(f)
                for entry in data:
                    self.version_history.append(SchemaVersion(
                        version=entry["version"],
                        timestamp=datetime.fromisoformat(entry["timestamp"]),
                        changes=entry["changes"],
                        breaking_changes=entry.get("breaking_changes", False),
                        checksum=entry.get("checksum", "")
                    ))
    
    def validate_schemas(self) -> Tuple[bool, List[str]]:
        """
        Validate all schemas for consistency
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check for common entities across APIs
        common_entities = self._find_common_entities()
        
        for entity in common_entities:
            # Check field consistency
            rest_schema = self.rest_schemas.get(entity)
            grpc_schema = self.grpc_schemas.get(entity)
            graphql_schema = self.graphql_schemas.get(entity)
            
            schemas = [s for s in [rest_schema, grpc_schema, graphql_schema] if s]
            
            if len(schemas) > 1:
                # Compare schemas
                field_sets = [set(f.name for f in s.fields) for s in schemas]
                
                # Check for field mismatches
                common_fields = set.intersection(*field_sets)
                all_fields = set.union(*field_sets)
                
                if common_fields != all_fields:
                    missing_fields = all_fields - common_fields
                    errors.append(
                        f"Schema mismatch for {entity}: fields {missing_fields} "
                        f"not present in all API schemas"
                    )
        
        # Validate individual schemas
        for schema_type, schemas in [
            ("REST", self.rest_schemas),
            ("gRPC", self.grpc_schemas),
            ("GraphQL", self.graphql_schemas)
        ]:
            for name, schema in schemas.items():
                # Check for empty schemas
                if not schema.fields:
                    errors.append(f"{schema_type} schema {name} has no fields")
                
                # Check for duplicate fields
                field_names = [f.name for f in schema.fields]
                if len(field_names) != len(set(field_names)):
                    errors.append(f"{schema_type} schema {name} has duplicate fields")
        
        return len(errors) == 0, errors
    
    def _find_common_entities(self) -> Set[str]:
        """Find entities that exist across multiple APIs"""
        rest_entities = set(self.rest_schemas.keys())
        grpc_entities = set(self.grpc_schemas.keys())
        graphql_entities = set(self.graphql_schemas.keys())
        
        # Find entities that exist in at least 2 APIs
        common = (
            (rest_entities & grpc_entities) |
            (rest_entities & graphql_entities) |
            (grpc_entities & graphql_entities)
        )
        
        return common
    
    def update_schema(self, api_type: str, schema_name: str, 
                     updates: Dict[str, Any]) -> bool:
        """
        Update a schema definition
        
        Args:
            api_type: API type (rest, grpc, graphql)
            schema_name: Name of the schema
            updates: Updates to apply
            
        Returns:
            Success status
        """
        schemas_map = {
            "rest": self.rest_schemas,
            "grpc": self.grpc_schemas,
            "graphql": self.graphql_schemas
        }
        
        if api_type not in schemas_map:
            console.print(f"[red]Invalid API type: {api_type}[/red]")
            return False
        
        schemas = schemas_map[api_type]
        
        if schema_name not in schemas:
            console.print(f"[red]Schema {schema_name} not found[/red]")
            return False
        
        schema = schemas[schema_name]
        
        # Apply updates
        changes = []
        
        # Add new fields
        if "add_fields" in updates:
            for field_def in updates["add_fields"]:
                field = SchemaField(**field_def)
                schema.fields.append(field)
                changes.append(f"Added field {field.name}")
        
        # Remove fields
        if "remove_fields" in updates:
            for field_name in updates["remove_fields"]:
                schema.fields = [f for f in schema.fields if f.name != field_name]
                changes.append(f"Removed field {field_name}")
        
        # Modify fields
        if "modify_fields" in updates:
            for field_name, modifications in updates["modify_fields"].items():
                for field in schema.fields:
                    if field.name == field_name:
                        for key, value in modifications.items():
                            setattr(field, key, value)
                        changes.append(f"Modified field {field_name}")
        
        # Update version
        if changes:
            old_version = schema.version
            schema.version = self._increment_version(
                old_version,
                breaking=updates.get("breaking", False)
            )
            
            # Record in version history
            self.version_history.append(SchemaVersion(
                version=schema.version,
                timestamp=datetime.now(),
                changes=changes,
                breaking_changes=updates.get("breaking", False),
                checksum=self._calculate_checksum(schema)
            ))
            
            console.print(f"[green]Updated {api_type} schema {schema_name} "
                         f"from {old_version} to {schema.version}[/green]")
            
            return True
        
        return False
    
    def _increment_version(self, version: str, breaking: bool = False) -> str:
        """Increment schema version using semantic versioning"""
        try:
            ver = semver.VersionInfo.parse(version)
            if breaking:
                return str(ver.bump_major())
            else:
                return str(ver.bump_minor())
        except:
            # Fallback to simple increment
            parts = version.split('.')
            if breaking:
                return f"{int(parts[0]) + 1}.0.0"
            else:
                return f"{parts[0]}.{int(parts[1]) + 1}.0"
    
    def _calculate_checksum(self, schema: SchemaDefinition) -> str:
        """Calculate checksum for schema"""
        content = json.dumps({
            "name": schema.name,
            "fields": [
                {
                    "name": f.name,
                    "type": f.type,
                    "required": f.required
                }
                for f in schema.fields
            ]
        }, sort_keys=True)
        
        return hashlib.sha256(content.encode()).hexdigest()
    
    def generate_migrations(self) -> List[Dict[str, Any]]:
        """Generate migration scripts for schema changes"""
        migrations = []
        
        if len(self.version_history) < 2:
            return migrations
        
        for i in range(len(self.version_history) - 1):
            from_version = self.version_history[i]
            to_version = self.version_history[i + 1]
            
            migration = {
                "from_version": from_version.version,
                "to_version": to_version.version,
                "timestamp": to_version.timestamp.isoformat(),
                "breaking": to_version.breaking_changes,
                "changes": to_version.changes,
                "scripts": self._generate_migration_scripts(from_version, to_version)
            }
            migrations.append(migration)
        
        return migrations
    
    def _generate_migration_scripts(self, from_version: SchemaVersion, 
                                   to_version: SchemaVersion) -> Dict[str, str]:
        """Generate migration scripts for different databases/systems"""
        scripts = {}
        
        # SQL migration
        sql_statements = []
        for change in to_version.changes:
            if "Added field" in change:
                field_name = change.split()[-1]
                sql_statements.append(f"ALTER TABLE table_name ADD COLUMN {field_name} VARCHAR(255);")
            elif "Removed field" in change:
                field_name = change.split()[-1]
                sql_statements.append(f"ALTER TABLE table_name DROP COLUMN {field_name};")
        
        if sql_statements:
            scripts["sql"] = "\n".join(sql_statements)
        
        # MongoDB migration
        mongo_operations = []
        for change in to_version.changes:
            if "Added field" in change:
                field_name = change.split()[-1]
                mongo_operations.append(f'db.collection.updateMany({{}}, {{$set: {{{field_name}: null}}}});')
            elif "Removed field" in change:
                field_name = change.split()[-1]
                mongo_operations.append(f'db.collection.updateMany({{}}, {{$unset: {{{field_name}: ""}}}});')
        
        if mongo_operations:
            scripts["mongodb"] = "\n".join(mongo_operations)
        
        return scripts
    
    def synchronize_schemas(self):
        """Synchronize schemas across different API types"""
        console.print("\n[bold blue]Synchronizing API Schemas[/bold blue]\n")
        
        common_entities = self._find_common_entities()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for entity in common_entities:
                task = progress.add_task(f"Synchronizing {entity}...", total=1)
                
                # Get all versions of the entity
                schemas = []
                if entity in self.rest_schemas:
                    schemas.append(("REST", self.rest_schemas[entity]))
                if entity in self.grpc_schemas:
                    schemas.append(("gRPC", self.grpc_schemas[entity]))
                if entity in self.graphql_schemas:
                    schemas.append(("GraphQL", self.graphql_schemas[entity]))
                
                if len(schemas) > 1:
                    # Find the most complete schema (with most fields)
                    master_schema = max(schemas, key=lambda x: len(x[1].fields))
                    
                    # Update other schemas to match
                    for api_type, schema in schemas:
                        if schema != master_schema[1]:
                            missing_fields = set(f.name for f in master_schema[1].fields) - \
                                           set(f.name for f in schema.fields)
                            
                            if missing_fields:
                                console.print(f"[yellow]Adding missing fields to {api_type} "
                                            f"schema {entity}: {missing_fields}[/yellow]")
                                
                                for field in master_schema[1].fields:
                                    if field.name in missing_fields:
                                        schema.fields.append(field)
                
                progress.update(task, completed=1)
        
        console.print("[green]Schema synchronization complete[/green]")
    
    def export_schemas(self, output_dir: Optional[str] = None):
        """Export all schemas to files"""
        output_dir = Path(output_dir or self.schema_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export REST schemas as Pydantic models
        rest_file = output_dir / "rest_schemas.py"
        with open(rest_file, 'w') as f:
            f.write("# Generated REST API Schemas\n")
            f.write("# Author: Võ Hải Dũng\n\n")
            f.write("from pydantic import BaseModel, Field\n")
            f.write("from typing import Optional, List, Dict, Any\n\n")
            
            for name, schema in self.rest_schemas.items():
                f.write(f"class {name}(BaseModel):\n")
                f.write(f'    """Generated schema for {name}"""\n')
                for field in schema.fields:
                    field_type = field.type
                    if not field.required:
                        field_type = f"Optional[{field_type}]"
                    f.write(f"    {field.name}: {field_type}\n")
                f.write("\n")
        
        console.print(f"[green]REST schemas exported to {rest_file}[/green]")
        
        # Export version history
        history_file = output_dir / "version_history.json"
        with open(history_file, 'w') as f:
            json.dump([
                {
                    "version": v.version,
                    "timestamp": v.timestamp.isoformat(),
                    "changes": v.changes,
                    "breaking_changes": v.breaking_changes,
                    "checksum": v.checksum
                }
                for v in self.version_history
            ], f, indent=2)
        
        console.print(f"[green]Version history exported to {history_file}[/green]")
    
    def generate_report(self):
        """Generate schema status report"""
        console.print("\n[bold green]Schema Status Report[/bold green]\n")
        
        # Summary table
        table = Table(title="Schema Summary")
        table.add_column("API Type", style="cyan")
        table.add_column("Total Schemas", style="yellow")
        table.add_column("Total Fields", style="green")
        table.add_column("Latest Version", style="magenta")
        
        for api_type, schemas in [
            ("REST", self.rest_schemas),
            ("gRPC", self.grpc_schemas),
            ("GraphQL", self.graphql_schemas)
        ]:
            total_fields = sum(len(s.fields) for s in schemas.values())
            latest_version = max((s.version for s in schemas.values()), default="N/A")
            
            table.add_row(
                api_type,
                str(len(schemas)),
                str(total_fields),
                latest_version
            )
        
        console.print(table)
        
        # Validation results
        is_valid, errors = self.validate_schemas()
        
        if is_valid:
            console.print("\n[green]✓ All schemas are valid[/green]")
        else:
            console.print("\n[red]Schema validation errors:[/red]")
            for error in errors:
                console.print(f"  - {error}")
        
        # Common entities
        common = self._find_common_entities()
        if common:
            console.print(f"\n[cyan]Common entities across APIs: {', '.join(common)}[/cyan]")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Update and manage API schemas for AG News Text Classification"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate schemas for consistency"
    )
    parser.add_argument(
        "--synchronize",
        action="store_true",
        help="Synchronize schemas across APIs"
    )
    parser.add_argument(
        "--update",
        type=str,
        help="Update schema from JSON file"
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export schemas to directory"
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Generate migration scripts"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate schema status report"
    )
    
    args = parser.parse_args()
    
    # Initialize schema manager
    manager = SchemaManager()
    
    # Perform requested operations
    if args.validate:
        is_valid, errors = manager.validate_schemas()
        if is_valid:
            console.print("[green]All schemas are valid[/green]")
        else:
            console.print("[red]Validation errors found:[/red]")
            for error in errors:
                console.print(f"  - {error}")
    
    if args.synchronize:
        manager.synchronize_schemas()
    
    if args.update:
        with open(args.update, 'r') as f:
            updates = json.load(f)
        
        for update in updates:
            success = manager.update_schema(
                update["api_type"],
                update["schema_name"],
                update["changes"]
            )
            if not success:
                console.print(f"[red]Failed to update {update['schema_name']}[/red]")
    
    if args.export:
        manager.export_schemas(args.export)
    
    if args.migrate:
        migrations = manager.generate_migrations()
        output_file = Path(args.export or ".") / "migrations.json"
        with open(output_file, 'w') as f:
            json.dump(migrations, f, indent=2, default=str)
        console.print(f"[green]Migrations saved to {output_file}[/green]")
    
    if args.report:
        manager.generate_report()


if __name__ == "__main__":
    main()
