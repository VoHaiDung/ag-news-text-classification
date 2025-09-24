"""
Workflow Orchestrator Implementation for AG News Text Classification
================================================================================
This module implements workflow orchestration for coordinating complex
multi-step operations across services.

The orchestrator provides:
- DAG-based workflow execution
- Error handling and retry logic
- Parallel and sequential execution
- State persistence

References:
    - van der Aalst, W. M. (2013). Business Process Management: A Comprehensive Survey
    - Apache Airflow Architecture

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
from collections import defaultdict
import json

from src.services.base_service import BaseService, ServiceConfig
from src.core.exceptions import ServiceException
from src.utils.logging_config import get_logger


class WorkflowStatus(Enum):
    """
    Workflow execution status enumeration.
    
    States:
        PENDING: Workflow created but not started
        RUNNING: Workflow is executing
        PAUSED: Workflow is temporarily suspended
        COMPLETED: Workflow finished successfully
        FAILED: Workflow encountered an error
        CANCELLED: Workflow was cancelled
    """
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """
    Individual step in a workflow.
    
    Attributes:
        name: Step identifier
        service_name: Service to execute
        operation: Operation to perform
        parameters: Operation parameters
        dependencies: Previous steps that must complete
        retry_policy: Step-specific retry configuration
        timeout: Step execution timeout
    """
    name: str
    service_name: str
    operation: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {
        "max_retries": 3,
        "retry_delay": 1
    })
    timeout: int = 300
    
    def validate(self) -> bool:
        """Validate step configuration."""
        if not self.name or not self.service_name or not self.operation:
            raise ValueError("Step name, service_name, and operation are required")
        return True


@dataclass
class WorkflowDefinition:
    """
    Workflow definition with steps and metadata.
    
    Attributes:
        name: Workflow name
        version: Workflow version
        description: Workflow description
        steps: List of workflow steps
        metadata: Additional workflow metadata
        on_success: Success callback
        on_failure: Failure callback
    """
    name: str
    version: str = "1.0.0"
    description: str = ""
    steps: List[WorkflowStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    on_success: Optional[Callable] = None
    on_failure: Optional[Callable] = None
    
    def validate(self) -> bool:
        """
        Validate workflow definition.
        
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If validation fails
        """
        if not self.name:
            raise ValueError("Workflow name is required")
        
        # Validate all steps
        for step in self.steps:
            step.validate()
        
        # Check for circular dependencies
        if self._has_circular_dependencies():
            raise ValueError("Circular dependencies detected in workflow")
        
        return True
    
    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies in workflow steps."""
        visited = set()
        rec_stack = set()
        
        def has_cycle(step_name: str) -> bool:
            visited.add(step_name)
            rec_stack.add(step_name)
            
            step = next((s for s in self.steps if s.name == step_name), None)
            if step:
                for dep in step.dependencies:
                    if dep not in visited:
                        if has_cycle(dep):
                            return True
                    elif dep in rec_stack:
                        return True
            
            rec_stack.remove(step_name)
            return False
        
        for step in self.steps:
            if step.name not in visited:
                if has_cycle(step.name):
                    return True
        
        return False


@dataclass
class WorkflowExecution:
    """
    Workflow execution instance.
    
    Attributes:
        id: Unique execution identifier
        workflow: Workflow definition
        status: Current execution status
        started_at: Execution start time
        completed_at: Execution completion time
        step_results: Results from each step
        error: Error message if failed
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow: Optional[WorkflowDefinition] = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    step_results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class WorkflowOrchestrator(BaseService):
    """
    Orchestrator for managing and executing workflows.
    
    This service coordinates complex multi-step workflows across
    multiple services, handling dependencies, retries, and state management.
    """
    
    def __init__(self, config: Optional[ServiceConfig] = None):
        """
        Initialize workflow orchestrator.
        
        Args:
            config: Service configuration
        """
        if config is None:
            config = ServiceConfig(name="workflow_orchestrator")
        super().__init__(config)
        
        # Workflow storage
        self._workflows: Dict[str, WorkflowDefinition] = {}
        self._executions: Dict[str, WorkflowExecution] = {}
        
        # Execution management
        self._running_executions: Set[str] = set()
        self._execution_tasks: Dict[str, asyncio.Task] = {}
        
        # Service registry reference
        from src.services.service_registry import registry
        self._registry = registry
        
        self.logger = get_logger("service.orchestrator")
    
    async def _initialize(self) -> None:
        """Initialize orchestrator resources."""
        self.logger.info("Initializing workflow orchestrator")
    
    async def _start(self) -> None:
        """Start orchestrator service."""
        self.logger.info("Workflow orchestrator started")
    
    async def _stop(self) -> None:
        """Stop orchestrator service."""
        # Cancel all running executions
        for task in self._execution_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self._execution_tasks:
            await asyncio.gather(*self._execution_tasks.values(), return_exceptions=True)
        
        self.logger.info("Workflow orchestrator stopped")
    
    async def _cleanup(self) -> None:
        """Cleanup orchestrator resources."""
        self._execution_tasks.clear()
        self._running_executions.clear()
    
    async def _check_health(self) -> bool:
        """Check orchestrator health."""
        return len(self._running_executions) < 100  # Max concurrent workflows
    
    def register_workflow(self, workflow: WorkflowDefinition) -> None:
        """
        Register a workflow definition.
        
        Args:
            workflow: Workflow to register
            
        Raises:
            ValueError: If workflow is invalid
        """
        workflow.validate()
        workflow_key = f"{workflow.name}:{workflow.version}"
        self._workflows[workflow_key] = workflow
        self.logger.info(f"Registered workflow: {workflow_key}")
    
    async def execute_workflow(
        self,
        workflow_name: str,
        version: str = "1.0.0",
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Execute a registered workflow.
        
        Args:
            workflow_name: Name of workflow to execute
            version: Workflow version
            context: Execution context
            
        Returns:
            str: Execution ID
            
        Raises:
            ValueError: If workflow not found
            ServiceException: If execution fails
        """
        workflow_key = f"{workflow_name}:{version}"
        
        if workflow_key not in self._workflows:
            raise ValueError(f"Workflow not found: {workflow_key}")
        
        workflow = self._workflows[workflow_key]
        
        # Create execution instance
        execution = WorkflowExecution(
            workflow=workflow,
            status=WorkflowStatus.RUNNING,
            started_at=datetime.now()
        )
        
        self._executions[execution.id] = execution
        self._running_executions.add(execution.id)
        
        # Start execution task
        task = asyncio.create_task(
            self._execute_workflow_async(execution, context or {})
        )
        self._execution_tasks[execution.id] = task
        
        self.logger.info(f"Started workflow execution: {execution.id}")
        return execution.id
    
    async def _execute_workflow_async(
        self,
        execution: WorkflowExecution,
        context: Dict[str, Any]
    ) -> None:
        """
        Execute workflow asynchronously.
        
        Args:
            execution: Workflow execution instance
            context: Execution context
        """
        try:
            # Get execution order
            execution_order = self._get_execution_order(execution.workflow)
            
            # Execute steps
            for step_group in execution_order:
                # Execute parallel steps
                tasks = []
                for step in step_group:
                    task = asyncio.create_task(
                        self._execute_step(step, execution, context)
                    )
                    tasks.append((step.name, task))
                
                # Wait for parallel steps to complete
                for step_name, task in tasks:
                    try:
                        result = await task
                        execution.step_results[step_name] = result
                    except Exception as e:
                        execution.status = WorkflowStatus.FAILED
                        execution.error = f"Step {step_name} failed: {e}"
                        raise
            
            # Workflow completed successfully
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.now()
            
            # Execute success callback
            if execution.workflow.on_success:
                await self._execute_callback(
                    execution.workflow.on_success,
                    execution
                )
            
            self.logger.info(f"Workflow completed: {execution.id}")
            
        except Exception as e:
            # Workflow failed
            execution.status = WorkflowStatus.FAILED
            execution.completed_at = datetime.now()
            execution.error = str(e)
            
            # Execute failure callback
            if execution.workflow.on_failure:
                await self._execute_callback(
                    execution.workflow.on_failure,
                    execution
                )
            
            self.logger.error(f"Workflow failed: {execution.id} - {e}")
            
        finally:
            self._running_executions.discard(execution.id)
            self._execution_tasks.pop(execution.id, None)
    
    async def _execute_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
        context: Dict[str, Any]
    ) -> Any:
        """
        Execute a single workflow step.
        
        Args:
            step: Step to execute
            execution: Workflow execution instance
            context: Execution context
            
        Returns:
            Any: Step result
            
        Raises:
            ServiceException: If step execution fails
        """
        self.logger.info(f"Executing step: {step.name}")
        
        # Get service
        service = self._registry.get(step.service_name)
        if not service:
            raise ServiceException(f"Service not found: {step.service_name}")
        
        # Get operation
        if not hasattr(service, step.operation):
            raise ServiceException(
                f"Operation not found: {step.service_name}.{step.operation}"
            )
        
        operation = getattr(service, step.operation)
        
        # Prepare parameters with context
        params = {**context, **step.parameters}
        
        # Add previous step results to parameters
        for dep in step.dependencies:
            if dep in execution.step_results:
                params[f"{dep}_result"] = execution.step_results[dep]
        
        # Execute with retry
        retry_count = 0
        max_retries = step.retry_policy["max_retries"]
        retry_delay = step.retry_policy["retry_delay"]
        
        while retry_count <= max_retries:
            try:
                # Execute operation with timeout
                result = await asyncio.wait_for(
                    operation(**params),
                    timeout=step.timeout
                )
                return result
                
            except asyncio.TimeoutError:
                raise ServiceException(f"Step {step.name} timed out")
                
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    raise ServiceException(
                        f"Step {step.name} failed after {max_retries} retries: {e}"
                    )
                
                self.logger.warning(
                    f"Step {step.name} failed (attempt {retry_count}/{max_retries}): {e}"
                )
                await asyncio.sleep(retry_delay * retry_count)
    
    async def _execute_callback(
        self,
        callback: Callable,
        execution: WorkflowExecution
    ) -> None:
        """Execute workflow callback."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(execution)
            else:
                callback(execution)
        except Exception as e:
            self.logger.error(f"Callback execution failed: {e}")
    
    def _get_execution_order(
        self,
        workflow: WorkflowDefinition
    ) -> List[List[WorkflowStep]]:
        """
        Get step execution order respecting dependencies.
        
        Args:
            workflow: Workflow definition
            
        Returns:
            List[List[WorkflowStep]]: Steps grouped by parallel execution
        """
        # Build dependency graph
        in_degree = defaultdict(int)
        graph = defaultdict(list)
        
        for step in workflow.steps:
            for dep in step.dependencies:
                graph[dep].append(step.name)
                in_degree[step.name] += 1
        
        # Topological sort with level grouping
        execution_order = []
        current_level = []
        
        # Find steps with no dependencies
        for step in workflow.steps:
            if in_degree[step.name] == 0:
                current_level.append(step)
        
        while current_level:
            execution_order.append(current_level)
            next_level = []
            
            for step in current_level:
                for dependent in graph[step.name]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        dependent_step = next(
                            s for s in workflow.steps if s.name == dependent
                        )
                        next_level.append(dependent_step)
            
            current_level = next_level
        
        return execution_order
    
    def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """
        Get workflow execution status.
        
        Args:
            execution_id: Execution identifier
            
        Returns:
            WorkflowExecution: Execution instance or None
        """
        return self._executions.get(execution_id)
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel a running workflow execution.
        
        Args:
            execution_id: Execution identifier
            
        Returns:
            bool: True if cancelled successfully
        """
        if execution_id in self._execution_tasks:
            task = self._execution_tasks[execution_id]
            task.cancel()
            
            execution = self._executions.get(execution_id)
            if execution:
                execution.status = WorkflowStatus.CANCELLED
                execution.completed_at = datetime.now()
            
            self.logger.info(f"Cancelled workflow execution: {execution_id}")
            return True
        
        return False
