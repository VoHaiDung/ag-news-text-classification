"""
Service Orchestration Module for AG News Text Classification
================================================================================
This module implements workflow orchestration and pipeline management for
coordinating complex multi-service operations.

The orchestration layer provides:
- Workflow definition and execution
- Pipeline management
- Job scheduling
- State management

References:
    - Hohpe, G., & Woolf, B. (2003). Enterprise Integration Patterns
    - Apache Airflow Documentation

Author: Võ Hải Dũng
License: MIT
"""

from src.services.orchestration.workflow_orchestrator import (
    WorkflowOrchestrator,
    WorkflowDefinition,
    WorkflowStep,
    WorkflowStatus
)
from src.services.orchestration.pipeline_manager import (
    PipelineManager,
    Pipeline,
    PipelineStage
)
from src.services.orchestration.job_scheduler import (
    JobScheduler,
    Job,
    JobStatus,
    Schedule
)
from src.services.orchestration.state_manager import (
    StateManager,
    WorkflowState,
    StateTransition
)

__all__ = [
    "WorkflowOrchestrator",
    "WorkflowDefinition",
    "WorkflowStep",
    "WorkflowStatus",
    "PipelineManager",
    "Pipeline",
    "PipelineStage",
    "JobScheduler",
    "Job",
    "JobStatus",
    "Schedule",
    "StateManager",
    "WorkflowState",
    "StateTransition"
]
