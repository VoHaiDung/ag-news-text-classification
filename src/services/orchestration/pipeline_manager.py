"""
Pipeline Manager Implementation for AG News Text Classification
================================================================================
This module implements pipeline management for data processing and model training
workflows, providing stage-based execution with data flow management.

The pipeline manager supports:
- Sequential and parallel stage execution
- Data transformation between stages
- Checkpoint and recovery mechanisms
- Performance monitoring

References:
    - Kleppmann, M. (2017). Designing Data-Intensive Applications
    - Apache Beam Programming Guide

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import pickle
import json
from pathlib import Path

from src.services.base_service import BaseService, ServiceConfig
from src.core.exceptions import ServiceException
from src.utils.logging_config import get_logger
from src.utils.io_utils import save_json, load_json


class PipelineStatus(Enum):
    """
    Pipeline execution status enumeration.
    
    States:
        INITIALIZED: Pipeline created but not started
        RUNNING: Pipeline is executing stages
        PAUSED: Pipeline execution is paused
        COMPLETED: All stages completed successfully
        FAILED: Pipeline encountered an error
        CANCELLED: Pipeline was cancelled by user
    """
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineStage:
    """
    Individual stage in a pipeline.
    
    Attributes:
        name: Stage identifier
        processor: Processing function or callable
        input_keys: Keys to extract from previous stage output
        output_keys: Keys that this stage will produce
        config: Stage-specific configuration
        parallel: Whether to execute in parallel with other stages
        checkpoint: Whether to save checkpoint after stage
        retry_on_failure: Number of retry attempts
    """
    name: str
    processor: Callable
    input_keys: List[str] = field(default_factory=list)
    output_keys: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    parallel: bool = False
    checkpoint: bool = False
    retry_on_failure: int = 3
    
    def validate(self) -> bool:
        """Validate stage configuration."""
        if not self.name:
            raise ValueError("Stage name is required")
        if not callable(self.processor):
            raise ValueError(f"Stage {self.name} processor must be callable")
        return True


@dataclass
class Pipeline:
    """
    Pipeline definition with stages and configuration.
    
    Attributes:
        name: Pipeline identifier
        version: Pipeline version
        description: Pipeline description
        stages: List of pipeline stages
        config: Pipeline-wide configuration
        checkpoint_dir: Directory for saving checkpoints
        enable_monitoring: Whether to collect metrics
    """
    name: str
    version: str = "1.0.0"
    description: str = ""
    stages: List[PipelineStage] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    checkpoint_dir: Optional[Path] = None
    enable_monitoring: bool = True
    
    def validate(self) -> bool:
        """
        Validate pipeline configuration.
        
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If validation fails
        """
        if not self.name:
            raise ValueError("Pipeline name is required")
        
        # Validate all stages
        stage_names = set()
        for stage in self.stages:
            stage.validate()
            if stage.name in stage_names:
                raise ValueError(f"Duplicate stage name: {stage.name}")
            stage_names.add(stage.name)
        
        return True


@dataclass
class PipelineExecution:
    """
    Pipeline execution instance with state tracking.
    
    Attributes:
        pipeline: Pipeline being executed
        status: Current execution status
        current_stage: Currently executing stage index
        stage_outputs: Outputs from completed stages
        metrics: Execution metrics
        started_at: Execution start time
        completed_at: Execution completion time
        error: Error message if failed
    """
    pipeline: Pipeline
    status: PipelineStatus = PipelineStatus.INITIALIZED
    current_stage: int = 0
    stage_outputs: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class PipelineManager(BaseService):
    """
    Manager for creating and executing data processing pipelines.
    
    This service orchestrates multi-stage pipelines with support for
    parallel execution, checkpointing, and monitoring.
    """
    
    def __init__(self, config: Optional[ServiceConfig] = None):
        """
        Initialize pipeline manager.
        
        Args:
            config: Service configuration
        """
        if config is None:
            config = ServiceConfig(name="pipeline_manager")
        super().__init__(config)
        
        # Pipeline storage
        self._pipelines: Dict[str, Pipeline] = {}
        self._executions: Dict[str, PipelineExecution] = {}
        self._running_pipelines: Dict[str, asyncio.Task] = {}
        
        # Checkpoint management
        self._checkpoint_dir = Path("outputs/pipeline_checkpoints")
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger("service.pipeline_manager")
    
    async def _initialize(self) -> None:
        """Initialize pipeline manager resources."""
        self.logger.info("Initializing pipeline manager")
    
    async def _start(self) -> None:
        """Start pipeline manager service."""
        self.logger.info("Pipeline manager started")
    
    async def _stop(self) -> None:
        """Stop pipeline manager service."""
        # Cancel all running pipelines
        for task in self._running_pipelines.values():
            task.cancel()
        
        if self._running_pipelines:
            await asyncio.gather(
                *self._running_pipelines.values(),
                return_exceptions=True
            )
        
        self.logger.info("Pipeline manager stopped")
    
    async def _cleanup(self) -> None:
        """Cleanup pipeline manager resources."""
        self._running_pipelines.clear()
    
    async def _check_health(self) -> bool:
        """Check pipeline manager health."""
        return len(self._running_pipelines) < 50  # Max concurrent pipelines
    
    def register_pipeline(self, pipeline: Pipeline) -> None:
        """
        Register a pipeline definition.
        
        Args:
            pipeline: Pipeline to register
            
        Raises:
            ValueError: If pipeline is invalid
        """
        pipeline.validate()
        pipeline_key = f"{pipeline.name}:{pipeline.version}"
        self._pipelines[pipeline_key] = pipeline
        self.logger.info(f"Registered pipeline: {pipeline_key}")
    
    async def execute_pipeline(
        self,
        pipeline_name: str,
        initial_data: Optional[Dict[str, Any]] = None,
        version: str = "1.0.0"
    ) -> str:
        """
        Execute a registered pipeline.
        
        Args:
            pipeline_name: Name of pipeline to execute
            initial_data: Initial input data
            version: Pipeline version
            
        Returns:
            str: Execution ID
            
        Raises:
            ValueError: If pipeline not found
            ServiceException: If execution fails
        """
        pipeline_key = f"{pipeline_name}:{version}"
        
        if pipeline_key not in self._pipelines:
            raise ValueError(f"Pipeline not found: {pipeline_key}")
        
        pipeline = self._pipelines[pipeline_key]
        
        # Create execution instance
        execution = PipelineExecution(
            pipeline=pipeline,
            status=PipelineStatus.RUNNING,
            started_at=datetime.now()
        )
        
        # Generate execution ID
        exec_id = f"{pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._executions[exec_id] = execution
        
        # Initialize with input data
        if initial_data:
            execution.stage_outputs["__input__"] = initial_data
        
        # Start execution task
        task = asyncio.create_task(self._execute_pipeline_async(exec_id, execution))
        self._running_pipelines[exec_id] = task
        
        self.logger.info(f"Started pipeline execution: {exec_id}")
        return exec_id
    
    async def _execute_pipeline_async(
        self,
        exec_id: str,
        execution: PipelineExecution
    ) -> None:
        """
        Execute pipeline asynchronously.
        
        Args:
            exec_id: Execution identifier
            execution: Pipeline execution instance
        """
        try:
            # Group stages by parallel execution
            stage_groups = self._group_stages(execution.pipeline.stages)
            
            for group_idx, stage_group in enumerate(stage_groups):
                self.logger.info(f"Executing stage group {group_idx + 1}/{len(stage_groups)}")
                
                if len(stage_group) == 1:
                    # Sequential execution
                    stage = stage_group[0]
                    await self._execute_stage(execution, stage)
                else:
                    # Parallel execution
                    tasks = [
                        self._execute_stage(execution, stage)
                        for stage in stage_group
                    ]
                    await asyncio.gather(*tasks)
                
                # Update current stage
                execution.current_stage = group_idx + 1
                
                # Save checkpoint if enabled
                if any(stage.checkpoint for stage in stage_group):
                    await self._save_checkpoint(exec_id, execution)
            
            # Pipeline completed successfully
            execution.status = PipelineStatus.COMPLETED
            execution.completed_at = datetime.now()
            
            # Calculate metrics
            execution.metrics["duration_seconds"] = (
                execution.completed_at - execution.started_at
            ).total_seconds()
            execution.metrics["stages_completed"] = len(execution.pipeline.stages)
            
            self.logger.info(f"Pipeline completed: {exec_id}")
            
        except asyncio.CancelledError:
            execution.status = PipelineStatus.CANCELLED
            execution.completed_at = datetime.now()
            self.logger.info(f"Pipeline cancelled: {exec_id}")
            
        except Exception as e:
            execution.status = PipelineStatus.FAILED
            execution.completed_at = datetime.now()
            execution.error = str(e)
            self.logger.error(f"Pipeline failed: {exec_id} - {e}")
            
        finally:
            self._running_pipelines.pop(exec_id, None)
    
    async def _execute_stage(
        self,
        execution: PipelineExecution,
        stage: PipelineStage
    ) -> None:
        """
        Execute a single pipeline stage.
        
        Args:
            execution: Pipeline execution instance
            stage: Stage to execute
            
        Raises:
            ServiceException: If stage execution fails
        """
        self.logger.info(f"Executing stage: {stage.name}")
        start_time = datetime.now()
        
        retry_count = 0
        while retry_count <= stage.retry_on_failure:
            try:
                # Prepare input data
                input_data = self._prepare_stage_input(execution, stage)
                
                # Execute processor
                if asyncio.iscoroutinefunction(stage.processor):
                    output = await stage.processor(input_data, **stage.config)
                else:
                    output = stage.processor(input_data, **stage.config)
                
                # Store output
                execution.stage_outputs[stage.name] = output
                
                # Update metrics
                duration = (datetime.now() - start_time).total_seconds()
                execution.metrics[f"stage_{stage.name}_duration"] = duration
                
                self.logger.info(f"Stage completed: {stage.name} ({duration:.2f}s)")
                return
                
            except Exception as e:
                retry_count += 1
                if retry_count > stage.retry_on_failure:
                    raise ServiceException(
                        f"Stage {stage.name} failed after {stage.retry_on_failure} retries: {e}"
                    )
                
                self.logger.warning(
                    f"Stage {stage.name} failed (attempt {retry_count}/{stage.retry_on_failure}): {e}"
                )
                await asyncio.sleep(2 ** retry_count)  # Exponential backoff
    
    def _prepare_stage_input(
        self,
        execution: PipelineExecution,
        stage: PipelineStage
    ) -> Dict[str, Any]:
        """
        Prepare input data for a stage.
        
        Args:
            execution: Pipeline execution instance
            stage: Stage to prepare input for
            
        Returns:
            Dict[str, Any]: Prepared input data
        """
        input_data = {}
        
        # Extract specified input keys from previous outputs
        for key in stage.input_keys:
            # Search for key in all previous stage outputs
            for stage_name, output in execution.stage_outputs.items():
                if isinstance(output, dict) and key in output:
                    input_data[key] = output[key]
                    break
        
        # If no specific keys, pass all previous outputs
        if not stage.input_keys:
            input_data = {
                k: v for k, v in execution.stage_outputs.items()
                if k != "__input__"
            }
        
        return input_data
    
    def _group_stages(self, stages: List[PipelineStage]) -> List[List[PipelineStage]]:
        """
        Group stages by parallel execution capability.
        
        Args:
            stages: List of pipeline stages
            
        Returns:
            List[List[PipelineStage]]: Grouped stages
        """
        groups = []
        current_group = []
        
        for stage in stages:
            if stage.parallel and current_group:
                current_group.append(stage)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [stage]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    async def _save_checkpoint(
        self,
        exec_id: str,
        execution: PipelineExecution
    ) -> None:
        """
        Save pipeline execution checkpoint.
        
        Args:
            exec_id: Execution identifier
            execution: Pipeline execution instance
        """
        checkpoint_path = self._checkpoint_dir / f"{exec_id}_stage_{execution.current_stage}.pkl"
        
        checkpoint_data = {
            "exec_id": exec_id,
            "pipeline_name": execution.pipeline.name,
            "current_stage": execution.current_stage,
            "stage_outputs": execution.stage_outputs,
            "metrics": execution.metrics,
            "timestamp": datetime.now()
        }
        
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)
        
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    async def resume_from_checkpoint(
        self,
        checkpoint_path: Path
    ) -> str:
        """
        Resume pipeline execution from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            str: New execution ID
            
        Raises:
            ValueError: If checkpoint is invalid
        """
        with open(checkpoint_path, "rb") as f:
            checkpoint_data = pickle.load(f)
        
        pipeline_name = checkpoint_data["pipeline_name"]
        
        # Find pipeline
        pipeline = None
        for key, p in self._pipelines.items():
            if p.name == pipeline_name:
                pipeline = p
                break
        
        if not pipeline:
            raise ValueError(f"Pipeline not found: {pipeline_name}")
        
        # Create new execution from checkpoint
        execution = PipelineExecution(
            pipeline=pipeline,
            status=PipelineStatus.RUNNING,
            current_stage=checkpoint_data["current_stage"],
            stage_outputs=checkpoint_data["stage_outputs"],
            metrics=checkpoint_data["metrics"],
            started_at=datetime.now()
        )
        
        # Generate new execution ID
        exec_id = f"{pipeline_name}_resumed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._executions[exec_id] = execution
        
        # Start execution from checkpoint
        task = asyncio.create_task(self._execute_pipeline_async(exec_id, execution))
        self._running_pipelines[exec_id] = task
        
        self.logger.info(f"Resumed pipeline from checkpoint: {exec_id}")
        return exec_id
    
    def get_execution_status(self, exec_id: str) -> Optional[PipelineExecution]:
        """
        Get pipeline execution status.
        
        Args:
            exec_id: Execution identifier
            
        Returns:
            PipelineExecution: Execution instance or None
        """
        return self._executions.get(exec_id)
    
    async def pause_pipeline(self, exec_id: str) -> bool:
        """
        Pause a running pipeline.
        
        Args:
            exec_id: Execution identifier
            
        Returns:
            bool: True if paused successfully
        """
        if exec_id in self._running_pipelines:
            task = self._running_pipelines[exec_id]
            task.cancel()
            
            execution = self._executions.get(exec_id)
            if execution:
                execution.status = PipelineStatus.PAUSED
                await self._save_checkpoint(exec_id, execution)
            
            self.logger.info(f"Paused pipeline: {exec_id}")
            return True
        
        return False
