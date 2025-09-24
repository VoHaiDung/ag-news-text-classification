"""
State Manager Implementation for AG News Text Classification
================================================================================
This module implements state management for workflows and pipelines, providing
persistence, recovery, and state transition tracking.

The state manager provides:
- State persistence and recovery
- State transition validation
- Checkpoint management
- State history tracking

References:
    - van der Aalst, W. M. (2016). Process Mining: Data Science in Action
    - Workflow Management Coalition Standards

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
import json
import pickle
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
import sqlite3

from src.services.base_service import BaseService, ServiceConfig
from src.core.exceptions import ServiceException
from src.utils.logging_config import get_logger
from src.utils.io_utils import save_json, load_json


class StateType(Enum):
    """
    Types of workflow states.
    
    Types:
        INITIAL: Initial state
        INTERMEDIATE: Intermediate processing state
        DECISION: Decision point state
        PARALLEL: Parallel execution state
        FINAL: Final/terminal state
        ERROR: Error state
    """
    INITIAL = "initial"
    INTERMEDIATE = "intermediate"
    DECISION = "decision"
    PARALLEL = "parallel"
    FINAL = "final"
    ERROR = "error"


@dataclass
class StateTransition:
    """
    State transition definition.
    
    Attributes:
        from_state: Source state
        to_state: Target state
        condition: Transition condition
        action: Action to perform on transition
        metadata: Additional transition metadata
    """
    from_state: str
    to_state: str
    condition: Optional[Callable] = None
    action: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self, context: Dict[str, Any]) -> bool:
        """
        Check if transition is valid given context.
        
        Args:
            context: Current execution context
            
        Returns:
            bool: True if transition is valid
        """
        if self.condition:
            try:
                return self.condition(context)
            except Exception:
                return False
        return True


@dataclass
class WorkflowState:
    """
    Workflow execution state.
    
    Attributes:
        workflow_id: Workflow identifier
        current_state: Current state name
        state_type: Type of current state
        context: State context data
        history: State transition history
        created_at: State creation time
        updated_at: Last update time
        checkpoints: Saved checkpoints
    """
    workflow_id: str
    current_state: str
    state_type: StateType = StateType.INITIAL
    context: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    checkpoints: List[str] = field(default_factory=list)
    
    def add_history_entry(self, from_state: str, to_state: str, metadata: Dict[str, Any] = None):
        """
        Add entry to state history.
        
        Args:
            from_state: Previous state
            to_state: New state
            metadata: Additional metadata
        """
        entry = {
            "from_state": from_state,
            "to_state": to_state,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.history.append(entry)
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "current_state": self.current_state,
            "state_type": self.state_type.value,
            "context": self.context,
            "history": self.history,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "checkpoints": self.checkpoints
        }


@dataclass
class StateMachine:
    """
    State machine definition.
    
    Attributes:
        name: State machine name
        states: Set of valid states
        transitions: List of state transitions
        initial_state: Initial state name
        final_states: Set of final states
        error_states: Set of error states
    """
    name: str
    states: Set[str] = field(default_factory=set)
    transitions: List[StateTransition] = field(default_factory=list)
    initial_state: str = ""
    final_states: Set[str] = field(default_factory=set)
    error_states: Set[str] = field(default_factory=set)
    
    def validate(self) -> bool:
        """
        Validate state machine configuration.
        
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.name:
            raise ValueError("State machine name is required")
        
        if not self.initial_state:
            raise ValueError("Initial state is required")
        
        if self.initial_state not in self.states:
            raise ValueError(f"Initial state {self.initial_state} not in states")
        
        # Validate transitions
        for transition in self.transitions:
            if transition.from_state not in self.states:
                raise ValueError(f"Invalid from_state: {transition.from_state}")
            if transition.to_state not in self.states:
                raise ValueError(f"Invalid to_state: {transition.to_state}")
        
        return True
    
    def get_valid_transitions(self, current_state: str) -> List[StateTransition]:
        """
        Get valid transitions from current state.
        
        Args:
            current_state: Current state name
            
        Returns:
            List[StateTransition]: Valid transitions
        """
        return [t for t in self.transitions if t.from_state == current_state]


class StateManager(BaseService):
    """
    Manager for workflow state persistence and transitions.
    
    This service manages workflow states with support for persistence,
    recovery, validation, and history tracking.
    """
    
    def __init__(
        self,
        config: Optional[ServiceConfig] = None,
        storage_path: Optional[Path] = None,
        enable_persistence: bool = True
    ):
        """
        Initialize state manager.
        
        Args:
            config: Service configuration
            storage_path: Path for state storage
            enable_persistence: Enable state persistence
        """
        if config is None:
            config = ServiceConfig(name="state_manager")
        super().__init__(config)
        
        self.storage_path = storage_path or Path("outputs/state_storage")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.enable_persistence = enable_persistence
        
        # State storage
        self._states: Dict[str, WorkflowState] = {}
        self._state_machines: Dict[str, StateMachine] = {}
        
        # Database connection
        self._db_path = self.storage_path / "states.db"
        self._db_conn: Optional[sqlite3.Connection] = None
        
        # State change callbacks
        self._state_callbacks: Dict[str, List[Callable]] = {}
        
        self.logger = get_logger("service.state_manager")
    
    async def _initialize(self) -> None:
        """Initialize state manager resources."""
        self.logger.info("Initializing state manager")
        
        if self.enable_persistence:
            await self._initialize_database()
            await self._load_persisted_states()
    
    async def _start(self) -> None:
        """Start state manager service."""
        self.logger.info("State manager started")
    
    async def _stop(self) -> None:
        """Stop state manager service."""
        if self.enable_persistence:
            await self._persist_all_states()
        
        if self._db_conn:
            self._db_conn.close()
        
        self.logger.info("State manager stopped")
    
    async def _cleanup(self) -> None:
        """Cleanup state manager resources."""
        self._states.clear()
        self._state_machines.clear()
    
    async def _check_health(self) -> bool:
        """Check state manager health."""
        if self.enable_persistence and self._db_conn:
            try:
                cursor = self._db_conn.cursor()
                cursor.execute("SELECT 1")
                return True
            except Exception:
                return False
        return True
    
    async def _initialize_database(self) -> None:
        """Initialize SQLite database for state persistence."""
        self._db_conn = sqlite3.connect(str(self._db_path))
        
        # Create tables
        cursor = self._db_conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workflow_states (
                workflow_id TEXT PRIMARY KEY,
                current_state TEXT NOT NULL,
                state_type TEXT NOT NULL,
                context TEXT,
                history TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                checkpoints TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS state_machines (
                name TEXT PRIMARY KEY,
                definition TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        self._db_conn.commit()
        
        self.logger.info(f"Initialized database at {self._db_path}")
    
    async def _load_persisted_states(self) -> None:
        """Load persisted states from database."""
        cursor = self._db_conn.cursor()
        
        # Load workflow states
        cursor.execute("SELECT * FROM workflow_states")
        rows = cursor.fetchall()
        
        for row in rows:
            state = WorkflowState(
                workflow_id=row[0],
                current_state=row[1],
                state_type=StateType(row[2]),
                context=json.loads(row[3]) if row[3] else {},
                history=json.loads(row[4]) if row[4] else [],
                created_at=datetime.fromisoformat(row[5]),
                updated_at=datetime.fromisoformat(row[6]),
                checkpoints=json.loads(row[7]) if row[7] else []
            )
            self._states[state.workflow_id] = state
        
        # Load state machines
        cursor.execute("SELECT * FROM state_machines")
        rows = cursor.fetchall()
        
        for row in rows:
            definition = json.loads(row[1])
            machine = StateMachine(
                name=row[0],
                states=set(definition["states"]),
                initial_state=definition["initial_state"],
                final_states=set(definition["final_states"]),
                error_states=set(definition["error_states"])
            )
            
            # Reconstruct transitions (without callables)
            for trans_def in definition["transitions"]:
                transition = StateTransition(
                    from_state=trans_def["from_state"],
                    to_state=trans_def["to_state"],
                    metadata=trans_def.get("metadata", {})
                )
                machine.transitions.append(transition)
            
            self._state_machines[machine.name] = machine
        
        self.logger.info(
            f"Loaded {len(self._states)} states and "
            f"{len(self._state_machines)} state machines"
        )
    
    async def _persist_all_states(self) -> None:
        """Persist all states to database."""
        if not self._db_conn:
            return
        
        cursor = self._db_conn.cursor()
        
        # Persist workflow states
        for state in self._states.values():
            cursor.execute("""
                INSERT OR REPLACE INTO workflow_states 
                (workflow_id, current_state, state_type, context, history, 
                 created_at, updated_at, checkpoints)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                state.workflow_id,
                state.current_state,
                state.state_type.value,
                json.dumps(state.context),
                json.dumps(state.history),
                state.created_at.isoformat(),
                state.updated_at.isoformat(),
                json.dumps(state.checkpoints)
            ))
        
        self._db_conn.commit()
        
        self.logger.info(f"Persisted {len(self._states)} states")
    
    def register_state_machine(self, machine: StateMachine) -> None:
        """
        Register a state machine definition.
        
        Args:
            machine: State machine to register
            
        Raises:
            ValueError: If machine is invalid
        """
        machine.validate()
        self._state_machines[machine.name] = machine
        
        if self.enable_persistence and self._db_conn:
            # Persist machine definition
            definition = {
                "states": list(machine.states),
                "initial_state": machine.initial_state,
                "final_states": list(machine.final_states),
                "error_states": list(machine.error_states),
                "transitions": [
                    {
                        "from_state": t.from_state,
                        "to_state": t.to_state,
                        "metadata": t.metadata
                    }
                    for t in machine.transitions
                ]
            }
            
            cursor = self._db_conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO state_machines (name, definition, created_at)
                VALUES (?, ?, ?)
            """, (
                machine.name,
                json.dumps(definition),
                datetime.now().isoformat()
            ))
            self._db_conn.commit()
        
        self.logger.info(f"Registered state machine: {machine.name}")
    
    async def create_workflow_state(
        self,
        workflow_id: str,
        machine_name: str,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> WorkflowState:
        """
        Create a new workflow state.
        
        Args:
            workflow_id: Workflow identifier
            machine_name: State machine name
            initial_context: Initial context data
            
        Returns:
            WorkflowState: Created state
            
        Raises:
            ValueError: If machine not found
        """
        if machine_name not in self._state_machines:
            raise ValueError(f"State machine not found: {machine_name}")
        
        machine = self._state_machines[machine_name]
        
        state = WorkflowState(
            workflow_id=workflow_id,
            current_state=machine.initial_state,
            state_type=StateType.INITIAL,
            context=initial_context or {}
        )
        
        self._states[workflow_id] = state
        
        if self.enable_persistence:
            await self._persist_state(state)
        
        self.logger.info(f"Created workflow state: {workflow_id}")
        return state
    
    async def transition_state(
        self,
        workflow_id: str,
        to_state: str,
        context_update: Optional[Dict[str, Any]] = None
    ) -> WorkflowState:
        """
        Transition workflow to new state.
        
        Args:
            workflow_id: Workflow identifier
            to_state: Target state
            context_update: Context updates
            
        Returns:
            WorkflowState: Updated state
            
        Raises:
            ValueError: If transition is invalid
        """
        if workflow_id not in self._states:
            raise ValueError(f"Workflow state not found: {workflow_id}")
        
        state = self._states[workflow_id]
        from_state = state.current_state
        
        # Find applicable state machine
        machine = None
        for m in self._state_machines.values():
            if from_state in m.states and to_state in m.states:
                machine = m
                break
        
        if not machine:
            raise ValueError(f"No state machine found for transition {from_state} -> {to_state}")
        
        # Validate transition
        valid_transitions = machine.get_valid_transitions(from_state)
        transition = None
        
        for t in valid_transitions:
            if t.to_state == to_state and t.is_valid(state.context):
                transition = t
                break
        
        if not transition:
            raise ValueError(f"Invalid transition: {from_state} -> {to_state}")
        
        # Execute transition action
        if transition.action:
            try:
                if asyncio.iscoroutinefunction(transition.action):
                    await transition.action(state.context)
                else:
                    transition.action(state.context)
            except Exception as e:
                self.logger.error(f"Transition action failed: {e}")
                raise ServiceException(f"Transition action failed: {e}")
        
        # Update state
        state.current_state = to_state
        
        # Determine state type
        if to_state in machine.final_states:
            state.state_type = StateType.FINAL
        elif to_state in machine.error_states:
            state.state_type = StateType.ERROR
        else:
            state.state_type = StateType.INTERMEDIATE
        
        # Update context
        if context_update:
            state.context.update(context_update)
        
        # Add history entry
        state.add_history_entry(from_state, to_state, transition.metadata)
        
        # Persist state
        if self.enable_persistence:
            await self._persist_state(state)
        
        # Trigger callbacks
        await self._trigger_state_callbacks(workflow_id, from_state, to_state)
        
        self.logger.info(f"Transitioned {workflow_id}: {from_state} -> {to_state}")
        return state
    
    async def _persist_state(self, state: WorkflowState) -> None:
        """
        Persist state to database.
        
        Args:
            state: State to persist
        """
        if not self._db_conn:
            return
        
        cursor = self._db_conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO workflow_states 
            (workflow_id, current_state, state_type, context, history, 
             created_at, updated_at, checkpoints)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            state.workflow_id,
            state.current_state,
            state.state_type.value,
            json.dumps(state.context),
            json.dumps(state.history),
            state.created_at.isoformat(),
            state.updated_at.isoformat(),
            json.dumps(state.checkpoints)
        ))
        self._db_conn.commit()
    
    async def save_checkpoint(
        self,
        workflow_id: str,
        checkpoint_name: Optional[str] = None
    ) -> str:
        """
        Save workflow state checkpoint.
        
        Args:
            workflow_id: Workflow identifier
            checkpoint_name: Optional checkpoint name
            
        Returns:
            str: Checkpoint identifier
        """
        if workflow_id not in self._states:
            raise ValueError(f"Workflow state not found: {workflow_id}")
        
        state = self._states[workflow_id]
        
        # Generate checkpoint name
        if not checkpoint_name:
            checkpoint_name = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save checkpoint to file
        checkpoint_path = self.storage_path / f"{workflow_id}_{checkpoint_name}.pkl"
        
        with open(checkpoint_path, "wb") as f:
            pickle.dump(state, f)
        
        # Add to checkpoint list
        state.checkpoints.append(checkpoint_name)
        
        if self.enable_persistence:
            await self._persist_state(state)
        
        self.logger.info(f"Saved checkpoint for {workflow_id}: {checkpoint_name}")
        return checkpoint_name
    
    async def restore_checkpoint(
        self,
        workflow_id: str,
        checkpoint_name: str
    ) -> WorkflowState:
        """
        Restore workflow state from checkpoint.
        
        Args:
            workflow_id: Workflow identifier
            checkpoint_name: Checkpoint name
            
        Returns:
            WorkflowState: Restored state
        """
        checkpoint_path = self.storage_path / f"{workflow_id}_{checkpoint_name}.pkl"
        
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_name}")
        
        with open(checkpoint_path, "rb") as f:
            state = pickle.load(f)
        
        self._states[workflow_id] = state
        
        if self.enable_persistence:
            await self._persist_state(state)
        
        self.logger.info(f"Restored checkpoint for {workflow_id}: {checkpoint_name}")
        return state
    
    def get_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """
        Get workflow state.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            WorkflowState: Current state or None
        """
        return self._states.get(workflow_id)
    
    def add_state_callback(
        self,
        workflow_id: str,
        callback: Callable
    ) -> None:
        """
        Add state change callback.
        
        Args:
            workflow_id: Workflow identifier
            callback: Callback function
        """
        if workflow_id not in self._state_callbacks:
            self._state_callbacks[workflow_id] = []
        
        self._state_callbacks[workflow_id].append(callback)
        self.logger.debug(f"Added state callback for {workflow_id}")
    
    async def _trigger_state_callbacks(
        self,
        workflow_id: str,
        from_state: str,
        to_state: str
    ) -> None:
        """
        Trigger state change callbacks.
        
        Args:
            workflow_id: Workflow identifier
            from_state: Previous state
            to_state: New state
        """
        if workflow_id in self._state_callbacks:
            for callback in self._state_callbacks[workflow_id]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(workflow_id, from_state, to_state)
                    else:
                        callback(workflow_id, from_state, to_state)
                except Exception as e:
                    self.logger.error(f"State callback error: {e}")
