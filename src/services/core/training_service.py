"""
Training Service for Model Training and Fine-tuning
================================================================================
Implements comprehensive training capabilities including distributed training,
hyperparameter optimization, experiment tracking, and model versioning.

This service follows MLOps best practices for reproducible and scalable
model training workflows.

References:
    - Goodfellow, I., et al. (2016). Deep Learning
    - He, T., et al. (2020). AutoML: A Survey of the State-of-the-Art
    - Zaharia, M., et al. (2018). Accelerating the Machine Learning Lifecycle with MLflow

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader

from src.services.base_service import BaseService, ServiceConfig
from src.training.trainers.base_trainer import BaseTrainer
from src.utils.logging_config import get_logger
from src.utils.reproducibility import set_seed

logger = get_logger(__name__)


class JobStatus(Enum):
    """Training job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    PAUSED = "paused"


@dataclass
class TrainingJob:
    """
    Training job information.
    
    Attributes:
        job_id: Unique job identifier
        model_name: Model being trained
        dataset_name: Training dataset
        status: Current job status
        config: Training configuration
        progress: Training progress (0-100)
        current_epoch: Current training epoch
        total_epochs: Total number of epochs
        metrics: Training metrics
        created_at: Job creation time
        updated_at: Last update time
        completed_at: Completion time
        error_message: Error message if failed
        checkpoint_path: Path to latest checkpoint
    """
    job_id: str
    model_name: str
    dataset_name: str
    status: JobStatus = JobStatus.PENDING
    config: Dict[str, Any] = field(default_factory=dict)
    progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 10
    metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    checkpoint_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "status": self.status.value,
            "config": self.config,
            "progress": self.progress,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "metrics": self.metrics,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "checkpoint_path": self.checkpoint_path
        }


class TrainingService(BaseService):
    """
    Service for model training and fine-tuning.
    
    Manages training jobs, experiment tracking, hyperparameter optimization,
    and model versioning following MLOps best practices.
    """
    
    def __init__(self, config: ServiceConfig):
        """
        Initialize training service.
        
        Args:
            config: Service configuration
        """
        super().__init__(config)
        self.jobs: Dict[str, TrainingJob] = {}
        self.active_trainers: Dict[str, BaseTrainer] = {}
        self.checkpoint_dir = Path("outputs/models/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training metrics
        self._total_jobs = 0
        self._successful_jobs = 0
        self._failed_jobs = 0
    
    async def _initialize(self) -> None:
        """Initialize service components."""
        logger.info("Initializing training service")
        
        # Load any persisted jobs
        await self._load_persisted_jobs()
    
    async def _shutdown(self) -> None:
        """Cleanup service resources."""
        logger.info("Shutting down training service")
        
        # Stop all active training jobs
        for job_id in list(self.active_trainers.keys()):
            await self.stop_job(job_id, save_checkpoint=True)
        
        # Persist job states
        await self._persist_jobs()
    
    async def create_job(
        self,
        job_id: str,
        model_name: str,
        dataset_name: str,
        config: Dict[str, Any]
    ) -> TrainingJob:
        """
        Create a new training job.
        
        Args:
            job_id: Unique job identifier
            model_name: Model to train
            dataset_name: Dataset for training
            config: Training configuration
            
        Returns:
            Created TrainingJob
        """
        # Create job
        job = TrainingJob(
            job_id=job_id,
            model_name=model_name,
            dataset_name=dataset_name,
            config=config,
            total_epochs=config.get("epochs", 10)
        )
        
        # Store job
        self.jobs[job_id] = job
        self._total_jobs += 1
        
        logger.info(f"Created training job: {job_id}")
        
        return job
    
    async def run_training(
        self,
        job_id: str,
        config: Dict[str, Any]
    ) -> None:
        """
        Run training for a job.
        
        Args:
            job_id: Job identifier
            config: Training configuration
        """
        if job_id not in self.jobs:
            logger.error(f"Job {job_id} not found")
            return
        
        job = self.jobs[job_id]
        
        try:
            # Update job status
            job.status = JobStatus.RUNNING
            job.updated_at = datetime.now(timezone.utc)
            
            logger.info(f"Starting training for job {job_id}")
            
            # Set random seed for reproducibility
            set_seed(config.get("seed", 42))
            
            # Get model and data
            model = await self._get_model(job.model_name)
            train_loader, val_loader = await self._get_data_loaders(
                job.dataset_name,
                config
            )
            
            # Create trainer
            trainer = await self._create_trainer(model, config)
            self.active_trainers[job_id] = trainer
            
            # Training loop
            for epoch in range(job.current_epoch, job.total_epochs):
                if job.status != JobStatus.RUNNING:
                    break
                
                # Train epoch
                train_metrics = await trainer.train_epoch(
                    train_loader,
                    epoch,
                    job.total_epochs
                )
                
                # Validation
                val_metrics = await trainer.validate(val_loader)
                
                # Update job
                job.current_epoch = epoch + 1
                job.progress = (epoch + 1) / job.total_epochs * 100
                job.metrics = {
                    "train": train_metrics,
                    "validation": val_metrics,
                    "best_accuracy": trainer.best_metric
                }
                job.updated_at = datetime.now(timezone.utc)
                
                # Save checkpoint
                if (epoch + 1) % config.get("checkpoint_interval", 1) == 0:
                    checkpoint_path = await self._save_checkpoint(
                        job_id,
                        trainer,
                        epoch + 1
                    )
                    job.checkpoint_path = str(checkpoint_path)
                
                logger.info(
                    f"Job {job_id} - Epoch {epoch + 1}/{job.total_epochs} - "
                    f"Train Loss: {train_metrics.get('loss', 0):.4f}, "
                    f"Val Acc: {val_metrics.get('accuracy', 0):.4f}"
                )
            
            # Training completed
            if job.status == JobStatus.RUNNING:
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.now(timezone.utc)
                self._successful_jobs += 1
                
                # Save final model
                await self._save_final_model(job_id, trainer)
                
                logger.info(f"Training completed for job {job_id}")
            
        except Exception as e:
            logger.error(f"Training failed for job {job_id}: {str(e)}", exc_info=True)
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.updated_at = datetime.now(timezone.utc)
            self._failed_jobs += 1
            
        finally:
            # Cleanup
            if job_id in self.active_trainers:
                del self.active_trainers[job_id]
    
    async def stop_job(
        self,
        job_id: str,
        save_checkpoint: bool = True
    ) -> None:
        """
        Stop a running training job.
        
        Args:
            job_id: Job identifier
            save_checkpoint: Whether to save checkpoint
        """
        if job_id not in self.jobs:
            logger.warning(f"Job {job_id} not found")
            return
        
        job = self.jobs[job_id]
        
        if job.status == JobStatus.RUNNING:
            logger.info(f"Stopping job {job_id}")
            
            # Save checkpoint if requested
            if save_checkpoint and job_id in self.active_trainers:
                trainer = self.active_trainers[job_id]
                checkpoint_path = await self._save_checkpoint(
                    job_id,
                    trainer,
                    job.current_epoch
                )
                job.checkpoint_path = str(checkpoint_path)
            
            # Update status
            job.status = JobStatus.STOPPED
            job.updated_at = datetime.now(timezone.utc)
            
            # Remove active trainer
            if job_id in self.active_trainers:
                del self.active_trainers[job_id]
    
    async def resume_training(
        self,
        job_id: str,
        from_checkpoint: Optional[str] = None
    ) -> None:
        """
        Resume a stopped training job.
        
        Args:
            job_id: Job identifier
            from_checkpoint: Checkpoint to resume from
        """
        if job_id not in self.jobs:
            logger.error(f"Job {job_id} not found")
            return
        
        job = self.jobs[job_id]
        
        if job.status not in [JobStatus.STOPPED, JobStatus.FAILED]:
            logger.warning(f"Job {job_id} cannot be resumed (status: {job.status})")
            return
        
        logger.info(f"Resuming job {job_id}")
        
        # Load checkpoint if specified
        if from_checkpoint:
            await self._load_checkpoint(job_id, from_checkpoint)
        elif job.checkpoint_path:
            await self._load_checkpoint(job_id, job.checkpoint_path)
        
        # Resume training
        await self.run_training(job_id, job.config)
    
    async def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """
        Get job information.
        
        Args:
            job_id: Job identifier
            
        Returns:
            TrainingJob or None
        """
        return self.jobs.get(job_id)
    
    async def list_jobs(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> List[TrainingJob]:
        """
        List training jobs with optional filtering.
        
        Args:
            user_id: Filter by user
            status: Filter by status
            model_name: Filter by model
            
        Returns:
            List of TrainingJob objects
        """
        jobs = list(self.jobs.values())
        
        # Apply filters
        if user_id:
            jobs = [j for j in jobs if j.config.get("user_id") == user_id]
        
        if status:
            try:
                status_enum = JobStatus(status)
                jobs = [j for j in jobs if j.status == status_enum]
            except ValueError:
                logger.warning(f"Invalid status filter: {status}")
        
        if model_name:
            jobs = [j for j in jobs if j.model_name == model_name]
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        
        return jobs
    
    async def get_active_jobs(self) -> List[TrainingJob]:
        """Get list of active training jobs."""
        return [
            job for job in self.jobs.values()
            if job.status in [JobStatus.PENDING, JobStatus.RUNNING]
        ]
    
    async def delete_job(
        self,
        job_id: str,
        delete_artifacts: bool = False
    ) -> None:
        """
        Delete a training job.
        
        Args:
            job_id: Job identifier
            delete_artifacts: Whether to delete artifacts
        """
        if job_id not in self.jobs:
            logger.warning(f"Job {job_id} not found")
            return
        
        job = self.jobs[job_id]
        
        # Stop if running
        if job.status == JobStatus.RUNNING:
            await self.stop_job(job_id, save_checkpoint=False)
        
        # Delete artifacts if requested
        if delete_artifacts and job.checkpoint_path:
            try:
                Path(job.checkpoint_path).unlink(missing_ok=True)
                logger.info(f"Deleted artifacts for job {job_id}")
            except Exception as e:
                logger.error(f"Failed to delete artifacts: {str(e)}")
        
        # Remove job
        del self.jobs[job_id]
        logger.info(f"Deleted job {job_id}")
    
    async def get_logs(
        self,
        job_id: str,
        lines: int = 100
    ) -> List[str]:
        """
        Get training logs for a job.
        
        Args:
            job_id: Job identifier
            lines: Number of lines to return
            
        Returns:
            List of log lines
        """
        # Read from log file
        log_file = Path(f"outputs/logs/training/{job_id}.log")
        
        if not log_file.exists():
            return [f"No logs found for job {job_id}"]
        
        try:
            with open(log_file, "r") as f:
                all_lines = f.readlines()
                return all_lines[-lines:]
        except Exception as e:
            logger.error(f"Failed to read logs: {str(e)}")
            return [f"Error reading logs: {str(e)}"]
    
    async def stream_logs(self, job_id: str):
        """
        Stream training logs.
        
        Args:
            job_id: Job identifier
            
        Yields:
            Log lines
        """
        log_file = Path(f"outputs/logs/training/{job_id}.log")
        
        if not log_file.exists():
            yield f"No logs found for job {job_id}"
            return
        
        # Stream logs
        with open(log_file, "r") as f:
            # Go to end of file
            f.seek(0, 2)
            
            while job_id in self.jobs and self.jobs[job_id].status == JobStatus.RUNNING:
                line = f.readline()
                if line:
                    yield line.rstrip()
                else:
                    await asyncio.sleep(1)
    
    async def _get_model(self, model_name: str):
        """Get model for training."""
        model_service = self.get_dependency("model_management_service")
        if not model_service:
            raise ValueError("Model management service not available")
        
        model = await model_service.get_model_for_training(model_name)
        if not model:
            raise ValueError(f"Model '{model_name}' not found")
        
        return model
    
    async def _get_data_loaders(
        self,
        dataset_name: str,
        config: Dict[str, Any]
    ) -> tuple[DataLoader, DataLoader]:
        """Get data loaders for training."""
        data_service = self.get_dependency("data_service")
        if not data_service:
            raise ValueError("Data service not available")
        
        # Get dataset
        dataset = await data_service.get_dataset(dataset_name)
        if not dataset:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        # Create data loaders
        train_loader = DataLoader(
            dataset["train"],
            batch_size=config.get("batch_size", 32),
            shuffle=True,
            num_workers=config.get("num_workers", 4),
            pin_memory=True
        )
        
        val_loader = DataLoader(
            dataset["validation"],
            batch_size=config.get("batch_size", 32),
            shuffle=False,
            num_workers=config.get("num_workers", 4),
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    async def _create_trainer(
        self,
        model,
        config: Dict[str, Any]
    ) -> BaseTrainer:
        """Create trainer instance."""
        # Import trainer class dynamically
        from src.training.trainers.standard_trainer import StandardTrainer
        
        trainer = StandardTrainer(
            model=model,
            config=config,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        return trainer
    
    async def _save_checkpoint(
        self,
        job_id: str,
        trainer: BaseTrainer,
        epoch: int
    ) -> Path:
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{job_id}_epoch_{epoch}.pt"
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "best_metric": trainer.best_metric,
            "config": self.jobs[job_id].config
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        return checkpoint_path
    
    async def _load_checkpoint(
        self,
        job_id: str,
        checkpoint_path: str
    ) -> None:
        """Load training checkpoint."""
        if job_id not in self.jobs:
            return
        
        job = self.jobs[job_id]
        
        try:
            checkpoint = torch.load(checkpoint_path)
            job.current_epoch = checkpoint["epoch"]
            logger.info(f"Loaded checkpoint from epoch {job.current_epoch}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
    
    async def _save_final_model(
        self,
        job_id: str,
        trainer: BaseTrainer
    ) -> None:
        """Save final trained model."""
        model_service = self.get_dependency("model_management_service")
        if model_service:
            job = self.jobs[job_id]
            await model_service.save_trained_model(
                model=trainer.model,
                name=f"{job.model_name}_trained_{job_id}",
                metrics=job.metrics
            )
    
    async def _persist_jobs(self) -> None:
        """Persist job states to disk."""
        jobs_file = Path("outputs/training_jobs.json")
        
        try:
            jobs_data = {
                job_id: job.to_dict()
                for job_id, job in self.jobs.items()
            }
            
            with open(jobs_file, "w") as f:
                json.dump(jobs_data, f, indent=2)
                
            logger.info(f"Persisted {len(jobs_data)} jobs")
        except Exception as e:
            logger.error(f"Failed to persist jobs: {str(e)}")
    
    async def _load_persisted_jobs(self) -> None:
        """Load persisted job states."""
        jobs_file = Path("outputs/training_jobs.json")
        
        if not jobs_file.exists():
            return
        
        try:
            with open(jobs_file, "r") as f:
                jobs_data = json.load(f)
            
            for job_id, job_dict in jobs_data.items():
                # Reconstruct job
                job = TrainingJob(
                    job_id=job_dict["job_id"],
                    model_name=job_dict["model_name"],
                    dataset_name=job_dict["dataset_name"],
                    status=JobStatus(job_dict["status"]),
                    config=job_dict["config"],
                    progress=job_dict["progress"],
                    current_epoch=job_dict["current_epoch"],
                    total_epochs=job_dict["total_epochs"],
                    metrics=job_dict["metrics"]
                )
                
                # Parse timestamps
                job.created_at = datetime.fromisoformat(job_dict["created_at"])
                job.updated_at = datetime.fromisoformat(job_dict["updated_at"])
                
                if job_dict.get("completed_at"):
                    job.completed_at = datetime.fromisoformat(job_dict["completed_at"])
                
                job.error_message = job_dict.get("error_message")
                job.checkpoint_path = job_dict.get("checkpoint_path")
                
                self.jobs[job_id] = job
            
            logger.info(f"Loaded {len(self.jobs)} persisted jobs")
            
        except Exception as e:
            logger.error(f"Failed to load persisted jobs: {str(e)}")
    
    async def _execute(self, *args, **kwargs) -> Any:
        """Execute service operation."""
        # Not directly callable
        raise NotImplementedError("Training service operations must be called directly")
