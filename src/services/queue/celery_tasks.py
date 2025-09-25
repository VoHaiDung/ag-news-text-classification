"""
Celery Task Definitions for AG News Text Classification
================================================================================
This module defines asynchronous tasks for the AG News classification system
using Celery for distributed task processing.

The tasks include:
- Model training tasks
- Batch prediction tasks
- Data processing tasks
- Model evaluation tasks
- Notification tasks

References:
    - Celery Documentation
    - Celery Best Practices
    - Task Design Patterns

Author: Võ Hải Dũng
License: MIT
"""

import os
import json
import pickle
import traceback
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path

from celery import Celery, Task, group, chain, chord
from celery.utils.log import get_task_logger
from celery.result import AsyncResult
from kombu import Queue, Exchange

from src.core.registry import Registry
from src.core.exceptions import ServiceException
from src.services.core.prediction_service import PredictionService
from src.services.core.training_service import TrainingService
from src.services.core.data_service import DataService
from src.services.core.model_management_service import ModelManagementService
from src.services.notification.notification_service import NotificationService
from src.utils.logging_config import get_logger


# Celery configuration
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

# Create Celery app
app = Celery(
    "ag_news_tasks",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)

# Configure Celery
app.conf.update(
    task_serializer="json",
    accept_content=["json", "pickle"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour hard limit
    task_soft_time_limit=3000,  # 50 minutes soft limit
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    result_expires=86400,  # Results expire after 1 day
    task_routes={
        "tasks.training.*": {"queue": "training"},
        "tasks.prediction.*": {"queue": "prediction"},
        "tasks.data.*": {"queue": "data"},
        "tasks.evaluation.*": {"queue": "evaluation"},
        "tasks.notification.*": {"queue": "notification"}
    }
)

# Define queues
app.conf.task_queues = (
    Queue("default", Exchange("default"), routing_key="default"),
    Queue("training", Exchange("training"), routing_key="training", priority=5),
    Queue("prediction", Exchange("prediction"), routing_key="prediction", priority=10),
    Queue("data", Exchange("data"), routing_key="data", priority=3),
    Queue("evaluation", Exchange("evaluation"), routing_key="evaluation", priority=3),
    Queue("notification", Exchange("notification"), routing_key="notification", priority=1)
)

# Task logger
logger = get_task_logger(__name__)


class BaseTask(Task):
    """
    Base task class with automatic resource management.
    
    This class provides common functionality for all tasks including
    error handling, logging, and service initialization.
    """
    
    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 3, "countdown": 5}
    
    def __init__(self):
        """Initialize base task."""
        super().__init__()
        self._services = {}
    
    def get_service(self, service_name: str):
        """
        Get or create a service instance.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Service instance
        """
        if service_name not in self._services:
            if service_name == "prediction":
                self._services[service_name] = PredictionService()
            elif service_name == "training":
                self._services[service_name] = TrainingService()
            elif service_name == "data":
                self._services[service_name] = DataService()
            elif service_name == "model_management":
                self._services[service_name] = ModelManagementService()
            elif service_name == "notification":
                self._services[service_name] = NotificationService()
            else:
                raise ValueError(f"Unknown service: {service_name}")
        
        return self._services[service_name]
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """
        Handle task failure.
        
        Args:
            exc: Exception instance
            task_id: Task ID
            args: Task arguments
            kwargs: Task keyword arguments
            einfo: Exception info
        """
        logger.error(
            f"Task {self.name}[{task_id}] failed: {str(exc)}\n"
            f"Args: {args}\n"
            f"Kwargs: {kwargs}\n"
            f"Traceback: {einfo}"
        )
        
        # Send failure notification
        try:
            notification_service = self.get_service("notification")
            notification_service.send_task_failure_notification(
                task_name=self.name,
                task_id=task_id,
                error=str(exc),
                traceback=str(einfo)
            )
        except:
            pass
    
    def on_success(self, retval, task_id, args, kwargs):
        """
        Handle task success.
        
        Args:
            retval: Task return value
            task_id: Task ID
            args: Task arguments
            kwargs: Task keyword arguments
        """
        logger.info(f"Task {self.name}[{task_id}] completed successfully")
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """
        Handle task retry.
        
        Args:
            exc: Exception instance
            task_id: Task ID
            args: Task arguments
            kwargs: Task keyword arguments
            einfo: Exception info
        """
        logger.warning(
            f"Task {self.name}[{task_id}] retrying due to: {str(exc)}"
        )


# Training Tasks
@app.task(base=BaseTask, name="tasks.training.train_model")
def train_model(
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    training_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Train a model asynchronously.
    
    Args:
        model_config: Model configuration
        data_config: Data configuration
        training_config: Training configuration
        
    Returns:
        Training results including model ID and metrics
    """
    logger.info(f"Starting model training with config: {model_config['name']}")
    
    try:
        # Get services
        training_service = train_model.get_service("training")
        data_service = train_model.get_service("data")
        
        # Load data
        dataset = data_service.load_dataset(data_config)
        
        # Train model
        result = training_service.train(
            model_config=model_config,
            dataset=dataset,
            training_config=training_config
        )
        
        logger.info(f"Model training completed: {result['model_id']}")
        
        return {
            "status": "success",
            "model_id": result["model_id"],
            "metrics": result["metrics"],
            "training_time": result["training_time"]
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


@app.task(base=BaseTask, name="tasks.training.finetune_model")
def finetune_model(
    model_id: str,
    data_config: Dict[str, Any],
    finetune_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Fine-tune an existing model.
    
    Args:
        model_id: ID of model to fine-tune
        data_config: Data configuration
        finetune_config: Fine-tuning configuration
        
    Returns:
        Fine-tuning results
    """
    logger.info(f"Starting model fine-tuning for: {model_id}")
    
    try:
        training_service = finetune_model.get_service("training")
        data_service = finetune_model.get_service("data")
        model_service = finetune_model.get_service("model_management")
        
        # Load model
        model = model_service.load_model(model_id)
        
        # Load data
        dataset = data_service.load_dataset(data_config)
        
        # Fine-tune
        result = training_service.finetune(
            model=model,
            dataset=dataset,
            finetune_config=finetune_config
        )
        
        return {
            "status": "success",
            "model_id": result["model_id"],
            "original_model_id": model_id,
            "metrics": result["metrics"]
        }
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {str(e)}")
        raise


# Prediction Tasks
@app.task(base=BaseTask, name="tasks.prediction.predict_batch")
def predict_batch(
    model_id: str,
    texts: List[str],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Perform batch prediction.
    
    Args:
        model_id: Model ID to use
        texts: List of texts to classify
        config: Prediction configuration
        
    Returns:
        Prediction results
    """
    logger.info(f"Starting batch prediction with model {model_id} for {len(texts)} texts")
    
    try:
        prediction_service = predict_batch.get_service("prediction")
        
        results = prediction_service.predict_batch(
            model_id=model_id,
            texts=texts,
            config=config or {}
        )
        
        return {
            "status": "success",
            "model_id": model_id,
            "num_samples": len(texts),
            "predictions": results["predictions"],
            "processing_time": results["processing_time"]
        }
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise


@app.task(base=BaseTask, name="tasks.prediction.predict_dataset")
def predict_dataset(
    model_id: str,
    dataset_path: str,
    output_path: str,
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Predict on entire dataset.
    
    Args:
        model_id: Model ID to use
        dataset_path: Path to dataset
        output_path: Path to save predictions
        batch_size: Batch size for processing
        
    Returns:
        Prediction statistics
    """
    logger.info(f"Starting dataset prediction with model {model_id}")
    
    try:
        prediction_service = predict_dataset.get_service("prediction")
        data_service = predict_dataset.get_service("data")
        
        # Load dataset
        dataset = data_service.load_dataset_from_path(dataset_path)
        
        # Process in batches
        all_predictions = []
        total_samples = len(dataset)
        
        for i in range(0, total_samples, batch_size):
            batch = dataset[i:i + batch_size]
            batch_results = prediction_service.predict_batch(
                model_id=model_id,
                texts=batch["text"],
                config={"batch_size": batch_size}
            )
            all_predictions.extend(batch_results["predictions"])
            
            # Update progress
            progress = (i + len(batch)) / total_samples * 100
            predict_dataset.update_state(
                state="PROGRESS",
                meta={"current": i + len(batch), "total": total_samples, "progress": progress}
            )
        
        # Save predictions
        data_service.save_predictions(all_predictions, output_path)
        
        return {
            "status": "success",
            "model_id": model_id,
            "total_samples": total_samples,
            "output_path": output_path
        }
        
    except Exception as e:
        logger.error(f"Dataset prediction failed: {str(e)}")
        raise


# Data Processing Tasks
@app.task(base=BaseTask, name="tasks.data.preprocess_data")
def preprocess_data(
    input_path: str,
    output_path: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Preprocess data for training.
    
    Args:
        input_path: Input data path
        output_path: Output path for processed data
        config: Preprocessing configuration
        
    Returns:
        Preprocessing statistics
    """
    logger.info(f"Starting data preprocessing: {input_path}")
    
    try:
        data_service = preprocess_data.get_service("data")
        
        result = data_service.preprocess_dataset(
            input_path=input_path,
            output_path=output_path,
            config=config
        )
        
        return {
            "status": "success",
            "input_path": input_path,
            "output_path": output_path,
            "num_samples": result["num_samples"],
            "processing_time": result["processing_time"]
        }
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {str(e)}")
        raise


@app.task(base=BaseTask, name="tasks.data.augment_data")
def augment_data(
    input_path: str,
    output_path: str,
    augmentation_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Augment training data.
    
    Args:
        input_path: Input data path
        output_path: Output path for augmented data
        augmentation_config: Augmentation configuration
        
    Returns:
        Augmentation statistics
    """
    logger.info(f"Starting data augmentation: {input_path}")
    
    try:
        data_service = augment_data.get_service("data")
        
        result = data_service.augment_dataset(
            input_path=input_path,
            output_path=output_path,
            config=augmentation_config
        )
        
        return {
            "status": "success",
            "original_samples": result["original_samples"],
            "augmented_samples": result["augmented_samples"],
            "augmentation_ratio": result["augmentation_ratio"]
        }
        
    except Exception as e:
        logger.error(f"Data augmentation failed: {str(e)}")
        raise


# Evaluation Tasks
@app.task(base=BaseTask, name="tasks.evaluation.evaluate_model")
def evaluate_model(
    model_id: str,
    test_data_path: str,
    metrics: List[str]
) -> Dict[str, Any]:
    """
    Evaluate model performance.
    
    Args:
        model_id: Model ID to evaluate
        test_data_path: Path to test data
        metrics: List of metrics to compute
        
    Returns:
        Evaluation results
    """
    logger.info(f"Starting model evaluation: {model_id}")
    
    try:
        from src.evaluation.metrics.classification_metrics import ClassificationMetrics
        
        prediction_service = evaluate_model.get_service("prediction")
        data_service = evaluate_model.get_service("data")
        
        # Load test data
        test_data = data_service.load_dataset_from_path(test_data_path)
        
        # Get predictions
        predictions = prediction_service.predict_batch(
            model_id=model_id,
            texts=test_data["text"]
        )
        
        # Calculate metrics
        evaluator = ClassificationMetrics()
        results = evaluator.calculate_metrics(
            y_true=test_data["labels"],
            y_pred=predictions["predictions"],
            metrics=metrics
        )
        
        return {
            "status": "success",
            "model_id": model_id,
            "metrics": results,
            "num_samples": len(test_data)
        }
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {str(e)}")
        raise


# Composite Tasks
@app.task(base=BaseTask, name="tasks.composite.train_and_evaluate")
def train_and_evaluate(
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    training_config: Dict[str, Any],
    evaluation_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Train and evaluate a model in sequence.
    
    Args:
        model_config: Model configuration
        data_config: Data configuration
        training_config: Training configuration
        evaluation_config: Evaluation configuration
        
    Returns:
        Combined training and evaluation results
    """
    # Create a chain of tasks
    workflow = chain(
        train_model.s(model_config, data_config, training_config),
        evaluate_model.s(
            evaluation_config["test_data_path"],
            evaluation_config["metrics"]
        )
    )
    
    # Execute workflow
    result = workflow.apply_async()
    
    return {
        "status": "started",
        "workflow_id": result.id,
        "tasks": ["training", "evaluation"]
    }


@app.task(base=BaseTask, name="tasks.composite.ensemble_prediction")
def ensemble_prediction(
    model_ids: List[str],
    texts: List[str],
    aggregation_method: str = "voting"
) -> Dict[str, Any]:
    """
    Perform ensemble prediction using multiple models.
    
    Args:
        model_ids: List of model IDs
        texts: Texts to classify
        aggregation_method: Method to aggregate predictions
        
    Returns:
        Ensemble prediction results
    """
    # Create parallel prediction tasks
    prediction_tasks = group(
        predict_batch.s(model_id, texts)
        for model_id in model_ids
    )
    
    # Execute in parallel
    job = prediction_tasks.apply_async()
    results = job.get()
    
    # Aggregate predictions
    from src.models.ensemble.base_ensemble import aggregate_predictions
    
    all_predictions = [r["predictions"] for r in results]
    final_predictions = aggregate_predictions(
        all_predictions,
        method=aggregation_method
    )
    
    return {
        "status": "success",
        "model_ids": model_ids,
        "num_models": len(model_ids),
        "predictions": final_predictions,
        "aggregation_method": aggregation_method
    }


# Notification Tasks
@app.task(base=BaseTask, name="tasks.notification.send_notification")
def send_notification(
    recipient: str,
    subject: str,
    message: str,
    priority: str = "medium",
    channel: str = "email"
) -> Dict[str, Any]:
    """
    Send notification asynchronously.
    
    Args:
        recipient: Notification recipient
        subject: Notification subject
        message: Notification message
        priority: Notification priority
        channel: Notification channel
        
    Returns:
        Notification status
    """
    try:
        notification_service = send_notification.get_service("notification")
        
        result = notification_service.send(
            recipient=recipient,
            subject=subject,
            message=message,
            priority=priority,
            channel=channel
        )
        
        return {
            "status": "success" if result else "failed",
            "recipient": recipient,
            "channel": channel
        }
        
    except Exception as e:
        logger.error(f"Notification failed: {str(e)}")
        raise


# Task Management Functions
def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Get status of a task.
    
    Args:
        task_id: Task ID
        
    Returns:
        Task status information
    """
    result = AsyncResult(task_id, app=app)
    
    return {
        "task_id": task_id,
        "state": result.state,
        "info": result.info,
        "result": result.result if result.ready() else None,
        "traceback": result.traceback if result.failed() else None
    }


def cancel_task(task_id: str) -> bool:
    """
    Cancel a running task.
    
    Args:
        task_id: Task ID
        
    Returns:
        True if cancelled successfully
    """
    result = AsyncResult(task_id, app=app)
    result.revoke(terminate=True)
    return True


def get_active_tasks() -> List[Dict[str, Any]]:
    """
    Get list of active tasks.
    
    Returns:
        List of active task information
    """
    inspect = app.control.inspect()
    active = inspect.active()
    
    tasks = []
    if active:
        for worker, task_list in active.items():
            for task in task_list:
                tasks.append({
                    "worker": worker,
                    "task_id": task["id"],
                    "name": task["name"],
                    "args": task.get("args", []),
                    "kwargs": task.get("kwargs", {})
                })
    
    return tasks
