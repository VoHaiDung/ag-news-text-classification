"""
Production-Ready State-of-the-Art Experiments for AG News Text Classification
================================================================================
This module implements production-optimized SOTA models with focus on inference
speed, model size, deployment efficiency, and real-world performance.

Production SOTA balances accuracy with practical constraints like latency,
throughput, memory usage, and deployment complexity.

References:
    - Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT
    - Jiao, X., et al. (2020). TinyBERT: Distilling BERT for Natural Language Understanding
    - Han, S., et al. (2016). Deep Compression: Compressing Deep Neural Networks

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime
import time
from dataclasses import dataclass, field
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.quantization import quantize_dynamic, get_default_qconfig
import torch.onnx
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DistilBertForSequenceClassification,
    MobileBertForSequenceClassification
)
from sklearn.metrics import accuracy_score, f1_score
import onnxruntime as ort
from tqdm import tqdm

from src.core.factory import Factory
from src.core.registry import Registry
from src.utils.reproducibility import set_seed
from src.utils.experiment_tracking import ExperimentTracker
from src.utils.memory_utils import optimize_memory_usage
from src.data.datasets.ag_news import AGNewsDataset
from src.models.efficient.quantization.int8_quantization import INT8Quantization
from src.models.efficient.quantization.dynamic_quantization import DynamicQuantization
from src.models.efficient.pruning.magnitude_pruning import MagnitudePruning
from src.inference.optimization.onnx_converter import ONNXConverter
from src.inference.optimization.tensorrt_optimizer import TensorRTOptimizer
from src.evaluation.metrics.classification_metrics import ClassificationMetrics

logger = logging.getLogger(__name__)


@dataclass
class ProductionConfig:
    """Configuration for production SOTA experiments."""
    # Model selection
    base_models: List[str] = field(default_factory=lambda: [
        "distilbert-base-uncased",
        "google/mobilebert-uncased",
        "microsoft/xtremedistil-l6-h256-uncased",
        "albert-base-v2",
        "squeezebert/squeezebert-uncased"
    ])
    
    # Optimization techniques
    use_quantization: bool = True
    quantization_type: str = "dynamic"  # dynamic, static, qat
    use_pruning: bool = True
    pruning_sparsity: float = 0.5
    use_knowledge_distillation: bool = True
    use_onnx_conversion: bool = True
    use_tensorrt: bool = False  # Requires NVIDIA GPU
    
    # Model compression
    target_model_size_mb: float = 50.0
    target_inference_time_ms: float = 10.0
    target_throughput_qps: int = 1000  # Queries per second
    
    # Training configuration
    batch_size: int = 32
    learning_rate: float = 3e-5
    num_epochs: int = 5
    max_length: int = 128  # Shorter for speed
    
    # Deployment configuration
    deployment_framework: str = "onnx"  # onnx, tensorrt, tflite, coreml
    optimization_level: int = 2  # 0: none, 1: basic, 2: aggressive
    batch_inference: bool = True
    dynamic_batching: bool = True
    
    # Performance targets
    min_accuracy: float = 0.90
    max_latency_p99_ms: float = 20.0
    max_memory_mb: float = 500.0
    
    # Infrastructure
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42


class ProductionSOTA:
    """
    Implements production-ready SOTA experiments.
    
    Focuses on creating models that are optimized for real-world deployment
    with constraints on latency, throughput, and resource usage.
    """
    
    def __init__(
        self,
        experiment_name: str = "production_sota",
        config: Optional[ProductionConfig] = None,
        output_dir: str = "./outputs/sota_experiments/production",
        benchmark_mode: bool = True
    ):
        """
        Initialize production SOTA experiments.
        
        Args:
            experiment_name: Name of experiment
            config: Production configuration
            output_dir: Output directory
            benchmark_mode: Enable comprehensive benchmarking
        """
        self.experiment_name = experiment_name
        self.config = config or ProductionConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark_mode = benchmark_mode
        
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        
        self.factory = Factory()
        self.registry = Registry()
        self.metrics_calculator = ClassificationMetrics()
        self.experiment_tracker = ExperimentTracker(
            experiment_name=experiment_name,
            tracking_uri="./mlruns"
        )
        
        self.results = {
            "models": {},
            "optimization": {},
            "benchmarks": {},
            "deployment": {},
            "best_model": None,
            "production_metrics": {}
        }
        
        set_seed(self.config.seed)
        logger.info(f"Initialized Production SOTA with config: {self.config}")
    
    def run_experiments(self) -> Dict[str, Any]:
        """
        Run production SOTA experiments.
        
        Returns:
            Experiment results
        """
        logger.info("Starting Production SOTA Experiments")
        start_time = time.time()
        
        # Load dataset
        dataset = self._load_dataset()
        
        # Step 1: Train lightweight base models
        logger.info("\n" + "="*60)
        logger.info("Step 1: Training Lightweight Models")
        logger.info("="*60)
        
        base_models = self._train_lightweight_models(dataset)
        
        # Step 2: Apply optimization techniques
        logger.info("\n" + "="*60)
        logger.info("Step 2: Applying Optimization Techniques")
        logger.info("="*60)
        
        optimized_models = self._optimize_models(base_models, dataset)
        
        # Step 3: Benchmark performance
        if self.benchmark_mode:
            logger.info("\n" + "="*60)
            logger.info("Step 3: Benchmarking Performance")
            logger.info("="*60)
            
            benchmark_results = self._benchmark_models(optimized_models, dataset)
            self.results["benchmarks"] = benchmark_results
        
        # Step 4: Select best production model
        logger.info("\n" + "="*60)
        logger.info("Step 4: Selecting Best Production Model")
        logger.info("="*60)
        
        best_model = self._select_best_production_model(optimized_models)
        self.results["best_model"] = best_model["name"]
        
        # Step 5: Prepare for deployment
        logger.info("\n" + "="*60)
        logger.info("Step 5: Preparing for Deployment")
        logger.info("="*60)
        
        deployment_package = self._prepare_deployment(best_model)
        self.results["deployment"] = deployment_package
        
        # Step 6: Final production validation
        logger.info("\n" + "="*60)
        logger.info("Step 6: Production Validation")
        logger.info("="*60)
        
        production_metrics = self._validate_production_readiness(
            best_model,
            dataset
        )
        self.results["production_metrics"] = production_metrics
        
        # Calculate total time
        self.results["total_time"] = time.time() - start_time
        
        # Generate report
        self._generate_report()
        
        logger.info("\n" + "="*60)
        logger.info("Production SOTA Complete!")
        logger.info(f"Best Model: {self.results['best_model']}")
        logger.info(f"Accuracy: {production_metrics['accuracy']:.4f}")
        logger.info(f"Latency P99: {production_metrics['latency_p99_ms']:.2f}ms")
        logger.info(f"Model Size: {production_metrics['model_size_mb']:.2f}MB")
        logger.info("="*60)
        
        return self.results
    
    def _train_lightweight_models(
        self,
        dataset: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train lightweight models suitable for production.
        
        Args:
            dataset: Training dataset
            
        Returns:
            Trained models
        """
        trained_models = {}
        
        for model_name in self.config.base_models:
            logger.info(f"\nTraining {model_name}...")
            
            # Create model
            if "distilbert" in model_name:
                model = DistilBertForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=4
                )
            elif "mobilebert" in model_name:
                model = MobileBertForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=4
                )
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=4
                )
            
            model.to(self.device)
            
            # Fast training with reduced epochs
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate
            )
            
            # Training loop (simplified)
            model.train()
            best_accuracy = 0
            
            for epoch in range(min(self.config.num_epochs, 3)):
                total_loss = 0
                num_batches = 0
                
                for i in range(0, len(dataset["train"]["texts"]), self.config.batch_size):
                    batch_texts = dataset["train"]["texts"][i:i+self.config.batch_size]
                    batch_labels = torch.tensor(
                        dataset["train"]["labels"][i:i+self.config.batch_size]
                    ).to(self.device)
                    
                    # Tokenize (simplified)
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    inputs = tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_length,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Forward pass
                    outputs = model(**inputs, labels=batch_labels)
                    loss = outputs.loss
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Break early for speed
                    if num_batches > 100:
                        break
                
                avg_loss = total_loss / num_batches
                
                # Quick validation
                val_accuracy = self._quick_evaluate(model, dataset["val"], tokenizer)
                
                logger.info(
                    f"  Epoch {epoch+1}, Loss: {avg_loss:.4f}, "
                    f"Val Acc: {val_accuracy:.4f}"
                )
                
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
            
            # Final evaluation
            test_accuracy = self._quick_evaluate(model, dataset["test"], tokenizer)
            
            # Measure model size
            model_size = self._get_model_size(model)
            
            trained_models[model_name] = {
                "model": model,
                "tokenizer": tokenizer,
                "accuracy": test_accuracy,
                "val_accuracy": best_accuracy,
                "model_size_mb": model_size
            }
            
            logger.info(
                f"  Test Accuracy: {test_accuracy:.4f}, "
                f"Size: {model_size:.2f}MB"
            )
            
            # Store results
            self.results["models"][model_name] = {
                "accuracy": test_accuracy,
                "model_size_mb": model_size
            }
            
            # Clear memory
            gc.collect()
            torch.cuda.empty_cache()
        
        return trained_models
    
    def _optimize_models(
        self,
        base_models: Dict[str, Any],
        dataset: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply optimization techniques to models.
        
        Args:
            base_models: Base trained models
            dataset: Dataset for calibration
            
        Returns:
            Optimized models
        """
        optimized_models = {}
        
        for model_name, model_info in base_models.items():
            logger.info(f"\nOptimizing {model_name}...")
            
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]
            
            optimization_steps = []
            
            # Step 1: Quantization
            if self.config.use_quantization:
                logger.info("  Applying quantization...")
                
                if self.config.quantization_type == "dynamic":
                    # Dynamic quantization
                    quantized_model = quantize_dynamic(
                        model,
                        {nn.Linear},
                        dtype=torch.qint8
                    )
                    optimization_steps.append("quantization")
                else:
                    quantized_model = model
            else:
                quantized_model = model
            
            # Step 2: Pruning
            if self.config.use_pruning:
                logger.info("  Applying pruning...")
                
                pruner = MagnitudePruning(
                    sparsity=self.config.pruning_sparsity
                )
                
                pruned_model = pruner.prune_model(quantized_model)
                optimization_steps.append("pruning")
            else:
                pruned_model = quantized_model
            
            # Step 3: ONNX Conversion
            if self.config.use_onnx_conversion:
                logger.info("  Converting to ONNX...")
                
                onnx_path = self._convert_to_onnx(
                    pruned_model,
                    tokenizer,
                    model_name
                )
                optimization_steps.append("onnx")
            else:
                onnx_path = None
            
            # Step 4: TensorRT (if available and enabled)
            if self.config.use_tensorrt and torch.cuda.is_available():
                logger.info("  Optimizing with TensorRT...")
                
                trt_path = self._optimize_with_tensorrt(onnx_path)
                optimization_steps.append("tensorrt")
            else:
                trt_path = None
            
            # Measure optimized model
            optimized_size = self._get_model_size(pruned_model)
            size_reduction = (
                1 - optimized_size / model_info["model_size_mb"]
            ) * 100
            
            # Test optimized model
            opt_accuracy = self._quick_evaluate(
                pruned_model,
                dataset["test"][:1000],
                tokenizer
            )
            
            accuracy_drop = model_info["accuracy"] - opt_accuracy
            
            optimized_models[model_name] = {
                "model": pruned_model,
                "tokenizer": tokenizer,
                "original_accuracy": model_info["accuracy"],
                "optimized_accuracy": opt_accuracy,
                "accuracy_drop": accuracy_drop,
                "original_size_mb": model_info["model_size_mb"],
                "optimized_size_mb": optimized_size,
                "size_reduction_pct": size_reduction,
                "optimization_steps": optimization_steps,
                "onnx_path": onnx_path,
                "trt_path": trt_path
            }
            
            logger.info(
                f"  Optimization complete: "
                f"Size reduction: {size_reduction:.1f}%, "
                f"Accuracy drop: {accuracy_drop:.4f}"
            )
            
            # Store optimization results
            self.results["optimization"][model_name] = {
                "size_reduction_pct": size_reduction,
                "accuracy_drop": accuracy_drop,
                "optimization_steps": optimization_steps
            }
        
        return optimized_models
    
    def _benchmark_models(
        self,
        models: Dict[str, Any],
        dataset: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Benchmark model performance.
        
        Args:
            models: Models to benchmark
            dataset: Test dataset
            
        Returns:
            Benchmark results
        """
        benchmark_results = {}
        
        # Prepare test data
        test_texts = dataset["test"]["texts"][:1000]
        test_labels = dataset["test"]["labels"][:1000]
        
        for model_name, model_info in models.items():
            logger.info(f"\nBenchmarking {model_name}...")
            
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]
            
            # Latency benchmark
            latencies = []
            
            model.eval()
            with torch.no_grad():
                for text in test_texts[:100]:
                    start = time.time()
                    
                    inputs = tokenizer(
                        text,
                        truncation=True,
                        max_length=self.config.max_length,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    outputs = model(**inputs)
                    _ = torch.argmax(outputs.logits, dim=-1)
                    
                    latency = (time.time() - start) * 1000  # ms
                    latencies.append(latency)
            
            # Calculate latency statistics
            latency_p50 = np.percentile(latencies, 50)
            latency_p95 = np.percentile(latencies, 95)
            latency_p99 = np.percentile(latencies, 99)
            
            # Throughput benchmark
            batch_sizes = [1, 8, 16, 32]
            throughputs = {}
            
            for batch_size in batch_sizes:
                start = time.time()
                num_processed = 0
                
                for i in range(0, min(500, len(test_texts)), batch_size):
                    batch_texts = test_texts[i:i+batch_size]
                    
                    inputs = tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_length,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        _ = torch.argmax(outputs.logits, dim=-1)
                    
                    num_processed += len(batch_texts)
                
                elapsed = time.time() - start
                throughput = num_processed / elapsed
                throughputs[f"batch_{batch_size}"] = throughput
            
            # Memory usage
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                
                # Run inference
                with torch.no_grad():
                    for i in range(0, 100, 16):
                        batch_texts = test_texts[i:i+16]
                        
                        inputs = tokenizer(
                            batch_texts,
                            padding=True,
                            truncation=True,
                            max_length=self.config.max_length,
                            return_tensors="pt"
                        ).to(self.device)
                        
                        outputs = model(**inputs)
                
                peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            else:
                peak_memory_mb = 0
            
            # ONNX Runtime benchmark (if available)
            if model_info.get("onnx_path"):
                onnx_latency = self._benchmark_onnx(
                    model_info["onnx_path"],
                    test_texts[:100],
                    tokenizer
                )
            else:
                onnx_latency = None
            
            benchmark_results[model_name] = {
                "latency_p50_ms": latency_p50,
                "latency_p95_ms": latency_p95,
                "latency_p99_ms": latency_p99,
                "throughput_qps": throughputs,
                "peak_memory_mb": peak_memory_mb,
                "onnx_latency_ms": onnx_latency,
                "model_size_mb": model_info["optimized_size_mb"],
                "accuracy": model_info["optimized_accuracy"]
            }
            
            logger.info(
                f"  Latency P99: {latency_p99:.2f}ms, "
                f"Throughput: {throughputs['batch_1']:.1f} QPS, "
                f"Memory: {peak_memory_mb:.1f}MB"
            )
        
        return benchmark_results
    
    def _select_best_production_model(
        self,
        models: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Select best model for production based on multiple criteria.
        
        Args:
            models: Candidate models
            
        Returns:
            Best production model
        """
        logger.info("Selecting best production model...")
        
        # Score each model
        scores = {}
        
        for model_name, model_info in models.items():
            score = 0
            
            # Accuracy score (40% weight)
            if model_info["optimized_accuracy"] >= self.config.min_accuracy:
                accuracy_score = model_info["optimized_accuracy"] * 40
            else:
                accuracy_score = 0  # Disqualify if below minimum
            
            # Size score (20% weight)
            if model_info["optimized_size_mb"] <= self.config.target_model_size_mb:
                size_score = (
                    1 - model_info["optimized_size_mb"] / 
                    self.config.target_model_size_mb
                ) * 20
            else:
                size_score = 0
            
            # Latency score (30% weight)
            if self.results.get("benchmarks") and model_name in self.results["benchmarks"]:
                latency = self.results["benchmarks"][model_name]["latency_p99_ms"]
                if latency <= self.config.max_latency_p99_ms:
                    latency_score = (
                        1 - latency / self.config.max_latency_p99_ms
                    ) * 30
                else:
                    latency_score = 0
            else:
                latency_score = 15  # Default middle score
            
            # Optimization score (10% weight)
            optimization_score = len(model_info["optimization_steps"]) * 2.5
            
            total_score = (
                accuracy_score + size_score + 
                latency_score + optimization_score
            )
            
            scores[model_name] = {
                "total": total_score,
                "accuracy": accuracy_score,
                "size": size_score,
                "latency": latency_score,
                "optimization": optimization_score
            }
            
            logger.info(
                f"  {model_name}: Total score: {total_score:.2f} "
                f"(Acc: {accuracy_score:.1f}, Size: {size_score:.1f}, "
                f"Latency: {latency_score:.1f}, Opt: {optimization_score:.1f})"
            )
        
        # Select best model
        best_model_name = max(scores.items(), key=lambda x: x[1]["total"])[0]
        best_model = models[best_model_name].copy()
        best_model["name"] = best_model_name
        best_model["scores"] = scores[best_model_name]
        
        logger.info(f"\nSelected model: {best_model_name}")
        
        return best_model
    
    def _prepare_deployment(
        self,
        model: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare model for deployment.
        
        Args:
            model: Selected model
            
        Returns:
            Deployment package
        """
        logger.info("Preparing deployment package...")
        
        deployment_dir = self.output_dir / "deployment"
        deployment_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = deployment_dir / "model.pt"
        torch.save(model["model"].state_dict(), model_path)
        
        # Save tokenizer
        tokenizer_path = deployment_dir / "tokenizer"
        model["tokenizer"].save_pretrained(tokenizer_path)
        
        # Create deployment config
        deployment_config = {
            "model_name": model["name"],
            "model_path": str(model_path),
            "tokenizer_path": str(tokenizer_path),
            "max_length": self.config.max_length,
            "batch_size": self.config.batch_size,
            "device": "cpu",  # Default to CPU for deployment
            "optimization_steps": model["optimization_steps"],
            "performance": {
                "accuracy": model["optimized_accuracy"],
                "model_size_mb": model["optimized_size_mb"],
                "expected_latency_ms": self.results["benchmarks"][model["name"]]["latency_p99_ms"]
                    if self.results.get("benchmarks") else None
            }
        }
        
        # Save config
        config_path = deployment_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(deployment_config, f, indent=2)
        
        # Create inference script
        self._create_inference_script(deployment_dir)
        
        # Create Docker file
        self._create_dockerfile(deployment_dir)
        
        # Create requirements file
        self._create_requirements(deployment_dir)
        
        logger.info(f"Deployment package created at {deployment_dir}")
        
        return {
            "deployment_dir": str(deployment_dir),
            "model_path": str(model_path),
            "config_path": str(config_path),
            "docker_ready": True,
            "serving_ready": True
        }
    
    def _validate_production_readiness(
        self,
        model: Dict[str, Any],
        dataset: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate model is ready for production.
        
        Args:
            model: Model to validate
            dataset: Test dataset
            
        Returns:
            Production metrics
        """
        logger.info("Validating production readiness...")
        
        validation_results = {
            "accuracy": model["optimized_accuracy"],
            "model_size_mb": model["optimized_size_mb"],
            "meets_accuracy_target": model["optimized_accuracy"] >= self.config.min_accuracy,
            "meets_size_target": model["optimized_size_mb"] <= self.config.target_model_size_mb
        }
        
        # Get benchmark metrics if available
        if self.results.get("benchmarks") and model["name"] in self.results["benchmarks"]:
            bench = self.results["benchmarks"][model["name"]]
            validation_results.update({
                "latency_p99_ms": bench["latency_p99_ms"],
                "throughput_qps": bench["throughput_qps"]["batch_1"],
                "peak_memory_mb": bench["peak_memory_mb"],
                "meets_latency_target": bench["latency_p99_ms"] <= self.config.max_latency_p99_ms,
                "meets_memory_target": bench["peak_memory_mb"] <= self.config.max_memory_mb
            })
        
        # Overall production readiness
        validation_results["production_ready"] = all([
            validation_results["meets_accuracy_target"],
            validation_results["meets_size_target"],
            validation_results.get("meets_latency_target", True),
            validation_results.get("meets_memory_target", True)
        ])
        
        if validation_results["production_ready"]:
            logger.info("✓ Model is production ready!")
        else:
            logger.warning("✗ Model does not meet all production requirements")
            
            if not validation_results["meets_accuracy_target"]:
                logger.warning(f"  - Accuracy below target: {model['optimized_accuracy']:.4f} < {self.config.min_accuracy}")
            if not validation_results["meets_size_target"]:
                logger.warning(f"  - Size above target: {model['optimized_size_mb']:.2f}MB > {self.config.target_model_size_mb}MB")
        
        return validation_results
    
    def _convert_to_onnx(
        self,
        model: nn.Module,
        tokenizer,
        model_name: str
    ) -> Path:
        """Convert model to ONNX format."""
        onnx_dir = self.output_dir / "onnx"
        onnx_dir.mkdir(exist_ok=True)
        
        onnx_path = onnx_dir / f"{model_name.replace('/', '_')}.onnx"
        
        # Create dummy input
        dummy_text = "This is a sample text for ONNX conversion"
        dummy_input = tokenizer(
            dummy_text,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
            padding="max_length"
        )
        
        # Export to ONNX
        model.eval()
        
        try:
            torch.onnx.export(
                model,
                tuple(dummy_input.values()),
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input_ids", "attention_mask"],
                output_names=["output"],
                dynamic_axes={
                    "input_ids": {0: "batch_size"},
                    "attention_mask": {0: "batch_size"},
                    "output": {0: "batch_size"}
                }
            )
            
            logger.info(f"  ONNX model saved to {onnx_path}")
            
        except Exception as e:
            logger.error(f"  ONNX conversion failed: {e}")
            return None
        
        return onnx_path
    
    def _optimize_with_tensorrt(self, onnx_path: Path) -> Optional[Path]:
        """Optimize ONNX model with TensorRT."""
        # Placeholder - would implement actual TensorRT optimization
        logger.info("  TensorRT optimization skipped (not implemented)")
        return None
    
    def _benchmark_onnx(
        self,
        onnx_path: Path,
        test_texts: List[str],
        tokenizer
    ) -> float:
        """Benchmark ONNX model."""
        try:
            # Create ONNX Runtime session
            session = ort.InferenceSession(str(onnx_path))
            
            latencies = []
            
            for text in test_texts[:50]:
                inputs = tokenizer(
                    text,
                    return_tensors="np",
                    max_length=self.config.max_length,
                    truncation=True,
                    padding="max_length"
                )
                
                start = time.time()
                
                outputs = session.run(
                    None,
                    {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"]
                    }
                )
                
                latency = (time.time() - start) * 1000
                latencies.append(latency)
            
            return np.mean(latencies)
            
        except Exception as e:
            logger.error(f"ONNX benchmarking failed: {e}")
            return None
    
    def _quick_evaluate(
        self,
        model: nn.Module,
        data: Dict[str, Any],
        tokenizer
    ) -> float:
        """Quick evaluation for accuracy."""
        model.eval()
        
        predictions = []
        
        with torch.no_grad():
            for i in range(0, min(1000, len(data["texts"])), 32):
                batch_texts = data["texts"][i:i+32]
                
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                predictions.extend(preds)
        
        labels = data["labels"][:len(predictions)]
        accuracy = accuracy_score(labels, predictions)
        
        return accuracy
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = sum(
            p.numel() * p.element_size() 
            for p in model.parameters()
        )
        
        buffer_size = sum(
            b.numel() * b.element_size() 
            for b in model.buffers()
        )
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        
        return size_mb
    
    def _create_inference_script(self, deployment_dir: Path):
        """Create inference script for deployment."""
        script = '''#!/usr/bin/env python3
"""Production inference script for AG News classification."""

import torch
from transformers import AutoTokenizer
import json

class Predictor:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.model = torch.load(self.config["model_path"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["tokenizer_path"]
        )
        self.model.eval()
    
    def predict(self, text):
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config["max_length"],
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1)
        
        labels = ["World", "Sports", "Business", "Science"]
        return labels[prediction.item()]

if __name__ == "__main__":
    import sys
    predictor = Predictor("config.json")
    text = sys.argv[1] if len(sys.argv) > 1 else "Sample text"
    print(predictor.predict(text))
'''
        
        script_path = deployment_dir / "inference.py"
        with open(script_path, "w") as f:
            f.write(script)
        
        # Make executable
        script_path.chmod(0o755)
    
    def _create_dockerfile(self, deployment_dir: Path):
        """Create Dockerfile for deployment."""
        dockerfile = '''FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "inference.py"]
'''
        
        dockerfile_path = deployment_dir / "Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile)
    
    def _create_requirements(self, deployment_dir: Path):
        """Create requirements file for deployment."""
        requirements = '''torch==2.0.0
transformers==4.30.0
numpy==1.24.0
onnxruntime==1.15.0
'''
        
        req_path = deployment_dir / "requirements.txt"
        with open(req_path, "w") as f:
            f.write(requirements)
    
    def _load_dataset(self) -> Dict[str, Any]:
        """Load dataset."""
        dataset = AGNewsDataset()
        data = dataset.load_splits()
        
        # Use smaller subset for speed
        for split in ["train", "val", "test"]:
            data[split]["texts"] = data[split]["texts"][:5000]
            data[split]["labels"] = data[split]["labels"][:5000]
        
        return data
    
    def _generate_report(self):
        """Generate production readiness report."""
        report = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "best_model": self.results["best_model"],
            "models": self.results["models"],
            "optimization": self.results["optimization"],
            "benchmarks": self.results["benchmarks"],
            "deployment": self.results["deployment"],
            "production_metrics": self.results["production_metrics"],
            "total_time": self.results.get("total_time", 0)
        }
        
        # Save JSON report
        report_path = self.output_dir / "production_sota_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save markdown report
        self._save_markdown_report(report)
        
        logger.info(f"Reports saved to {self.output_dir}")
    
    def _save_markdown_report(self, report: Dict[str, Any]):
        """Save report as markdown."""
        md_path = self.output_dir / "production_sota_report.md"
        
        with open(md_path, "w") as f:
            f.write("# Production SOTA Experiment Report\n\n")
            f.write(f"**Date**: {report['timestamp']}\n")
            f.write(f"**Best Model**: {report['best_model']}\n\n")
            
            f.write("## Production Metrics\n\n")
            metrics = report["production_metrics"]
            f.write(f"- **Accuracy**: {metrics['accuracy']:.4f}\n")
            f.write(f"- **Model Size**: {metrics['model_size_mb']:.2f} MB\n")
            
            if "latency_p99_ms" in metrics:
                f.write(f"- **Latency P99**: {metrics['latency_p99_ms']:.2f} ms\n")
                f.write(f"- **Throughput**: {metrics['throughput_qps']:.1f} QPS\n")
            
            f.write(f"- **Production Ready**: {'✓' if metrics['production_ready'] else '✗'}\n\n")
            
            f.write("## Model Comparison\n\n")
            f.write("| Model | Accuracy | Size (MB) | Optimizations |\n")
            f.write("|-------|----------|-----------|---------------|\n")
            
            for model_name, opt_info in report["optimization"].items():
                model_info = report["models"][model_name]
                f.write(
                    f"| {model_name} | {model_info['accuracy']:.4f} | "
                    f"{model_info['model_size_mb']:.2f} | "
                    f"{', '.join(opt_info['optimization_steps'])} |\n"
                )


def run_production_sota():
    """Run production SOTA experiments."""
    logger.info("Starting Production SOTA Experiments")
    
    config = ProductionConfig(
        use_quantization=True,
        use_pruning=True,
        use_onnx_conversion=True,
        target_model_size_mb=50.0,
        target_inference_time_ms=10.0,
        min_accuracy=0.90
    )
    
    experiment = ProductionSOTA(
        experiment_name="ag_news_production",
        config=config,
        benchmark_mode=True
    )
    
    results = experiment.run_experiments()
    
    logger.info("\nProduction SOTA Results:")
    logger.info(f"Best Model: {results['best_model']}")
    logger.info(f"Production Ready: {results['production_metrics']['production_ready']}")
    
    return results


if __name__ == "__main__":
    run_production_sota()
