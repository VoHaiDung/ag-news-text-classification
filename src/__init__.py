__version__ = "1.0.0"
__author__ = "Võ Hải Dũng"
__email__ = "vohaidung.work@gmail.com"
__license__ = "MIT"

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

# Version information
from src.__version__ import __version__ as version

# Core components
from src.core.exceptions import (
    AGNewsException,
    ConfigurationError,
    DataError,
    ModelError,
    TrainingError,
    InferenceError,
    APIError,
)
from src.core.factory import ModelFactory, DatasetFactory, TrainerFactory
from src.core.registry import Registry, register_model, register_trainer, register_dataset
from src.core.types import (
    ModelType,
    DatasetType,
    TrainerType,
    PredictionOutput,
    EvaluationMetrics,
    ExperimentConfig,
)

# Configuration management
from src.configs import (
    load_config,
    save_config,
    merge_configs,
    validate_config,
    get_default_config,
    ConfigLoader,
    ConfigValidator,
)

# Data components
from src.data.datasets import (
    AGNewsDataset,
    ExternalNewsDataset,
    CombinedDataset,
    PromptedDataset,
    ContrastDataset,
)
from src.data.preprocessing import (
    TextCleaner,
    Tokenizer,
    FeatureExtractor,
    SlidingWindowProcessor,
    PromptFormatter,
)
from src.data.augmentation import (
    BackTranslationAugmenter,
    ParaphraseAugmenter,
    TokenReplacementAugmenter,
    MixupAugmenter,
    CutMixAugmenter,
    AdversarialAugmenter,
    ContrastSetGenerator,
    create_augmentation_pipeline,
)
from src.data.sampling import (
    BalancedSampler,
    CurriculumSampler,
    ActiveLearningSampler,
    UncertaintySampler,
    CoresetSampler,
)
from src.data.selection import (
    InfluenceFunctionSelector,
    GradientMatchingSelector,
    DiversitySelector,
    QualityFilter,
)
from src.data.loaders import (
    create_dataloader,
    DynamicBatchingDataLoader,
    PrefetchDataLoader,
    DistributedDataLoader,
)

# Model components
from src.models.base import BaseModel, ModelWrapper, PoolingStrategy
from src.models.transformers import (
    DeBERTaV3Model,
    RoBERTaEnhancedModel,
    XLNetClassifier,
    ElectraDiscriminator,
    LongformerGlobalModel,
    GPT2Classifier,
    T5Classifier,
)
from src.models.prompt_based import (
    PromptModel,
    SoftPromptModel,
    InstructionModel,
    TemplateManager,
)
from src.models.efficient import (
    LoRAModel,
    AdapterModel,
    AdapterFusionModel,
    Int8Model,
    DynamicQuantModel,
    PrunedModel,
)
from src.models.ensemble import (
    VotingEnsemble,
    WeightedVotingEnsemble,
    StackingEnsemble,
    BlendingEnsemble,
    BayesianEnsemble,
    SnapshotEnsemble,
    create_ensemble,
)
from src.models.heads import (
    ClassificationHead,
    MultitaskHead,
    HierarchicalHead,
    AttentionHead,
    PromptHead,
)

# Training components
from src.training.trainers import (
    StandardTrainer,
    DistributedTrainer,
    ApexTrainer,
    PromptTrainer,
    InstructionTrainer,
    MultiStageTrainer,
    create_trainer,
)
from src.training.strategies import (
    CurriculumLearning,
    AdversarialTraining,
    KnowledgeDistillation,
    MetaLearning,
    PromptTuning,
    PrefixTuning,
    ContrastiveLearning,
    MultiTaskLearning,
)
from src.training.objectives import (
    FocalLoss,
    LabelSmoothingLoss,
    ContrastiveLoss,
    TripletLoss,
    CustomCrossEntropyLoss,
    InstructionLoss,
    R_DropLoss,
)
from src.training.optimization import (
    create_optimizer,
    create_scheduler,
    SAMOptimizer,
    LAMBOptimizer,
    LookaheadOptimizer,
    CosineAnnealingWarmupScheduler,
    PolynomialDecayScheduler,
)
from src.training.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoardLogger,
    WandbLogger,
    LearningRateMonitor,
    GradientAccumulator,
    MemoryMonitor,
)

# Domain adaptation
from src.domain_adaptation import (
    MLMPretrainer,
    NewsCorpusBuilder,
    AdaptivePretrainer,
    GradualUnfreezing,
    DiscriminativeLR,
    PseudoLabeler,
    SelfTrainer,
)

# Distillation
from src.distillation import (
    GPT4Teacher,
    EnsembleTeacher,
    MultiTeacher,
    DistillationDataGenerator,
    QualityFilter as DistillationQualityFilter,
    create_distillation_pipeline,
)

# Evaluation components
from src.evaluation.metrics import (
    ClassificationMetrics,
    EnsembleMetrics,
    RobustnessMetrics,
    EfficiencyMetrics,
    FairnessMetrics,
    EnvironmentalMetrics,
    ContrastConsistencyMetrics,
    compute_all_metrics,
)
from src.evaluation.analysis import (
    ErrorAnalyzer,
    ConfusionAnalyzer,
    ClassWiseAnalyzer,
    FailureCaseAnalyzer,
    DatasetShiftAnalyzer,
    BiasAnalyzer,
    ContrastSetAnalyzer,
)
from src.evaluation.interpretability import (
    AttentionAnalyzer,
    SHAPInterpreter,
    LIMEInterpreter,
    IntegratedGradients,
    LayerWiseRelevance,
    ProbingClassifier,
    PromptAnalyzer,
)
from src.evaluation.statistical import (
    SignificanceTest,
    BootstrapConfidence,
    McNemarTest,
    EffectSizeCalculator,
)
from src.evaluation.visualization import (
    plot_performance,
    plot_learning_curves,
    plot_attention_heatmap,
    visualize_embeddings,
    generate_report,
)

# Inference components
from src.inference.predictors import (
    SinglePredictor,
    BatchPredictor,
    StreamingPredictor,
    EnsemblePredictor,
    PromptPredictor,
    create_predictor,
)
from src.inference.optimization import (
    ONNXConverter,
    TensorRTOptimizer,
    QuantizationOptimizer,
    PruningOptimizer,
    optimize_for_inference,
)
from src.inference.serving import (
    ModelServer,
    BatchServer,
    LoadBalancer,
    create_model_server,
)
from src.inference.post_processing import (
    ConfidenceCalibrator,
    ThresholdOptimizer,
    OutputFormatter,
)

# API components
from src.api.rest import create_app, EndpointRouter, APIMiddleware, RequestValidator
from src.api.grpc import create_grpc_server, AGNewsServicer
from src.api.graphql import create_graphql_schema, QueryResolver, MutationResolver

# Service layer
from src.services import (
    PredictionService,
    TrainingService,
    DataService,
    ModelManagementService,
    ExperimentService,
)

# Utilities
from src.utils import (
    set_seed,
    get_device,
    count_parameters,
    log_metrics,
    save_checkpoint,
    load_checkpoint,
    setup_logging,
    setup_distributed,
    cleanup_distributed,
    profile_memory,
    profile_time,
    track_experiment,
    generate_uid,
    validate_inputs,
    sanitize_text,
    batch_generator,
    parallel_process,
    cache_result,
    retry_on_failure,
    deprecated,
)

# Environment setup
def setup_environment(
    config_path: Optional[str] = None,
    log_level: str = "INFO",
    device: Optional[str] = None,
    seed: Optional[int] = 42,
    distributed: bool = False,
) -> Dict[str, Any]:
    """Setup the environment for AG News framework."""
    env_config = {}
    
    # Setup logging
    setup_logging(level=log_level)
    
    # Set random seed for reproducibility
    if seed is not None:
        set_seed(seed)
        env_config["seed"] = seed
    
    # Setup device
    if device is None:
        device = get_device()
    env_config["device"] = device
    
    # Setup distributed training if needed
    if distributed:
        setup_distributed()
        env_config["distributed"] = True
    
    # Load configuration
    if config_path:
        config = load_config(config_path)
        env_config["config"] = config
    
    return env_config

# Quick start functions
def quick_train(
    model_name: str = "deberta-v3-xlarge",
    dataset_name: str = "ag_news",
    num_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    output_dir: str = "./outputs",
    **kwargs
) -> Dict[str, Any]:
    """Quick training function for getting started."""
    from src.pipelines import TrainingPipeline
    
    pipeline = TrainingPipeline(
        model_name=model_name,
        dataset_name=dataset_name,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=output_dir,
        **kwargs
    )
    
    return pipeline.run()

def quick_evaluate(
    model_path: str,
    dataset_name: str = "ag_news",
    split: str = "test",
    batch_size: int = 64,
    **kwargs
) -> Dict[str, float]:
    """Quick evaluation function."""
    from src.pipelines import EvaluationPipeline
    
    pipeline = EvaluationPipeline(
        model_path=model_path,
        dataset_name=dataset_name,
        split=split,
        batch_size=batch_size,
        **kwargs
    )
    
    return pipeline.run()

def quick_predict(
    text: Union[str, List[str]],
    model_path: str,
    return_probabilities: bool = False,
    **kwargs
) -> Union[str, List[str], Dict[str, Any]]:
    """Quick prediction function."""
    from src.pipelines import InferencePipeline
    
    pipeline = InferencePipeline(
        model_path=model_path,
        **kwargs
    )
    
    return pipeline.predict(
        text=text,
        return_probabilities=return_probabilities
    )

# Registry initialization
model_registry = Registry("models")
trainer_registry = Registry("trainers")
dataset_registry = Registry("datasets")
augmenter_registry = Registry("augmenters")
metric_registry = Registry("metrics")
callback_registry = Registry("callbacks")

# Register default components
def _register_defaults():
    """Register default components in registries."""
    # Models
    model_registry.register("deberta-v3", DeBERTaV3Model)
    model_registry.register("roberta", RoBERTaEnhancedModel)
    model_registry.register("xlnet", XLNetClassifier)
    model_registry.register("electra", ElectraDiscriminator)
    model_registry.register("longformer", LongformerGlobalModel)
    model_registry.register("gpt2", GPT2Classifier)
    model_registry.register("t5", T5Classifier)
    model_registry.register("lora", LoRAModel)
    model_registry.register("voting-ensemble", VotingEnsemble)
    model_registry.register("stacking-ensemble", StackingEnsemble)
    
    # Trainers
    trainer_registry.register("standard", StandardTrainer)
    trainer_registry.register("distributed", DistributedTrainer)
    trainer_registry.register("prompt", PromptTrainer)
    trainer_registry.register("instruction", InstructionTrainer)
    trainer_registry.register("multistage", MultiStageTrainer)
    
    # Datasets
    dataset_registry.register("ag_news", AGNewsDataset)
    dataset_registry.register("external_news", ExternalNewsDataset)
    dataset_registry.register("combined", CombinedDataset)
    dataset_registry.register("prompted", PromptedDataset)
    dataset_registry.register("contrast", ContrastDataset)
    
    # Augmenters
    augmenter_registry.register("backtranslation", BackTranslationAugmenter)
    augmenter_registry.register("paraphrase", ParaphraseAugmenter)
    augmenter_registry.register("mixup", MixupAugmenter)
    augmenter_registry.register("adversarial", AdversarialAugmenter)
    
    # Metrics
    metric_registry.register("accuracy", ClassificationMetrics)
    metric_registry.register("robustness", RobustnessMetrics)
    metric_registry.register("efficiency", EfficiencyMetrics)
    metric_registry.register("fairness", FairnessMetrics)
    
    # Callbacks
    callback_registry.register("early_stopping", EarlyStopping)
    callback_registry.register("checkpoint", ModelCheckpoint)
    callback_registry.register("wandb", WandbLogger)
    callback_registry.register("tensorboard", TensorBoardLogger)

# Initialize defaults on import
_register_defaults()

# Plugin system
class PluginManager:
    """Manage plugins and extensions."""
    
    def __init__(self):
        self.plugins = {}
        self.hooks = {}
    
    def register_plugin(self, name: str, plugin: Any) -> None:
        """Register a plugin."""
        self.plugins[name] = plugin
    
    def register_hook(self, event: str, callback: callable) -> None:
        """Register a hook for an event."""
        if event not in self.hooks:
            self.hooks[event] = []
        self.hooks[event].append(callback)
    
    def trigger_hook(self, event: str, *args, **kwargs) -> List[Any]:
        """Trigger all hooks for an event."""
        results = []
        if event in self.hooks:
            for callback in self.hooks[event]:
                results.append(callback(*args, **kwargs))
        return results
    
    def get_plugin(self, name: str) -> Any:
        """Get a registered plugin."""
        return self.plugins.get(name)

plugin_manager = PluginManager()

# Environment variables setup
def _setup_env_vars():
    """Setup environment variables from .env file."""
    from dotenv import load_dotenv
    
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)

_setup_env_vars()

# Package metadata
__all__ = [
    # Version
    "__version__",
    "version",
    
    # Core
    "ModelFactory",
    "DatasetFactory", 
    "TrainerFactory",
    "Registry",
    "register_model",
    "register_trainer",
    "register_dataset",
    
    # Models
    "DeBERTaV3Model",
    "RoBERTaEnhancedModel",
    "XLNetClassifier",
    "ElectraDiscriminator",
    "LongformerGlobalModel",
    "VotingEnsemble",
    "StackingEnsemble",
    "LoRAModel",
    "PromptModel",
    "create_ensemble",
    
    # Training
    "StandardTrainer",
    "DistributedTrainer",
    "PromptTrainer",
    "create_trainer",
    "CurriculumLearning",
    "AdversarialTraining",
    "KnowledgeDistillation",
    
    # Data
    "AGNewsDataset",
    "TextCleaner",
    "BackTranslationAugmenter",
    "create_dataloader",
    "create_augmentation_pipeline",
    
    # Evaluation
    "compute_all_metrics",
    "ErrorAnalyzer",
    "AttentionAnalyzer",
    "plot_performance",
    "generate_report",
    
    # Inference
    "SinglePredictor",
    "BatchPredictor",
    "create_predictor",
    "optimize_for_inference",
    "create_model_server",
    
    # Quick start
    "quick_train",
    "quick_evaluate",
    "quick_predict",
    "setup_environment",
    
    # Utilities
    "set_seed",
    "get_device",
    "setup_logging",
    "track_experiment",
    
    # Registries
    "model_registry",
    "trainer_registry",
    "dataset_registry",
    "augmenter_registry",
    "metric_registry",
    "callback_registry",
    
    # Plugin system
    "plugin_manager",
]

# Package info
def get_package_info() -> Dict[str, str]:
    """Get package information."""
    return {
        "name": "ag-news-text-classification",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "python_requires": ">=3.8,<3.12",
        "pytorch_version": ">=2.0.0",
        "transformers_version": ">=4.35.0",
    }

# Check dependencies on import
def _check_dependencies():
    """Check if required dependencies are installed."""
    required = {
        "torch": "2.0.0",
        "transformers": "4.35.0",
        "numpy": "1.24.0",
        "pandas": "2.0.0",
    }
    
    missing = []
    for package, min_version in required.items():
        try:
            __import__(package)
        except ImportError:
            missing.append(f"{package}>={min_version}")
    
    if missing:
        warnings.warn(
            f"Missing required dependencies: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}",
            ImportWarning
        )

_check_dependencies()

# Lazy imports for heavy modules
def __getattr__(name: str) -> Any:
    """Lazy import heavy modules."""
    lazy_modules = {
        "experiments": "src.experiments",
        "research": "src.research",
        "benchmarks": "src.benchmarks",
        "monitoring": "src.monitoring",
        "deployment": "src.deployment",
    }
    
    if name in lazy_modules:
        import importlib
        return importlib.import_module(lazy_modules[name])
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Initialize package
def initialize(config: Optional[Dict[str, Any]] = None) -> None:
    """Initialize the AG News framework with custom configuration."""
    if config:
        # Apply custom configuration
        for key, value in config.items():
            if key == "plugins":
                for plugin_name, plugin in value.items():
                    plugin_manager.register_plugin(plugin_name, plugin)
            elif key == "hooks":
                for event, callbacks in value.items():
                    for callback in callbacks:
                        plugin_manager.register_hook(event, callback)

# Clean up on exit
import atexit

def _cleanup():
    """Cleanup resources on exit."""
    try:
        cleanup_distributed()
    except:
        pass

atexit.register(_cleanup)
