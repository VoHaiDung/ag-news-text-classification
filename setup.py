import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any

from setuptools import setup, find_packages, Command
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.test import test as TestCommand


# Project root directory
ROOT_DIR = Path(__file__).parent.resolve()
SRC_DIR = ROOT_DIR / "src"

# Read version
VERSION_FILE = SRC_DIR / "__version__.py"
VERSION_DICT = {}
if VERSION_FILE.exists():
    with open(VERSION_FILE, "r", encoding="utf-8") as f:
        exec(f.read(), VERSION_DICT)
    VERSION = VERSION_DICT.get("__version__", "0.1.0")
else:
    VERSION = "0.1.0"

# Read long description
with open(ROOT_DIR / "README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()


def read_requirements(filename: str) -> List[str]:
    """Read requirements from file, handling various formats."""
    requirements = []
    filepath = ROOT_DIR / filename
    
    if not filepath.exists():
        print(f"Warning: {filename} not found")
        return requirements
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            # Handle git+ requirements
            if "git+" in line:
                if "#egg=" in line:
                    pkg_name = line.split("#egg=")[-1]
                    requirements.append(f"{pkg_name} @ {line}")
                else:
                    # Extract package name from git URL
                    pkg_name = line.split("/")[-1].replace(".git", "")
                    requirements.append(f"{pkg_name} @ {line}")
            else:
                requirements.append(line)
    
    return requirements


# Core requirements (minimal for basic functionality)
INSTALL_REQUIRES = [
    "torch>=2.0.0",
    "transformers>=4.35.0",
    "datasets>=2.14.0",
    "tokenizers>=0.15.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "tqdm>=4.66.0",
    "pyyaml>=6.0.0",
    "python-dotenv>=1.0.0",
    "click>=8.1.0",
    "rich>=13.5.0",
    "einops>=0.7.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]

# Development requirements
DEV_REQUIRES = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
    "pytest-benchmark>=4.0.0",
    "pytest-mock>=3.12.0",
    "pytest-xdist>=3.5.0",
    "hypothesis>=6.90.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.5.0",
    "pylint>=3.0.0",
    "pre-commit>=3.4.0",
    "ipdb>=0.13.13",
    "jupyterlab>=4.0.0",
    "notebook>=7.0.0",
    "nbformat>=5.9.0",
    "nbconvert>=7.12.0",
]

# All requirements from requirements.txt
try:
    ALL_REQUIRES = read_requirements("requirements.txt")
except Exception as e:
    print(f"Warning: Could not read requirements.txt: {e}")
    ALL_REQUIRES = []


# Custom command for MLOps setup
class SetupMLOps(Command):
    """Setup MLOps environment with monitoring and deployment tools."""
    
    description = "Setup MLOps environment"
    user_options = []
    
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass
    
    def run(self):
        """Setup MLOps tools."""
        print("Setting up MLOps environment...")
        
        # Setup monitoring
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                              "prometheus-client", "grafana-api", "evidently", "whylogs"])
        
        # Setup deployment tools
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                              "bentoml", "mlflow", "kubeflow", "seldon-core"])
        
        # Setup Docker
        if sys.platform != "win32":
            subprocess.call(["docker", "--version"])
            subprocess.call(["docker-compose", "--version"])
        
        # Setup Kubernetes tools
        subprocess.call(["kubectl", "version", "--client"])
        
        print("MLOps environment setup complete!")


# Custom command for research setup
class SetupResearch(Command):
    """Setup research environment with all tools."""
    
    description = "Setup complete research environment"
    user_options = []
    
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass
    
    def run(self):
        """Setup research environment."""
        print("Setting up research environment...")
        
        # Install spaCy models
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_lg"])
        
        # Download NLTK data
        import nltk
        nltk_downloads = [
            'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
            'maxent_ne_chunker', 'words', 'vader_lexicon', 'reuters',
            'brown', 'conll2000', 'movie_reviews'
        ]
        for item in nltk_downloads:
            nltk.download(item)
        
        # Setup Jupyter extensions
        subprocess.check_call([sys.executable, "-m", "jupyter", "nbextension", "enable", "--py", "widgetsnbextension"])
        subprocess.check_call([sys.executable, "-m", "jupyter", "labextension", "install", "@jupyter-widgets/jupyterlab-manager"])
        
        # Setup experiment tracking
        print("Configuring experiment tracking...")
        subprocess.check_call([sys.executable, "-m", "wandb", "login", "--relogin"])
        
        print("Research environment setup complete!")


# Custom command for production setup
class SetupProduction(Command):
    """Setup production environment."""
    
    description = "Setup production environment"
    user_options = []
    
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass
    
    def run(self):
        """Setup production environment."""
        print("Setting up production environment...")
        
        # Install production dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                              "gunicorn", "nginx", "supervisor", "redis", "celery"])
        
        # Setup monitoring
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                              "prometheus-client", "grafana-api", "sentry-sdk"])
        
        # Setup security
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                              "python-jose", "passlib", "bcrypt"])
        
        print("Production environment setup complete!")


# Custom pytest command
class PyTest(TestCommand):
    """Custom pytest command."""
    
    user_options = [("pytest-args=", "a", "Arguments to pass to pytest")]
    
    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []
    
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True
    
    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


# Entry points for CLI commands
CONSOLE_SCRIPTS = [
    # Main CLI
    "ag-news=src.cli.main:main",
    
    # Data commands
    "ag-news-data=src.cli.data:main",
    "ag-news-prepare=scripts.data_preparation.prepare_ag_news:main",
    "ag-news-prepare-external=scripts.data_preparation.prepare_external_data:main",
    "ag-news-augment=scripts.data_preparation.create_augmented_data:main",
    "ag-news-splits=scripts.data_preparation.create_data_splits:main",
    "ag-news-contrast=scripts.data_preparation.generate_contrast_sets:main",
    "ag-news-pseudo=scripts.data_preparation.generate_pseudo_labels:main",
    "ag-news-quality=scripts.data_preparation.select_quality_data:main",
    "ag-news-instruction-data=scripts.data_preparation.prepare_instruction_data:main",
    
    # Training commands
    "ag-news-train=scripts.training.train_single_model:main",
    "ag-news-train-all=scripts.training.train_all_models:main",
    "ag-news-train-ensemble=scripts.training.train_ensemble:main",
    "ag-news-train-distributed=scripts.training.distributed_training:main",
    "ag-news-resume=scripts.training.resume_training:main",
    "ag-news-prompt=scripts.training.train_with_prompts:main",
    "ag-news-instruction=scripts.training.instruction_tuning:main",
    "ag-news-multistage=scripts.training.multi_stage_training:main",
    "ag-news-distill=scripts.training.distill_from_gpt4:main",
    
    # Domain Adaptation
    "ag-news-pretrain=scripts.domain_adaptation.pretrain_on_news:main",
    "ag-news-download-news=scripts.domain_adaptation.download_news_corpus:main",
    "ag-news-domain-adapt=scripts.domain_adaptation.run_dapt:main",
    
    # Evaluation commands
    "ag-news-evaluate=scripts.evaluation.evaluate_all_models:main",
    "ag-news-benchmark=scripts.evaluation.create_leaderboard:main",
    "ag-news-analyze=scripts.evaluation.statistical_analysis:main",
    "ag-news-report=scripts.evaluation.generate_reports:main",
    "ag-news-contrast-eval=scripts.evaluation.evaluate_contrast_sets:main",
    
    # Analysis commands
    "ag-news-error-analysis=src.cli.error_analysis:main",
    "ag-news-interpretability=src.cli.interpretability:main",
    "ag-news-attention=src.cli.attention_analysis:main",
    "ag-news-embeddings=src.cli.embedding_analysis:main",
    
    # Optimization commands
    "ag-news-optimize=scripts.optimization.hyperparameter_search:main",
    "ag-news-nas=scripts.optimization.architecture_search:main",
    "ag-news-ensemble-opt=scripts.optimization.ensemble_optimization:main",
    "ag-news-prompt-opt=scripts.optimization.prompt_optimization:main",
    
    # Deployment commands
    "ag-news-export=scripts.deployment.export_models:main",
    "ag-news-serve=scripts.deployment.optimize_for_inference:main",
    "ag-news-deploy=scripts.deployment.deploy_to_cloud:main",
    "ag-news-docker=scripts.deployment.create_docker_image:main",
    
    # API and App commands
    "ag-news-api=src.api.rest.main:main",
    "ag-news-grpc=src.api.grpc.services:main",
    "ag-news-graphql=src.api.graphql.server:main",
    "ag-news-streamlit=app.streamlit_app:main",
    
    # Utility commands
    "ag-news-setup=scripts.setup.verify_installation:main",
    "ag-news-download=scripts.setup.download_all_data:main",
    "ag-news-monitor=src.cli.monitor:main",
    "ag-news-profile=src.cli.profile:main",
    
    # Tools
    "ag-news-debug=tools.debugging.model_debugger:main",
    "ag-news-validate=tools.debugging.data_validator:main",
    "ag-news-visualize=tools.visualization.training_monitor:main",
    "ag-news-memory-profile=tools.profiling.memory_profiler:main",
    "ag-news-speed-profile=tools.profiling.speed_profiler:main",
    
    # Research tools
    "ag-news-experiment=experiments.experiment_runner:main",
    "ag-news-ablation=experiments.ablation_studies.component_ablation:main",
    "ag-news-leaderboard=experiments.results.leaderboard_generator:main",
    
    # Monitoring and security
    "ag-news-metrics=monitoring.metrics.metric_collectors:main",
    "ag-news-alerts=monitoring.alerts.alert_manager:main",
    "ag-news-audit=security.audit_logs.audit_logger:main",
    "ag-news-security-scan=security.model_security.security_scanner:main",
]


# Package data
PACKAGE_DATA = {
    "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md", "*.toml", "*.cfg", "*.ini"],
    "src": ["**/*.yaml", "**/*.yml", "**/*.json", "**/*.txt", "**/*.pyi"],
    "configs": ["**/*.yaml", "**/*.yml", "**/*.json", "**/*.toml"],
    "prompts": ["**/*.txt", "**/*.json", "**/*.md", "**/*.jinja2"],
    "data": ["**/.gitkeep", "**/*.txt"],
    "notebooks": ["**/*.ipynb", "**/*.py"],
    "deployment": ["**/*.yaml", "**/*.yml", "**/*.sh", "**/*Dockerfile*", "**/*.env"],
    "app": ["**/*.py", "**/*.css", "**/*.js", "**/*.png", "**/*.jpg", "**/*.html", "**/*.svg"],
    "docs": ["**/*.md", "**/*.rst", "**/*.png", "**/*.jpg", "**/*.svg", "**/*.puml", "**/*.css"],
    "benchmarks": ["**/*.json", "**/*.yaml", "**/*.csv"],
    "templates": ["**/*.py", "**/*.yaml", "**/*.md", "**/*.jinja2"],
    "tests": ["**/*.py", "**/*.json", "**/*.yaml", "**/*.txt"],
    "tools": ["**/*.py", "**/*.sh"],
    "monitoring": ["**/*.yaml", "**/*.json", "**/*.py"],
    "security": ["**/*.py", "**/*.yaml", "**/*.pem", "**/*.key"],
    "experiments": ["**/*.yaml", "**/*.json", "**/*.py"],
    "research": ["**/*.bib", "**/*.md", "**/*.txt"],
    "plugins": ["**/*.py", "**/*.yaml"],
    "migrations": ["**/*.py", "**/*.sql"],
    "cache": ["**/*.yaml", "**/*.py"],
    "load_testing": ["**/*.yaml", "**/*.py", "**/*.js"],
    "backup": ["**/*.yaml", "**/*.sh", "**/*.md"],
    "quality": ["**/*.md", "**/*.yaml"],
    "quickstart": ["**/*.py", "**/*.ipynb", "**/*.md", "**/Dockerfile*"],
    ".github": ["**/*.yml", "**/*.md"],
    "ci": ["**/*.sh", "**/*.py", "**/*.yml"],
}

# Data files
data_files = [
    # Configuration files
    ('config', ['configs/**/*.yaml', 'configs/**/*.yml', 'configs/**/*.json']),
    # Prompts
    ('prompts', ['prompts/**/*.txt', 'prompts/**/*.json', 'prompts/**/*.md']),
    # Documentation
    ('docs', ['docs/**/*.md', 'docs/**/*.rst', 'docs/**/*.png']),
    # Notebooks
    ('notebooks', ['notebooks/**/*.ipynb']),
    # Deployment configurations
    ('deployment', ['deployment/**/*.yaml', 'deployment/**/*.yml', 'deployment/**/Dockerfile*']),
    # Benchmarks
    ('benchmarks', ['benchmarks/**/*.json']),
    # Scripts
    ('scripts', ['scripts/**/*.sh', 'scripts/**/*.py']),
]

# Main setup configuration
setup(
    # Metadata
    name="ag-news-text-classification",
    version=VERSION,
    author="Your Name",
    author_email="your.email@example.com",
    maintainer="AG News Research Team",
    maintainer_email="team@agnews-research.ai",
    
    # Description
    description="A comprehensive state-of-the-art research framework for news text classification with advanced ML/DL techniques",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    
    # URLs
    url="https://github.com/yourusername/ag-news-text-classification",
    download_url=f"https://github.com/yourusername/ag-news-text-classification/archive/v{VERSION}.tar.gz",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/ag-news-text-classification/issues",
        "Documentation": "https://ag-news-text-classification.readthedocs.io",
        "Source Code": "https://github.com/yourusername/ag-news-text-classification",
        "Research Paper": "https://arxiv.org/abs/your-paper-id",
        "Model Hub": "https://huggingface.co/your-org/ag-news-models",
        "Demo": "https://huggingface.co/spaces/your-org/ag-news-demo",
        "Changelog": "https://github.com/yourusername/ag-news-text-classification/blob/main/CHANGELOG.md",
        "Benchmarks": "https://ag-news-benchmarks.github.io",
        "CI/CD": "https://github.com/yourusername/ag-news-text-classification/actions",
    },
    
    # License
    license="MIT",
    license_files=["LICENSE"],
    
    # Packages
    packages=find_packages(where=".", include=[
        "src*", "scripts*", "experiments*", "configs*",
        "app*", "notebooks*", "prompts*", "docs*",
        "deployment*", "benchmarks*", "tests*", "tools*",
        "research*", "monitoring*", "security*", "plugins*",
        "migrations*", "cache*", "load_testing*", "backup*",
        "quality*", "quickstart*", "templates*"
    ]),
    package_dir={"": "."},
    py_modules=["quickstart.minimal_example"],
    include_package_data=True,
    package_data=PACKAGE_DATA,
    data_files=data_files,
    
    # Python version
    python_requires=">=3.8,<3.12",
    
    # Dependencies
    install_requires=INSTALL_REQUIRES,
    
    # Optional dependencies
    extras_require={
        # Development environment
        "dev": DEV_REQUIRES,
        
        # Complete research environment
        "research": ALL_REQUIRES,
        
        # Documentation
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-autodoc-typehints>=1.25.0",
            "myst-parser>=2.0.0",
            "sphinx-copybutton>=0.5.2",
            "nbsphinx>=0.9.3",
            "sphinxcontrib-napoleon>=0.7",
            "sphinxcontrib-bibtex>=2.5.0",
            "sphinx-panels>=0.6.0",
            "sphinx-tabs>=3.4.0",
        ],
        
        # Testing
        "test": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "pytest-benchmark>=4.0.0",
            "pytest-xdist>=3.5.0",
            "pytest-timeout>=2.2.0",
            "pytest-mock>=3.12.0",
            "pytest-env>=1.1.3",
            "hypothesis>=6.90.0",
            "locust>=2.17.0",
            "schemathesis>=3.20.0",
            "faker>=20.1.0",
            "factory-boy>=3.3.0",
        ],
        
        # Model optimization
        "optimization": [
            "optuna>=3.3.0",
            "ray[tune]>=2.8.0",
            "hyperopt>=0.2.7",
            "onnx>=1.15.0",
            "onnxruntime-gpu>=1.16.0",
            "tensorrt>=8.6.0",
            "openvino-dev>=2023.2.0",
            "neural-compressor>=2.3.0",
        ],
        
        # Efficient training
        "efficient": [
            "peft>=0.6.0",
            "bitsandbytes>=0.41.0",
            "deepspeed>=0.12.0",
            "fairscale>=0.4.13",
            "apex @ git+https://github.com/NVIDIA/apex",
            "flash-attn>=2.4.0",
            "xformers>=0.0.23",
            "triton>=2.1.0",
        ],
        
        # NLP specific
        "nlp": [
            "spacy>=3.6.0",
            "nltk>=3.8.0",
            "gensim>=4.3.0",
            "textblob>=0.17.0",
            "sentence-transformers>=2.2.0",
            "bertopic>=0.16.0",
            "newspaper3k>=0.2.8",
            "newsapi-python>=0.2.7",
            "lexnlp>=2.3.0",
            "textstat>=0.7.3",
        ],
        
        # Data processing
        "data": [
            "dask>=2023.12.0",
            "pyarrow>=12.0.0",
            "great-expectations>=0.18.0",
            "ydata-profiling>=4.6.0",
            "dvc>=3.0.0",
            "dvc-s3>=2.23.0",
            "nlpaug>=1.1.0",
            "textaugment>=1.3.4",
            "augly>=1.0.0",
        ],
        
        # Model interpretability
        "interpretability": [
            "shap>=0.44.0",
            "lime>=0.2.0.1",
            "captum>=0.6.0",
            "eli5>=0.13.0",
            "interpret>=0.5.0",
            "alibi>=0.9.4",
            "anchor-exp>=0.0.2.0",
        ],
        
        # Fairness & bias
        "fairness": [
            "fairlearn>=0.10.0",
            "aif360>=0.5.0",
            "evidently>=0.4.0",
            "fairness-indicators>=0.1.0",
        ],
        
        # LLM integration
        "llm": [
            "openai>=1.3.0",
            "anthropic>=0.7.0",
            "langchain>=0.0.340",
            "guidance>=0.1.0",
            "trl>=0.7.0",
            "promptsource>=0.2.0",
            "llama-index>=0.9.0",
            "chromadb>=0.4.0",
        ],
        
        # API & serving
        "api": [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "streamlit>=1.28.0",
            "gradio>=4.0.0",
            "bentoml>=1.1.0",
            "grpcio>=1.60.0",
            "grpcio-tools>=1.60.0",
            "strawberry-graphql>=0.209.0",
            "flask>=3.0.0",
            "django>=4.2.0",
        ],
        
        # Cloud deployment
        "cloud": [
            "boto3>=1.28.0",
            "google-cloud-storage>=2.10.0",
            "azure-storage-blob>=12.19.0",
            "sagemaker>=2.199.0",
            "google-cloud-aiplatform>=1.38.0",
            "kubeflow>=1.7.0",
        ],
        
        # Monitoring
        "monitoring": [
            "prometheus-client>=0.19.0",
            "grafana-api>=1.0.0",
            "evidently>=0.4.0",
            "whylogs>=1.3.0",
            "codecarbon>=2.3.0",
            "tensorboard>=2.13.0",
            "mlflow>=2.8.0",
            "neptune>=1.8.0",
        ],
        
        # Adversarial
        "adversarial": [
            "textattack>=0.3.0",
            "foolbox>=3.3.0",
            "checklist>=0.0.11",
            "adversarial-robustness-toolbox>=1.15.0",
        ],
        
        # Research tools
        "research-tools": [
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
            "papermill>=2.5.0",
            "bibtexparser>=1.4.0",
            "wandb>=0.15.0",
            "wandb-callbacks>=0.2.0",
            "mlflow>=2.8.0",
            "mlflow-skinny>=2.9.0",
            "sacred>=0.8.4",
            "comet-ml>=3.35.3",
        ],
        
        # Database
        "database": [
            "sqlalchemy>=2.0.0",
            "alembic>=1.12.0",
            "redis>=5.0.0",
            "pymongo>=4.5.0",
            "elasticsearch>=8.11.0",
            "pinecone-client>=2.2.0",
            "weaviate-client>=3.25.0",
        ],
        
        # Platform specific
        "colab": [
            "google-colab",
            "ipywidgets>=8.1.0",
            "gdown>=4.7.0",
        ],
        "kaggle": [
            "kaggle>=1.5.0",
        ],
        "sagemaker": [
            "sagemaker>=2.199.0",
            "sagemaker-experiments>=0.1.0",
        ],
        
        # All extras combined
        "all": ALL_REQUIRES + DEV_REQUIRES,
        
        # Quick start (minimal)
        "minimal": [
            "torch>=2.0.0",
            "transformers>=4.35.0",
            "datasets>=2.14.0",
            "scikit-learn>=1.3.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "matplotlib>=3.7.0",
            "tqdm>=4.66.0",
        ],
        
        # Week-specific bundles
        "week1": [
            "scikit-learn>=1.3.0",
            "xgboost>=2.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "jupyter>=1.0.0",
            "wandb>=0.15.0",
        ],
        "week2-3": [
            "optuna>=3.3.0",
            "ray[tune]>=2.8.0",
            "wandb>=0.15.0",
            "pytorch-lightning>=2.0.0",
        ],
        "week4-5": [
            "nlpaug>=1.1.0",
            "textaugment>=1.3.4",
            "dvc>=3.0.0",
            "newspaper3k>=0.2.8",
        ],
        "week6-7": [
            "openai>=1.3.0",
            "peft>=0.6.0",
            "trl>=0.7.0",
            "langchain>=0.0.340",
        ],
        "week8-9": [
            "onnx>=1.15.0",
            "tensorrt>=8.6.0",
            "fastapi>=0.104.0",
            "bentoml>=1.1.0",
        ],
        "week10": [
            "docker>=6.1.0",
            "kubernetes>=28.1.0",
            "bentoml>=1.1.0",
            "prometheus-client>=0.19.0",
        ],
    },
    
    # Entry points
    entry_points={
        "console_scripts": CONSOLE_SCRIPTS,
        
        # Plugin system for extensibility
        "ag_news.models": [
            # Transformer models
            "deberta = src.models.transformers.deberta.deberta_v3:DeBERTaV3Model",
            "deberta_sliding = src.models.transformers.deberta.deberta_sliding:DeBERTaSlidingModel",
            "deberta_hierarchical = src.models.transformers.deberta.deberta_hierarchical:DeBERTaHierarchicalModel",
            "roberta = src.models.transformers.roberta.roberta_enhanced:RoBERTaEnhancedModel",
            "roberta_domain = src.models.transformers.roberta.roberta_domain:RoBERTaDomainModel",
            "xlnet = src.models.transformers.xlnet.xlnet_classifier:XLNetClassifier",
            "electra = src.models.transformers.electra.electra_discriminator:ElectraDiscriminator",
            "longformer = src.models.transformers.longformer.longformer_global:LongformerGlobalModel",
            "gpt2 = src.models.transformers.generative.gpt2_classifier:GPT2Classifier",
            "t5 = src.models.transformers.generative.t5_classifier:T5Classifier",
        ],
        
        "ag_news.prompt_models": [
            "prompt_model = src.models.prompt_based.prompt_model:PromptModel",
            "soft_prompt = src.models.prompt_based.soft_prompt:SoftPromptModel",
            "instruction = src.models.prompt_based.instruction_model:InstructionModel",
        ],
        
        "ag_news.efficient_models": [
            "lora = src.models.efficient.lora.lora_model:LoRAModel",
            "qlora = src.models.efficient.lora.qlora_model:QLoRAModel",
            "adapter = src.models.efficient.adapters.adapter_model:AdapterModel",
            "adapter_fusion = src.models.efficient.adapters.adapter_fusion:AdapterFusionModel",
            "int8 = src.models.efficient.quantization.int8_quantization:Int8Model",
            "dynamic_quant = src.models.efficient.quantization.dynamic_quantization:DynamicQuantModel",
            "pruned = src.models.efficient.pruning.magnitude_pruning:PrunedModel",
        ],
        
        "ag_news.ensemble_models": [
            "soft_voting = src.models.ensemble.voting.soft_voting:SoftVotingEnsemble",
            "weighted_voting = src.models.ensemble.voting.weighted_voting:WeightedVotingEnsemble",
            "rank_averaging = src.models.ensemble.voting.rank_averaging:RankAveragingEnsemble",
            "stacking = src.models.ensemble.stacking.stacking_classifier:StackingEnsemble",
            "blending = src.models.ensemble.blending.blending_ensemble:BlendingEnsemble",
            "bayesian = src.models.ensemble.advanced.bayesian_ensemble:BayesianEnsemble",
            "snapshot = src.models.ensemble.advanced.snapshot_ensemble:SnapshotEnsemble",
        ],
        
        "ag_news.trainers": [
            "standard = src.training.trainers.standard_trainer:StandardTrainer",
            "distributed = src.training.trainers.distributed_trainer:DistributedTrainer",
            "apex = src.training.trainers.apex_trainer:ApexTrainer",
            "prompt = src.training.trainers.prompt_trainer:PromptTrainer",
            "instruction = src.training.trainers.instruction_trainer:InstructionTrainer",
            "multistage = src.training.trainers.multi_stage_trainer:MultiStageTrainer",
        ],
        
        "ag_news.augmenters": [
            "backtranslation = src.data.augmentation.back_translation:BackTranslationAugmenter",
            "paraphrase = src.data.augmentation.paraphrase:ParaphraseAugmenter",
            "token_replacement = src.data.augmentation.token_replacement:TokenReplacementAugmenter",
            "mixup = src.data.augmentation.mixup:MixupAugmenter",
            "cutmix = src.data.augmentation.cutmix:CutMixAugmenter",
            "adversarial = src.data.augmentation.adversarial:AdversarialAugmenter",
            "contrast_set = src.data.augmentation.contrast_set_generator:ContrastSetGenerator",
        ],
        
        "ag_news.evaluators": [
            "classification = src.evaluation.metrics.classification_metrics:ClassificationEvaluator",
            "ensemble = src.evaluation.metrics.ensemble_metrics:EnsembleEvaluator",
            "robustness = src.evaluation.metrics.robustness_metrics:RobustnessEvaluator",
            "efficiency = src.evaluation.metrics.efficiency_metrics:EfficiencyEvaluator",
            "fairness = src.evaluation.metrics.fairness_metrics:FairnessEvaluator",
            "environmental = src.evaluation.metrics.environmental_impact:EnvironmentalEvaluator",
            "contrast = src.evaluation.metrics.contrast_consistency:ContrastConsistencyEvaluator",
        ],
        
        "ag_news.domain_adapters": [
            "mlm_pretrain = src.domain_adaptation.pretraining.mlm_pretrain:MLMPretrainer",
            "adaptive = src.domain_adaptation.pretraining.adaptive_pretrain:AdaptivePretrainer",
            "news_corpus = src.domain_adaptation.pretraining.news_corpus_builder:NewsCorpusBuilder",
            "pseudo_label = src.domain_adaptation.pseudo_labeling.self_training:SelfTrainer",
            "confidence = src.domain_adaptation.pseudo_labeling.confidence_based:ConfidenceBasedLabeler",
        ],
        
        "ag_news.optimizers": [
            "adamw_custom = src.training.optimization.optimizers.adamw_custom:AdamWCustom",
            "sam = src.training.optimization.optimizers.sam:SAMOptimizer",
            "lamb = src.training.optimization.optimizers.lamb:LAMBOptimizer",
            "lookahead = src.training.optimization.optimizers.lookahead:LookaheadOptimizer",
        ],
        
        "ag_news.schedulers": [
            "cosine_warmup = src.training.optimization.schedulers.cosine_warmup:CosineWarmupScheduler",
            "polynomial = src.training.optimization.schedulers.polynomial_decay:PolynomialDecayScheduler",
            "cyclic = src.training.optimization.schedulers.cyclic_scheduler:CyclicScheduler",
        ],
        
        "ag_news.losses": [
            "focal = src.training.objectives.losses.focal_loss:FocalLoss",
            "label_smoothing = src.training.objectives.losses.label_smoothing:LabelSmoothingLoss",
            "contrastive = src.training.objectives.losses.contrastive_loss:ContrastiveLoss",
            "triplet = src.training.objectives.losses.triplet_loss:TripletLoss",
            "instruction = src.training.objectives.losses.instruction_loss:InstructionLoss",
        ],
        
        "ag_news.data_loaders": [
            "standard = src.data.loaders.dataloader:StandardDataLoader",
            "dynamic = src.data.loaders.dynamic_batching:DynamicBatchLoader",
            "prefetch = src.data.loaders.prefetch_loader:PrefetchLoader",
        ],
        
        "ag_news.serving": [
            "fastapi = src.inference.serving.model_server:FastAPIServer",
            "batch = src.inference.serving.batch_server:BatchServer",
            "load_balancer = src.inference.serving.load_balancer:LoadBalancer",
            "onnx = src.inference.optimization.onnx_converter:ONNXServer",
            "tensorrt = src.inference.optimization.tensorrt_optimizer:TensorRTServer",
            "quantized = src.inference.optimization.quantization_optimizer:QuantizedServer",
        ],
        
        "ag_news.monitoring": [
            "prometheus = monitoring.metrics.metric_collectors:PrometheusCollector",
            "custom = monitoring.metrics.custom_metrics:CustomMetrics",
            "log_parser = monitoring.logs_analysis.log_parser:LogParser",
            "anomaly = monitoring.logs_analysis.anomaly_detector:AnomalyDetector",
        ],
        
        "ag_news.security": [
            "jwt = security.api_auth.jwt_handler:JWTHandler",
            "api_keys = security.api_auth.api_keys:APIKeyManager",
            "pii = security.data_privacy.pii_detector:PIIDetector",
            "masking = security.data_privacy.data_masking:DataMasker",
            "adversarial_defense = security.model_security.adversarial_defense:AdversarialDefense",
        ],
        
        "ag_news.interpreters": [
            "shap = src.evaluation.interpretability.shap_interpreter:SHAPInterpreter",
            "lime = src.evaluation.interpretability.lime_interpreter:LIMEInterpreter",
            "attention = src.evaluation.interpretability.attention_analysis:AttentionAnalyzer",
            "integrated_gradients = src.evaluation.interpretability.integrated_gradients:IntegratedGradients",
        ],
    },
    
    # Classifiers
    classifiers=[
        # Development Status
        "Development Status :: 4 - Beta",
        
        # Intended Audience
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        
        # Topics
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Programming Language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
        
        # Operating System
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        
        # Environment
        "Environment :: Console",
        "Environment :: Web Environment",
        "Environment :: GPU :: NVIDIA CUDA :: 11.8",
        "Environment :: GPU :: NVIDIA CUDA :: 12",
        
        # Framework
        "Framework :: Pytest",
        "Framework :: Jupyter",
        "Framework :: Flask",
        "Framework :: FastAPI",
        
        # Natural Language
        "Natural Language :: English",
        
        # Typing
        "Typing :: Typed",
    ],
    
    # Keywords
    keywords=[
        "nlp", "natural-language-processing", "text-classification",
        "news-classification", "ag-news", "deep-learning", "machine-learning",
        "transformers", "pytorch", "huggingface", "bert", "deberta", "roberta",
        "ensemble-learning", "multi-class-classification", "research", "benchmark",
        "state-of-the-art", "sota", "artificial-intelligence", "ai", "data-science",
        "neural-networks", "pretrained-models", "fine-tuning", "transfer-learning",
        "nlp-research", "text-analysis", "document-classification", "news-analysis",
        "media-analysis", "interpretability", "explainable-ai", "model-optimization",
        "efficient-training", "distributed-training", "production-ml", "mlops",
        "knowledge-distillation", "prompt-engineering", "few-shot-learning",
        "zero-shot-learning", "multi-task-learning", "domain-adaptation",
        "adversarial-training", "model-compression", "quantization", "pruning",
        "contrast-sets", "robustness", "fairness", "bias-detection",
        "experiment-tracking", "hyperparameter-optimization", "neural-architecture-search",
        "model-serving", "api-deployment", "cloud-deployment", "edge-deployment",
    ],
    
    # Command classes
    cmdclass={
        "test": PyTest,
        "setup_mlops": SetupMLOps,
        "setup_research": SetupResearch,
        "setup_production": SetupProduction,
    },
    
    # Test configuration
    test_suite="tests",
    tests_require=["pytest>=7.4.0", "pytest-cov>=4.1.0"],
    
    # Other options
    zip_safe=False,
    platforms=["any"],
    
    # Additional metadata
    long_description_content_type="text/markdown",
    provides=["ag_news_text_classification"],
)


# Post-installation message
if "install" in sys.argv or "develop" in sys.argv:
    print("\n" + "="*80)
    print("AG News Text Classification Framework Installation Complete!")
    print("="*80)
    print("\nQuick Start:")
    print("  1. Verify installation: ag-news-setup")
    print("  2. Download data: ag-news-download")
    print("  3. Prepare data: ag-news-prepare")
    print("  4. Train baseline: ag-news-train --model bert")
    print("  5. Evaluate: ag-news-evaluate")
    print("\nSetup Commands:")
    print("  - Research environment: python setup.py setup_research")
    print("  - MLOps environment: python setup.py setup_mlops")
    print("  - Production environment: python setup.py setup_production")
    print("\nDocumentation: https://ag-news-text-classification.readthedocs.io")
    print("Model Hub: https://huggingface.co/your-org/ag-news-models")
    print("="*80 + "\n")


# Create MANIFEST.in if it doesn't exist
manifest_content = """
# Include all documentation
include README.md LICENSE CITATION.cff CHANGELOG.md
include ARCHITECTURE.md PERFORMANCE.md SECURITY.md
include TROUBLESHOOTING.md ROADMAP.md

# Include configuration files
recursive-include configs *.yaml *.yml *.json *.toml
recursive-include prompts *.txt *.json *.md *.jinja2
recursive-include deployment *.yaml *.yml *.sh Dockerfile*
recursive-include .github *.yml *.md

# Include data files
recursive-include data .gitkeep
recursive-include benchmarks *.json *.yaml *.csv

# Include notebooks
recursive-include notebooks *.ipynb

# Include documentation
recursive-include docs *.md *.rst *.png *.jpg *.svg *.puml

# Include app assets
recursive-include app/assets *

# Include test fixtures
recursive-include tests/fixtures *

# Include scripts
recursive-include scripts *.sh *.py

# Include CI/CD
recursive-include ci *.sh *.yml

# Include tools
recursive-include tools *.py *.sh

# Exclude unnecessary files
global-exclude *.pyc
global-exclude *.pyo
global-exclude *.swp
global-exclude .DS_Store
global-exclude __pycache__
global-exclude .git
global-exclude .pytest_cache
global-exclude .mypy_cache
"""

manifest_path = ROOT_DIR / "MANIFEST.in"
if not manifest_path.exists():
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(manifest_content)
    print("Created MANIFEST.in file")
