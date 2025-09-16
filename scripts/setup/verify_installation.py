#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Installation Verification Script for AG News Classification Framework
=====================================================================

This script performs comprehensive verification of the installation following
best practices from:
- Sculley et al. (2015): "Hidden Technical Debt in Machine Learning Systems"
- Amershi et al. (2019): "Software Engineering for Machine Learning"
- Paleyes et al. (2022): "Challenges in Deploying Machine Learning"

Author: Võ Hải Dũng
License: MIT
"""

import os
import sys
import json
import platform
import subprocess
import importlib
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging
from src.core.exceptions import ConfigurationError
from configs.constants import (
    PROJECT_NAME, PROJECT_VERSION, PYTHON_VERSION,
    SUPPORTED_MODELS, DEFAULT_HYPERPARAMETERS
)

# Setup logging
logger = setup_logging(
    name=__name__,
    log_dir=PROJECT_ROOT / "outputs" / "logs" / "setup",
    log_file="verify_installation.log"
)

# Suppress warnings during import checks
warnings.filterwarnings("ignore")

@dataclass
class VerificationResult:
    """
    Result of verification check following verification methodology from:
    - Beizer (1990): "Software Testing Techniques"
    """
    name: str
    status: str  # "pass", "fail", "warning"
    message: str
    details: Optional[Dict[str, Any]] = None
    severity: str = "info"  # "critical", "error", "warning", "info"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class InstallationVerifier:
    """
    Comprehensive installation verification system.
    
    Implements verification strategies from:
    - Hunt & Thomas (1999): "The Pragmatic Programmer" - Defensive Programming
    - McConnell (2004): "Code Complete" - System Testing
    """
    
    def __init__(self):
        """Initialize verifier."""
        self.results: List[VerificationResult] = []
        self.critical_failures = 0
        self.warnings = 0
        self.system_info = self._collect_system_info()
        
    def _collect_system_info(self) -> Dict[str, Any]:
        """
        Collect system information for reproducibility.
        
        Following guidelines from:
        - Pineau et al. (2021): "Improving Reproducibility in Machine Learning Research"
        """
        info = {
            "timestamp": datetime.now().isoformat(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
            },
            "environment": {
                "cwd": str(Path.cwd()),
                "project_root": str(PROJECT_ROOT),
                "python_path": sys.path,
                "env_vars": {
                    k: v for k, v in os.environ.items()
                    if any(pattern in k for pattern in [
                        "PYTHON", "CUDA", "PATH", "LD_LIBRARY"
                    ])
                },
            },
        }
        return info
    
    def verify_all(self) -> bool:
        """
        Run all verification checks.
        
        Returns:
            True if all critical checks pass
        """
        logger.info("Starting comprehensive installation verification...")
        
        # Run verification checks in order of importance
        self._verify_python_version()
        self._verify_project_structure()
        self._verify_core_packages()
        self._verify_ml_packages()
        self._verify_transformers_models()
        self._verify_cuda_setup()
        self._verify_data_availability()
        self._verify_configs()
        self._verify_scripts()
        self._verify_api_keys()
        self._verify_disk_space()
        self._verify_memory()
        self._verify_network()
        self._verify_reproducibility()
        
        # Generate report
        self._generate_report()
        
        # Return success if no critical failures
        return self.critical_failures == 0
    
    def _verify_python_version(self):
        """Verify Python version meets requirements."""
        result_name = "Python Version"
        
        try:
            current_version = sys.version_info
            required_version = tuple(map(int, PYTHON_VERSION.split(".")))
            
            if current_version[:2] >= required_version[:2]:
                self._add_result(
                    result_name, "pass",
                    f"Python {current_version.major}.{current_version.minor} meets requirements",
                    {"current": f"{current_version.major}.{current_version.minor}",
                     "required": PYTHON_VERSION}
                )
            else:
                self._add_result(
                    result_name, "fail",
                    f"Python {PYTHON_VERSION}+ required, found {current_version.major}.{current_version.minor}",
                    severity="critical"
                )
        except Exception as e:
            self._add_result(result_name, "fail", str(e), severity="critical")
    
    def _verify_project_structure(self):
        """Verify project directory structure."""
        result_name = "Project Structure"
        
        required_dirs = [
            "src", "configs", "data", "scripts", "tests",
            "notebooks", "experiments", "outputs", "deployment"
        ]
        
        missing_dirs = []
        for dir_name in required_dirs:
            dir_path = PROJECT_ROOT / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            self._add_result(
                result_name, "warning",
                f"Missing directories: {', '.join(missing_dirs)}",
                {"missing": missing_dirs},
                severity="warning"
            )
        else:
            self._add_result(
                result_name, "pass",
                "All required directories present"
            )
    
    def _verify_core_packages(self):
        """Verify core Python packages."""
        result_name = "Core Packages"
        
        core_packages = {
            "numpy": "1.24.0",
            "pandas": "2.0.0",
            "scipy": "1.10.0",
            "scikit-learn": "1.3.0",
            "tqdm": "4.66.0",
            "pyyaml": "6.0.0",
            "pydantic": "2.4.0",
        }
        
        missing = []
        outdated = []
        
        for package, min_version in core_packages.items():
            try:
                module = importlib.import_module(package)
                
                # Check version if available
                if hasattr(module, "__version__"):
                    current_version = module.__version__
                    if self._compare_versions(current_version, min_version) < 0:
                        outdated.append(f"{package} ({current_version} < {min_version})")
                        
            except ImportError:
                missing.append(package)
        
        if missing:
            self._add_result(
                result_name, "fail",
                f"Missing packages: {', '.join(missing)}",
                {"missing": missing},
                severity="critical"
            )
        elif outdated:
            self._add_result(
                result_name, "warning",
                f"Outdated packages: {', '.join(outdated)}",
                {"outdated": outdated},
                severity="warning"
            )
        else:
            self._add_result(result_name, "pass", "All core packages installed")
    
    def _verify_ml_packages(self):
        """
        Verify ML framework packages.
        
        Following framework requirements from:
        - Paszke et al. (2019): "PyTorch: An Imperative Style, High-Performance Deep Learning Library"
        - Wolf et al. (2020): "Transformers: State-of-the-Art Natural Language Processing"
        """
        result_name = "ML Frameworks"
        
        ml_packages = {
            "torch": ("2.0.0", "PyTorch"),
            "transformers": ("4.35.0", "Hugging Face Transformers"),
            "datasets": ("2.14.0", "Hugging Face Datasets"),
            "tokenizers": ("0.15.0", "Fast Tokenizers"),
            "accelerate": ("0.24.0", "Hugging Face Accelerate"),
        }
        
        issues = []
        details = {}
        
        for package, (min_version, full_name) in ml_packages.items():
            try:
                module = importlib.import_module(package)
                
                if hasattr(module, "__version__"):
                    version = module.__version__
                    details[package] = {
                        "installed": version,
                        "required": min_version,
                        "name": full_name
                    }
                    
                    # Special checks for PyTorch
                    if package == "torch":
                        import torch
                        details["torch"]["cuda_available"] = torch.cuda.is_available()
                        details["torch"]["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
                        details["torch"]["gpu_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
                        
            except ImportError:
                issues.append(f"{full_name} not installed")
                details[package] = {"installed": False, "required": min_version}
        
        if issues:
            self._add_result(
                result_name, "fail",
                f"ML framework issues: {'; '.join(issues)}",
                details,
                severity="critical"
            )
        else:
            self._add_result(
                result_name, "pass",
                "All ML frameworks properly installed",
                details
            )
    
    def _verify_transformers_models(self):
        """Verify access to required transformer models."""
        result_name = "Transformer Models"
        
        try:
            from transformers import AutoModel, AutoTokenizer
            
            # Test model loading (just config, not weights)
            test_models = [
                "bert-base-uncased",
                "microsoft/deberta-v3-base",  # Test smaller version
            ]
            
            accessible = []
            inaccessible = []
            
            for model_name in test_models:
                try:
                    # Only load config to verify access
                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(model_name)
                    accessible.append(model_name)
                except Exception as e:
                    inaccessible.append(f"{model_name}: {str(e)[:50]}")
            
            if inaccessible:
                self._add_result(
                    result_name, "warning",
                    "Some models not accessible (may need download)",
                    {"accessible": accessible, "issues": inaccessible},
                    severity="warning"
                )
            else:
                self._add_result(
                    result_name, "pass",
                    "Transformer models accessible",
                    {"verified_models": accessible}
                )
                
        except ImportError:
            self._add_result(
                result_name, "fail",
                "Transformers library not available",
                severity="critical"
            )
    
    def _verify_cuda_setup(self):
        """
        Verify CUDA and GPU setup.
        
        GPU requirements based on:
        - Strubell et al. (2019): "Energy and Policy Considerations for Deep Learning in NLP"
        """
        result_name = "CUDA/GPU Setup"
        
        try:
            import torch
            
            cuda_available = torch.cuda.is_available()
            
            if cuda_available:
                gpu_count = torch.cuda.device_count()
                cuda_version = torch.version.cuda
                
                # Get GPU details
                gpu_details = []
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                    gpu_details.append({
                        "id": i,
                        "name": gpu_name,
                        "memory_gb": round(gpu_memory, 2)
                    })
                
                # Check for sufficient GPU memory (need at least 16GB for DeBERTa-xlarge)
                max_memory = max(gpu["memory_gb"] for gpu in gpu_details) if gpu_details else 0
                
                if max_memory < 16:
                    self._add_result(
                        result_name, "warning",
                        f"GPU memory ({max_memory:.1f}GB) may be insufficient for large models",
                        {"cuda_version": cuda_version, "gpus": gpu_details},
                        severity="warning"
                    )
                else:
                    self._add_result(
                        result_name, "pass",
                        f"CUDA {cuda_version} with {gpu_count} GPU(s) available",
                        {"cuda_version": cuda_version, "gpus": gpu_details}
                    )
            else:
                self._add_result(
                    result_name, "warning",
                    "No CUDA GPUs available - will use CPU (slower training)",
                    severity="warning"
                )
                
        except ImportError:
            self._add_result(
                result_name, "fail",
                "PyTorch not installed",
                severity="critical"
            )
    
    def _verify_data_availability(self):
        """Verify data files are available."""
        result_name = "Data Availability"
        
        data_dir = PROJECT_ROOT / "data"
        ag_news_dir = data_dir / "raw" / "ag_news"
        
        required_files = {
            "train.csv": "Training data",
            "test.csv": "Test data",
            "classes.txt": "Class labels",
        }
        
        missing_files = []
        found_files = []
        
        for filename, description in required_files.items():
            file_path = ag_news_dir / filename
            if file_path.exists():
                found_files.append(filename)
            else:
                missing_files.append(f"{filename} ({description})")
        
        if missing_files:
            self._add_result(
                result_name, "warning",
                f"Data files missing: {', '.join(missing_files)}. Run 'python scripts/setup/download_all_data.py'",
                {"missing": missing_files, "found": found_files},
                severity="warning"
            )
        else:
            # Check file sizes
            file_sizes = {}
            for filename in required_files:
                file_path = ag_news_dir / filename
                size_mb = file_path.stat().st_size / 1e6
                file_sizes[filename] = f"{size_mb:.2f} MB"
            
            self._add_result(
                result_name, "pass",
                "All required data files present",
                {"files": file_sizes}
            )
    
    def _verify_configs(self):
        """Verify configuration files."""
        result_name = "Configuration Files"
        
        config_dir = PROJECT_ROOT / "configs"
        
        critical_configs = [
            "environments/dev.yaml",
            "models/single/deberta_v3_xlarge.yaml",
            "models/single/roberta_large.yaml",
            "training/standard/base_training.yaml",
        ]
        
        missing_configs = []
        
        for config_path in critical_configs:
            full_path = config_dir / config_path
            if not full_path.exists():
                missing_configs.append(config_path)
        
        if missing_configs:
            self._add_result(
                result_name, "fail",
                f"Missing configuration files: {', '.join(missing_configs)}",
                {"missing": missing_configs},
                severity="error"
            )
        else:
            self._add_result(
                result_name, "pass",
                "All critical configuration files present"
            )
    
    def _verify_scripts(self):
        """Verify executable scripts."""
        result_name = "Executable Scripts"
        
        scripts_dir = PROJECT_ROOT / "scripts"
        
        critical_scripts = [
            "setup/download_all_data.py",
            "setup/setup_environment.sh",
            "data_preparation/prepare_ag_news.py",
            "training/train_single_model.py",
        ]
        
        missing_scripts = []
        non_executable = []
        
        for script_path in critical_scripts:
            full_path = scripts_dir / script_path
            if not full_path.exists():
                missing_scripts.append(script_path)
            elif script_path.endswith(".sh") and not os.access(full_path, os.X_OK):
                non_executable.append(script_path)
        
        if missing_scripts:
            self._add_result(
                result_name, "warning",
                f"Missing scripts: {', '.join(missing_scripts)}",
                {"missing": missing_scripts},
                severity="warning"
            )
        elif non_executable:
            self._add_result(
                result_name, "warning",
                f"Scripts not executable: {', '.join(non_executable)}",
                {"non_executable": non_executable},
                severity="warning"
            )
        else:
            self._add_result(
                result_name, "pass",
                "All critical scripts available"
            )
    
    def _verify_api_keys(self):
        """Verify API keys for external services."""
        result_name = "API Keys"
        
        # Check for .env file
        env_file = PROJECT_ROOT / ".env"
        
        if not env_file.exists():
            self._add_result(
                result_name, "warning",
                "No .env file found. Copy .env.example to .env and configure",
                severity="warning"
            )
            return
        
        # Check for important API keys (optional)
        optional_keys = {
            "WANDB_API_KEY": "Weights & Biases tracking",
            "OPENAI_API_KEY": "GPT-4 distillation",
            "HF_TOKEN": "Hugging Face private models",
        }
        
        configured = []
        not_configured = []
        
        for key, description in optional_keys.items():
            value = os.getenv(key, "")
            if value and value != "your_api_key_here":
                configured.append(f"{key} ({description})")
            else:
                not_configured.append(f"{key} ({description})")
        
        if configured:
            self._add_result(
                result_name, "pass",
                f"API keys configured: {len(configured)}/{len(optional_keys)}",
                {"configured": configured, "not_configured": not_configured}
            )
        else:
            self._add_result(
                result_name, "info",
                "No API keys configured (optional for basic usage)",
                {"not_configured": not_configured},
                severity="info"
            )
    
    def _verify_disk_space(self):
        """Verify sufficient disk space."""
        result_name = "Disk Space"
        
        import shutil
        
        # Get disk usage
        total, used, free = shutil.disk_usage(PROJECT_ROOT)
        
        # Convert to GB
        total_gb = total / (1024**3)
        used_gb = used / (1024**3)
        free_gb = free / (1024**3)
        
        # Require at least 50GB free for full pipeline
        required_gb = 50
        
        details = {
            "total_gb": round(total_gb, 2),
            "used_gb": round(used_gb, 2),
            "free_gb": round(free_gb, 2),
            "required_gb": required_gb,
        }
        
        if free_gb < required_gb:
            self._add_result(
                result_name, "warning",
                f"Low disk space: {free_gb:.1f}GB free, {required_gb}GB recommended",
                details,
                severity="warning"
            )
        else:
            self._add_result(
                result_name, "pass",
                f"Sufficient disk space: {free_gb:.1f}GB available",
                details
            )
    
    def _verify_memory(self):
        """Verify system memory."""
        result_name = "System Memory"
        
        try:
            import psutil
            
            # Get memory info
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)
            
            # Require at least 16GB RAM
            required_gb = 16
            
            details = {
                "total_gb": round(total_gb, 2),
                "available_gb": round(available_gb, 2),
                "used_percent": memory.percent,
                "required_gb": required_gb,
            }
            
            if total_gb < required_gb:
                self._add_result(
                    result_name, "warning",
                    f"Limited RAM: {total_gb:.1f}GB total, {required_gb}GB recommended",
                    details,
                    severity="warning"
                )
            else:
                self._add_result(
                    result_name, "pass",
                    f"Sufficient memory: {total_gb:.1f}GB total, {available_gb:.1f}GB available",
                    details
                )
                
        except ImportError:
            self._add_result(
                result_name, "info",
                "psutil not installed - cannot check memory",
                severity="info"
            )
    
    def _verify_network(self):
        """Verify network connectivity."""
        result_name = "Network Connectivity"
        
        import urllib.request
        
        test_urls = {
            "GitHub": "https://github.com",
            "Hugging Face": "https://huggingface.co",
            "PyPI": "https://pypi.org",
        }
        
        accessible = []
        inaccessible = []
        
        for name, url in test_urls.items():
            try:
                urllib.request.urlopen(url, timeout=5)
                accessible.append(name)
            except Exception:
                inaccessible.append(name)
        
        if inaccessible:
            self._add_result(
                result_name, "warning",
                f"Cannot access: {', '.join(inaccessible)}",
                {"accessible": accessible, "inaccessible": inaccessible},
                severity="warning"
            )
        else:
            self._add_result(
                result_name, "pass",
                "Network connectivity verified",
                {"accessible": accessible}
            )
    
    def _verify_reproducibility(self):
        """
        Verify reproducibility setup.
        
        Following guidelines from:
        - Pineau et al. (2021): "Improving Reproducibility in Machine Learning Research"
        """
        result_name = "Reproducibility Setup"
        
        issues = []
        
        # Check for seed configuration
        if "SEED" not in os.environ and "PYTHONHASHSEED" not in os.environ:
            issues.append("Random seeds not configured in environment")
        
        # Check for deterministic operations
        try:
            import torch
            if not torch.backends.cudnn.deterministic:
                issues.append("CUDNN deterministic mode not enabled")
        except:
            pass
        
        # Check for version pinning in requirements
        req_file = PROJECT_ROOT / "requirements" / "base.txt"
        if req_file.exists():
            with open(req_file) as f:
                unpinned = [line for line in f if "==" not in line and not line.startswith("#") and line.strip()]
                if unpinned:
                    issues.append(f"{len(unpinned)} packages without version pinning")
        
        if issues:
            self._add_result(
                result_name, "warning",
                "Reproducibility issues detected",
                {"issues": issues},
                severity="warning"
            )
        else:
            self._add_result(
                result_name, "pass",
                "Reproducibility measures in place"
            )
    
    def _add_result(
        self,
        name: str,
        status: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "info"
    ):
        """Add verification result."""
        result = VerificationResult(
            name=name,
            status=status,
            message=message,
            details=details,
            severity=severity
        )
        
        self.results.append(result)
        
        # Update counters
        if status == "fail" and severity == "critical":
            self.critical_failures += 1
        elif status == "warning":
            self.warnings += 1
        
        # Log result
        if status == "pass":
            logger.info(f"✓ {name}: {message}")
        elif status == "warning":
            logger.warning(f"⚠ {name}: {message}")
        elif status == "fail":
            logger.error(f"✗ {name}: {message}")
        else:
            logger.info(f"ℹ {name}: {message}")
    
    def _compare_versions(self, v1: str, v2: str) -> int:
        """Compare version strings."""
        from packaging import version
        
        try:
            return -1 if version.parse(v1) < version.parse(v2) else 1
        except:
            return 0
    
    def _generate_report(self):
        """Generate verification report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self.system_info,
            "summary": {
                "total_checks": len(self.results),
                "passed": len([r for r in self.results if r.status == "pass"]),
                "warnings": self.warnings,
                "failures": len([r for r in self.results if r.status == "fail"]),
                "critical_failures": self.critical_failures,
            },
            "results": [r.to_dict() for r in self.results],
            "project_info": {
                "name": PROJECT_NAME,
                "version": PROJECT_VERSION,
                "root": str(PROJECT_ROOT),
            },
        }
        
        # Save report
        report_path = PROJECT_ROOT / "outputs" / "logs" / "setup" / "verification_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "=" * 80)
        print("INSTALLATION VERIFICATION SUMMARY")
        print("=" * 80)
        print(f"Total checks: {report['summary']['total_checks']}")
        print(f"✓ Passed: {report['summary']['passed']}")
        print(f"⚠ Warnings: {report['summary']['warnings']}")
        print(f"✗ Failures: {report['summary']['failures']}")
        print(f"🔴 Critical failures: {report['summary']['critical_failures']}")
        print(f"\nDetailed report saved to: {report_path}")
        print("=" * 80)
        
        if self.critical_failures > 0:
            print("\n❌ Installation verification FAILED")
            print("Please address critical issues before proceeding.")
        elif self.warnings > 0:
            print("\n⚠️ Installation verified with WARNINGS")
            print("The system should work but some features may be limited.")
        else:
            print("\n✅ Installation verification PASSED")
            print("Your environment is ready for AG News classification!")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verify AG News Classification Framework installation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick verification only"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run verification
    verifier = InstallationVerifier()
    success = verifier.verify_all()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
