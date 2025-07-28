import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    """Centralized configuration loader"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    @staticmethod
    def load_model_config(model_name: str) -> Dict[str, Any]:
        """Load model-specific configuration"""
        config_path = Path(f"configs/models/single/{model_name}.yaml")
        return ConfigLoader.load_config(config_path)
    
    @staticmethod
    def load_training_config(training_type: str = "standard") -> Dict[str, Any]:
        """Load training configuration"""
        config_path = Path(f"configs/training/{training_type}/base_training.yaml")
        return ConfigLoader.load_config(config_path)
