import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import wandb
import yaml
from datetime import datetime
from src.utils.logging_config import setup_logger
from src.utils.reproducibility import set_seed
from src.core.types import ExperimentConfig
from configs.config_loader import ConfigLoader

class ExperimentRunner:
    """Base experiment runner"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = ConfigLoader.load_config(config_path)
        self.logger = setup_logger(self.__class__.__name__)
        
        # Setup experiment
        self.setup_experiment()
        
    def setup_experiment(self):
        """Setup experiment environment"""
        # Set seed
        seed = self.config.get('seed', 42)
        set_seed(seed)
        self.logger.info(f"Set random seed: {seed}")
        
        # Initialize wandb if enabled
        if self.config.get('use_wandb', True):
            wandb.init(
                project=self.config.get('project_name', 'ag-news-sota'),
                name=self.config.get('experiment_name', 
                     f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                config=self.config,
                tags=self.config.get('tags', [])
            )
            self.logger.info("Initialized Weights & Biases")
    
    def run(self):
        """Run experiment - to be implemented by subclasses"""
        raise NotImplementedError
