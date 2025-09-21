"""
AG News Text Classification Application Module
==============================================

This module provides the main application interface for AG News text classification,
implementing Model-View-Controller (MVC) pattern following:
- Krasner & Pope (1988): "A Description of the Model-View-Controller User Interface Paradigm"
- Fowler (2002): "Patterns of Enterprise Application Architecture"

The application architecture follows principles from:
- Nielsen (1993): "Usability Engineering" - User interface design
- Shneiderman et al. (2016): "Designing the User Interface" - Interaction patterns
- Johnson (2020): "Designing with the Mind in Mind" - Cognitive principles

Author: Võ Hải Dũng
License: MIT
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import json
from datetime import datetime

# Add project root to path
APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging

# Setup logging
logger = setup_logging(__name__)

# Application version
__version__ = "1.0.0"

# Application metadata
APP_METADATA = {
    "name": "AG News Text Classification",
    "version": __version__,
    "author": "Võ Hải Dũng",
    "description": "Interactive application for news article classification",
    "categories": ["World", "Sports", "Business", "Sci/Tech"],
    "framework": "Streamlit",
    "license": "MIT"
}

@dataclass
class AppConfig:
    """
    Application configuration.
    
    Following configuration management patterns from:
    - Hunt & Thomas (1999): "The Pragmatic Programmer" - Configuration as code
    """
    # Application settings
    app_name: str = APP_METADATA["name"]
    app_version: str = APP_METADATA["version"]
    
    # Model settings
    default_model_path: Path = PROJECT_ROOT / "outputs" / "models" / "fine_tuned"
    model_cache_size: int = 3
    
    # UI settings
    theme: str = "light"
    layout: str = "wide"
    sidebar_state: str = "expanded"
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    max_batch_size: int = 128
    
    # Feature flags
    enable_batch_processing: bool = True
    enable_visualization: bool = True
    enable_interpretability: bool = True
    enable_model_comparison: bool = True
    enable_real_time_demo: bool = True
    enable_api_testing: bool = False
    
    # Paths
    assets_dir: Path = field(default_factory=lambda: APP_DIR / "assets")
    components_dir: Path = field(default_factory=lambda: APP_DIR / "components")
    pages_dir: Path = field(default_factory=lambda: APP_DIR / "pages")
    utils_dir: Path = field(default_factory=lambda: APP_DIR / "utils")
    
    # Security
    enable_auth: bool = False
    api_key_required: bool = False
    max_text_length: int = 10000
    
    # Monitoring
    enable_logging: bool = True
    enable_metrics: bool = True
    track_usage: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure paths exist
        for path_attr in ["assets_dir", "components_dir", "pages_dir", "utils_dir"]:
            path = getattr(self, path_attr)
            if not path.exists():
                logger.warning(f"Directory not found: {path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AppConfig':
        """Create configuration from dictionary."""
        # Convert string paths back to Path objects
        for key in ["default_model_path", "assets_dir", "components_dir", "pages_dir", "utils_dir"]:
            if key in config_dict and isinstance(config_dict[key], str):
                config_dict[key] = Path(config_dict[key])
        
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'AppConfig':
        """Load configuration from file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def save(self, config_path: Path):
        """Save configuration to file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

class AppState:
    """
    Application state management.
    
    Implements state management pattern from:
    - Gamma et al. (1994): "Design Patterns" - State pattern
    """
    
    def __init__(self):
        """Initialize application state."""
        self._state = {}
        self._callbacks = {}
        self._history = []
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get state value."""
        return self._state.get(key, default)
    
    def set(self, key: str, value: Any, notify: bool = True):
        """Set state value."""
        old_value = self._state.get(key)
        self._state[key] = value
        
        # Add to history
        self._history.append({
            "timestamp": datetime.now().isoformat(),
            "key": key,
            "old_value": old_value,
            "new_value": value
        })
        
        # Notify callbacks
        if notify and key in self._callbacks:
            for callback in self._callbacks[key]:
                callback(value, old_value)
    
    def subscribe(self, key: str, callback: Callable):
        """Subscribe to state changes."""
        if key not in self._callbacks:
            self._callbacks[key] = []
        self._callbacks[key].append(callback)
    
    def unsubscribe(self, key: str, callback: Callable):
        """Unsubscribe from state changes."""
        if key in self._callbacks and callback in self._callbacks[key]:
            self._callbacks[key].remove(callback)
    
    def clear(self):
        """Clear all state."""
        self._state.clear()
        self._history.clear()
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get state history."""
        if limit:
            return self._history[-limit:]
        return self._history.copy()

class AppController:
    """
    Application controller.
    
    Implements controller logic following MVC pattern from:
    - Reenskaug (1979): "Models-Views-Controllers" - Original MVC formulation
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize application controller.
        
        Args:
            config: Application configuration
        """
        self.config = config or AppConfig()
        self.state = AppState()
        self._models = {}
        self._tokenizers = {}
        self._components = {}
        
        logger.info(f"Initialized {self.config.app_name} v{self.config.app_version}")
    
    def load_model(self, model_name: str, model_path: Optional[Path] = None):
        """
        Load a model for use in the application.
        
        Args:
            model_name: Name identifier for the model
            model_path: Path to model directory
        """
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        model_path = model_path or self.config.default_model_path / model_name
        
        try:
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.eval()
            
            # Cache
            self._models[model_name] = model
            self._tokenizers[model_name] = tokenizer
            
            # Update state
            self.state.set("current_model", model_name)
            
            logger.info(f"Loaded model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def get_model(self, model_name: Optional[str] = None):
        """Get loaded model."""
        model_name = model_name or self.state.get("current_model")
        return self._models.get(model_name), self._tokenizers.get(model_name)
    
    def register_component(self, name: str, component: Any):
        """Register a UI component."""
        self._components[name] = component
        logger.debug(f"Registered component: {name}")
    
    def get_component(self, name: str) -> Optional[Any]:
        """Get registered component."""
        return self._components.get(name)
    
    def cleanup(self):
        """Cleanup resources."""
        self._models.clear()
        self._tokenizers.clear()
        self._components.clear()
        self.state.clear()
        logger.info("Application cleanup completed")

# Global instances
_app_config: Optional[AppConfig] = None
_app_controller: Optional[AppController] = None

def initialize_app(config: Optional[AppConfig] = None) -> AppController:
    """
    Initialize the application.
    
    Args:
        config: Application configuration
        
    Returns:
        Application controller instance
    """
    global _app_config, _app_controller
    
    _app_config = config or AppConfig()
    _app_controller = AppController(_app_config)
    
    return _app_controller

def get_app_controller() -> Optional[AppController]:
    """Get the global application controller."""
    return _app_controller

def get_app_config() -> Optional[AppConfig]:
    """Get the global application configuration."""
    return _app_config

# Export public API
__all__ = [
    # Metadata
    "__version__",
    "APP_METADATA",
    
    # Configuration
    "AppConfig",
    
    # State management
    "AppState",
    
    # Controller
    "AppController",
    
    # Functions
    "initialize_app",
    "get_app_controller",
    "get_app_config",
]
