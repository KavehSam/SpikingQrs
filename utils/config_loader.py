import yaml
import os
from typing import Dict, Any


class ConfigLoader:
    """Configuration loader class for SpikingQRS."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize ConfigLoader with configuration file path."""
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        return config
    
    def _validate_config(self):
        """Validate configuration parameters."""
        required_sections = ['data', 'network', 'learning', 'stdp', 'homeostasis', 'detection', 'logging']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def get(self, key_path: str, default=None):
        """Get a nested configuration value using dot notation."""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value):
        """Set a nested configuration value using dot notation."""
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data processing configuration."""
        return self.config.get('data', {})
    
    def get_network_config(self) -> Dict[str, Any]:
        """Get network architecture configuration."""
        return self.config.get('network', {})
    
    def get_learning_config(self) -> Dict[str, Any]:
        """Get learning parameters configuration."""
        return self.config.get('learning', {})
    
    def get_stdp_config(self) -> Dict[str, Any]:
        """Get STDP parameters configuration."""
        return self.config.get('stdp', {})
    
    def get_homeostasis_config(self) -> Dict[str, Any]:
        """Get homeostasis parameters configuration."""
        return self.config.get('homeostasis', {})
    
    def get_detection_config(self) -> Dict[str, Any]:
        """Get detection parameters configuration."""
        return self.config.get('detection', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get('logging', {})
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration."""
        # Ensure the 'processing' section exists, providing defaults if not
        processing_conf = self.config.get('processing', {})
        
        defaults = {
            'enable_plotting': False,
            'return_spikes': False,
            'is_ecg_raw': True,
            'parallel_jobs': -1
        }
        
        # Merge defaults with actual config
        for key, value in defaults.items():
            if key not in processing_conf:
                processing_conf[key] = value
                
        return processing_conf
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return {
            "tolerance": 36,
            "use_wsl": False,
            "wsl_path": "/mnt/c/Users/hoome/Documents/Github/bxb-competition-framework-v1-0-0/evaluate",
            "save_plots": False,
            "show_plots": False,
            "save_results": True,
            "output_dir": "results",
        }
