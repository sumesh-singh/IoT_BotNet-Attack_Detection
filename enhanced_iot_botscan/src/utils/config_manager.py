"""
Configuration Manager for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Manages system configuration and provides centralized config access.
"""

import yaml
import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """Centralized configuration management."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        
        if config_path is None:
            config_path = self._find_default_config()
            logger.info(f"No config path provided, using default: {config_path}")

        self.config_path = config_path
        self.config = self._load_config(config_path)
        logger.info(f"Configuration loaded from: {self.config_path}")

        # Apply environment variable overrides
        self._apply_env_overrides()

    def _find_default_config(self) -> str:
        """Find default configuration file."""

        possible_paths = [
            './config/config.yaml',
            '../config/config.yaml',
            '../../config/config.yaml',
            './config.yaml'
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # If no config found, create minimal config
        return self._create_minimal_config()

    def _create_minimal_config(self) -> str:
        """Create minimal configuration if none exists."""

        minimal_config = {
            'system': {
                'name': 'Enhanced IoT BotScan',
                'version': '1.0.0',
                'environment': 'development'
            },
            'machine_learning': {
                'ensemble': {
                    'algorithms': [
                        {'name': 'random_forest', 'enabled': True, 'n_estimators': 100},
                        {'name': 'xgboost', 'enabled': True, 'n_estimators': 100},
                        {'name': 'lightgbm', 'enabled': True, 'n_estimators': 100}
                    ]
                },
                'meta_learner': {
                    'algorithm': 'logistic_regression'
                }
            },
            'adversarial_training': {
                'enabled': True,
                'adversarial_ratio': 0.3,
                'attacks': {
                    'fgsm': {'enabled': True, 'epsilon': 0.1},
                    'pgd': {'enabled': True, 'epsilon': 0.1, 'alpha': 0.01, 'num_iter': 10},
                    'cw': {'enabled': True, 'c': 1.0}
                }
            },
            'concept_drift': {
                'detection': {
                    'enabled': True,
                    'methods': ['kolmogorov_smirnov', 'page_hinkley'],
                    'threshold': 0.05
                }
            },
            'data': {
                'data_paths': {
                    'n_baiot': './data/raw/n_baiot/',
                    'iot_23': './data/raw/iot_23/',
                    'bot_iot': './data/raw/bot_iot/'
                }
            }
        }

        # Save minimal config
        config_path = './config.yaml'
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, 'w') as f:
            yaml.dump(minimal_config, f, default_flow_style=False)

        return config_path

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        file_extension = Path(config_path).suffix.lower()

        with open(config_path, 'r') as f:
            if file_extension in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif file_extension == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {file_extension}")

        return config or {}

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""

        import os
        logger.info("Applying environment variable overrides...")
        # Common environment variable mappings
        env_mappings = {
            'DB_HOST': ['database', 'primary', 'host'],
            'DB_PORT': ['database', 'primary', 'port'],
            'DB_NAME': ['database', 'primary', 'database'],
            'DB_USER': ['database', 'primary', 'username'],
            'DB_PASSWORD': ['database', 'primary', 'password'],
            'API_HOST': ['api', 'rest', 'host'],
            'API_PORT': ['api', 'rest', 'port'],
            'LOG_LEVEL': ['logging', 'level'],
            'SECRET_KEY': ['security', 'secret_key'],
            'ML_BATCH_SIZE': ['machine_learning', 'batch_size']
        }

        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                logger.info(f"Overriding config from environment variable: {env_var}")
                self._set_nested_value(config_path, env_value)

    def _set_nested_value(self, path: list, value: str) -> None:
        """Set nested configuration value."""

        current = self.config

        # Navigate to parent
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set final value with type conversion
        final_key = path[-1]
        current[final_key] = self._convert_env_value(value)

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""

        # Boolean conversion
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'

        # Integer conversion
        try:
            if '.' not in value:
                return int(value)
        except ValueError:
            pass

        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""

        keys = key_path.split('.')
        current = self.config

        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default

    def get_ml_config(self) -> Dict[str, Any]:
        """Get machine learning configuration."""
        return self.get('machine_learning', {})

    def get_adversarial_config(self) -> Dict[str, Any]:
        """Get adversarial training configuration."""
        return self.get('adversarial_training', {})

    def get_drift_config(self) -> Dict[str, Any]:
        """Get concept drift configuration."""
        return self.get('concept_drift', {})

    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.get('data', {})

    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration."""
        return self.get('feature_engineering', {})

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""

        def deep_update(base: dict, updates: dict) -> dict:
            for key, value in updates.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_update(base[key], value)
                else:
                    base[key] = value
            return base

        deep_update(self.config, updates)

    def save_config(self, output_path: Optional[str] = None) -> None:
        """Save current configuration to file."""

        if output_path is None:
            output_path = self.config_path

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        logger.info("Validating configuration...")
        issues = []

        # Check required sections
        required_sections = ['system', 'machine_learning', 'data']
        for section in required_sections:
            if section not in self.config:
                issues.append(f"Missing required section: {section}")
                logger.warning(f"Validation issue: Missing required section: {section}")

        # Check ML configuration
        ml_config = self.get_ml_config()
        if 'ensemble' not in ml_config:
            issues.append("Missing ensemble configuration in machine_learning section")
            logger.warning("Validation issue: Missing ensemble configuration in machine_learning section")

        # Check data paths
        data_config = self.get_data_config()
        if 'data_paths' in data_config:
            for dataset, path in data_config['data_paths'].items():
                if not os.path.exists(path):
                    issues.append(f"Data path does not exist: {dataset} -> {path}")
                    logger.warning(f"Validation issue: Data path does not exist: {dataset} -> {path}")
        
        if not issues:
            logger.info("Configuration validation successful.")
        else:
            logger.warning(f"Configuration validation failed with {len(issues)} issues.")

        return issues

    def get_config_summary(self) -> str:
        """Get configuration summary as string."""

        summary = f"""
ENHANCED IOT BOTSCAN - CONFIGURATION SUMMARY
==========================================

System Information:
- Name: {self.get('system.name', 'Unknown')}
- Version: {self.get('system.version', 'Unknown')}
- Environment: {self.get('system.environment', 'Unknown')}

Machine Learning:
- Ensemble Models: {len(self.get('machine_learning.ensemble.algorithms', []))}
- Meta-learner: {self.get('machine_learning.meta_learner.algorithm', 'Unknown')}

Adversarial Training:
- Enabled: {self.get('adversarial_training.enabled', False)}
- Adversarial Ratio: {self.get('adversarial_training.adversarial_ratio', 'N/A')}

Concept Drift Detection:
- Enabled: {self.get('concept_drift.detection.enabled', False)}
- Methods: {', '.join(self.get('concept_drift.detection.methods', []))}

Data Sources:
- N-BaIoT: {self.get('data.data_paths.n_baiot', 'Not configured')}
- IoT-23: {self.get('data.data_paths.iot_23', 'Not configured')}
- BoT-IoT: {self.get('data.data_paths.bot_iot', 'Not configured')}

Configuration File: {self.config_path}
"""

        return summary
