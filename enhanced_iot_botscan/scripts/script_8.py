# Let's create more essential components to complete the system

# 12. Performance Evaluator
performance_evaluator_content = '''"""
Performance Evaluator for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Comprehensive evaluation of model performance including adversarial robustness.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, auc
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class PerformanceEvaluator:
    """Comprehensive performance evaluation for IoT botnet detection models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.evaluation_history = []
        
    def comprehensive_evaluation(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Perform comprehensive model evaluation."""
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get prediction probabilities if available
        try:
            y_proba = model.predict_proba(X_test)
        except:
            y_proba = None
        
        # Basic classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # ROC-AUC if probabilities available
        roc_auc = None
        if y_proba is not None:
            try:
                if len(np.unique(y_test)) == 2:  # Binary classification
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                else:  # Multi-class
                    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            except:
                pass
        
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(X_test),
            'n_features': len(X_test.columns),
            'n_classes': len(np.unique(y_test)),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': y_pred.tolist(),
            'true_labels': y_test.tolist()
        }
        
        if y_proba is not None:
            evaluation_results['prediction_probabilities'] = y_proba.tolist()
        
        self.evaluation_history.append(evaluation_results)
        return evaluation_results
    
    def evaluate_adversarial_robustness(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate model robustness against adversarial attacks."""
        
        from ..core.adversarial.attack_generator import AdversarialAttackGenerator
        
        # Initialize attack generator
        attack_config = self.config.get('adversarial_attacks', {
            'fgsm': {'enabled': True, 'epsilon': 0.1},
            'pgd': {'enabled': True, 'epsilon': 0.1, 'alpha': 0.01, 'num_iter': 10},
            'cw': {'enabled': True, 'c': 1.0}
        })
        
        attack_generator = AdversarialAttackGenerator(attack_config)
        
        # Evaluate robustness
        robustness_results = attack_generator.evaluate_robustness(
            X_test.values, y_test.values, model
        )
        
        return robustness_results
    
    def cross_dataset_evaluation(self, model, datasets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate model performance across different datasets."""
        
        cross_results = {}
        
        for dataset_name, dataset in datasets.items():
            X_test = pd.DataFrame(dataset['features'])
            y_test = pd.Series(dataset['labels'])
            
            results = self.comprehensive_evaluation(model, X_test, y_test)
            cross_results[dataset_name] = results
        
        return cross_results
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, save_path=None):
        """Plot confusion matrix."""
        
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
                
        except Exception as e:
            print(f"Error plotting confusion matrix: {e}")
    
    def plot_roc_curve(self, y_true, y_proba, save_path=None):
        """Plot ROC curve for binary classification."""
        
        try:
            if len(np.unique(y_true)) != 2:
                print("ROC curve only available for binary classification")
                return
            
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
                
        except Exception as e:
            print(f"Error plotting ROC curve: {e}")
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive evaluation report."""
        
        report = f"""
ENHANCED IOT BOTSCAN - MODEL EVALUATION REPORT
=============================================

Evaluation Timestamp: {results['timestamp']}
Dataset Size: {results['n_samples']} samples, {results['n_features']} features
Number of Classes: {results['n_classes']}

OVERALL PERFORMANCE METRICS:
----------------------------
Accuracy:  {results['accuracy']:.4f}
Precision: {results['precision']:.4f}
Recall:    {results['recall']:.4f}
F1-Score:  {results['f1_score']:.4f}"""

        if results.get('roc_auc'):
            report += f"\\nROC-AUC:   {results['roc_auc']:.4f}"

        report += "\\n\\nPER-CLASS PERFORMANCE:"
        report += "\\n" + "-" * 22
        
        class_report = results['classification_report']
        for class_name, metrics in class_report.items():
            if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                report += f"\\nClass {class_name}:"
                report += f"\\n  Precision: {metrics['precision']:.4f}"
                report += f"\\n  Recall:    {metrics['recall']:.4f}"
                report += f"\\n  F1-Score:  {metrics['f1-score']:.4f}"
                report += f"\\n  Support:   {metrics['support']}"

        return report

def create_evaluation_summary(evaluation_results: Dict[str, Any]) -> pd.DataFrame:
    """Create summary DataFrame from evaluation results."""
    
    summary_data = []
    
    for eval_name, results in evaluation_results.items():
        summary_row = {
            'Evaluation': eval_name,
            'Accuracy': results.get('accuracy', 0),
            'Precision': results.get('precision', 0),
            'Recall': results.get('recall', 0),
            'F1-Score': results.get('f1_score', 0),
            'ROC-AUC': results.get('roc_auc', 'N/A'),
            'Samples': results.get('n_samples', 0)
        }
        summary_data.append(summary_row)
    
    return pd.DataFrame(summary_data)
'''

with open('./enhanced_iot_botscan/src/evaluation/performance_evaluator.py', 'w') as f:
    f.write(performance_evaluator_content)

print("✅ Created performance_evaluator.py")

# 13. Config Manager
config_manager_content = '''"""
Configuration Manager for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Manages system configuration and provides centralized config access.
"""

import yaml
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigManager:
    """Centralized configuration management."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        
        if config_path is None:
            config_path = self._find_default_config()
        
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
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
        
        issues = []
        
        # Check required sections
        required_sections = ['system', 'machine_learning', 'data']
        for section in required_sections:
            if section not in self.config:
                issues.append(f"Missing required section: {section}")
        
        # Check ML configuration
        ml_config = self.get_ml_config()
        if 'ensemble' not in ml_config:
            issues.append("Missing ensemble configuration in machine_learning section")
        
        # Check data paths
        data_config = self.get_data_config()
        if 'data_paths' in data_config:
            for dataset, path in data_config['data_paths'].items():
                if not os.path.exists(path):
                    issues.append(f"Data path does not exist: {dataset} -> {path}")
        
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
'''

with open('./enhanced_iot_botscan/src/utils/config_manager.py', 'w') as f:
    f.write(config_manager_content)

print("✅ Created config_manager.py")

print("\n🔧 Configuration and evaluation components created! We're making excellent progress...")