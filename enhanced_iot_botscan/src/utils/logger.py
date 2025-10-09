"""
Logger Configuration for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Provides structured logging functionality across the system.
"""

import logging
import logging.config
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

def setup_logging(config: Dict[str, Any] = None) -> None:
    """Setup logging configuration for the system."""

    if config is None:
        config = get_default_logging_config()

    # Create logs directory
    log_dir = Path('./logs')
    log_dir.mkdir(exist_ok=True)

    # Configure logging
    logging.config.dictConfig(config)

    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("Enhanced IoT BotScan logging system initialized")

def get_default_logging_config() -> Dict[str, Any]:
    """Get default logging configuration."""

    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s (%(filename)s:%(funcName)s)',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'json': {
                'format': '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s", "module": "%(module)s", "function": "%(funcName)s", "line": %(lineno)d}',
                'datefmt': '%Y-%m-%dT%H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'standard',
                'stream': sys.stdout
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'detailed',
                'filename': './logs/iot_botscan.log',
                'maxBytes': 104857600,  # 100MB
                'backupCount': 10,
                'encoding': 'utf8'
            },
            'error_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'detailed',
                'filename': './logs/iot_botscan_errors.log',
                'maxBytes': 104857600,  # 100MB
                'backupCount': 5,
                'encoding': 'utf8'
            }
        },
        'loggers': {
            '': {  # Root logger
                'level': 'INFO',
                'handlers': ['console', 'file', 'error_file'],
                'propagate': False
            }
        }
    }

def get_logger(name: str) -> logging.Logger:
    """Get logger instance with specified name."""
    return logging.getLogger(name)

class StructuredLogger:
    """Structured logger for enhanced logging capabilities."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def log_training_start(self, dataset: str, model: str, samples: int):
        """Log training start event."""
        self.logger.info(f"Training started - Dataset: {dataset}, Model: {model}, Samples: {samples}")

    def log_training_complete(self, accuracy: float, training_time: float):
        """Log training completion event."""
        self.logger.info(f"Training completed - Accuracy: {accuracy:.4f}, Time: {training_time:.2f}s")

    def log_drift_detection(self, method: str, drift_detected: bool, p_value: float = None):
        """Log drift detection event."""
        status = "DETECTED" if drift_detected else "NOT DETECTED"
        message = f"Drift {status} - Method: {method}"
        if p_value is not None:
            message += f", P-value: {p_value:.6f}"

        if drift_detected:
            self.logger.warning(message)
        else:
            self.logger.info(message)

    def log_adversarial_attack(self, attack_type: str, success_rate: float, samples: int):
        """Log adversarial attack results."""
        self.logger.info(f"Adversarial attack - Type: {attack_type}, Success rate: {success_rate:.3f}, Samples: {samples}")

    def log_performance_metrics(self, metrics: Dict[str, float]):
        """Log performance metrics."""
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Performance metrics - {metrics_str}")

    def log_dataset_loaded(self, dataset_name: str, samples: int, features: int, classes: int):
        """Log dataset loading event."""
        self.logger.info(f"Dataset loaded - {dataset_name}: {samples} samples, {features} features, {classes} classes")

    def log_model_saved(self, model_path: str, model_type: str):
        """Log model save event."""
        self.logger.info(f"Model saved - Type: {model_type}, Path: {model_path}")

    def log_error(self, error: Exception, context: str = None):
        """Log error with context."""
        message = f"Error occurred: {str(error)}"
        if context:
            message = f"{context} - {message}"
        self.logger.error(message, exc_info=True)
