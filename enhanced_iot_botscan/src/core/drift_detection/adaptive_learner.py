"""
Adaptive Learner Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Implements adaptive learning for handling concept drift in streaming data.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.base import BaseEstimator
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class AdaptiveLearner:
    """Adaptive learning system for handling concept drift."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize adaptive learner with configuration."""

        self.config = config or {}
        self.learning_history = []

        # Learning configuration
        self.learning_rate = self.config.get('learning_rate', 0.1)
        self.forgetting_factor = self.config.get('forgetting_factor', 0.95)
        self.min_samples_for_update = self.config.get(
            'min_samples_for_update', 100)
        self.update_frequency = self.config.get('update_frequency', 1000)
        self.performance_threshold = self.config.get(
            'performance_threshold', 0.8)

        # State variables
        self.model = None
        self.is_initialized = False
        self.sample_count = 0
        self.performance_history = []
        self.drift_detected = False

        logger.info(
            f"AdaptiveLearner initialized with learning_rate={self.learning_rate}")

    def initialize(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> None:
        """
        Initialize the adaptive learner with a base model.

        Args:
            model: Base model to adapt
            X: Initial training data
            y: Initial training labels
        """

        if len(X) < self.min_samples_for_update:
            raise ValueError(
                f"Initial data must have at least {self.min_samples_for_update} samples")

        # Store the base model
        self.model = model

        # Initial training
        self.model.fit(X, y)

        # Initialize state
        self.is_initialized = True
        self.sample_count = len(X)
        self.performance_history = [self.model.score(X, y)]

        logger.info(f"AdaptiveLearner initialized with {len(X)} samples")

    def update(self, X_new: np.ndarray, y_new: np.ndarray,
               drift_detected: bool = False) -> Dict[str, Any]:
        """
        Update the model with new data.

        Args:
            X_new: New data
            y_new: New labels
            drift_detected: Whether concept drift was detected

        Returns:
            Update results
        """

        if not self.is_initialized:
            raise ValueError(
                "AdaptiveLearner must be initialized before updating")

        logger.info(
            f"Updating model with {len(X_new)} new samples, drift_detected={drift_detected}")

        # Update sample count
        self.sample_count += len(X_new)

        # Determine update strategy based on drift detection
        if drift_detected:
            update_strategy = 'drift_adaptation'
            update_results = self._handle_drift_adaptation(X_new, y_new)
        else:
            update_strategy = 'incremental_update'
            update_results = self._handle_incremental_update(X_new, y_new)

        # Calculate performance
        performance = self.model.score(X_new, y_new)
        self.performance_history.append(performance)

        # Create update results
        results = {
            'update_strategy': update_strategy,
            'performance': performance,
            'sample_count': self.sample_count,
            'drift_detected': drift_detected,
            'update_results': update_results,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        # Store in history
        self.learning_history.append(results)

        logger.info(f"Model update completed: performance={performance:.4f}")

        return results

    def _handle_drift_adaptation(self, X_new: np.ndarray, y_new: np.ndarray) -> Dict[str, Any]:
        """Handle model adaptation when drift is detected."""

        logger.info("Handling drift adaptation")

        # Strategy 1: Retrain with recent data
        if len(X_new) >= self.min_samples_for_update:
            # Retrain model with recent data
            self.model.fit(X_new, y_new)

            return {
                'adaptation_type': 'retrain_recent',
                'n_samples_used': len(X_new),
                'adaptation_success': True
            }

        # Strategy 2: Incremental update with higher learning rate
        else:
            # Use higher learning rate for incremental update
            original_learning_rate = self.learning_rate
            self.learning_rate = min(1.0, self.learning_rate * 2)

            try:
                # Perform incremental update
                self._incremental_fit(X_new, y_new)

                return {
                    'adaptation_type': 'incremental_high_lr',
                    'learning_rate_used': self.learning_rate,
                    'adaptation_success': True
                }
            finally:
                # Restore original learning rate
                self.learning_rate = original_learning_rate

    def _handle_incremental_update(self, X_new: np.ndarray, y_new: np.ndarray) -> Dict[str, Any]:
        """Handle incremental model update."""

        logger.info("Handling incremental update")

        # Check if we have enough samples for update
        if len(X_new) < self.min_samples_for_update:
            return {
                'adaptation_type': 'insufficient_samples',
                'n_samples': len(X_new),
                'adaptation_success': False
            }

        # Perform incremental update
        self._incremental_fit(X_new, y_new)

        return {
            'adaptation_type': 'incremental',
            'n_samples_used': len(X_new),
            'adaptation_success': True
        }

    def _incremental_fit(self, X_new: np.ndarray, y_new: np.ndarray) -> None:
        """Perform incremental fitting of the model."""

        # This is a simplified version - in practice, you'd need to implement
        # proper incremental learning for different model types

        if hasattr(self.model, 'partial_fit'):
            # Use partial_fit if available
            self.model.partial_fit(X_new, y_new)
        else:
            # For models without partial_fit, we need to retrain
            # In practice, you'd implement proper incremental learning
            logger.warning(
                "Model does not support incremental learning, retraining")
            self.model.fit(X_new, y_new)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the current model."""

        if not self.is_initialized:
            raise ValueError(
                "AdaptiveLearner must be initialized before making predictions")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities using the current model."""

        if not self.is_initialized:
            raise ValueError(
                "AdaptiveLearner must be initialized before making predictions")

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba, return hard predictions
            predictions = self.model.predict(X)
            # Convert to probabilities (simplified)
            proba = np.zeros((len(predictions), 2))
            proba[np.arange(len(predictions)), predictions] = 1
            return proba

    def get_performance_history(self) -> List[float]:
        """Get performance history."""

        return self.performance_history.copy()

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics."""

        if not self.learning_history:
            return {'n_updates': 0}

        # Calculate statistics
        performances = [result['performance']
                        for result in self.learning_history]
        drift_detections = [result['drift_detected']
                            for result in self.learning_history]

        # Calculate performance trends
        if len(performances) > 1:
            performance_trend = np.polyfit(
                range(len(performances)), performances, 1)[0]
        else:
            performance_trend = 0

        statistics = {
            'n_updates': len(self.learning_history),
            'sample_count': self.sample_count,
            'mean_performance': np.mean(performances),
            'std_performance': np.std(performances),
            'min_performance': np.min(performances),
            'max_performance': np.max(performances),
            'performance_trend': performance_trend,
            'drift_detection_rate': np.mean(drift_detections),
            'last_update': self.learning_history[-1]['timestamp'] if self.learning_history else None
        }

        return statistics

    def save_model(self, filepath: str) -> None:
        """Save the current model to disk."""

        if not self.is_initialized:
            raise ValueError("Cannot save uninitialized model")

        model_data = {
            'model': self.model,
            'learning_history': self.learning_history,
            'performance_history': self.performance_history,
            'sample_count': self.sample_count,
            'config': self.config,
            'is_initialized': self.is_initialized
        }

        joblib.dump(model_data, filepath)
        logger.info(f"AdaptiveLearner model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a saved model from disk."""

        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.learning_history = model_data['learning_history']
        self.performance_history = model_data['performance_history']
        self.sample_count = model_data['sample_count']
        self.config = model_data['config']
        self.is_initialized = model_data['is_initialized']

        logger.info(f"AdaptiveLearner model loaded from {filepath}")

    def reset_learner(self) -> None:
        """Reset the adaptive learner."""

        self.model = None
        self.is_initialized = False
        self.sample_count = 0
        self.performance_history = []
        self.learning_history = []
        self.drift_detected = False

        logger.info("AdaptiveLearner reset")


# Example usage and testing
if __name__ == '__main__':
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # Initial data
    X_initial = np.random.normal(0, 1, (500, n_features))
    y_initial = np.random.randint(0, 2, 500)

    # Streaming data with drift
    X_stream = []
    y_stream = []

    for i in range(5):
        # Gradually introduce drift
        drift_amount = i * 0.5
        X_batch = np.random.normal(drift_amount, 1, (100, n_features))
        y_batch = np.random.randint(0, 2, 100)

        X_stream.append(X_batch)
        y_stream.append(y_batch)

    # Create base model
    from sklearn.linear_model import SGDClassifier
    base_model = SGDClassifier(random_state=42)

    # Initialize adaptive learner
    learner = AdaptiveLearner({
        'learning_rate': 0.1,
        'min_samples_for_update': 50,
        'performance_threshold': 0.7
    })

    # Initialize with initial data
    learner.initialize(base_model, X_initial, y_initial)

    print("Initial performance:", learner.performance_history[0])

    # Process streaming data
    for i, (X_batch, y_batch) in enumerate(zip(X_stream, y_stream)):
        # Simulate drift detection (detect drift after batch 2)
        drift_detected = i >= 2

        # Update model
        update_results = learner.update(
            X_batch, y_batch, drift_detected=drift_detected)

        print(f"Batch {i+1}: performance={update_results['performance']:.4f}, "
              f"strategy={update_results['update_strategy']}, "
              f"drift_detected={drift_detected}")

    # Get learning statistics
    stats = learner.get_learning_statistics()
    print(f"\nLearning Statistics:")
    print(f"Total updates: {stats['n_updates']}")
    print(f"Sample count: {stats['sample_count']}")
    print(f"Mean performance: {stats['mean_performance']:.4f}")
    print(f"Performance trend: {stats['performance_trend']:.4f}")
    print(f"Drift detection rate: {stats['drift_detection_rate']:.4f}")

    # Test predictions
    X_test = np.random.normal(1, 1, (100, n_features))
    y_test = np.random.randint(0, 2, 100)

    predictions = learner.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"\nTest accuracy: {accuracy:.4f}")

    # Reset learner
    learner.reset_learner()
    print("\nLearner reset completed")
