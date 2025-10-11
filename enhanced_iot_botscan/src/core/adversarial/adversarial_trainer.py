"""
Adversarial Trainer Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Implements adversarial training pipeline for robust model training.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.base import BaseEstimator
import joblib
from pathlib import Path

# Import attack generator
from .attack_generator import AdversarialAttackGenerator

logger = logging.getLogger(__name__)


class AdversarialTrainer:
    """Adversarial training pipeline for robust model training."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize adversarial trainer with configuration."""

        self.config = config or {}
        self.training_history = []

        # Training configuration
        self.adversarial_ratio = self.config.get('adversarial_ratio', 0.3)
        self.attack_types = self.config.get('attack_types', ['fgsm', 'pgd'])
        self.robustness_threshold = self.config.get(
            'robustness_threshold', 0.8)
        self.max_epochs = self.config.get('max_epochs', 10)
        self.early_stopping_patience = self.config.get(
            'early_stopping_patience', 3)

        # Attack configuration
        self.attack_config = self.config.get('attack_config', {
            'fgsm': {'epsilon': 0.1, 'norm': 'inf'},
            'pgd': {'epsilon': 0.1, 'alpha': 0.01, 'num_iter': 10, 'norm': 'inf'}
        })

        # Initialize attack generator
        self.attack_generator = AdversarialAttackGenerator({
            'enabled_attacks': self.attack_types,
            'attacks': self.attack_config
        })

        logger.info(
            f"AdversarialTrainer initialized with adversarial_ratio={self.adversarial_ratio}")

    def train_robust_model(self, model: BaseEstimator, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        Train a robust model using adversarial training.

        Args:
            model: Base model to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Training results
        """

        logger.info(f"Starting adversarial training on {len(X_train)} samples")

        # Initialize training state
        best_model = None
        best_robustness = 0
        patience_counter = 0
        training_results = []

        # Create a copy of the model for training
        robust_model = self._clone_model(model)

        for epoch in range(self.max_epochs):
            logger.info(
                f"Adversarial training epoch {epoch + 1}/{self.max_epochs}")

            # Generate adversarial examples
            adversarial_data = self._generate_adversarial_batch(
                robust_model, X_train, y_train
            )

            # Combine clean and adversarial data
            X_mixed, y_mixed = self._mix_clean_adversarial_data(
                X_train, y_train, adversarial_data
            )

            # Train model on mixed data
            epoch_results = self._train_epoch(
                robust_model, X_mixed, y_mixed, X_val, y_val)

            # Evaluate robustness
            robustness_eval = self.attack_generator.evaluate_robustness(
                robust_model, X_val, y_val
            )

            epoch_results['robustness'] = robustness_eval['overall_robustness']
            epoch_results['epoch'] = epoch + 1

            training_results.append(epoch_results)

            # Check if this is the best model
            if robustness_eval['overall_robustness'] > best_robustness:
                best_robustness = robustness_eval['overall_robustness']
                best_model = self._clone_model(robust_model)
                patience_counter = 0
                logger.info(f"New best robustness: {best_robustness:.4f}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            # Check if robustness threshold is met
            if robustness_eval['overall_robustness'] >= self.robustness_threshold:
                logger.info(
                    f"Robustness threshold {self.robustness_threshold} met")
                break

        # Final results
        final_results = {
            'training_history': training_results,
            'best_robustness': best_robustness,
            'final_epoch': len(training_results),
            'robustness_threshold_met': best_robustness >= self.robustness_threshold,
            'best_model': best_model,
            'adversarial_ratio': self.adversarial_ratio,
            'attack_types': self.attack_types
        }

        self.training_history.append(final_results)
        logger.info(
            f"Adversarial training completed. Best robustness: {best_robustness:.4f}")

        return final_results

    def _generate_adversarial_batch(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate adversarial examples for training batch."""

        adversarial_data = {}

        for attack_type in self.attack_types:
            try:
                # Generate adversarial examples
                attack_result = self.attack_generator.generate_single_attack(
                    attack_type, model, X, y, **self.attack_config[attack_type]
                )

                if 'error' not in attack_result:
                    adversarial_data[attack_type] = attack_result['adversarial_examples']
                    logger.debug(
                        f"Generated {len(adversarial_data[attack_type])} {attack_type} examples")

            except Exception as e:
                logger.error(f"Failed to generate {attack_type} examples: {e}")

        return adversarial_data

    def _mix_clean_adversarial_data(self, X_clean: np.ndarray, y_clean: np.ndarray,
                                    adversarial_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Mix clean and adversarial data according to adversarial ratio."""

        if not adversarial_data:
            return X_clean, y_clean

        # Calculate number of adversarial samples to use
        n_adversarial = int(len(X_clean) * self.adversarial_ratio)

        # Select adversarial examples
        X_adv_list = []
        y_adv_list = []

        for attack_type, X_adv in adversarial_data.items():
            # Use equal number from each attack type
            n_per_attack = n_adversarial // len(adversarial_data)
            if n_per_attack > 0:
                indices = np.random.choice(
                    len(X_adv), n_per_attack, replace=False)
                X_adv_list.append(X_adv[indices])
                y_adv_list.append(y_clean[indices])  # Use original labels

        if X_adv_list:
            X_adversarial = np.vstack(X_adv_list)
            y_adversarial = np.hstack(y_adv_list)

            # Combine clean and adversarial data
            X_mixed = np.vstack([X_clean, X_adversarial])
            y_mixed = np.hstack([y_clean, y_adversarial])

            # Shuffle the data
            indices = np.random.permutation(len(X_mixed))
            X_mixed = X_mixed[indices]
            y_mixed = y_mixed[indices]

            logger.debug(
                f"Mixed data: {len(X_clean)} clean + {len(X_adversarial)} adversarial")

        else:
            X_mixed = X_clean
            y_mixed = y_clean

        return X_mixed, y_mixed

    def _train_epoch(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train model for one epoch."""

        # Train model (this is a simplified version - in practice, you'd need to implement
        # proper incremental training for sklearn models)
        model.fit(X, y)

        # Evaluate on validation set
        train_accuracy = model.score(X, y)
        val_accuracy = model.score(X_val, y_val)

        return {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'n_samples': len(X)
        }

    def _clone_model(self, model: BaseEstimator) -> BaseEstimator:
        """Create a copy of the model."""

        # This is a simplified version - in practice, you'd need to implement
        # proper model cloning for different model types
        return model

    def evaluate_robustness(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate model robustness against adversarial attacks."""

        return self.attack_generator.evaluate_robustness(model, X, y)

    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history."""

        return self.training_history

    def save_training_results(self, filepath: str) -> None:
        """Save training results to disk."""

        if not self.training_history:
            raise ValueError("No training history to save")

        results_data = {
            'training_history': self.training_history,
            'config': self.config,
            'adversarial_ratio': self.adversarial_ratio,
            'attack_types': self.attack_types
        }

        joblib.dump(results_data, filepath)
        logger.info(f"Training results saved to {filepath}")

    def load_training_results(self, filepath: str) -> None:
        """Load training results from disk."""

        if not Path(filepath).exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")

        results_data = joblib.load(filepath)

        self.training_history = results_data['training_history']
        self.config = results_data['config']
        self.adversarial_ratio = results_data['adversarial_ratio']
        self.attack_types = results_data['attack_types']

        logger.info(f"Training results loaded from {filepath}")

    def compare_clean_vs_robust(self, clean_model: BaseEstimator, robust_model: BaseEstimator,
                                X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Compare clean and robust model performance."""

        logger.info("Comparing clean vs robust model performance")

        # Evaluate clean model
        clean_robustness = self.attack_generator.evaluate_robustness(
            clean_model, X_test, y_test)
        clean_accuracy = clean_model.score(X_test, y_test)

        # Evaluate robust model
        robust_robustness = self.attack_generator.evaluate_robustness(
            robust_model, X_test, y_test)
        robust_accuracy = robust_model.score(X_test, y_test)

        comparison = {
            'clean_model': {
                'accuracy': clean_accuracy,
                'robustness': clean_robustness['overall_robustness'],
                'attack_metrics': clean_robustness['attack_metrics']
            },
            'robust_model': {
                'accuracy': robust_accuracy,
                'robustness': robust_robustness['overall_robustness'],
                'attack_metrics': robust_robustness['attack_metrics']
            },
            'improvements': {
                'robustness_gain': robust_robustness['overall_robustness'] - clean_robustness['overall_robustness'],
                'accuracy_change': robust_accuracy - clean_accuracy
            }
        }

        logger.info(
            f"Robustness improvement: {comparison['improvements']['robustness_gain']:.4f}")
        logger.info(
            f"Accuracy change: {comparison['improvements']['accuracy_change']:.4f}")

        return comparison


# Example usage and testing
if __name__ == '__main__':
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Create labels
    y = pd.Series(
        (X.iloc[:, 0] + X.iloc[:, 1] +
         np.random.randn(n_samples) * 0.1 > 0).astype(int)
    )

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    # Create base model
    from sklearn.linear_model import LogisticRegression
    base_model = LogisticRegression(random_state=42)

    print("Original model accuracy:", base_model.score(X_test, y_test))

    # Initialize adversarial trainer
    trainer = AdversarialTrainer({
        'adversarial_ratio': 0.3,
        'attack_types': ['fgsm', 'pgd'],
        'robustness_threshold': 0.7,
        'max_epochs': 5,
        'early_stopping_patience': 2
    })

    # Train robust model
    training_results = trainer.train_robust_model(
        base_model, X_train.values, y_train.values,
        X_val.values, y_val.values
    )

    print(f"\nAdversarial training completed:")
    print(f"Best robustness: {training_results['best_robustness']:.4f}")
    print(f"Final epoch: {training_results['final_epoch']}")
    print(f"Threshold met: {training_results['robustness_threshold_met']}")

    # Compare clean vs robust
    if training_results['best_model'] is not None:
        comparison = trainer.compare_clean_vs_robust(
            base_model, training_results['best_model'],
            X_test.values, y_test.values
        )

        print("\nClean vs Robust Comparison:")
        print(
            f"Clean model robustness: {comparison['clean_model']['robustness']:.4f}")
        print(
            f"Robust model robustness: {comparison['robust_model']['robustness']:.4f}")
        print(
            f"Robustness improvement: {comparison['improvements']['robustness_gain']:.4f}")
        print(
            f"Accuracy change: {comparison['improvements']['accuracy_change']:.4f}")

    # Get training history
    history = trainer.get_training_history()
    print(f"\nTraining history: {len(history)} training sessions")
