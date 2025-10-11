"""
FGSM Attack Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Implements Fast Gradient Sign Method (FGSM) adversarial attack for testing model robustness.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.base import BaseEstimator
import warnings

logger = logging.getLogger(__name__)


class FGSMAttack:
    """Fast Gradient Sign Method (FGSM) adversarial attack implementation."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize FGSM attack with configuration."""

        self.config = config or {}
        self.epsilon = self.config.get('epsilon', 0.1)
        self.norm = self.config.get('norm', 'inf')
        self.targeted = self.config.get('targeted', False)
        self.random_start = self.config.get('random_start', False)

        logger.info(
            f"FGSMAttack initialized with epsilon={self.epsilon}, norm={self.norm}")

    def generate_attack(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                        target_labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate FGSM adversarial examples.

        Args:
            model: Target model to attack
            X: Input features
            y: True labels
            target_labels: Target labels for targeted attack

        Returns:
            Adversarial examples
        """

        logger.info(
            f"Generating FGSM attack on {len(X)} samples with epsilon={self.epsilon}")

        # Convert to PyTorch tensors for gradient computation
        X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)

        # Create a wrapper for sklearn models
        model_wrapper = SklearnModelWrapper(model)

        # Forward pass
        logits = model_wrapper(X_tensor)

        # Compute loss
        if self.targeted and target_labels is not None:
            # Targeted attack: minimize loss for target class
            target_tensor = torch.tensor(target_labels, dtype=torch.long)
            loss = nn.CrossEntropyLoss()(logits, target_tensor)
        else:
            # Untargeted attack: maximize loss for true class
            true_tensor = torch.tensor(y, dtype=torch.long)
            loss = -nn.CrossEntropyLoss()(logits, true_tensor)

        # Compute gradients
        loss.backward()

        # Generate adversarial perturbation
        if self.norm == 'inf':
            # L-infinity norm
            perturbation = self.epsilon * X_tensor.grad.sign()
        elif self.norm == 2:
            # L2 norm
            grad_norm = torch.norm(X_tensor.grad, dim=1, keepdim=True)
            perturbation = self.epsilon * X_tensor.grad / (grad_norm + 1e-8)
        else:
            raise ValueError(f"Unsupported norm: {self.norm}")

        # Add perturbation
        X_adv = X_tensor + perturbation

        # Clip to valid range
        X_adv = torch.clamp(X_adv, 0, 1)

        return X_adv.detach().numpy()

    def evaluate_attack(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                        X_adv: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of the attack.

        Args:
            model: Target model
            X: Original features
            y: True labels
            X_adv: Adversarial examples

        Returns:
            Attack evaluation results
        """

        # Original predictions
        y_pred_orig = model.predict(X)
        y_pred_adv = model.predict(X_adv)

        # Calculate attack success rate
        if self.targeted:
            # For targeted attack, success is when prediction matches target
            success_rate = np.mean(y_pred_adv == self.target_labels)
        else:
            # For untargeted attack, success is when prediction changes
            success_rate = np.mean(y_pred_orig != y_pred_adv)

        # Calculate accuracy drop
        orig_accuracy = np.mean(y_pred_orig == y)
        adv_accuracy = np.mean(y_pred_adv == y)
        accuracy_drop = orig_accuracy - adv_accuracy

        # Calculate perturbation magnitude
        perturbation = X_adv - X
        if self.norm == 'inf':
            perturbation_norm = np.max(np.abs(perturbation), axis=1)
        elif self.norm == 2:
            perturbation_norm = np.linalg.norm(perturbation, axis=1)
        else:
            perturbation_norm = np.linalg.norm(perturbation, axis=1)

        results = {
            'attack_type': 'FGSM',
            'epsilon': self.epsilon,
            'norm': self.norm,
            'targeted': self.targeted,
            'success_rate': success_rate,
            'original_accuracy': orig_accuracy,
            'adversarial_accuracy': adv_accuracy,
            'accuracy_drop': accuracy_drop,
            'mean_perturbation_norm': np.mean(perturbation_norm),
            'max_perturbation_norm': np.max(perturbation_norm),
            'n_samples': len(X)
        }

        logger.info(f"FGSM attack evaluation: success_rate={success_rate:.4f}, "
                    f"accuracy_drop={accuracy_drop:.4f}")

        return results


class SklearnModelWrapper(nn.Module):
    """Wrapper to make sklearn models compatible with PyTorch for gradient computation."""

    def __init__(self, sklearn_model: BaseEstimator):
        super().__init__()
        self.sklearn_model = sklearn_model

        # Extract model parameters for gradient computation
        if hasattr(sklearn_model, 'coef_'):
            self.coef = nn.Parameter(torch.tensor(
                sklearn_model.coef_, dtype=torch.float32))
        if hasattr(sklearn_model, 'intercept_'):
            self.intercept = nn.Parameter(torch.tensor(
                sklearn_model.intercept_, dtype=torch.float32))

    def forward(self, x):
        """Forward pass through the model."""

        if hasattr(self, 'coef'):
            # Linear model
            if len(self.coef.shape) == 1:
                # Binary classification
                logits = torch.matmul(x, self.coef) + self.intercept
                return torch.stack([-logits, logits], dim=1)
            else:
                # Multi-class classification
                logits = torch.matmul(x, self.coef.T) + self.intercept
                return logits
        else:
            # For tree-based models, we need to approximate with a neural network
            # This is a simplified approximation
            with torch.no_grad():
                predictions = self.sklearn_model.predict_proba(x.numpy())
            return torch.tensor(predictions, dtype=torch.float32)


class FGSMAttackGenerator:
    """Generator for FGSM attacks with multiple configurations."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize FGSM attack generator."""

        self.config = config or {}
        self.epsilon_range = self.config.get(
            'epsilon_range', [0.01, 0.05, 0.1, 0.2, 0.3])
        self.norms = self.config.get('norms', ['inf', '2'])

        logger.info("FGSMAttackGenerator initialized")

    def generate_multiple_attacks(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Generate FGSM attacks with multiple configurations.

        Args:
            model: Target model
            X: Input features
            y: True labels

        Returns:
            Dictionary of attack results
        """

        results = {}

        for epsilon in self.epsilon_range:
            for norm in self.norms:
                attack_name = f"FGSM_eps{epsilon}_norm{norm}"

                try:
                    # Create attack
                    attack = FGSMAttack({
                        'epsilon': epsilon,
                        'norm': norm,
                        'targeted': False
                    })

                    # Generate adversarial examples
                    X_adv = attack.generate_attack(model, X, y)

                    # Evaluate attack
                    eval_results = attack.evaluate_attack(model, X, y, X_adv)

                    results[attack_name] = {
                        'adversarial_examples': X_adv,
                        'evaluation': eval_results
                    }

                    logger.info(
                        f"Generated {attack_name}: success_rate={eval_results['success_rate']:.4f}")

                except Exception as e:
                    logger.error(f"Failed to generate {attack_name}: {e}")
                    results[attack_name] = {'error': str(e)}

        return results

    def find_minimal_epsilon(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                             target_success_rate: float = 0.5) -> float:
        """
        Find minimal epsilon that achieves target success rate.

        Args:
            model: Target model
            X: Input features
            y: True labels
            target_success_rate: Target attack success rate

        Returns:
            Minimal epsilon value
        """

        logger.info(
            f"Finding minimal epsilon for success rate {target_success_rate}")

        # Binary search for minimal epsilon
        low, high = 0.001, 1.0
        tolerance = 0.01

        while high - low > tolerance:
            mid = (low + high) / 2

            try:
                attack = FGSMAttack({'epsilon': mid, 'norm': 'inf'})
                X_adv = attack.generate_attack(model, X, y)
                eval_results = attack.evaluate_attack(model, X, y, X_adv)

                if eval_results['success_rate'] >= target_success_rate:
                    high = mid
                else:
                    low = mid

            except Exception as e:
                logger.error(f"Error in binary search: {e}")
                break

        minimal_epsilon = (low + high) / 2
        logger.info(f"Minimal epsilon: {minimal_epsilon:.4f}")

        return minimal_epsilon


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

    # Create a simple model
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    print("Original model accuracy:", model.score(X, y))

    # Test FGSM attack
    attack = FGSMAttack({'epsilon': 0.1, 'norm': 'inf'})
    X_adv = attack.generate_attack(model, X.values, y.values)

    # Evaluate attack
    results = attack.evaluate_attack(model, X.values, y.values, X_adv)
    print("\nFGSM Attack Results:")
    print(f"Success Rate: {results['success_rate']:.4f}")
    print(f"Accuracy Drop: {results['accuracy_drop']:.4f}")
    print(f"Mean Perturbation Norm: {results['mean_perturbation_norm']:.4f}")

    # Test multiple attacks
    generator = FGSMAttackGenerator({
        'epsilon_range': [0.05, 0.1, 0.2],
        'norms': ['inf']
    })

    multiple_results = generator.generate_multiple_attacks(
        model, X.values, y.values)
    print("\nMultiple Attack Results:")
    for attack_name, result in multiple_results.items():
        if 'error' not in result:
            eval_results = result['evaluation']
            print(f"{attack_name}: success_rate={eval_results['success_rate']:.4f}, "
                  f"accuracy_drop={eval_results['accuracy_drop']:.4f}")

    # Find minimal epsilon
    minimal_eps = generator.find_minimal_epsilon(
        model, X.values, y.values, target_success_rate=0.3)
    print(f"\nMinimal epsilon for 30% success rate: {minimal_eps:.4f}")
