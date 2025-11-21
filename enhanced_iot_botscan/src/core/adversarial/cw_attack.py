"""
Carlini & Wagner Attack Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Implements Carlini & Wagner (C&W) adversarial attack for testing model robustness.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.base import BaseEstimator
import warnings

logger = logging.getLogger(__name__)


class CWAttack:
    """Carlini & Wagner (C&W) adversarial attack implementation."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize C&W attack with configuration."""

        self.config = config or {}
        self.c = self.config.get('c', 1.0)
        self.kappa = self.config.get('kappa', 0.0)
        self.max_iter = self.config.get('max_iter', 1000)
        self.lr = self.config.get('lr', 0.01)
        self.binary_search_steps = self.config.get('binary_search_steps', 9)
        self.targeted = self.config.get('targeted', False)
        self.norm = self.config.get('norm', '2')

        logger.info(f"CWAttack initialized with c={self.c}, kappa={self.kappa}, "
                    f"max_iter={self.max_iter}, norm={self.norm}")

    def generate_attack(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                        target_labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate C&W adversarial examples.

        Args:
            model: Target model to attack
            X: Input features
            y: True labels
            target_labels: Target labels for targeted attack

        Returns:
            Adversarial examples
        """

        logger.info(f"Generating C&W attack on {len(X)} samples")

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        # Create a wrapper for sklearn models
        model_wrapper = SklearnModelWrapper(model)

        # Initialize adversarial examples
        X_adv = torch.zeros_like(X_tensor)

        # Process each sample
        for i in range(len(X)):
            X_adv[i] = self._attack_single_sample(
                model_wrapper, X_tensor[i], y_tensor[i],
                target_labels[i] if target_labels is not None else None
            )

        return X_adv.detach().numpy()

    def _attack_single_sample(self, model_wrapper: nn.Module, x: torch.Tensor, y: torch.Tensor,
                              target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Attack a single sample using C&W method."""

        # Initialize perturbation
        if self.norm == '2':
            # L2 norm: initialize with small random perturbation
            delta = torch.randn_like(x) * 0.01
        else:
            # L-infinity norm: initialize with small random perturbation
            delta = torch.rand_like(x) * 0.01 - 0.005

        delta.requires_grad = True

        # Binary search for optimal c
        c_low = 0.0
        c_high = 1.0

        for _ in range(self.binary_search_steps):
            c = (c_low + c_high) / 2

            # Optimize perturbation
            optimizer = optim.Adam([delta], lr=self.lr)

            for iteration in range(self.max_iter):
                optimizer.zero_grad()

                # Compute adversarial example
                x_adv = x + delta

                # Ensure x_adv is in valid range
                x_adv = torch.clamp(x_adv, 0, 1)

                # Forward pass
                logits = model_wrapper(x_adv.unsqueeze(0))

                # Compute loss
                if self.targeted and target is not None:
                    # Targeted attack: minimize loss for target class
                    target_loss = nn.CrossEntropyLoss()(logits, target.unsqueeze(0))
                    loss = target_loss + c * self._distance_loss(x, x_adv)
                else:
                    # Untargeted attack: maximize loss for true class
                    true_loss = nn.CrossEntropyLoss()(logits, y.unsqueeze(0))
                    loss = -true_loss + c * self._distance_loss(x, x_adv)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Check if attack is successful
                with torch.no_grad():
                    x_adv_clamped = torch.clamp(x + delta, 0, 1)
                    logits_check = model_wrapper(x_adv_clamped.unsqueeze(0))
                    pred = torch.argmax(logits_check, dim=1)

                    if self.targeted and target is not None:
                        success = pred.item() == target.item()
                    else:
                        success = pred.item() != y.item()

                    if success:
                        break

            # Update binary search bounds
            if success:
                c_high = c
            else:
                c_low = c

        # Return final adversarial example
        with torch.no_grad():
            x_adv_final = torch.clamp(x + delta, 0, 1)

        return x_adv_final

    def _distance_loss(self, x_orig: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
        """Compute distance loss between original and adversarial examples."""

        if self.norm == '2':
            # L2 distance
            return torch.norm(x_adv - x_orig, p=2)
        elif self.norm == 'inf':
            # L-infinity distance
            return torch.max(torch.abs(x_adv - x_orig))
        else:
            raise ValueError(f"Unsupported norm: {self.norm}")

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
        if self.norm == '2':
            perturbation_norm = np.linalg.norm(perturbation, axis=1)
        elif self.norm == 'inf':
            perturbation_norm = np.max(np.abs(perturbation), axis=1)
        else:
            perturbation_norm = np.linalg.norm(perturbation, axis=1)

        results = {
            'attack_type': 'C&W',
            'c': self.c,
            'kappa': self.kappa,
            'max_iter': self.max_iter,
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

        logger.info(f"C&W attack evaluation: success_rate={success_rate:.4f}, "
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
            if len(self.coef.shape) == 1 or (len(self.coef.shape) == 2 and self.coef.shape[0] == 1):
                # Binary classification
                if len(self.coef.shape) == 2:
                    logits = torch.matmul(x, self.coef.T) + self.intercept
                else:
                    logits = torch.matmul(x, self.coef) + self.intercept
                return torch.cat([-logits, logits], dim=1)
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


class CWAttackGenerator:
    """Generator for C&W attacks with multiple configurations."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize C&W attack generator."""

        self.config = config or {}
        self.c_range = self.config.get('c_range', [0.1, 0.5, 1.0, 2.0, 5.0])
        self.max_iter_range = self.config.get(
            'max_iter_range', [100, 500, 1000])
        self.norms = self.config.get('norms', ['2', 'inf'])

        logger.info("CWAttackGenerator initialized")

    def generate_multiple_attacks(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Generate C&W attacks with multiple configurations.

        Args:
            model: Target model
            X: Input features
            y: True labels

        Returns:
            Dictionary of attack results
        """

        results = {}

        for c in self.c_range:
            for max_iter in self.max_iter_range:
                for norm in self.norms:
                    attack_name = f"CW_c{c}_iter{max_iter}_norm{norm}"

                    try:
                        # Create attack
                        attack = CWAttack({
                            'c': c,
                            'max_iter': max_iter,
                            'norm': norm,
                            'targeted': False,
                            'lr': 0.01
                        })

                        # Generate adversarial examples
                        X_adv = attack.generate_attack(model, X, y)

                        # Evaluate attack
                        eval_results = attack.evaluate_attack(
                            model, X, y, X_adv)

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

    def find_optimal_c(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                       target_success_rate: float = 0.5) -> float:
        """
        Find optimal c parameter for target success rate.

        Args:
            model: Target model
            X: Input features
            y: True labels
            target_success_rate: Target attack success rate

        Returns:
            Optimal c value
        """

        logger.info(
            f"Finding optimal c for success rate {target_success_rate}")

        # Binary search for optimal c
        low, high = 0.01, 10.0
        tolerance = 0.1

        while high - low > tolerance:
            mid = (low + high) / 2

            try:
                attack = CWAttack({
                    'c': mid,
                    'max_iter': 500,
                    'norm': '2',
                    'targeted': False
                })

                X_adv = attack.generate_attack(model, X, y)
                eval_results = attack.evaluate_attack(model, X, y, X_adv)

                if eval_results['success_rate'] >= target_success_rate:
                    high = mid
                else:
                    low = mid

            except Exception as e:
                logger.error(f"Error in binary search: {e}")
                break

        optimal_c = (low + high) / 2
        logger.info(f"Optimal c: {optimal_c:.4f}")

        return optimal_c

    def compare_attack_methods(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Compare C&W with other attack methods.

        Args:
            model: Target model
            X: Input features
            y: True labels

        Returns:
            Comparison results
        """

        logger.info("Comparing C&W with other attack methods")

        # FGSM attack
        from .fgsm_attack import FGSMAttack

        fgsm_attack = FGSMAttack({'epsilon': 0.1, 'norm': 'inf'})
        X_adv_fgsm = fgsm_attack.generate_attack(model, X, y)
        fgsm_results = fgsm_attack.evaluate_attack(model, X, y, X_adv_fgsm)

        # PGD attack
        from .pgd_attack import PGDAttack

        pgd_attack = PGDAttack({
            'epsilon': 0.1,
            'alpha': 0.01,
            'num_iter': 10,
            'norm': 'inf'
        })
        X_adv_pgd = pgd_attack.generate_attack(model, X, y)
        pgd_results = pgd_attack.evaluate_attack(model, X, y, X_adv_pgd)

        # C&W attack
        cw_attack = CWAttack({
            'c': 1.0,
            'max_iter': 500,
            'norm': '2',
            'targeted': False
        })
        X_adv_cw = cw_attack.generate_attack(model, X, y)
        cw_results = cw_attack.evaluate_attack(model, X, y, X_adv_cw)

        comparison = {
            'fgsm': fgsm_results,
            'pgd': pgd_results,
            'cw': cw_results,
            'ranking': {
                'success_rate': sorted([
                    ('FGSM', fgsm_results['success_rate']),
                    ('PGD', pgd_results['success_rate']),
                    ('C&W', cw_results['success_rate'])
                ], key=lambda x: x[1], reverse=True),
                'accuracy_drop': sorted([
                    ('FGSM', fgsm_results['accuracy_drop']),
                    ('PGD', pgd_results['accuracy_drop']),
                    ('C&W', cw_results['accuracy_drop'])
                ], key=lambda x: x[1], reverse=True)
            }
        }

        logger.info(f"Attack comparison: FGSM={fgsm_results['success_rate']:.4f}, "
                    f"PGD={pgd_results['success_rate']:.4f}, C&W={cw_results['success_rate']:.4f}")

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

    # Create a simple model
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    print("Original model accuracy:", model.score(X, y))

    # Test C&W attack
    attack = CWAttack({
        'c': 1.0,
        'max_iter': 500,
        'norm': '2',
        'targeted': False
    })
    X_adv = attack.generate_attack(model, X.values, y.values)

    # Evaluate attack
    results = attack.evaluate_attack(model, X.values, y.values, X_adv)
    print("\nC&W Attack Results:")
    print(f"Success Rate: {results['success_rate']:.4f}")
    print(f"Accuracy Drop: {results['accuracy_drop']:.4f}")
    print(f"Mean Perturbation Norm: {results['mean_perturbation_norm']:.4f}")

    # Test multiple attacks
    generator = CWAttackGenerator({
        'c_range': [0.5, 1.0, 2.0],
        'max_iter_range': [100, 500],
        'norms': ['2']
    })

    multiple_results = generator.generate_multiple_attacks(
        model, X.values, y.values)
    print("\nMultiple Attack Results:")
    for attack_name, result in multiple_results.items():
        if 'error' not in result:
            eval_results = result['evaluation']
            print(f"{attack_name}: success_rate={eval_results['success_rate']:.4f}, "
                  f"accuracy_drop={eval_results['accuracy_drop']:.4f}")

    # Find optimal c
    optimal_c = generator.find_optimal_c(
        model, X.values, y.values, target_success_rate=0.3)
    print(f"\nOptimal c for 30% success rate: {optimal_c:.4f}")

    # Compare attack methods
    comparison = generator.compare_attack_methods(model, X.values, y.values)
    print("\nAttack Method Comparison:")
    print("Success Rate Ranking:")
    for method, rate in comparison['ranking']['success_rate']:
        print(f"  {method}: {rate:.4f}")
    print("Accuracy Drop Ranking:")
    for method, drop in comparison['ranking']['accuracy_drop']:
        print(f"  {method}: {drop:.4f}")
