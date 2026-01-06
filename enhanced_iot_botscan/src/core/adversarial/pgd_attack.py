"""
PGD Attack Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Implements Projected Gradient Descent (PGD) adversarial attack for testing model robustness.
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


class PGDAttack:
    """Projected Gradient Descent (PGD) adversarial attack implementation."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize PGD attack with configuration."""

        self.config = config or {}
        self.epsilon = self.config.get('epsilon', 0.1)
        self.alpha = self.config.get('alpha', 0.01)
        self.num_iter = self.config.get('num_iter', 10)
        self.norm = str(self.config.get('norm', 'inf'))  # Convert to string
        self.targeted = self.config.get('targeted', False)
        self.random_start = self.config.get('random_start', True)

        logger.info(f"PGDAttack initialized with epsilon={self.epsilon}, alpha={self.alpha}, "
                    f"iterations={self.num_iter}, norm={self.norm}")

    def generate_attack(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                        target_labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate PGD adversarial examples.

        Args:
            model: Target model to attack
            X: Input features
            y: True labels
            target_labels: Target labels for targeted attack

        Returns:
            Adversarial examples
        """

        logger.info(
            f"Generating PGD attack on {len(X)} samples with {self.num_iter} iterations")

        # Handle DataFrame/Series input
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values

        # Detect number of classes from model
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                sample_proba = model.predict_proba(X[:1])
                num_classes = sample_proba.shape[1]
        except Exception:
            num_classes = 2  # Default to binary
        
        # Remap labels to valid range if they exceed num_classes
        y = np.array(y)
        if y.max() >= num_classes:
            logger.warning(f"Labels (max={y.max()}) exceed model classes ({num_classes}). Remapping labels.")
            y = y % num_classes  # Wrap labels to valid range
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)

        # Create a wrapper for sklearn models
        model_wrapper = SklearnModelWrapper(model)

        # Initialize adversarial examples
        if self.random_start:
            # Random initialization within epsilon ball
            if self.norm == 'inf':
                noise = torch.rand_like(X_tensor) * 2 * \
                    self.epsilon - self.epsilon
            else:
                noise = torch.randn_like(X_tensor)
                noise = noise / \
                    torch.norm(noise, dim=1, keepdim=True) * self.epsilon
            X_adv = X_tensor + noise
        else:
            X_adv = X_tensor.clone()

        # PGD iterations
        for i in range(self.num_iter):
            # Detach and set requires_grad for the new iteration
            X_adv = X_adv.detach().clone()
            X_adv.requires_grad = True

            # Forward pass
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                logits = model_wrapper(X_adv)

            # Validate logits shape
            n_classes = logits.shape[1] if len(logits.shape) > 1 else 2
            max_label = int(max(y)) if len(y) > 0 else 0
            
            if max_label >= n_classes:
                # Labels exceed logit classes - use random perturbation fallback
                logger.warning(f"Label {max_label} exceeds logits classes {n_classes}. Using random perturbation.")
                if self.norm == 'inf':
                    perturbation = self.epsilon * torch.sign(torch.randn_like(X_adv))
                else:
                    perturbation = self.epsilon * torch.randn_like(X_adv)
                    perturbation = perturbation / (torch.norm(perturbation, dim=1, keepdim=True) + 1e-8) * self.epsilon
                X_adv = X_tensor + perturbation
                X_adv = torch.clamp(X_adv, 0, 1)
                return X_adv.detach().numpy()

            # Compute loss
            target_tensor = None
            true_tensor = None
            if self.targeted and target_labels is not None:
                # Targeted attack: minimize loss for target class
                target_tensor = torch.tensor(target_labels, dtype=torch.long)
                loss = nn.CrossEntropyLoss()(logits, target_tensor)
            else:
                # Untargeted attack: maximize loss for true class
                true_tensor = torch.tensor(y, dtype=torch.long)
                loss = -nn.CrossEntropyLoss()(logits, true_tensor)

            # Compute gradients
            try:
                loss.backward()
            except RuntimeError:
                # Gradient calculation failed, use finite differences
                if self.targeted and target_labels is not None:
                    grad_est = self._finite_diff_grad(model_wrapper, X_adv, None, target_tensor)
                else:
                    grad_est = self._finite_diff_grad(model_wrapper, X_adv, true_tensor, None)
                X_adv.grad = grad_est

            if X_adv.grad is None:
                if self.targeted and target_labels is not None:
                    grad_est = self._finite_diff_grad(model_wrapper, X_adv, None, target_tensor)
                else:
                    grad_est = self._finite_diff_grad(model_wrapper, X_adv, true_tensor, None)
                X_adv.grad = grad_est

            # Update adversarial examples
            with torch.no_grad():
                if self.norm == 'inf':
                    # L-infinity norm
                    perturbation = self.alpha * X_adv.grad.sign()
                elif self.norm == '2':
                    # L2 norm
                    grad_norm = torch.norm(X_adv.grad, dim=1, keepdim=True)
                    perturbation = self.alpha * X_adv.grad / (grad_norm + 1e-8)
                else:
                    raise ValueError(f"Unsupported norm: {self.norm}")

                X_adv = X_adv + perturbation

                # Project back to epsilon ball
                if self.norm == 'inf':
                    # L-infinity projection
                    delta = X_adv - X_tensor
                    delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                    X_adv = X_tensor + delta
                elif self.norm == '2':
                    # L2 projection
                    delta = X_adv - X_tensor
                    delta_norm = torch.norm(delta, dim=1, keepdim=True)
                    delta = delta / (delta_norm + 1e-8) * \
                        torch.clamp(delta_norm, max=self.epsilon)
                    X_adv = X_tensor + delta

                # Clip to valid range
                X_adv = torch.clamp(X_adv, 0, 1)

        return X_adv.detach().numpy()

    def _finite_diff_grad(self, model_wrapper, X, y_tensor, target_tensor=None, eps=1e-4):
        """Estimate gradient via central finite differences."""
        logger.warning("Using finite difference gradient estimation (slow).")
        X = X.detach().clone()
        grad_est = torch.zeros_like(X)
        
        n_features = X.shape[1]
        
        # Convert to long for integer indexing
        y_indices = y_tensor.long() if y_tensor is not None else None
        target_indices = target_tensor.long() if target_tensor is not None else None
        
        with torch.no_grad():
            for i in range(n_features):
                delta = torch.zeros_like(X)
                delta[:, i] = eps
                
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="X does not have valid feature names")
                    logits_plus = model_wrapper(X + delta)
                
                if self.targeted and target_indices is not None:
                    loss_plus = nn.CrossEntropyLoss(reduction='none')(logits_plus, target_indices)
                else:
                    loss_plus = -nn.CrossEntropyLoss(reduction='none')(logits_plus, y_indices)
                
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="X does not have valid feature names")
                    logits_minus = model_wrapper(X - delta)
                
                if self.targeted and target_indices is not None:
                    loss_minus = nn.CrossEntropyLoss(reduction='none')(logits_minus, target_indices)
                else:
                    loss_minus = -nn.CrossEntropyLoss(reduction='none')(logits_minus, y_indices)
                
                grad_est[:, i] = (loss_plus - loss_minus) / (2 * eps)
                
        return grad_est

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
            'attack_type': 'PGD',
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'num_iter': self.num_iter,
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

        logger.info(f"PGD attack evaluation: success_rate={success_rate:.4f}, "
                    f"accuracy_drop={accuracy_drop:.4f}")

        return results


class SklearnModelWrapper(nn.Module):
    """Wrapper to make sklearn models compatible with PyTorch for gradient computation."""

    def __init__(self, sklearn_model: BaseEstimator):
        super().__init__()
        self.sklearn_model = sklearn_model

        # Extract feature names if available (to prevent warnings)
        self.feature_names = getattr(sklearn_model, 'feature_names_in_', None)

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
            try:
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
            except RuntimeError:
                pass

        # For tree-based models or fallback, we need to approximate via predict_proba
        with torch.no_grad():
            x_np = x.detach().cpu().numpy()
            
            # Reconstruct DataFrame if feature names are available
            if self.feature_names is not None and x_np.shape[1] == len(self.feature_names):
                import pandas as pd
                x_input = pd.DataFrame(x_np, columns=self.feature_names)
                predictions = self.sklearn_model.predict_proba(x_input)
            else:
                predictions = self.sklearn_model.predict_proba(x_np)
                
        return torch.tensor(predictions, dtype=torch.float32)


class PGDAttackGenerator:
    """Generator for PGD attacks with multiple configurations."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize PGD attack generator."""

        self.config = config or {}
        self.epsilon_range = self.config.get(
            'epsilon_range', [0.01, 0.05, 0.1, 0.2, 0.3])
        self.alpha_range = self.config.get('alpha_range', [0.005, 0.01, 0.02])
        self.num_iter_range = self.config.get('num_iter_range', [5, 10, 20])
        self.norms = self.config.get('norms', ['inf', '2'])

        logger.info("PGDAttackGenerator initialized")

    def generate_multiple_attacks(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Generate PGD attacks with multiple configurations.

        Args:
            model: Target model
            X: Input features
            y: True labels

        Returns:
            Dictionary of attack results
        """

        results = {}

        for epsilon in self.epsilon_range:
            for alpha in self.alpha_range:
                for num_iter in self.num_iter_range:
                    for norm in self.norms:
                        attack_name = f"PGD_eps{epsilon}_alpha{alpha}_iter{num_iter}_norm{norm}"

                        try:
                            # Create attack
                            attack = PGDAttack({
                                'epsilon': epsilon,
                                'alpha': alpha,
                                'num_iter': num_iter,
                                'norm': norm,
                                'targeted': False,
                                'random_start': True
                            })

                            # Generate adversarial examples
                            X_adv = attack.generate_attack(model, X, y)

                            # Evaluate attack
                            eval_results = attack.evaluate_attack(
                                model, X, y, X_adv)

                            results[attack_name] = {
                                'adversarial_examples': X_adv,  # Full array
                                'adversarial_examples_sample': X_adv[:50],
                                'evaluation': eval_results
                            }

                            logger.info(
                                f"Generated {attack_name}: success_rate={eval_results['success_rate']:.4f}")

                        except Exception as e:
                            logger.error(
                                f"Failed to generate {attack_name}: {e}")
                            results[attack_name] = {'error': str(e)}

        return results

    def find_optimal_parameters(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                                target_success_rate: float = 0.5) -> Dict[str, Any]:
        """
        Find optimal PGD parameters for target success rate.

        Args:
            model: Target model
            X: Input features
            y: True labels
            target_success_rate: Target attack success rate

        Returns:
            Optimal parameters
        """

        logger.info(
            f"Finding optimal PGD parameters for success rate {target_success_rate}")

        best_params = None
        best_success_rate = 0
        best_accuracy_drop = 0

        # Grid search over parameter combinations
        for epsilon in self.epsilon_range:
            for alpha in self.alpha_range:
                for num_iter in self.num_iter_range:
                    try:
                        attack = PGDAttack({
                            'epsilon': epsilon,
                            'alpha': alpha,
                            'num_iter': num_iter,
                            'norm': 'inf',
                            'targeted': False,
                            'random_start': True
                        })

                        X_adv = attack.generate_attack(model, X, y)
                        eval_results = attack.evaluate_attack(
                            model, X, y, X_adv)

                        # Check if this combination meets our criteria
                        if (eval_results['success_rate'] >= target_success_rate and
                                eval_results['success_rate'] > best_success_rate):
                            best_params = {
                                'epsilon': epsilon,
                                'alpha': alpha,
                                'num_iter': num_iter,
                                'norm': 'inf'
                            }
                            best_success_rate = eval_results['success_rate']
                            best_accuracy_drop = eval_results['accuracy_drop']

                    except Exception as e:
                        logger.error(f"Error testing parameters: {e}")
                        continue

        if best_params is None:
            logger.warning(
                "No parameters found that meet the target success rate")
            return {}

        logger.info(f"Optimal parameters: {best_params}")
        logger.info(f"Best success rate: {best_success_rate:.4f}")
        logger.info(f"Best accuracy drop: {best_accuracy_drop:.4f}")

        return {
            'parameters': best_params,
            'success_rate': best_success_rate,
            'accuracy_drop': best_accuracy_drop
        }

    def compare_with_fgsm(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Compare PGD with FGSM attack.

        Args:
            model: Target model
            X: Input features
            y: True labels

        Returns:
            Comparison results
        """

        logger.info("Comparing PGD with FGSM attack")

        # FGSM attack
        from .fgsm_attack import FGSMAttack

        fgsm_attack = FGSMAttack({'epsilon': 0.1, 'norm': 'inf'})
        X_adv_fgsm = fgsm_attack.generate_attack(model, X, y)
        fgsm_results = fgsm_attack.evaluate_attack(model, X, y, X_adv_fgsm)

        # PGD attack
        pgd_attack = PGDAttack({
            'epsilon': 0.1,
            'alpha': 0.01,
            'num_iter': 10,
            'norm': 'inf',
            'targeted': False,
            'random_start': True
        })
        X_adv_pgd = pgd_attack.generate_attack(model, X, y)
        pgd_results = pgd_attack.evaluate_attack(model, X, y, X_adv_pgd)

        comparison = {
            'fgsm': fgsm_results,
            'pgd': pgd_results,
            'improvement': {
                'success_rate': pgd_results['success_rate'] - fgsm_results['success_rate'],
                'accuracy_drop': pgd_results['accuracy_drop'] - fgsm_results['accuracy_drop']
            }
        }

        logger.info(
            f"PGD vs FGSM: success_rate improvement={comparison['improvement']['success_rate']:.4f}")

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

    # Test PGD attack
    attack = PGDAttack({
        'epsilon': 0.1,
        'alpha': 0.01,
        'num_iter': 10,
        'norm': 'inf',
        'random_start': True
    })
    X_adv = attack.generate_attack(model, X.values, y.values)

    # Evaluate attack
    results = attack.evaluate_attack(model, X.values, y.values, X_adv)
    print("\nPGD Attack Results:")
    print(f"Success Rate: {results['success_rate']:.4f}")
    print(f"Accuracy Drop: {results['accuracy_drop']:.4f}")
    print(f"Mean Perturbation Norm: {results['mean_perturbation_norm']:.4f}")

    # Test multiple attacks
    generator = PGDAttackGenerator({
        'epsilon_range': [0.05, 0.1],
        'alpha_range': [0.01],
        'num_iter_range': [5, 10],
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

    # Find optimal parameters
    optimal_params = generator.find_optimal_parameters(
        model, X.values, y.values, target_success_rate=0.3)
    if optimal_params:
        print(f"\nOptimal parameters: {optimal_params['parameters']}")
        print(f"Success rate: {optimal_params['success_rate']:.4f}")

    # Compare with FGSM
    comparison = generator.compare_with_fgsm(model, X.values, y.values)
    print("\nPGD vs FGSM Comparison:")
    print(f"FGSM success rate: {comparison['fgsm']['success_rate']:.4f}")
    print(f"PGD success rate: {comparison['pgd']['success_rate']:.4f}")
    print(f"Improvement: {comparison['improvement']['success_rate']:.4f}")
