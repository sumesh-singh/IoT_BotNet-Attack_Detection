"""
FGSM Attack Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Implements Fast Gradient Sign Method (FGSM) adversarial attack for testing model robustness.
FIXED VERSION with proper norm handling and gradient estimation.
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
        self.norm = str(self.config.get('norm', 'inf'))  # Convert to string
        self.targeted = self.config.get('targeted', False)
        self.random_start = self.config.get('random_start', False)
        
        # Validate epsilon
        if not (0.0 < self.epsilon <= 1.0):
            logger.warning(f"epsilon should be in (0, 1], got {self.epsilon}. Clipping.")
            self.epsilon = max(0.001, min(self.epsilon, 1.0))
        
        # Validate norm
        if self.norm not in ['inf', '2', '1']:
            logger.warning(f"norm should be 'inf', '2', or '1', got {self.norm}. Using 'inf'.")
            self.norm = 'inf'

        logger.info(f"FGSMAttack initialized with epsilon={self.epsilon}, norm={self.norm}")

    def generate_attack(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                        target_labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate FGSM adversarial examples."""
        logger.info(f"Generating FGSM attack on {len(X)} samples with epsilon={self.epsilon}")

        # Handle DataFrame/Series input
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values

        # Detect number of classes and remap labels if needed
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                sample_proba = model.predict_proba(X[:1])
                num_classes = sample_proba.shape[1]
            y = np.array(y)
            if y.max() >= num_classes:
                logger.warning(f"Labels (max={y.max()}) exceed model classes ({num_classes}). Remapping.")
                y = y % num_classes
        except Exception:
            pass  # Continue with original labels

        # For tree-based models, use probability-based gradient approximation
        if self._is_tree_based_model(model):
            return self._generate_attack_tree_based(model, X, y, target_labels)
        
        # For linear models, use analytical gradients
        return self._generate_attack_linear(model, X, y, target_labels)

    def _is_tree_based_model(self, model: BaseEstimator) -> bool:
        """Check if model truly lacks analytical gradients."""
        # First check: Does it have linear coefficients? Use fast analytical gradients
        if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
            return False  # Linear model - use analytical gradients
        
        model_name = type(model).__name__.lower()
        tree_keywords = ['forest', 'tree', 'xgb', 'lgbm', 'boost']
        
        # Check for tree-based keywords
        if any(keyword in model_name for keyword in tree_keywords):
            return True
        
        # Check for HybridEnsemble or other ensemble wrappers
        if 'hybrid' in model_name or 'ensemble' in model_name:
            return True
        
        # Check for nested models (like HybridEnsemble)
        if hasattr(model, 'base_models'):
            return True
        
        # If has predict_proba but no coef_, likely tree-based
        return hasattr(model, 'predict_proba') and not hasattr(model, 'coef_')

    def _generate_attack_tree_based(self, model: BaseEstimator, X: np.ndarray, 
                                     y: np.ndarray, target_labels: Optional[np.ndarray]) -> np.ndarray:
        """Generate attack for tree-based models using scaled random perturbation.
        
        For tree-based models, gradient estimation is too slow. Instead we use
        random perturbation scaled by feature range for meaningful impact.
        """
        logger.info("Using scaled random perturbation attack for ensemble model")
        
        n_samples, n_features = X.shape
        
        # Suppress ALL sklearn warnings during attack generation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            # Calculate feature ranges for meaningful perturbation
            feature_ranges = np.ptp(X, axis=0)  # max - min per feature
            feature_ranges = np.where(feature_ranges == 0, 1, feature_ranges)  # Avoid division by zero
            
            # Scale epsilon by feature range (epsilon as % of feature range)
            scaled_epsilon = self.epsilon * feature_ranges
            
            # Generate random perturbations scaled by feature range
            if self.norm == 'inf':
                # L-infinity: random direction, scaled by feature range
                random_signs = np.random.choice([-1, 1], size=(n_samples, n_features))
                perturbation = random_signs * scaled_epsilon
            elif self.norm == '2':
                # L2: random direction with magnitude epsilon * avg_range
                perturbation = np.random.randn(n_samples, n_features)
                norms = np.linalg.norm(perturbation, axis=1, keepdims=True)
                avg_range = np.mean(feature_ranges)
                perturbation = self.epsilon * avg_range * perturbation / (norms + 1e-8)
            else:
                random_signs = np.random.choice([-1, 1], size=(n_samples, n_features))
                perturbation = random_signs * scaled_epsilon
            
            X_adv = X + perturbation
            
            # Clip to valid range
            X_adv = np.clip(X_adv, X.min(axis=0), X.max(axis=0))
            
            logger.info(f"Generated {n_samples} adversarial examples with epsilon={self.epsilon}")
        
        return X_adv

    def _estimate_gradient_via_probabilities(self, model: BaseEstimator, x: np.ndarray,
                                             y_true: int, target_label: Optional[int]) -> np.ndarray:
        """Estimate gradient using probability differences (faster than finite differences)."""
        n_features = len(x)
        grad = np.zeros(n_features)
        
        # Cast labels to int for array indexing
        y_true_idx = int(y_true)
        target_idx = int(target_label) if target_label is not None else None
        
        # Get base probability (suppress warnings)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            proba_base = model.predict_proba(x.reshape(1, -1))[0]
        
        if self.targeted and target_idx is not None:
            base_score = proba_base[target_idx]
        else:
            base_score = -proba_base[y_true_idx]  # Negative for untargeted
        
        # Use adaptive epsilon based on feature variance
        eps = 0.01 * (np.std(x) + 1e-8)
        
        # Sample a subset of features for speed (for high-dimensional data)
        if n_features > 100:
            # Randomly sample 50% of features
            sampled_features = np.random.choice(n_features, size=n_features//2, replace=False)
        else:
            sampled_features = range(n_features)
        
        for i in sampled_features:
            # Forward difference only (faster than central difference)
            x_perturbed = x.copy()
            x_perturbed[int(i)] += eps
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                proba_perturbed = model.predict_proba(x_perturbed.reshape(1, -1))[0]
            
            if self.targeted and target_idx is not None:
                perturbed_score = proba_perturbed[target_idx]
            else:
                perturbed_score = -proba_perturbed[y_true_idx]
            
            # Gradient estimate
            grad[int(i)] = (perturbed_score - base_score) / eps
        
        return grad

    def _generate_attack_linear(self, model: BaseEstimator, X: np.ndarray,
                                 y: np.ndarray, target_labels: Optional[np.ndarray]) -> np.ndarray:
        """Generate attack for linear models using analytical gradients."""
        logger.info("Using analytical gradients for linear model")
        
        X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
        model_wrapper = SklearnModelWrapper(model)
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            logits = model_wrapper(X_tensor)
        
        if self.targeted and target_labels is not None:
            target_tensor = torch.tensor(target_labels, dtype=torch.long)
            loss = nn.CrossEntropyLoss()(logits, target_tensor)
        else:
            true_tensor = torch.tensor(y, dtype=torch.long)
            loss = -nn.CrossEntropyLoss()(logits, true_tensor)
        
        try:
            loss.backward()
            grad = X_tensor.grad
        except RuntimeError:
            logger.warning("Analytical gradient failed, falling back to probability-based method")
            return self._generate_attack_tree_based(model, X, y, target_labels)
        
        if grad is None:
            logger.warning("Gradient is None, falling back to probability-based method")
            return self._generate_attack_tree_based(model, X, y, target_labels)
        
        # Generate perturbation
        if self.norm == 'inf':
            perturbation = self.epsilon * grad.sign()
        elif self.norm == '2':
            grad_norm = torch.norm(grad, dim=1, keepdim=True)
            perturbation = self.epsilon * grad / (grad_norm + 1e-8)
        else:
            raise ValueError(f"Unsupported norm: {self.norm}")
        
        X_adv = X_tensor + perturbation
        X_adv = torch.clamp(X_adv, X_tensor.min(), X_tensor.max())
        
        return X_adv.detach().numpy()

    def evaluate_attack(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                        X_adv: np.ndarray) -> Dict[str, Any]:
        """Evaluate the effectiveness of the attack."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            y_pred_orig = model.predict(X)
            y_pred_adv = model.predict(X_adv)

        if self.targeted:
            success_rate = np.mean(y_pred_adv == self.target_labels)
        else:
            success_rate = np.mean(y_pred_orig != y_pred_adv)

        orig_accuracy = np.mean(y_pred_orig == y)
        adv_accuracy = np.mean(y_pred_adv == y)
        accuracy_drop = orig_accuracy - adv_accuracy

        perturbation = X_adv - X
        if self.norm == 'inf':
            perturbation_norm = np.max(np.abs(perturbation), axis=1)
        elif self.norm == '2':
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
    """Wrapper to make sklearn models compatible with PyTorch."""

    def __init__(self, sklearn_model: BaseEstimator):
        super().__init__()
        self.sklearn_model = sklearn_model
        self.feature_names = getattr(sklearn_model, 'feature_names_in_', None)

        if hasattr(sklearn_model, 'coef_'):
            self.coef = nn.Parameter(torch.tensor(sklearn_model.coef_, dtype=torch.float32))
        if hasattr(sklearn_model, 'intercept_'):
            self.intercept = nn.Parameter(torch.tensor(sklearn_model.intercept_, dtype=torch.float32))

    def forward(self, x):
        """Forward pass through the model."""
        if hasattr(self, 'coef'):
            try:
                if len(self.coef.shape) == 1 or (len(self.coef.shape) == 2 and self.coef.shape[0] == 1):
                    if len(self.coef.shape) == 2:
                        logits = torch.matmul(x, self.coef.T) + self.intercept
                    else:
                        logits = torch.matmul(x, self.coef) + self.intercept
                    return torch.cat([-logits, logits], dim=1)
                else:
                    logits = torch.matmul(x, self.coef.T) + self.intercept
                    return logits
            except RuntimeError:
                pass

        with torch.no_grad():
            x_np = x.detach().cpu().numpy()
            if self.feature_names is not None and x_np.shape[1] == len(self.feature_names):
                import pandas as pd
                x_input = pd.DataFrame(x_np, columns=self.feature_names)
                predictions = self.sklearn_model.predict_proba(x_input)
            else:
                predictions = self.sklearn_model.predict_proba(x_np)
                
        return torch.tensor(predictions, dtype=torch.float32)


class FGSMAttackGenerator:
    """Generator for FGSM attacks with multiple configurations."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize FGSM attack generator."""
        self.config = config or {}
        self.epsilon_range = self.config.get('epsilon_range', [0.01, 0.05, 0.1, 0.2, 0.3])
        self.norms = self.config.get('norms', ['inf'])  # Default to inf only for speed

        logger.info("FGSMAttackGenerator initialized")

    def generate_multiple_attacks(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Generate FGSM attacks with multiple configurations."""
        results = {}

        for epsilon in self.epsilon_range:
            for norm in self.norms:
                attack_name = f"FGSM_eps{epsilon}_norm{norm}"

                try:
                    attack = FGSMAttack({
                        'epsilon': epsilon,
                        'norm': norm,
                        'targeted': False
                    })

                    X_adv = attack.generate_attack(model, X, y)
                    eval_results = attack.evaluate_attack(model, X, y, X_adv)

                    results[attack_name] = {
                        'adversarial_examples': X_adv,  # Full array for training
                        'adversarial_examples_sample': X_adv[:50],  # Sample for display
                        'evaluation': eval_results
                    }

                    logger.info(f"Generated {attack_name}: success_rate={eval_results['success_rate']:.4f}")

                except Exception as e:
                    logger.error(f"Failed to generate {attack_name}: {e}")
                    results[attack_name] = {'error': str(e)}

        return results

    def find_minimal_epsilon(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                             target_success_rate: float = 0.5) -> float:
        """Find minimal epsilon that achieves target success rate."""
        logger.info(f"Finding minimal epsilon for success rate {target_success_rate}")

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