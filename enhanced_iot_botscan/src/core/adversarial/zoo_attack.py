"""
ZOO (Zeroth-Order Optimization) Black-Box Attack for Tree Ensembles
Author: Kotiwale Sumesh Singh
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

class ZOOAttackGenerator:
    """
    Implements a Zeroth-Order Optimization (ZOO) block-box evasion attack.
    Estimates gradients using finite differences on predict_proba() to attack
    non-differentiable models like Random Forest, XGBoost, and LightGBM.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.h = self.config.get('zoo_h', 1e-4)
        self.learning_rate = self.config.get('zoo_lr', 0.1)
        self.max_iter = self.config.get('zoo_max_iter', 25)
        self.clip_min = self.config.get('clip_min', -5.0)
        self.clip_max = self.config.get('clip_max', 5.0)

    def generate(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Generate adversarial examples using ZOO attack.
        """
        X_adv = np.copy(X)
        success_count = 0
        perturbations = []

        # Ensure X is 2D
        if len(X.shape) == 1:
            X_adv = X_adv.reshape(1, -1)
            y = np.array([y])

        n_samples, n_features = X_adv.shape

        for i in range(min(n_samples, 200)):  # Cap at 200 for evaluation speed
            x_orig = X_adv[i].copy()
            y_true = y[i]
            
            # Target class (flip binary label)
            target = 1 - y_true 

            x_curr = x_orig.copy()
            is_successful = False

            for iteration in range(self.max_iter):
                # Check if current prediction is target (success)
                p = self._predict_proba_safe(model, x_curr.reshape(1, -1))[0]
                if np.argmax(p) == target:
                    is_successful = True
                    break

                # Estimate gradients
                grad = np.zeros(n_features)
                for j in range(n_features):
                    # Forward difference
                    x_plus = x_curr.copy()
                    x_plus[j] += self.h
                    p_plus = self._predict_proba_safe(model, x_plus.reshape(1, -1))[0][target]
                    
                    # Backward difference
                    x_minus = x_curr.copy()
                    x_minus[j] -= self.h
                    p_minus = self._predict_proba_safe(model, x_minus.reshape(1, -1))[0][target]
                    
                    grad[j] = (p_plus - p_minus) / (2 * self.h)

                # Update step (Gradient Ascent on target class probability)
                x_curr += self.learning_rate * np.sign(grad)
                
                # Clip bounds
                x_curr = np.clip(x_curr, self.clip_min, self.clip_max)
            
            # Post-check
            final_pred = np.argmax(self._predict_proba_safe(model, x_curr.reshape(1, -1))[0])
            if final_pred == target:
                success_count += 1
                dist = np.linalg.norm(x_curr - x_orig, ord=2)
                perturbations.append(dist)
                X_adv[i] = x_curr
                
        metrics = {
            'attack_success_rate': success_count / min(n_samples, 200) if n_samples > 0 else 0,
            'mean_l2_perturbation': np.mean(perturbations) if perturbations else 0.0,
            'samples_attacked': min(n_samples, 200)
        }
        
        return {
            'adversarial_examples': X_adv,
            'attack_metrics': metrics
        }

    def _predict_proba_safe(self, model: BaseEstimator, X: np.ndarray) -> np.ndarray:
        if isinstance(X, np.ndarray) and hasattr(model, 'feature_names_in_'):
            import pandas as pd
            X = pd.DataFrame(X, columns=model.feature_names_in_)
        elif not isinstance(X, pd.DataFrame) and hasattr(model, 'base_models'):
            # It's our HybridEnsemble, might require DataFrame depending on internal state
            # but usually it handles DataFrame natively. We'll pass it normally and see.
            pass
            
        try:
            return model.predict_proba(X)
        except Exception:
            # Fallback if tree wrapper gets angry
            preds = model.predict(X)
            # Fake one-hot if it doesn't give probabilities
            probs = np.zeros((len(preds), 2))
            probs[np.arange(len(preds)), preds.astype(int)] = 1.0
            return probs
