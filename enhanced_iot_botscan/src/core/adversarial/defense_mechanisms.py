"""
Defense Mechanisms Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Implements various defense mechanisms against adversarial attacks.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class DefenseMechanisms:
    """Collection of defense mechanisms against adversarial attacks."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize defense mechanisms with configuration."""

        self.config = config or {}
        self.defense_stats = {}

        # Defense configuration
        self.enabled_defenses = self.config.get('enabled_defenses', [
                                                'gradient_masking', 'feature_squeezing', 'adversarial_training'])
        self.gradient_masking_threshold = self.config.get(
            'gradient_masking_threshold', 0.1)
        self.feature_squeezing_bits = self.config.get(
            'feature_squeezing_bits', 8)
        self.detection_threshold = self.config.get('detection_threshold', 0.5)

        logger.info(
            f"DefenseMechanisms initialized with defenses: {self.enabled_defenses}")

    def apply_defenses(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray = None) -> Dict[str, Any]:
        """
        Apply all enabled defense mechanisms.

        Args:
            model: Target model
            X: Input features
            y: True labels (optional)

        Returns:
            Defense results
        """

        logger.info(f"Applying defense mechanisms to {len(X)} samples")

        defense_results = {}

        for defense_type in self.enabled_defenses:
            try:
                if defense_type == 'gradient_masking':
                    result = self._apply_gradient_masking(model, X)
                elif defense_type == 'feature_squeezing':
                    result = self._apply_feature_squeezing(model, X)
                elif defense_type == 'adversarial_training':
                    result = self._apply_adversarial_training(model, X, y)
                elif defense_type == 'input_validation':
                    result = self._apply_input_validation(model, X)
                elif defense_type == 'ensemble_defense':
                    result = self._apply_ensemble_defense(model, X)
                else:
                    logger.warning(f"Unknown defense type: {defense_type}")
                    continue

                defense_results[defense_type] = result
                logger.info(f"Applied {defense_type} defense")

            except Exception as e:
                logger.error(f"Failed to apply {defense_type} defense: {e}")
                defense_results[defense_type] = {'error': str(e)}

        self.defense_stats = defense_results
        return defense_results

    def _apply_gradient_masking(self, model: BaseEstimator, X: np.ndarray) -> Dict[str, Any]:
        """Apply gradient masking defense."""

        logger.info("Applying gradient masking defense")

        # Gradient masking involves adding noise to gradients during training
        # For inference, we can add small random noise to inputs
        noise_std = self.gradient_masking_threshold
        X_masked = X + np.random.normal(0, noise_std, X.shape)

        # Make predictions with masked inputs
        y_pred_masked = model.predict(X_masked)
        y_pred_original = model.predict(X)

        # Calculate defense effectiveness
        prediction_change_rate = np.mean(y_pred_masked != y_pred_original)

        return {
            'defense_type': 'gradient_masking',
            'noise_std': noise_std,
            'prediction_change_rate': prediction_change_rate,
            'masked_predictions': y_pred_masked,
            'original_predictions': y_pred_original
        }

    def _apply_feature_squeezing(self, model: BaseEstimator, X: np.ndarray) -> Dict[str, Any]:
        """Apply feature squeezing defense."""

        logger.info("Applying feature squeezing defense")

        # Feature squeezing reduces precision of input features
        bits = self.feature_squeezing_bits
        max_val = 2 ** bits - 1

        # Quantize features
        X_squeezed = np.round(X * max_val) / max_val

        # Make predictions with squeezed inputs
        y_pred_squeezed = model.predict(X_squeezed)
        y_pred_original = model.predict(X)

        # Calculate defense effectiveness
        prediction_change_rate = np.mean(y_pred_squeezed != y_pred_original)

        return {
            'defense_type': 'feature_squeezing',
            'squeezing_bits': bits,
            'prediction_change_rate': prediction_change_rate,
            'squeezed_predictions': y_pred_squeezed,
            'original_predictions': y_pred_original
        }

    def _apply_adversarial_training(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray = None) -> Dict[str, Any]:
        """Apply adversarial training defense."""

        logger.info("Applying adversarial training defense")

        # This is a simplified version - in practice, you'd use the AdversarialTrainer
        if y is None:
            return {'defense_type': 'adversarial_training', 'error': 'Labels required for adversarial training'}

        # Generate adversarial examples (simplified)
        from .attack_generator import AdversarialAttackGenerator

        attack_generator = AdversarialAttackGenerator({
            'enabled_attacks': ['fgsm'],
            'attacks': {'fgsm': {'epsilon': 0.1, 'norm': 'inf'}}
        })

        # Generate FGSM attacks
        fgsm_result = attack_generator.generate_single_attack(
            'fgsm', model, X, y, epsilon=0.1, norm='inf'
        )

        if 'error' in fgsm_result:
            return {'defense_type': 'adversarial_training', 'error': fgsm_result['error']}

        X_adv = fgsm_result['adversarial_examples']

        # Mix clean and adversarial data
        X_mixed = np.vstack([X, X_adv])
        y_mixed = np.hstack([y, y])

        # Train robust model
        robust_model = self._clone_model(model)
        robust_model.fit(X_mixed, y_mixed)

        # Evaluate robustness
        original_accuracy = model.score(X, y)
        robust_accuracy = robust_model.score(X, y)

        return {
            'defense_type': 'adversarial_training',
            'original_accuracy': original_accuracy,
            'robust_accuracy': robust_accuracy,
            'accuracy_change': robust_accuracy - original_accuracy,
            'robust_model': robust_model
        }

    def _apply_input_validation(self, model: BaseEstimator, X: np.ndarray) -> Dict[str, Any]:
        """Apply input validation defense."""

        logger.info("Applying input validation defense")

        # Detect anomalous inputs
        anomalies = self._detect_anomalies(X)

        # Filter out anomalous inputs
        valid_indices = ~anomalies
        X_valid = X[valid_indices]

        # Make predictions only on valid inputs
        if len(X_valid) > 0:
            y_pred_valid = model.predict(X_valid)
        else:
            y_pred_valid = np.array([])

        # Create full prediction array
        y_pred_full = np.full(len(X), -1)  # -1 for invalid inputs
        y_pred_full[valid_indices] = y_pred_valid

        return {
            'defense_type': 'input_validation',
            'anomaly_rate': np.mean(anomalies),
            'valid_predictions': y_pred_valid,
            'full_predictions': y_pred_full,
            'anomaly_mask': anomalies
        }

    def _apply_ensemble_defense(self, model: BaseEstimator, X: np.ndarray) -> Dict[str, Any]:
        """Apply ensemble defense."""

        logger.info("Applying ensemble defense")

        # Create ensemble of models with different architectures
        ensemble_models = self._create_ensemble_models(model)

        # Get predictions from all models
        ensemble_predictions = []
        for ensemble_model in ensemble_models:
            pred = ensemble_model.predict(X)
            ensemble_predictions.append(pred)

        ensemble_predictions = np.array(ensemble_predictions)

        # Use majority voting
        y_pred_ensemble = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=ensemble_predictions
        )

        # Calculate agreement rate
        agreement_rate = np.mean([
            np.mean(ensemble_predictions[i] == y_pred_ensemble)
            for i in range(len(ensemble_models))
        ])

        return {
            'defense_type': 'ensemble_defense',
            'n_models': len(ensemble_models),
            'agreement_rate': agreement_rate,
            'ensemble_predictions': y_pred_ensemble,
            'individual_predictions': ensemble_predictions
        }

    def _detect_anomalies(self, X: np.ndarray) -> np.ndarray:
        """Detect anomalous inputs."""

        # Simple anomaly detection using statistical methods
        # In practice, you'd use more sophisticated methods

        # Check for extreme values
        extreme_values = np.any(np.abs(X) > 10, axis=1)

        # Check for NaN or infinite values
        invalid_values = np.any(np.isnan(X) | np.isinf(X), axis=1)

        # Check for constant features (potential adversarial)
        constant_features = np.any(np.std(X, axis=0) < 1e-8)
        if constant_features:
            constant_mask = np.std(X, axis=0) < 1e-8
            constant_samples = np.any(
                X[:, constant_mask] != X[0, constant_mask], axis=1)
        else:
            constant_samples = np.zeros(len(X), dtype=bool)

        # Combine all anomaly indicators
        anomalies = extreme_values | invalid_values | constant_samples

        return anomalies

    def _create_ensemble_models(self, base_model: BaseEstimator) -> List[BaseEstimator]:
        """Create ensemble of models with different architectures."""

        ensemble_models = []

        # Add the original model
        ensemble_models.append(base_model)

        # Create additional models with different parameters
        if hasattr(base_model, 'random_state'):
            # Random Forest with different random states
            for i in range(2):
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42 + i,
                    max_depth=10
                )
                ensemble_models.append(model)

        return ensemble_models

    def _clone_model(self, model: BaseEstimator) -> BaseEstimator:
        """Create a copy of the model."""

        # This is a simplified version - in practice, you'd need to implement
        # proper model cloning for different model types
        return model

    def evaluate_defense_effectiveness(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                                       X_adv: np.ndarray) -> Dict[str, Any]:
        """Evaluate effectiveness of defense mechanisms against adversarial attacks."""

        logger.info("Evaluating defense effectiveness")

        # Test original model
        original_accuracy = model.score(X, y)
        adversarial_accuracy = model.score(X_adv, y)

        # Apply defenses
        defense_results = self.apply_defenses(model, X_adv, y)

        # Calculate defense effectiveness
        effectiveness = {}

        for defense_type, result in defense_results.items():
            if 'error' in result:
                continue

            if defense_type == 'gradient_masking':
                # Use masked predictions
                masked_predictions = result['masked_predictions']
                masked_accuracy = np.mean(masked_predictions == y)
                effectiveness[defense_type] = {
                    'accuracy': masked_accuracy,
                    'improvement': masked_accuracy - adversarial_accuracy
                }

            elif defense_type == 'feature_squeezing':
                # Use squeezed predictions
                squeezed_predictions = result['squeezed_predictions']
                squeezed_accuracy = np.mean(squeezed_predictions == y)
                effectiveness[defense_type] = {
                    'accuracy': squeezed_accuracy,
                    'improvement': squeezed_accuracy - adversarial_accuracy
                }

            elif defense_type == 'adversarial_training':
                # Use robust model
                robust_model = result['robust_model']
                robust_accuracy = robust_model.score(X_adv, y)
                effectiveness[defense_type] = {
                    'accuracy': robust_accuracy,
                    'improvement': robust_accuracy - adversarial_accuracy
                }

            elif defense_type == 'input_validation':
                # Use valid predictions only
                valid_predictions = result['valid_predictions']
                valid_mask = result['anomaly_mask']
                if len(valid_predictions) > 0:
                    valid_accuracy = np.mean(
                        valid_predictions == y[~valid_mask])
                    effectiveness[defense_type] = {
                        'accuracy': valid_accuracy,
                        'improvement': valid_accuracy - adversarial_accuracy,
                        'anomaly_rate': result['anomaly_rate']
                    }

            elif defense_type == 'ensemble_defense':
                # Use ensemble predictions
                ensemble_predictions = result['ensemble_predictions']
                ensemble_accuracy = np.mean(ensemble_predictions == y)
                effectiveness[defense_type] = {
                    'accuracy': ensemble_accuracy,
                    'improvement': ensemble_accuracy - adversarial_accuracy,
                    'agreement_rate': result['agreement_rate']
                }

        # Overall effectiveness
        if effectiveness:
            overall_improvement = np.mean([
                eff['improvement'] for eff in effectiveness.values()
            ])
        else:
            overall_improvement = 0

        return {
            'original_accuracy': original_accuracy,
            'adversarial_accuracy': adversarial_accuracy,
            'defense_effectiveness': effectiveness,
            'overall_improvement': overall_improvement,
            'n_defenses': len(effectiveness)
        }

    def get_defense_report(self) -> Dict[str, Any]:
        """Get comprehensive defense report."""

        return {
            'enabled_defenses': self.enabled_defenses,
            'defense_stats': self.defense_stats,
            'gradient_masking_threshold': self.gradient_masking_threshold,
            'feature_squeezing_bits': self.feature_squeezing_bits,
            'detection_threshold': self.detection_threshold
        }


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

    # Initialize defense mechanisms
    defenses = DefenseMechanisms({
        'enabled_defenses': ['gradient_masking', 'feature_squeezing', 'input_validation'],
        'gradient_masking_threshold': 0.05,
        'feature_squeezing_bits': 8
    })

    # Apply defenses
    defense_results = defenses.apply_defenses(model, X.values, y.values)

    print("\nDefense Results:")
    for defense_type, result in defense_results.items():
        if 'error' not in result:
            print(
                f"{defense_type}: {result['prediction_change_rate']:.4f} change rate")

    # Test with adversarial examples
    from .attack_generator import AdversarialAttackGenerator

    attack_generator = AdversarialAttackGenerator({
        'enabled_attacks': ['fgsm'],
        'attacks': {'fgsm': {'epsilon': 0.1, 'norm': 'inf'}}
    })

    fgsm_result = attack_generator.generate_single_attack(
        'fgsm', model, X.values, y.values, epsilon=0.1, norm='inf'
    )

    if 'error' not in fgsm_result:
        X_adv = fgsm_result['adversarial_examples']

        # Evaluate defense effectiveness
        effectiveness = defenses.evaluate_defense_effectiveness(
            model, X.values, y.values, X_adv
        )

        print("\nDefense Effectiveness:")
        print(f"Original accuracy: {effectiveness['original_accuracy']:.4f}")
        print(
            f"Adversarial accuracy: {effectiveness['adversarial_accuracy']:.4f}")
        print(
            f"Overall improvement: {effectiveness['overall_improvement']:.4f}")

        for defense_type, eff in effectiveness['defense_effectiveness'].items():
            print(f"{defense_type}: {eff['improvement']:.4f} improvement")

    # Get defense report
    report = defenses.get_defense_report()
    print(
        f"\nDefense Report: {len(report['enabled_defenses'])} defenses enabled")
