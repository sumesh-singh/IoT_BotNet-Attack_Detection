"""
Hybrid Ensemble Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Implements hybrid ensemble using stacking architecture combining Random Forest, XGBoost, and LightGBM.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from datetime import datetime

# Import base models
from .random_forest_model import RandomForestModel
from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel
from .meta_learner import MetaLearner, StackingEnsemble

logger = logging.getLogger(__name__)


class HybridEnsemble:
    """Hybrid ensemble combining Random Forest, XGBoost, LightGBM with meta-learner."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize hybrid ensemble with configuration."""

        self.config = config or {}
        self.is_trained = False
        self.label_encoder = None
        self.training_history = []

        # Initialize base models
        self.base_models = {
            'random_forest': RandomForestModel(self.config),
            'xgboost': XGBoostModel(self.config),
            'lightgbm': LightGBMModel(self.config)
        }

        # Initialize meta-learner
        self.meta_learner = MetaLearner(self.config)

        # Ensemble configuration
        self.use_stacking = self.config.get('use_stacking', True)
        self.stacking_cv_folds = self.config.get('stacking_cv_folds', 5)
        self.optimize_base_models = self.config.get(
            'optimize_base_models', False)

        logger.info(
            "HybridEnsemble initialized with Random Forest, XGBoost, LightGBM, and Meta-Learner")

    def train(self, X: pd.DataFrame, y: pd.Series,
              validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> Dict[str, Any]:
        """
        Train hybrid ensemble.

        Args:
            X: Training features
            y: Training labels
            validation_data: Optional validation data (X_val, y_val)

        Returns:
            Training results dictionary
        """

        logger.info(
            f"Training Hybrid Ensemble on {len(X)} samples with {len(X.columns)} features")

        # Prepare validation data
        X_val, y_val = validation_data if validation_data else (None, None)

        # Train base models
        base_model_results = {}
        for model_name, model in self.base_models.items():
            logger.info(f"Training {model_name}...")

            try:
                results = model.train(
                    X, y,
                    validation_data=validation_data,
                    optimize_hyperparams=self.optimize_base_models
                )
                base_model_results[model_name] = results

            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                base_model_results[model_name] = {'error': str(e)}

        n_classes = len(np.unique(y))
        if n_classes > 2:
            # Multi-class detected: stacking now supported via StackingEnsemble update
            pass

        if self.use_stacking:
            logger.info("Generating stacking data...")

            # Create temporary models for stacking (to avoid overfitting)
            temp_models = []
            for model_name in ['random_forest', 'xgboost', 'lightgbm']:
                if model_name in self.base_models:
                    # Create fresh instance for stacking
                    if model_name == 'random_forest':
                        temp_model = RandomForestModel(self.config)
                    elif model_name == 'xgboost':
                        temp_model = XGBoostModel(self.config)
                    elif model_name == 'lightgbm':
                        temp_model = LightGBMModel(self.config)
                    temp_models.append(temp_model)

            # Generate stacking predictions
            stacking_predictions, _ = StackingEnsemble.generate_stacking_data(
                temp_models, X, y, self.stacking_cv_folds
            )

            # Train meta-learner on stacking data
            logger.info("Training meta-learner...")

            # Split stacking data for meta-learner training
            from sklearn.model_selection import train_test_split
            X_stack_train, X_stack_test, y_stack_train, y_stack_test = train_test_split(
                stacking_predictions, y, test_size=0.2, random_state=42, stratify=y
            )

            meta_results = self.meta_learner.train(
                X_stack_train, y_stack_train,
                validation_data=(X_stack_test, y_stack_test)
            )

        else:
            logger.info("Using simple averaging ensemble")
            meta_results = {'model_type': 'SimpleAveraging'}

        # Final ensemble training results
        ensemble_results = {
            'ensemble_type': 'HybridEnsemble',
            'use_stacking': self.use_stacking,
            'base_model_results': base_model_results,
            'meta_learner_results': meta_results,
            'n_features': len(X.columns),
            'n_samples': len(X),
            'training_timestamp': datetime.now().isoformat()
        }

        self.is_trained = True
        self.feature_names_in_ = X.columns.tolist() # Store feature names for compatibility (e.g. adversarial wrapper)
        if validation_data:
            ensemble_accuracy = self._evaluate_ensemble(X_val, y_val)
            ensemble_results['ensemble_validation_accuracy'] = ensemble_accuracy
        self.training_history.append(ensemble_results)

        logger.info("Hybrid Ensemble training completed")
        return ensemble_results

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using hybrid ensemble."""

        if not self.is_trained:
            raise ValueError(
                "Ensemble must be trained before making predictions")

        if self.use_stacking:
            # Get probabilities from base models for stacking (matching training data)
            base_probabilities = []
            for model_name, model in self.base_models.items():
                if hasattr(model, 'is_trained') and model.is_trained:
                    pred = model.predict_proba(X)
                    base_probabilities.append(pred)

            if not base_probabilities:
                raise ValueError("No trained base models available")

            # Stack probabilities (use column 1 for binary classification to match StackingEnsemble)
            base_feature_array = np.column_stack(
                [prob[:, 1] if prob.shape[1] > 1 else prob[:, 0] for prob in base_probabilities]
            )

            # Use meta-learner for final prediction
            return self.meta_learner.predict(base_feature_array)

        else:
            probabilities = []
            weights = []

            for model_name, model in self.base_models.items():
                if hasattr(model, 'is_trained') and model.is_trained:
                    proba = model.predict_proba(X)
                    probabilities.append(proba)
                    weight = 1.0
                    weights.append(weight)

            if not probabilities:
                raise ValueError("No trained base models available")

            probabilities_array = np.array(probabilities)
            weights_array = np.array(weights)
            weights_array = weights_array / weights_array.sum()

            ensemble_probabilities = np.average(
                probabilities_array, axis=0, weights=weights_array)

            if ensemble_probabilities.shape[1] > 1:
                return np.argmax(ensemble_probabilities, axis=1)
            else:
                return (ensemble_probabilities[:, 0] > 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities from hybrid ensemble."""

        if not self.is_trained:
            raise ValueError(
                "Ensemble must be trained before making predictions")

        if self.use_stacking:
            # Get probabilities from base models
            base_probabilities = []
            for model_name, model in self.base_models.items():
                if hasattr(model, 'is_trained') and model.is_trained:
                    proba = model.predict_proba(X)
                    base_probabilities.append(proba)

            if not base_probabilities:
                raise ValueError("No trained base models available")

            # Stack probabilities
            base_prob_array = np.column_stack(
                # Binary classification
                [prob[:, 1] for prob in base_probabilities])

            # Use meta-learner for final probabilities
            return self.meta_learner.predict_proba(base_prob_array)

        else:
            # Simple averaging of probabilities
            probabilities = []
            weights = []

            for model_name, model in self.base_models.items():
                if hasattr(model, 'is_trained') and model.is_trained:
                    proba = model.predict_proba(X)
                    probabilities.append(proba)

                    # Get model weight
                    weight = 1.0  # Equal weights for now
                    weights.append(weight)

            if not probabilities:
                raise ValueError("No trained base models available")

            # Weighted average probabilities
            probabilities_array = np.array(probabilities)
            weights_array = np.array(weights)
            weights_array = weights_array / weights_array.sum()  # Normalize weights

            # Weighted average probabilities
            ensemble_probabilities = np.average(
                probabilities_array, axis=0, weights=weights_array)

            return ensemble_probabilities

    def _evaluate_ensemble(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate ensemble performance."""

        try:
            predictions = self.predict(X)
            accuracy = (predictions == y).mean()
            return accuracy
        except Exception as e:
            logger.error(f"Ensemble evaluation failed: {e}")
            return 0.0

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, pd.Series]:
        """Get feature importance from all base models."""

        if not self.is_trained:
            raise ValueError(
                "Ensemble must be trained to get feature importance")

        importance_dict = {}

        for model_name, model in self.base_models.items():
            if hasattr(model, 'is_trained') and model.is_trained:
                try:
                    importance = model.get_feature_importance(top_n)
                    importance_dict[model_name] = importance
                except Exception as e:
                    logger.warning(
                        f"Could not get feature importance for {model_name}: {e}")

        return importance_dict

    def get_base_model_weights(self) -> Dict[str, float]:
        """Get weights of base models from meta-learner."""

        if not self.is_trained or not self.use_stacking:
            return {}

        return self.meta_learner.get_base_model_weights()

    def save_model(self, filepath: str, feature_engineer_state: Optional[Dict[str, Any]] = None) -> None:
        """Save trained ensemble to disk with feature engineer state."""

        if not self.is_trained:
            raise ValueError("Cannot save untrained ensemble")

        # Save base models
        base_models_data = {}
        for model_name, model in self.base_models.items():
            if hasattr(model, 'is_trained') and model.is_trained:
                base_models_data[model_name] = {
                    'model': model.model,
                    'feature_importance': model.feature_importance_,
                    'params': model.params,
                    'training_history': model.training_history,
                    'is_trained': model.is_trained
                }

        # Save meta-learner
        meta_learner_data = None
        if self.use_stacking and self.meta_learner.is_trained:
            meta_learner_data = {
                'model': self.meta_learner.model,
                'feature_importance': self.meta_learner.feature_importance_,
                'meta_learner_type': self.meta_learner.meta_learner_type,
                'meta_params': self.meta_learner.meta_params,
                'training_history': self.meta_learner.training_history,
                'is_trained': self.meta_learner.is_trained
            }
        
        # CRITICAL: Always save feature engineer state
        if feature_engineer_state is None:
            logger.warning("Feature engineer state not provided to save_model()!")
        else:
            # Update internal state so it's available in memory immediately without reloading
            self.feature_engineer_state = feature_engineer_state

        ensemble_data = {
            'base_models': base_models_data,
            'meta_learner': meta_learner_data,
            'use_stacking': self.use_stacking,
            'training_history': self.training_history,
            'is_trained': self.is_trained,
            'label_encoder': self.label_encoder,
            'config': self.config,
            'feature_names_in_': getattr(self, 'feature_names_in_', None),
            'feature_engineer_state': feature_engineer_state,  # Save with consistent key
            '_feature_engineer_state': feature_engineer_state  # Backup key for compatibility
        }

        joblib.dump(ensemble_data, filepath)
        logger.info(f"Hybrid Ensemble saved to {filepath}")
        if feature_engineer_state:
            logger.info(f"  - Feature engineer state saved (selected features: {len(feature_engineer_state.get('selected_features', []))})")
        else:
            logger.warning("  - WARNING: No feature engineer state saved!")

    def load_model(self, filepath: str) -> None:
        """Load trained ensemble from disk."""

        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        ensemble_data = joblib.load(filepath)

        # Load base models
        for model_name, model_data in ensemble_data['base_models'].items():
            if model_name in self.base_models:
                model = self.base_models[model_name]
                model.model = model_data['model']
                model.feature_importance_ = model_data['feature_importance']
                model.params = model_data['params']
                model.training_history = model_data['training_history']
                model.is_trained = model_data['is_trained']

        # Load meta-learner
        if ensemble_data['meta_learner'] is not None:
            meta_data = ensemble_data['meta_learner']
            self.meta_learner.model = meta_data['model']
            self.meta_learner.feature_importance_ = meta_data['feature_importance']
            self.meta_learner.meta_learner_type = meta_data['meta_learner_type']
            self.meta_learner.meta_params = meta_data['meta_params']
            self.meta_learner.training_history = meta_data['training_history']
            self.meta_learner.is_trained = meta_data['is_trained']

        self.use_stacking = ensemble_data['use_stacking']
        self.training_history = ensemble_data['training_history']
        self.is_trained = ensemble_data['is_trained']
        self.label_encoder = ensemble_data.get('label_encoder')
        self.config = ensemble_data['config']
        self.feature_names_in_ = ensemble_data.get('feature_names_in_')
        
        # CRITICAL: Load feature engineer state (check both keys)
        self.feature_engineer_state = ensemble_data.get('feature_engineer_state')
        if self.feature_engineer_state is None:
            self.feature_engineer_state = ensemble_data.get('_feature_engineer_state')

        logger.info(f"Hybrid Ensemble loaded from {filepath}")
        if self.feature_engineer_state:
            selected_count = len(self.feature_engineer_state.get('selected_features', []))
            logger.info(f"  - Feature engineer state loaded (selected features: {selected_count})")
        else:
            logger.warning("  - WARNING: No feature engineer state found in loaded model!")

        logger.info(f"Hybrid Ensemble loaded from {filepath}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive ensemble information."""

        if not self.is_trained:
            return {'status': 'not_trained'}

        base_model_info = {}
        for model_name, model in self.base_models.items():
            if hasattr(model, 'is_trained') and model.is_trained:
                base_model_info[model_name] = model.get_model_info()

        ensemble_info = {
            'ensemble_type': 'HybridEnsemble',
            'is_trained': self.is_trained,
            'use_stacking': self.use_stacking,
            'base_models': base_model_info,
            'n_base_models_trained': len(base_model_info),
            'training_history_count': len(self.training_history)
        }

        if self.use_stacking and self.meta_learner.is_trained:
            ensemble_info['meta_learner'] = self.meta_learner.get_model_info()

        return ensemble_info

    def explain_prediction(self, X: pd.DataFrame, sample_idx: int = 0) -> Dict[str, Any]:
        """Explain ensemble prediction for a specific sample."""

        if not self.is_trained:
            raise ValueError("Ensemble must be trained to explain predictions")

        if sample_idx >= len(X):
            raise ValueError(f"Sample index {sample_idx} out of range")

        # Get ensemble prediction
        prediction = self.predict(X.iloc[[sample_idx]])[0]
        probability = self.predict_proba(X.iloc[[sample_idx]])[0]

        # Get base model predictions and explanations
        base_model_explanations = {}
        base_model_predictions = {}

        for model_name, model in self.base_models.items():
            if hasattr(model, 'is_trained') and model.is_trained:
                try:
                    base_pred = model.predict(X.iloc[[sample_idx]])[0]
                    base_proba = model.predict_proba(X.iloc[[sample_idx]])[0]

                    base_model_predictions[model_name] = {
                        'prediction': int(base_pred),
                        'probability': base_proba.tolist()
                    }

                    # Get explanation if available
                    if hasattr(model, 'explain_prediction'):
                        explanation = model.explain_prediction(X, sample_idx)
                        base_model_explanations[model_name] = explanation

                except Exception as e:
                    logger.warning(
                        f"Could not explain prediction for {model_name}: {e}")

        explanation = {
            'ensemble_prediction': int(prediction),
            'ensemble_probability': probability.tolist(),
            'base_model_predictions': base_model_predictions,
            'base_model_explanations': base_model_explanations
        }

        # Add meta-learner explanation if using stacking
        if self.use_stacking and self.meta_learner.is_trained:
            try:
                # Get base model predictions for stacking
                base_preds = []
                for model_name in ['random_forest', 'xgboost', 'lightgbm']:
                    if model_name in base_model_predictions:
                        base_preds.append(
                            base_model_predictions[model_name]['prediction'])

                if base_preds:
                    base_pred_array = np.array([base_preds])
                    meta_explanation = self.meta_learner.explain_prediction(
                        base_pred_array, 0)
                    explanation['meta_learner_explanation'] = meta_explanation

            except Exception as e:
                logger.warning(f"Could not get meta-learner explanation: {e}")

        return explanation


# Example usage and testing
if __name__ == '__main__':
    # Create sample data for testing
    np.random.seed(42)
    n_samples, n_features = 1000, 50

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Create labels with some structure
    y = pd.Series(
        (X.iloc[:, 0] + X.iloc[:, 1] +
         np.random.randn(n_samples) * 0.1 > 0).astype(int)
    )

    # Initialize ensemble
    ensemble = HybridEnsemble({'use_stacking': True})

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train ensemble
    results = ensemble.train(
        X_train, y_train, validation_data=(X_test, y_test))

    print("Ensemble Training Results:")
    print(
        f"Ensemble Validation Accuracy: {results['ensemble_validation_accuracy']:.4f}")

    # Test predictions
    predictions = ensemble.predict(X_test)
    accuracy = (predictions == y_test).mean()
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Feature importance
    print("\nFeature Importance by Model:")
    importance_dict = ensemble.get_feature_importance(5)
    for model_name, importance in importance_dict.items():
        print(f"\n{model_name}:")
        print(importance)

    # Model info
    print("\nEnsemble Info:")
    print(ensemble.get_model_info())
