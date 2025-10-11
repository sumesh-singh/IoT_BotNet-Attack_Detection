"""
Meta-Learner Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Implements meta-learner for stacking ensemble architecture to combine predictions from base models.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class MetaLearner:
    """Meta-learner for stacking ensemble predictions."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize meta-learner with configuration."""

        self.config = config or {}
        self.model = None
        self.is_trained = False
        self.feature_importance_ = None
        self.training_history = []

        # Meta-learner configuration
        self.meta_learner_type = self.config.get(
            'meta_learner_type', 'logistic_regression')
        self.cv_folds = self.config.get('cv_folds', 5)

        # Default parameters for different meta-learners
        self.meta_params = {
            'logistic_regression': {
                'C': 1.0,
                'random_state': 42,
                'max_iter': 1000,
                'solver': 'lbfgs'
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            }
        }

        # Override with config if provided
        if 'meta_params' in self.config:
            self.meta_params[self.meta_learner_type].update(
                self.config['meta_params'])

        logger.info(
            f"MetaLearner initialized with type: {self.meta_learner_type}")

    def train(self, base_predictions: np.ndarray, y: pd.Series,
              validation_data: Optional[Tuple[np.ndarray, pd.Series]] = None) -> Dict[str, Any]:
        """
        Train meta-learner on base model predictions.

        Args:
            base_predictions: Predictions from base models (n_samples, n_base_models)
            y: True labels
            validation_data: Optional validation data (base_pred_val, y_val)

        Returns:
            Training results dictionary
        """

        logger.info(
            f"Training meta-learner on {len(base_predictions)} samples with {base_predictions.shape[1]} base models")

        # Initialize meta-learner based on type
        if self.meta_learner_type == 'logistic_regression':
            self.model = LogisticRegression(
                **self.meta_params['logistic_regression'])
        elif self.meta_learner_type == 'random_forest':
            self.model = RandomForestClassifier(
                **self.meta_params['random_forest'])
        else:
            raise ValueError(
                f"Unsupported meta-learner type: {self.meta_learner_type}")

        # Cross-validation during training
        cv_scores = cross_val_score(
            self.model, base_predictions, y,
            cv=self.cv_folds, scoring='accuracy'
        )

        # Train meta-learner
        self.model.fit(base_predictions, y)
        self.is_trained = True

        # Extract feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.Series(
                self.model.feature_importances_,
                index=[f'base_model_{i}' for i in range(
                    base_predictions.shape[1])]
            ).sort_values(ascending=False)
        elif hasattr(self.model, 'coef_'):
            # For logistic regression, use absolute coefficients
            self.feature_importance_ = pd.Series(
                np.abs(self.model.coef_[0]),
                index=[f'base_model_{i}' for i in range(
                    base_predictions.shape[1])]
            ).sort_values(ascending=False)

        # Training metrics
        train_score = self.model.score(base_predictions, y)

        results = {
            'model_type': f'MetaLearner_{self.meta_learner_type}',
            'train_accuracy': train_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_base_models': base_predictions.shape[1],
            'n_samples': len(base_predictions),
            'feature_importance': self.feature_importance_.to_dict() if self.feature_importance_ is not None else {},
            'hyperparameters': self.meta_params[self.meta_learner_type]
        }

        # Validation evaluation if provided
        if validation_data:
            base_pred_val, y_val = validation_data
            val_score = self.model.score(base_pred_val, y_val)
            results['validation_accuracy'] = val_score

            # Detailed validation metrics
            y_val_pred = self.model.predict(base_pred_val)
            results['validation_classification_report'] = classification_report(
                y_val, y_val_pred, output_dict=True
            )

        self.training_history.append(results)
        logger.info(
            f"Meta-learner training completed. Train accuracy: {train_score:.4f}")

        return results

    def predict(self, base_predictions: np.ndarray) -> np.ndarray:
        """Make predictions using meta-learner."""

        if not self.is_trained:
            raise ValueError(
                "Meta-learner must be trained before making predictions")

        return self.model.predict(base_predictions)

    def predict_proba(self, base_predictions: np.ndarray) -> np.ndarray:
        """Get prediction probabilities from meta-learner."""

        if not self.is_trained:
            raise ValueError(
                "Meta-learner must be trained before making predictions")

        return self.model.predict_proba(base_predictions)

    def get_base_model_weights(self) -> Dict[str, float]:
        """Get weights/importance of each base model."""

        if not self.is_trained or self.feature_importance_ is None:
            return {}

        return self.feature_importance_.to_dict()

    def save_model(self, filepath: str) -> None:
        """Save trained meta-learner to disk."""

        if not self.is_trained:
            raise ValueError("Cannot save untrained meta-learner")

        model_data = {
            'model': self.model,
            'feature_importance': self.feature_importance_,
            'meta_learner_type': self.meta_learner_type,
            'meta_params': self.meta_params,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Meta-learner saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load trained meta-learner from disk."""

        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.feature_importance_ = model_data['feature_importance']
        self.meta_learner_type = model_data['meta_learner_type']
        self.meta_params = model_data['meta_params']
        self.training_history = model_data['training_history']
        self.is_trained = model_data['is_trained']

        logger.info(f"Meta-learner loaded from {filepath}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive meta-learner information."""

        if not self.is_trained:
            return {'status': 'not_trained'}

        return {
            'model_type': f'MetaLearner_{self.meta_learner_type}',
            'is_trained': self.is_trained,
            'meta_learner_type': self.meta_learner_type,
            'n_base_models': len(self.feature_importance_) if self.feature_importance_ is not None else 0,
            'base_model_weights': self.get_base_model_weights(),
            'training_history_count': len(self.training_history)
        }

    def explain_prediction(self, base_predictions: np.ndarray, sample_idx: int = 0) -> Dict[str, Any]:
        """Explain meta-learner prediction for a specific sample."""

        if not self.is_trained:
            raise ValueError(
                "Meta-learner must be trained to explain predictions")

        if sample_idx >= len(base_predictions):
            raise ValueError(f"Sample index {sample_idx} out of range")

        # Get prediction and probability
        prediction = self.predict(base_predictions[sample_idx:sample_idx+1])[0]
        probability = self.predict_proba(
            base_predictions[sample_idx:sample_idx+1])[0]

        # Get base model contributions
        sample_predictions = base_predictions[sample_idx]
        base_model_contributions = {}

        for i, pred in enumerate(sample_predictions):
            weight = self.feature_importance_.iloc[i] if self.feature_importance_ is not None else 1.0
            base_model_contributions[f'base_model_{i}'] = {
                'prediction': pred,
                'weight': weight,
                'contribution': pred * weight
            }

        return {
            'prediction': int(prediction),
            'probability': probability.tolist(),
            'base_model_contributions': base_model_contributions,
            'base_model_predictions': sample_predictions.tolist()
        }


class StackingEnsemble:
    """Helper class for generating stacking predictions."""

    @staticmethod
    def generate_stacking_data(base_models: List[Any], X: pd.DataFrame,
                               y: pd.Series = None, cv_folds: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate stacking data using cross-validation to prevent overfitting.

        Args:
            base_models: List of trained base models
            X: Features
            y: Labels (optional, for stratified CV)
            cv_folds: Number of CV folds

        Returns:
            Tuple of (stacking_predictions, true_labels)
        """

        n_samples = len(X)
        n_models = len(base_models)

        # Initialize stacking predictions array
        stacking_predictions = np.zeros((n_samples, n_models))

        if y is not None:
            # Use stratified K-fold
            skf = StratifiedKFold(
                n_splits=cv_folds, shuffle=True, random_state=42)
            splits = skf.split(X, y)
        else:
            # Use regular K-fold
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            splits = kf.split(X)

        # Generate out-of-fold predictions for each base model
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]

            for model_idx, model in enumerate(base_models):
                # Train model on fold training data
                model.fit(X_train_fold,
                          y.iloc[train_idx] if y is not None else None)

                # Predict on fold validation data
                if hasattr(model, 'predict_proba'):
                    # Use probabilities if available
                    fold_predictions = model.predict_proba(
                        X_val_fold)[:, 1]  # Binary classification
                else:
                    # Use raw predictions
                    fold_predictions = model.predict(X_val_fold)

                stacking_predictions[val_idx, model_idx] = fold_predictions

        return stacking_predictions, y.values if y is not None else None


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

    # Create dummy base models for testing
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    base_models = [
        RandomForestClassifier(n_estimators=50, random_state=42),
        LogisticRegression(random_state=42)
    ]

    # Generate stacking data
    stacking_predictions, _ = StackingEnsemble.generate_stacking_data(
        base_models, X, y)

    # Initialize and train meta-learner
    meta_learner = MetaLearner({'meta_learner_type': 'logistic_regression'})

    # Split stacking data
    from sklearn.model_selection import train_test_split
    X_stack_train, X_stack_test, y_train, y_test = train_test_split(
        stacking_predictions, y, test_size=0.2, random_state=42
    )

    # Train meta-learner
    results = meta_learner.train(
        X_stack_train, y_train, validation_data=(X_stack_test, y_test))

    print("Meta-Learner Training Results:")
    print(f"Train Accuracy: {results['train_accuracy']:.4f}")
    print(f"Validation Accuracy: {results['validation_accuracy']:.4f}")
    print(f"CV Score: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")

    # Test predictions
    predictions = meta_learner.predict(X_stack_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Base model weights
    print("\nBase Model Weights:")
    print(meta_learner.get_base_model_weights())

    # Model info
    print("\nMeta-Learner Info:")
    print(meta_learner.get_model_info())
