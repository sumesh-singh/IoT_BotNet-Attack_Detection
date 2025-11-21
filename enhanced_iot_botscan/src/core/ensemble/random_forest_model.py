"""
Random Forest Model Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Implements Random Forest classifier with optimized hyperparameters for IoT botnet detection.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)


class RandomForestModel:
    """Random Forest classifier optimized for IoT botnet detection."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Random Forest model with configuration."""

        self.config = config or {}
        self.model = None
        self.is_trained = False
        self.feature_importance_ = None
        self.training_history = []

        # Default hyperparameters optimized for IoT botnet detection
        self.default_params = {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced'
        }

        # Override with config if provided
        self.params = {**self.default_params, **
                       self.config.get('random_forest', {})}

        logger.info(
            f"RandomForestModel initialized with params: {self.params}")

    def train(self, X: pd.DataFrame, y: pd.Series,
              validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
              optimize_hyperparams: bool = False) -> Dict[str, Any]:
        """
        Train Random Forest model.

        Args:
            X: Training features
            y: Training labels
            validation_data: Optional validation data (X_val, y_val)
            optimize_hyperparams: Whether to perform hyperparameter optimization

        Returns:
            Training results dictionary
        """

        logger.info(
            f"Training Random Forest on {len(X)} samples with {len(X.columns)} features")

        # Hyperparameter optimization if requested
        if optimize_hyperparams:
            self._optimize_hyperparameters(X, y)

        # Initialize and train model
        self.model = RandomForestClassifier(**self.params)

        # Cross-validation during training
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')

        # Train on full dataset
        self.model.fit(X, y)
        self.is_trained = True

        # Extract feature importance
        self.feature_importance_ = pd.Series(
            self.model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

        # Training metrics
        train_score = self.model.score(X, y)

        results = {
            'model_type': 'RandomForest',
            'train_accuracy': train_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_features': len(X.columns),
            'n_samples': len(X),
            'feature_importance': self.feature_importance_.to_dict(),
            'hyperparameters': self.params
        }

        # Validation evaluation if provided
        if validation_data:
            X_val, y_val = validation_data
            val_score = self.model.score(X_val, y_val)
            results['validation_accuracy'] = val_score

            # Detailed validation metrics
            y_val_pred = self.model.predict(X_val)
            results['validation_classification_report'] = classification_report(
                y_val, y_val_pred, output_dict=True
            )

        self.training_history.append(results)
        logger.info(
            f"Random Forest training completed. Train accuracy: {train_score:.4f}")

        return results

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RandomForestModel':
        """Scikit-learn compatible fit method."""
        self.train(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""

        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""

        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict_proba(X)

    def get_feature_importance(self, top_n: int = 20) -> pd.Series:
        """Get top N most important features."""

        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")

        return self.feature_importance_.head(top_n)

    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Perform hyperparameter optimization using GridSearchCV."""

        logger.info("Starting hyperparameter optimization for Random Forest")

        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }

        # Grid search with cross-validation
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X, y)

        # Update parameters with best found
        self.params.update(grid_search.best_params_)

        logger.info(f"Best parameters found: {grid_search.best_params_}")
        logger.info(
            f"Best cross-validation score: {grid_search.best_score_:.4f}")

    def save_model(self, filepath: str) -> None:
        """Save trained model to disk."""

        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_data = {
            'model': self.model,
            'feature_importance': self.feature_importance_,
            'params': self.params,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Random Forest model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load trained model from disk."""

        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.feature_importance_ = model_data['feature_importance']
        self.params = model_data['params']
        self.training_history = model_data['training_history']
        self.is_trained = model_data['is_trained']

        logger.info(f"Random Forest model loaded from {filepath}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""

        if not self.is_trained:
            return {'status': 'not_trained'}

        return {
            'model_type': 'RandomForest',
            'is_trained': self.is_trained,
            'n_estimators': self.params.get('n_estimators'),
            'max_depth': self.params.get('max_depth'),
            'feature_count': len(self.feature_importance_) if self.feature_importance_ is not None else 0,
            'top_features': self.get_feature_importance(10).to_dict() if self.feature_importance_ is not None else {},
            'training_history_count': len(self.training_history)
        }

    def explain_prediction(self, X: pd.DataFrame, sample_idx: int = 0) -> Dict[str, Any]:
        """Explain prediction for a specific sample using feature importance."""

        if not self.is_trained:
            raise ValueError("Model must be trained to explain predictions")

        if sample_idx >= len(X):
            raise ValueError(f"Sample index {sample_idx} out of range")

        # Get prediction and probability
        prediction = self.predict(X.iloc[[sample_idx]])[0]
        probability = self.predict_proba(X.iloc[[sample_idx]])[0]

        # Get feature contributions (simplified using feature importance)
        sample_features = X.iloc[sample_idx]
        feature_contributions = {}

        for feature, importance in self.feature_importance_.items():
            if feature in sample_features.index:
                feature_contributions[feature] = {
                    'value': sample_features[feature],
                    'importance': importance,
                    'contribution': sample_features[feature] * importance
                }

        return {
            'prediction': int(prediction),
            'probability': probability.tolist(),
            'feature_contributions': feature_contributions,
            'top_contributing_features': sorted(
                feature_contributions.items(),
                key=lambda x: abs(x[1]['contribution']),
                reverse=True
            )[:10]
        }


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

    # Initialize and train model
    rf_model = RandomForestModel()

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train model
    results = rf_model.train(
        X_train, y_train, validation_data=(X_test, y_test))

    print("Training Results:")
    print(f"Train Accuracy: {results['train_accuracy']:.4f}")
    print(f"Validation Accuracy: {results['validation_accuracy']:.4f}")
    print(f"CV Score: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")

    # Test predictions
    predictions = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Feature importance
    print("\nTop 10 Most Important Features:")
    print(rf_model.get_feature_importance(10))

    # Model info
    print("\nModel Info:")
    print(rf_model.get_model_info())
