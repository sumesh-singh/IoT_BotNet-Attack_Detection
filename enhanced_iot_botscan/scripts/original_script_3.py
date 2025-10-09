# Now let's start implementing the core components, beginning with the hybrid ensemble

# 1. Core ensemble implementation - hybrid_ensemble.py
hybrid_ensemble_content = '''"""
Enhanced IoT BotScan - Hybrid Ensemble Learning Module
Author: Kotiwale Sumesh Singh (160124862043)

This module implements the hybrid ensemble architecture combining 
Random Forest, XGBoost, and LightGBM with a meta-learner for 
superior botnet detection performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import joblib
import logging
import os
from datetime import datetime

from .random_forest_model import RandomForestModel
from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel
from .meta_learner import MetaLearner
from ..preprocessing.feature_engineer import FeatureEngineer
from ...utils.logger import get_logger
from ...utils.config_manager import ConfigManager

logger = get_logger(__name__)

class HybridEnsemble:
    """
    Hybrid Ensemble Learning system combining Random Forest, XGBoost, and LightGBM
    with a meta-learner for enhanced IoT botnet detection.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the hybrid ensemble system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigManager(config_path)
        self.logger = logger
        
        # Initialize base models
        self.base_models = self._initialize_base_models()
        self.meta_learner = MetaLearner(self.config.get_ml_config()['meta_learner'])
        self.feature_engineer = FeatureEngineer(self.config.get_feature_config())
        
        # Ensemble components
        self.ensemble = None
        self.is_trained = False
        self.model_weights = None
        self.feature_importance = None
        
        # Performance tracking
        self.training_history = {}
        self.validation_scores = {}
        
        self.logger.info("HybridEnsemble initialized successfully")
    
    def _initialize_base_models(self) -> Dict[str, Any]:
        """Initialize the base models with configuration parameters."""
        ml_config = self.config.get_ml_config()
        
        base_models = {
            'random_forest': RandomForestModel(ml_config['ensemble']['algorithms'][0]),
            'xgboost': XGBoostModel(ml_config['ensemble']['algorithms'][1]),
            'lightgbm': LightGBMModel(ml_config['ensemble']['algorithms'][2])
        }
        
        self.logger.info(f"Initialized base models: {list(base_models.keys())}")
        return base_models
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare and preprocess data for training.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Preprocessed features and targets
        """
        self.logger.info(f"Preparing data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Feature engineering
        X_processed = self.feature_engineer.fit_transform(X)
        
        # Convert to numpy arrays
        X_processed = np.array(X_processed)
        y_processed = np.array(y)
        
        self.logger.info(f"Data prepared: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
        return X_processed, y_processed
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_data: Optional[Tuple] = None) -> Dict[str, Any]:
        """
        Train the hybrid ensemble model.
        
        Args:
            X: Training feature matrix
            y: Training target vector
            validation_data: Optional validation data tuple (X_val, y_val)
            
        Returns:
            Training results and metrics
        """
        start_time = datetime.now()
        self.logger.info("Starting hybrid ensemble training")
        
        # Prepare data
        X_processed, y_processed = self.prepare_data(X, y)
        
        # Create base estimators for stacking
        estimators = [
            ('rf', self.base_models['random_forest'].get_sklearn_model()),
            ('xgb', self.base_models['xgboost'].get_sklearn_model()),
            ('lgb', self.base_models['lightgbm'].get_sklearn_model())
        ]
        
        # Create stacking classifier
        self.ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=self.meta_learner.get_sklearn_model(),
            cv=self.config.get_ml_config()['ensemble']['stacking']['cv_folds'],
            stack_method='auto',
            n_jobs=-1,
            verbose=1
        )
        
        # Train the ensemble
        self.ensemble.fit(X_processed, y_processed)
        self.is_trained = True
        
        # Evaluate training performance
        train_predictions = self.ensemble.predict(X_processed)
        train_accuracy = accuracy_score(y_processed, train_predictions)
        
        # Cross-validation evaluation
        cv_scores = self._cross_validate(X_processed, y_processed)
        
        # Validation evaluation if provided
        validation_results = None
        if validation_data is not None:
            validation_results = self._evaluate_validation(validation_data)
        
        # Calculate feature importance
        self.feature_importance = self._calculate_feature_importance(X_processed)
        
        # Training summary
        training_time = (datetime.now() - start_time).total_seconds()
        training_results = {
            'training_accuracy': train_accuracy,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'training_time': training_time,
            'n_samples': len(X_processed),
            'n_features': X_processed.shape[1],
            'validation_results': validation_results
        }
        
        # Store training history
        self.training_history[datetime.now().isoformat()] = training_results
        
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        self.logger.info(f"Training accuracy: {train_accuracy:.4f}")
        self.logger.info(f"CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores)*2:.4f})")
        
        return training_results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained ensemble.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Predicted class labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess features
        X_processed = self.feature_engineer.transform(X)
        X_processed = np.array(X_processed)
        
        # Make predictions
        predictions = self.ensemble.predict(X_processed)
        
        self.logger.info(f"Made predictions for {len(predictions)} samples")
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities using the trained ensemble.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Predicted class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess features
        X_processed = self.feature_engineer.transform(X)
        X_processed = np.array(X_processed)
        
        # Make probability predictions
        probabilities = self.ensemble.predict_proba(X_processed)
        
        self.logger.info(f"Made probability predictions for {len(probabilities)} samples")
        return probabilities
    
    def get_base_model_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get predictions from individual base models.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary of base model predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_processed = self.feature_engineer.transform(X)
        X_processed = np.array(X_processed)
        
        base_predictions = {}
        for name, estimator in self.ensemble.named_estimators_.items():
            base_predictions[name] = estimator.predict(X_processed)
        
        return base_predictions
    
    def _cross_validate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Perform cross-validation evaluation."""
        cv_folds = self.config.get_ml_config()['training_config']['cross_validation']['folds']
        
        cv = StratifiedKFold(
            n_splits=cv_folds, 
            shuffle=True, 
            random_state=42
        )
        
        scores = cross_val_score(
            self.ensemble, X, y, 
            cv=cv, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        
        return scores
    
    def _evaluate_validation(self, validation_data: Tuple) -> Dict[str, Any]:
        """Evaluate model on validation data."""
        X_val, y_val = validation_data
        X_val_processed = self.feature_engineer.transform(X_val)
        X_val_processed = np.array(X_val_processed)
        
        val_predictions = self.ensemble.predict(X_val_processed)
        val_probabilities = self.ensemble.predict_proba(X_val_processed)
        
        return {
            'accuracy': accuracy_score(y_val, val_predictions),
            'classification_report': classification_report(y_val, val_predictions, output_dict=True),
            'confusion_matrix': confusion_matrix(y_val, val_predictions).tolist(),
            'predictions': val_predictions.tolist(),
            'probabilities': val_probabilities.tolist()
        }
    
    def _calculate_feature_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Calculate aggregated feature importance from base models."""
        feature_names = self.feature_engineer.get_feature_names()
        importance_dict = {name: 0.0 for name in feature_names}
        
        # Get feature importance from models that support it
        for name, estimator in self.ensemble.named_estimators_.items():
            if hasattr(estimator, 'feature_importances_'):
                importances = estimator.feature_importances_
                for i, feature_name in enumerate(feature_names):
                    if i < len(importances):
                        importance_dict[feature_name] += importances[i]
        
        # Normalize importance scores
        total_importance = sum(importance_dict.values())
        if total_importance > 0:
            importance_dict = {k: v/total_importance for k, v in importance_dict.items()}
        
        return importance_dict
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'is_trained': self.is_trained,
            'base_models': list(self.base_models.keys()),
            'meta_learner': self.meta_learner.__class__.__name__,
            'feature_count': len(self.feature_engineer.get_feature_names()) if self.feature_engineer else 0,
            'training_history': self.training_history,
            'feature_importance': self.feature_importance
        }
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'ensemble': self.ensemble,
            'feature_engineer': self.feature_engineer,
            'config': self.config.config,
            'training_history': self.training_history,
            'feature_importance': self.feature_importance,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.ensemble = model_data['ensemble']
        self.feature_engineer = model_data['feature_engineer']
        self.training_history = model_data['training_history']
        self.feature_importance = model_data['feature_importance']
        self.is_trained = model_data['is_trained']
        
        self.logger.info(f"Model loaded from {filepath}")

    def get_ensemble_weights(self) -> Dict[str, float]:
        """Get the learned weights of base models in the ensemble."""
        if not self.is_trained:
            return {}
        
        # For stacking classifier, analyze meta-learner coefficients
        meta_model = self.ensemble.final_estimator_
        if hasattr(meta_model, 'coef_'):
            weights = {}
            coef = meta_model.coef_[0] if len(meta_model.coef_.shape) > 1 else meta_model.coef_
            
            model_names = ['rf', 'xgb', 'lgb']
            for i, name in enumerate(model_names):
                if i < len(coef):
                    weights[name] = float(coef[i])
            
            return weights
        
        return {}
'''

with open('./enhanced_iot_botscan/src/core/ensemble/hybrid_ensemble.py', 'w') as f:
    f.write(hybrid_ensemble_content)

print("✅ Created hybrid_ensemble.py - Main ensemble orchestrator")

# 2. Random Forest Model implementation
rf_model_content = '''"""
Random Forest Model Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Optimized Random Forest implementation for IoT botnet detection
with hyperparameter tuning and performance optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
import logging
from datetime import datetime

from ...utils.logger import get_logger

logger = get_logger(__name__)

class RandomForestModel:
    """
    Enhanced Random Forest model for IoT botnet detection.
    Includes hyperparameter optimization and performance monitoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Random Forest model with configuration.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.logger = logger
        
        # Default parameters
        self.default_params = {
            'n_estimators': config.get('n_estimators', 100),
            'max_depth': config.get('max_depth', 10),
            'min_samples_split': config.get('min_samples_split', 2),
            'min_samples_leaf': config.get('min_samples_leaf', 1),
            'max_features': config.get('max_features', 'sqrt'),
            'random_state': config.get('random_state', 42),
            'n_jobs': config.get('n_jobs', -1),
            'oob_score': True,
            'bootstrap': True
        }
        
        # Initialize model
        self.model = RandomForestClassifier(**self.default_params)
        self.is_tuned = False
        self.best_params = None
        self.feature_importance = None
        self.oob_score = None
        
        self.logger.info("RandomForestModel initialized")
    
    def get_sklearn_model(self) -> RandomForestClassifier:
        """Get the sklearn model instance."""
        return self.model
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Random Forest model.
        
        Args:
            X: Training features
            y: Training labels
        """
        start_time = datetime.now()
        self.logger.info(f"Training Random Forest on {len(X)} samples")
        
        # Train the model
        self.model.fit(X, y)
        
        # Store feature importance and OOB score
        self.feature_importance = self.model.feature_importances_
        self.oob_score = self.model.oob_score_
        
        training_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Random Forest training completed in {training_time:.2f} seconds")
        self.logger.info(f"OOB Score: {self.oob_score:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, 
                            method: str = 'grid_search', cv: int = 5) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning.
        
        Args:
            X: Training features
            y: Training labels
            method: Tuning method ('grid_search' or 'random_search')
            cv: Number of cross-validation folds
            
        Returns:
            Best parameters and scores
        """
        self.logger.info(f"Starting hyperparameter tuning using {method}")
        
        # Parameter grid for tuning
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        # Reduced parameter grid for random search
        if method == 'random_search':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }
        
        # Create search object
        if method == 'grid_search':
            search = GridSearchCV(
                RandomForestClassifier(random_state=42, n_jobs=-1),
                param_grid=param_grid,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
        else:  # random_search
            search = RandomizedSearchCV(
                RandomForestClassifier(random_state=42, n_jobs=-1),
                param_distributions=param_grid,
                n_iter=50,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
        
        # Perform search
        start_time = datetime.now()
        search.fit(X, y)
        search_time = (datetime.now() - start_time).total_seconds()
        
        # Update model with best parameters
        self.best_params = search.best_params_
        self.model = search.best_estimator_
        self.is_tuned = True
        
        tuning_results = {
            'best_params': self.best_params,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_,
            'search_time': search_time
        }
        
        self.logger.info(f"Hyperparameter tuning completed in {search_time:.2f} seconds")
        self.logger.info(f"Best CV score: {search.best_score_:.4f}")
        self.logger.info(f"Best parameters: {self.best_params}")
        
        return tuning_results
    
    def get_feature_importance(self, feature_names: Optional[list] = None) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature importance scores
        """
        if self.feature_importance is None:
            return {}
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importance))]
        
        importance_dict = dict(zip(feature_names, self.feature_importance))
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), 
                                    key=lambda x: x[1], reverse=True))
        
        return importance_dict
    
    def get_tree_info(self) -> Dict[str, Any]:
        """Get information about the trees in the forest."""
        if not hasattr(self.model, 'estimators_'):
            return {}
        
        tree_depths = []
        tree_nodes = []
        tree_leaves = []
        
        for tree in self.model.estimators_:
            tree_depths.append(tree.tree_.max_depth)
            tree_nodes.append(tree.tree_.node_count)
            tree_leaves.append(tree.tree_.n_leaves)
        
        return {
            'n_trees': len(self.model.estimators_),
            'avg_depth': np.mean(tree_depths),
            'max_depth': np.max(tree_depths),
            'min_depth': np.min(tree_depths),
            'avg_nodes': np.mean(tree_nodes),
            'avg_leaves': np.mean(tree_leaves),
            'oob_score': self.oob_score
        }
    
    def get_model_complexity(self) -> Dict[str, Any]:
        """Calculate model complexity metrics."""
        complexity_metrics = {
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'min_samples_split': self.model.min_samples_split,
            'min_samples_leaf': self.model.min_samples_leaf,
            'max_features': self.model.max_features,
        }
        
        # Add tree statistics if available
        tree_info = self.get_tree_info()
        complexity_metrics.update(tree_info)
        
        return complexity_metrics
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        evaluation_results = {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist(),
            'oob_score': self.oob_score
        }
        
        self.logger.info(f"Model evaluation - Accuracy: {accuracy:.4f}")
        
        return evaluation_results
'''

with open('./enhanced_iot_botscan/src/core/ensemble/random_forest_model.py', 'w') as f:
    f.write(rf_model_content)

print("✅ Created random_forest_model.py - Random Forest implementation")

print("\n🚀 Core ensemble implementation started!")
print("📁 Created core ensemble files:")
print("   - hybrid_ensemble.py (Main ensemble orchestrator)")
print("   - random_forest_model.py (RF implementation)")
print("\n🔄 Next steps:")
print("   1. XGBoost and LightGBM model implementations")
print("   2. Meta-learner implementation")
print("   3. Adversarial training modules")
print("   4. Concept drift detection"))