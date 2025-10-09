# Fix the syntax error and continue with XGBoost implementation

# 2. XGBoost Model implementation
xgboost_model_content = '''"""
XGBoost Model Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Optimized XGBoost implementation for IoT botnet detection
with advanced hyperparameter tuning and early stopping.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, log_loss
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from ...utils.logger import get_logger

logger = get_logger(__name__)

class XGBoostModel:
    """
    Enhanced XGBoost model for IoT botnet detection.
    Includes advanced hyperparameter optimization, early stopping,
    and GPU acceleration support.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize XGBoost model with configuration.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.logger = logger
        
        # Default parameters optimized for IoT botnet detection
        self.default_params = {
            'n_estimators': config.get('n_estimators', 100),
            'max_depth': config.get('max_depth', 6),
            'learning_rate': config.get('learning_rate', 0.1),
            'subsample': config.get('subsample', 0.8),
            'colsample_bytree': config.get('colsample_bytree', 0.8),
            'colsample_bylevel': config.get('colsample_bylevel', 1.0),
            'colsample_bynode': config.get('colsample_bynode', 1.0),
            'reg_alpha': config.get('reg_alpha', 0),
            'reg_lambda': config.get('reg_lambda', 1),
            'random_state': config.get('random_state', 42),
            'n_jobs': config.get('n_jobs', -1),
            'verbosity': 0,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'auto',  # 'gpu_hist' for GPU acceleration
            'early_stopping_rounds': 10
        }
        
        # Initialize model
        self.model = xgb.XGBClassifier(**self.default_params)
        self.is_tuned = False
        self.best_params = None
        self.feature_importance = None
        self.training_history = []
        self.validation_scores = []
        
        self.logger.info("XGBoostModel initialized")
    
    def get_sklearn_model(self) -> xgb.XGBClassifier:
        """Get the sklearn-compatible XGBoost model instance."""
        return self.model
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            eval_set: Optional[Tuple] = None, 
            early_stopping_rounds: Optional[int] = None) -> Dict[str, Any]:
        """
        Train the XGBoost model with optional early stopping.
        
        Args:
            X: Training features
            y: Training labels
            eval_set: Validation set for early stopping (X_val, y_val)
            early_stopping_rounds: Early stopping patience
            
        Returns:
            Training results and metrics
        """
        start_time = datetime.now()
        self.logger.info(f"Training XGBoost on {len(X)} samples")
        
        # Prepare training parameters
        fit_params = {}
        if eval_set is not None:
            fit_params['eval_set'] = [eval_set]
            fit_params['verbose'] = False
            
            if early_stopping_rounds is not None:
                fit_params['early_stopping_rounds'] = early_stopping_rounds
        
        # Train the model
        self.model.fit(X, y, **fit_params)
        
        # Store feature importance
        self.feature_importance = self.model.feature_importances_
        
        # Get training history if available
        if hasattr(self.model, 'evals_result_'):
            self.training_history = self.model.evals_result_
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate training metrics
        train_pred = self.model.predict(X)
        train_proba = self.model.predict_proba(X)
        train_accuracy = accuracy_score(y, train_pred)
        train_logloss = log_loss(y, train_proba)
        
        training_results = {
            'training_time': training_time,
            'train_accuracy': train_accuracy,
            'train_logloss': train_logloss,
            'best_iteration': getattr(self.model, 'best_iteration', None),
            'n_estimators_final': self.model.n_estimators
        }
        
        self.logger.info(f"XGBoost training completed in {training_time:.2f} seconds")
        self.logger.info(f"Training accuracy: {train_accuracy:.4f}")
        self.logger.info(f"Training log loss: {train_logloss:.4f}")
        
        if hasattr(self.model, 'best_iteration'):
            self.logger.info(f"Best iteration: {self.model.best_iteration}")
        
        return training_results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, 
                            method: str = 'grid_search', 
                            cv: int = 5,
                            eval_set: Optional[Tuple] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning with advanced parameter space.
        
        Args:
            X: Training features
            y: Training labels
            method: Tuning method ('grid_search', 'random_search', or 'bayesian')
            cv: Number of cross-validation folds
            eval_set: Validation set for early stopping
            
        Returns:
            Best parameters and scores
        """
        self.logger.info(f"Starting XGBoost hyperparameter tuning using {method}")
        
        # Comprehensive parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0, 1.5]
        }
        
        # Reduced grid for faster tuning
        if method == 'random_search':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [1.0, 1.5]
            }
        
        # Create base model for tuning
        base_model = xgb.XGBClassifier(
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            eval_metric='logloss'
        )
        
        # Create search object
        if method == 'grid_search':
            search = GridSearchCV(
                base_model,
                param_grid=param_grid,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
        else:  # random_search
            search = RandomizedSearchCV(
                base_model,
                param_distributions=param_grid,
                n_iter=100,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
        
        # Prepare fit parameters for search
        fit_params = {}
        if eval_set is not None:
            # For cross-validation, we can't use the same eval_set
            # This would need to be handled differently in practice
            pass
        
        # Perform search
        start_time = datetime.now()
        search.fit(X, y, **fit_params)
        search_time = (datetime.now() - start_time).total_seconds()
        
        # Update model with best parameters
        self.best_params = search.best_params_
        self.model = search.best_estimator_
        self.is_tuned = True
        
        tuning_results = {
            'best_params': self.best_params,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_,
            'search_time': search_time,
            'n_iterations': len(search.cv_results_['mean_test_score'])
        }
        
        self.logger.info(f"Hyperparameter tuning completed in {search_time:.2f} seconds")
        self.logger.info(f"Best CV score: {search.best_score_:.4f}")
        self.logger.info(f"Best parameters: {self.best_params}")
        
        return tuning_results
    
    def get_feature_importance(self, importance_type: str = 'weight', 
                             feature_names: Optional[list] = None) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover')
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature importance scores
        """
        if self.feature_importance is None:
            return {}
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importance))]
        
        # Get importance based on type
        if importance_type == 'weight':
            importance_values = self.feature_importance
        else:
            # For other types, use the booster method
            if hasattr(self.model, 'get_booster'):
                booster = self.model.get_booster()
                importance_dict = booster.get_score(importance_type=importance_type)
                # Convert to array format matching feature names
                importance_values = np.array([importance_dict.get(f'f{i}', 0.0) 
                                            for i in range(len(feature_names))])
            else:
                importance_values = self.feature_importance
        
        importance_dict = dict(zip(feature_names, importance_values))
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), 
                                    key=lambda x: x[1], reverse=True))
        
        return importance_dict
    
    def get_training_history(self) -> Dict[str, Any]:
        """Get training history if available."""
        return {
            'training_history': self.training_history,
            'validation_scores': self.validation_scores,
            'best_iteration': getattr(self.model, 'best_iteration', None)
        }
    
    def plot_importance(self, max_num_features: int = 20, 
                       importance_type: str = 'weight') -> None:
        """Plot feature importance (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            
            # Get feature importance
            importance = self.get_feature_importance(importance_type=importance_type)
            
            # Select top features
            top_features = dict(list(importance.items())[:max_num_features])
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(top_features)), list(top_features.values()))
            plt.yticks(range(len(top_features)), list(top_features.keys()))
            plt.xlabel(f'Feature Importance ({importance_type})')
            plt.title('XGBoost Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            self.logger.warning("matplotlib not available for plotting")
    
    def get_model_complexity(self) -> Dict[str, Any]:
        """Calculate model complexity metrics."""
        complexity_metrics = {
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'learning_rate': self.model.learning_rate,
            'subsample': self.model.subsample,
            'colsample_bytree': self.model.colsample_bytree,
            'reg_alpha': self.model.reg_alpha,
            'reg_lambda': self.model.reg_lambda,
        }
        
        # Add booster information if available
        if hasattr(self.model, 'get_booster'):
            try:
                booster = self.model.get_booster()
                complexity_metrics.update({
                    'num_boosted_rounds': booster.num_boosted_rounds(),
                    'best_iteration': getattr(self.model, 'best_iteration', None)
                })
            except Exception as e:
                self.logger.warning(f"Could not get booster info: {e}")
        
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
        logloss = log_loss(y_test, y_proba)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        evaluation_results = {
            'accuracy': accuracy,
            'log_loss': logloss,
            'classification_report': report,
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist()
        }
        
        self.logger.info(f"XGBoost evaluation - Accuracy: {accuracy:.4f}, Log Loss: {logloss:.4f}")
        
        return evaluation_results
    
    def save_model_native(self, filepath: str) -> None:
        """Save model in XGBoost native format."""
        if hasattr(self.model, 'get_booster'):
            booster = self.model.get_booster()
            booster.save_model(filepath)
            self.logger.info(f"XGBoost model saved in native format to {filepath}")
        else:
            self.logger.warning("Cannot save in native format - model not properly trained")
    
    def load_model_native(self, filepath: str) -> None:
        """Load model from XGBoost native format."""
        try:
            self.model.load_model(filepath)
            self.logger.info(f"XGBoost model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
'''

with open('./enhanced_iot_botscan/src/core/ensemble/xgboost_model.py', 'w') as f:
    f.write(xgboost_model_content)

print("✅ Created xgboost_model.py - XGBoost implementation")

# 3. LightGBM Model implementation
lightgbm_model_content = '''"""
LightGBM Model Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Optimized LightGBM implementation for IoT botnet detection
with categorical feature handling and advanced optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, log_loss
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from ...utils.logger import get_logger

logger = get_logger(__name__)

class LightGBMModel:
    """
    Enhanced LightGBM model for IoT botnet detection.
    Includes advanced hyperparameter optimization, categorical feature handling,
    and memory-efficient training.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LightGBM model with configuration.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.logger = logger
        
        # Default parameters optimized for IoT botnet detection
        self.default_params = {
            'n_estimators': config.get('n_estimators', 100),
            'max_depth': config.get('max_depth', 6),
            'learning_rate': config.get('learning_rate', 0.1),
            'num_leaves': config.get('num_leaves', 31),
            'subsample': config.get('subsample', 0.8),
            'subsample_freq': config.get('subsample_freq', 1),
            'colsample_bytree': config.get('colsample_bytree', 0.8),
            'reg_alpha': config.get('reg_alpha', 0),
            'reg_lambda': config.get('reg_lambda', 1),
            'min_child_samples': config.get('min_child_samples', 20),
            'min_child_weight': config.get('min_child_weight', 1e-3),
            'random_state': config.get('random_state', 42),
            'n_jobs': config.get('n_jobs', -1),
            'verbosity': -1,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'feature_fraction': config.get('feature_fraction', 0.9),
            'bagging_fraction': config.get('bagging_fraction', 0.8),
            'bagging_freq': config.get('bagging_freq', 5),
            'early_stopping_rounds': 10
        }
        
        # Initialize model
        self.model = lgb.LGBMClassifier(**self.default_params)
        self.is_tuned = False
        self.best_params = None
        self.feature_importance = None
        self.training_history = []
        self.categorical_features = []
        
        self.logger.info("LightGBMModel initialized")
    
    def get_sklearn_model(self) -> lgb.LGBMClassifier:
        """Get the sklearn-compatible LightGBM model instance."""
        return self.model
    
    def set_categorical_features(self, categorical_features: List[int]) -> None:
        """
        Set categorical feature indices.
        
        Args:
            categorical_features: List of categorical feature indices
        """
        self.categorical_features = categorical_features
        self.model.set_params(categorical_feature=categorical_features)
        self.logger.info(f"Set {len(categorical_features)} categorical features")
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            eval_set: Optional[Tuple] = None, 
            early_stopping_rounds: Optional[int] = None,
            categorical_features: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Train the LightGBM model with optional early stopping.
        
        Args:
            X: Training features
            y: Training labels
            eval_set: Validation set for early stopping (X_val, y_val)
            early_stopping_rounds: Early stopping patience
            categorical_features: List of categorical feature indices
            
        Returns:
            Training results and metrics
        """
        start_time = datetime.now()
        self.logger.info(f"Training LightGBM on {len(X)} samples")
        
        # Set categorical features if provided
        if categorical_features is not None:
            self.set_categorical_features(categorical_features)
        
        # Prepare training parameters
        fit_params = {}
        if eval_set is not None:
            fit_params['eval_set'] = [eval_set]
            fit_params['eval_metric'] = 'binary_logloss'
            fit_params['verbose'] = False
            
            if early_stopping_rounds is not None:
                fit_params['early_stopping_rounds'] = early_stopping_rounds
        
        if self.categorical_features:
            fit_params['categorical_feature'] = self.categorical_features
        
        # Train the model
        self.model.fit(X, y, **fit_params)
        
        # Store feature importance
        self.feature_importance = self.model.feature_importances_
        
        # Get training history if available
        if hasattr(self.model, 'evals_result_'):
            self.training_history = self.model.evals_result_
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate training metrics
        train_pred = self.model.predict(X)
        train_proba = self.model.predict_proba(X)
        train_accuracy = accuracy_score(y, train_pred)
        train_logloss = log_loss(y, train_proba)
        
        training_results = {
            'training_time': training_time,
            'train_accuracy': train_accuracy,
            'train_logloss': train_logloss,
            'best_iteration': getattr(self.model, 'best_iteration', None),
            'n_estimators_final': self.model.n_estimators,
            'num_leaves_final': self.model.num_leaves
        }
        
        self.logger.info(f"LightGBM training completed in {training_time:.2f} seconds")
        self.logger.info(f"Training accuracy: {train_accuracy:.4f}")
        self.logger.info(f"Training log loss: {train_logloss:.4f}")
        
        if hasattr(self.model, 'best_iteration'):
            self.logger.info(f"Best iteration: {self.model.best_iteration}")
        
        return training_results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, 
                            method: str = 'grid_search', 
                            cv: int = 5,
                            eval_set: Optional[Tuple] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning with LightGBM-specific parameters.
        
        Args:
            X: Training features
            y: Training labels
            method: Tuning method ('grid_search' or 'random_search')
            cv: Number of cross-validation folds
            eval_set: Validation set for early stopping
            
        Returns:
            Best parameters and scores
        """
        self.logger.info(f"Starting LightGBM hyperparameter tuning using {method}")
        
        # Comprehensive parameter grid for LightGBM
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, -1],  # -1 means no limit
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves': [15, 31, 63, 127],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0],
            'min_child_samples': [10, 20, 30],
            'feature_fraction': [0.8, 0.9, 1.0]
        }
        
        # Reduced grid for faster tuning
        if method == 'random_search':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 7, -1],
                'learning_rate': [0.05, 0.1, 0.15],
                'num_leaves': [31, 63],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [0.5, 1.0],
                'min_child_samples': [20, 30]
            }
        
        # Create base model for tuning
        base_model = lgb.LGBMClassifier(
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
            objective='binary',
            metric='binary_logloss',
            categorical_feature=self.categorical_features if self.categorical_features else 'auto'
        )
        
        # Create search object
        if method == 'grid_search':
            search = GridSearchCV(
                base_model,
                param_grid=param_grid,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
        else:  # random_search
            search = RandomizedSearchCV(
                base_model,
                param_distributions=param_grid,
                n_iter=100,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
        
        # Prepare fit parameters for search
        fit_params = {}
        if self.categorical_features:
            fit_params['categorical_feature'] = self.categorical_features
        
        # Perform search
        start_time = datetime.now()
        search.fit(X, y, **fit_params)
        search_time = (datetime.now() - start_time).total_seconds()
        
        # Update model with best parameters
        self.best_params = search.best_params_
        self.model = search.best_estimator_
        self.is_tuned = True
        
        tuning_results = {
            'best_params': self.best_params,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_,
            'search_time': search_time,
            'n_iterations': len(search.cv_results_['mean_test_score'])
        }
        
        self.logger.info(f"Hyperparameter tuning completed in {search_time:.2f} seconds")
        self.logger.info(f"Best CV score: {search.best_score_:.4f}")
        self.logger.info(f"Best parameters: {self.best_params}")
        
        return tuning_results
    
    def get_feature_importance(self, importance_type: str = 'split', 
                             feature_names: Optional[list] = None) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance ('split' or 'gain')
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature importance scores
        """
        if self.feature_importance is None:
            return {}
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importance))]
        
        # Get importance based on type
        if importance_type == 'split':
            importance_values = self.feature_importance
        else:  # gain
            # For gain importance, use the booster method if available
            if hasattr(self.model, 'booster_'):
                try:
                    importance_dict = self.model.booster_.feature_importance(
                        importance_type=importance_type,
                        iteration=self.model.best_iteration
                    )
                    importance_values = np.array([importance_dict[i] if i < len(importance_dict) 
                                                else 0.0 for i in range(len(feature_names))])
                except:
                    importance_values = self.feature_importance
            else:
                importance_values = self.feature_importance
        
        importance_dict = dict(zip(feature_names, importance_values))
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), 
                                    key=lambda x: x[1], reverse=True))
        
        return importance_dict
    
    def get_training_history(self) -> Dict[str, Any]:
        """Get training history if available."""
        return {
            'training_history': self.training_history,
            'best_iteration': getattr(self.model, 'best_iteration', None),
            'best_score': getattr(self.model, 'best_score', None)
        }
    
    def get_model_complexity(self) -> Dict[str, Any]:
        """Calculate model complexity metrics."""
        complexity_metrics = {
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'learning_rate': self.model.learning_rate,
            'num_leaves': self.model.num_leaves,
            'subsample': self.model.subsample,
            'colsample_bytree': self.model.colsample_bytree,
            'reg_alpha': self.model.reg_alpha,
            'reg_lambda': self.model.reg_lambda,
            'min_child_samples': self.model.min_child_samples,
            'categorical_features_count': len(self.categorical_features)
        }
        
        # Add booster information if available
        if hasattr(self.model, 'booster_'):
            try:
                complexity_metrics.update({
                    'num_trees': self.model.booster_.num_trees(),
                    'best_iteration': getattr(self.model, 'best_iteration', None)
                })
            except Exception as e:
                self.logger.warning(f"Could not get booster info: {e}")
        
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
        logloss = log_loss(y_test, y_proba)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        evaluation_results = {
            'accuracy': accuracy,
            'log_loss': logloss,
            'classification_report': report,
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist()
        }
        
        self.logger.info(f"LightGBM evaluation - Accuracy: {accuracy:.4f}, Log Loss: {logloss:.4f}")
        
        return evaluation_results
    
    def plot_importance(self, max_num_features: int = 20, 
                       importance_type: str = 'split') -> None:
        """Plot feature importance (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            
            # Get feature importance
            importance = self.get_feature_importance(importance_type=importance_type)
            
            # Select top features
            top_features = dict(list(importance.items())[:max_num_features])
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(top_features)), list(top_features.values()))
            plt.yticks(range(len(top_features)), list(top_features.keys()))
            plt.xlabel(f'Feature Importance ({importance_type})')
            plt.title('LightGBM Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            self.logger.warning("matplotlib not available for plotting")
    
    def save_model_native(self, filepath: str) -> None:
        """Save model in LightGBM native format."""
        if hasattr(self.model, 'booster_'):
            self.model.booster_.save_model(filepath)
            self.logger.info(f"LightGBM model saved in native format to {filepath}")
        else:
            self.logger.warning("Cannot save in native format - model not properly trained")
    
    def load_model_native(self, filepath: str) -> None:
        """Load model from LightGBM native format."""
        try:
            # Create new model instance and load booster
            booster = lgb.Booster(model_file=filepath)
            self.model.booster_ = booster
            self.logger.info(f"LightGBM model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
'''

with open('./enhanced_iot_botscan/src/core/ensemble/lightgbm_model.py', 'w') as f:
    f.write(lightgbm_model_content)

print("✅ Created lightgbm_model.py - LightGBM implementation")

print("\n🚀 Core ensemble models implemented!")
print("📁 Created ensemble model files:")
print("   - hybrid_ensemble.py (Main orchestrator)")  
print("   - random_forest_model.py (Random Forest)")
print("   - xgboost_model.py (XGBoost)")
print("   - lightgbm_model.py (LightGBM)")
print("\n🔄 Next steps:")
print("   1. Meta-learner implementation")
print("   2. Adversarial training modules")
print("   3. Concept drift detection")
print("   4. Preprocessing pipeline")