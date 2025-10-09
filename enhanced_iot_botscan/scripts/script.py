# Let's continue implementing ALL missing components
# First, complete the meta-learner implementation

meta_learner_content = '''"""
Meta-Learner Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Meta-learner component that combines predictions from base models
using stacking ensemble technique for superior performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime

from ...utils.logger import get_logger

logger = get_logger(__name__)

class MetaLearner:
    """
    Meta-learner for stacking ensemble that learns to combine
    predictions from base models optimally.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize meta-learner with configuration.
        
        Args:
            config: Meta-learner configuration
        """
        self.config = config
        self.logger = logger
        
        # Algorithm selection
        algorithm = config.get('algorithm', 'logistic_regression')
        self.algorithm_name = algorithm
        
        # Initialize the meta-learner based on config
        self.meta_model = self._initialize_meta_model(algorithm, config)
        self.scaler = StandardScaler()
        
        self.is_trained = False
        self.feature_names = []
        self.training_history = {}
        
        self.logger.info(f"MetaLearner initialized with {algorithm}")
    
    def _initialize_meta_model(self, algorithm: str, config: Dict[str, Any]):
        """Initialize meta-model based on algorithm choice."""
        
        if algorithm == 'logistic_regression':
            return LogisticRegression(
                C=config.get('C', 1.0),
                penalty=config.get('penalty', 'l2'),
                solver=config.get('solver', 'lbfgs'),
                max_iter=config.get('max_iter', 1000),
                random_state=config.get('random_state', 42),
                n_jobs=-1
            )
        
        elif algorithm == 'random_forest':
            return RandomForestClassifier(
                n_estimators=config.get('n_estimators', 100),
                max_depth=config.get('max_depth', 5),
                random_state=config.get('random_state', 42),
                n_jobs=-1
            )
        
        elif algorithm == 'svm':
            return SVC(
                C=config.get('C', 1.0),
                kernel=config.get('kernel', 'rbf'),
                probability=True,
                random_state=config.get('random_state', 42)
            )
        
        elif algorithm == 'neural_network':
            return MLPClassifier(
                hidden_layer_sizes=config.get('hidden_layer_sizes', (100, 50)),
                activation=config.get('activation', 'relu'),
                solver=config.get('solver', 'adam'),
                max_iter=config.get('max_iter', 500),
                random_state=config.get('random_state', 42)
            )
        
        else:
            self.logger.warning(f"Unknown algorithm {algorithm}, defaulting to LogisticRegression")
            return LogisticRegression(random_state=42, n_jobs=-1)
    
    def get_sklearn_model(self):
        """Get the sklearn model instance."""
        return self.meta_model
    
    def prepare_meta_features(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Prepare meta-features from base model predictions.
        
        Args:
            base_predictions: Dictionary of base model predictions
            
        Returns:
            Meta-feature matrix
        """
        # Stack predictions from all base models
        feature_arrays = []
        feature_names = []
        
        for model_name, predictions in base_predictions.items():
            if predictions.ndim == 1:
                # Binary predictions
                feature_arrays.append(predictions.reshape(-1, 1))
                feature_names.append(f"{model_name}_pred")
            else:
                # Probability predictions
                feature_arrays.append(predictions)
                for i in range(predictions.shape[1]):
                    feature_names.append(f"{model_name}_prob_{i}")
        
        self.feature_names = feature_names
        meta_features = np.hstack(feature_arrays)
        
        self.logger.info(f"Prepared meta-features: {meta_features.shape}")
        return meta_features
    
    def fit(self, base_predictions: Dict[str, np.ndarray], y: np.ndarray) -> Dict[str, Any]:
        """
        Train the meta-learner on base model predictions.
        
        Args:
            base_predictions: Dictionary of base model predictions
            y: True labels
            
        Returns:
            Training results
        """
        start_time = datetime.now()
        self.logger.info("Training meta-learner")
        
        # Prepare meta-features
        X_meta = self.prepare_meta_features(base_predictions)
        
        # Scale features if using certain algorithms
        if self.algorithm_name in ['svm', 'neural_network', 'logistic_regression']:
            X_meta_scaled = self.scaler.fit_transform(X_meta)
        else:
            X_meta_scaled = X_meta
        
        # Train meta-model
        self.meta_model.fit(X_meta_scaled, y)
        self.is_trained = True
        
        # Evaluate training performance
        train_pred = self.meta_model.predict(X_meta_scaled)
        train_accuracy = accuracy_score(y, train_pred)
        
        # Cross-validation
        cv_scores = self._cross_validate(X_meta_scaled, y)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        training_results = {
            'training_accuracy': train_accuracy,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'training_time': training_time,
            'n_meta_features': X_meta.shape[1],
            'algorithm': self.algorithm_name
        }
        
        self.training_history[datetime.now().isoformat()] = training_results
        
        self.logger.info(f"Meta-learner trained in {training_time:.2f} seconds")
        self.logger.info(f"Training accuracy: {train_accuracy:.4f}")
        self.logger.info(f"CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores)*2:.4f})")
        
        return training_results
    
    def predict(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Make predictions using trained meta-learner.
        
        Args:
            base_predictions: Dictionary of base model predictions
            
        Returns:
            Final ensemble predictions
        """
        if not self.is_trained:
            raise ValueError("Meta-learner must be trained first")
        
        # Prepare meta-features
        X_meta = self.prepare_meta_features(base_predictions)
        
        # Scale if necessary
        if self.algorithm_name in ['svm', 'neural_network', 'logistic_regression']:
            X_meta_scaled = self.scaler.transform(X_meta)
        else:
            X_meta_scaled = X_meta
        
        # Make predictions
        predictions = self.meta_model.predict(X_meta_scaled)
        
        self.logger.info(f"Meta-learner made predictions for {len(predictions)} samples")
        return predictions
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities using trained meta-learner.
        
        Args:
            base_predictions: Dictionary of base model predictions
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Meta-learner must be trained first")
        
        # Check if meta-model supports probability prediction
        if not hasattr(self.meta_model, 'predict_proba'):
            raise ValueError(f"Meta-model {self.algorithm_name} doesn't support probability prediction")
        
        # Prepare meta-features
        X_meta = self.prepare_meta_features(base_predictions)
        
        # Scale if necessary
        if self.algorithm_name in ['svm', 'neural_network', 'logistic_regression']:
            X_meta_scaled = self.scaler.transform(X_meta)
        else:
            X_meta_scaled = X_meta
        
        # Make probability predictions
        probabilities = self.meta_model.predict_proba(X_meta_scaled)
        
        self.logger.info(f"Meta-learner made probability predictions for {len(probabilities)} samples")
        return probabilities
    
    def _cross_validate(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> np.ndarray:
        """Perform cross-validation on meta-learner."""
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(self.meta_model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        return scores
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if supported by the meta-model."""
        if not self.is_trained:
            return {}
        
        importance_dict = {}
        
        if hasattr(self.meta_model, 'feature_importances_'):
            # Tree-based models
            importances = self.meta_model.feature_importances_
            importance_dict = dict(zip(self.feature_names, importances))
        
        elif hasattr(self.meta_model, 'coef_'):
            # Linear models
            coef = self.meta_model.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            else:
                coef = np.abs(coef)
            importance_dict = dict(zip(self.feature_names, coef))
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), 
                                    key=lambda x: x[1], reverse=True))
        
        return importance_dict
    
    def get_model_weights(self) -> Dict[str, float]:
        """Extract learned weights for each base model."""
        if not self.is_trained:
            return {}
        
        weights = {}
        
        # Group feature importance by base model
        importance = self.get_feature_importance()
        
        for feature_name, importance_value in importance.items():
            # Extract base model name from feature name
            base_model = feature_name.split('_')[0]
            
            if base_model not in weights:
                weights[base_model] = 0.0
            
            weights[base_model] += importance_value
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    def evaluate(self, base_predictions: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate meta-learner performance.
        
        Args:
            base_predictions: Base model predictions
            y_true: True labels
            
        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Meta-learner must be trained first")
        
        # Make predictions
        y_pred = self.predict(base_predictions)
        y_proba = self.predict_proba(base_predictions) if hasattr(self.meta_model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        classification_rep = classification_report(y_true, y_pred, output_dict=True)
        
        evaluation_results = {
            'accuracy': accuracy,
            'classification_report': classification_rep,
            'predictions': y_pred.tolist(),
            'feature_importance': self.get_feature_importance(),
            'model_weights': self.get_model_weights()
        }
        
        if y_proba is not None:
            evaluation_results['probabilities'] = y_proba.tolist()
        
        self.logger.info(f"Meta-learner evaluation - Accuracy: {accuracy:.4f}")
        
        return evaluation_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'algorithm': self.algorithm_name,
            'is_trained': self.is_trained,
            'n_meta_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'supports_probability': hasattr(self.meta_model, 'predict_proba')
        }
'''

with open('./enhanced_iot_botscan/src/core/ensemble/meta_learner.py', 'w') as f:
    f.write(meta_learner_content)

print("✅ Created meta_learner.py - Stacking meta-learner")

# Now implement the adversarial attack generator
attack_generator_content = '''"""
Adversarial Attack Generator for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Generates adversarial examples using FGSM, PGD, and C&W attacks
to train robust IoT botnet detection models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime

from .fgsm_attack import FGSMAttack
from .pgd_attack import PGDAttack
from .cw_attack import CWAttack
from ...utils.logger import get_logger

logger = get_logger(__name__)

class AdversarialAttackGenerator:
    """
    Comprehensive adversarial attack generator supporting multiple
    attack methods for robust model training.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize attack generator with configuration.
        
        Args:
            config: Adversarial attack configuration
        """
        self.config = config
        self.logger = logger
        
        # Initialize attack methods
        self.attack_methods = {}
        self.scaler = StandardScaler()
        
        # Initialize individual attack classes
        if config.get('fgsm', {}).get('enabled', True):
            self.attack_methods['fgsm'] = FGSMAttack(config.get('fgsm', {}))
        
        if config.get('pgd', {}).get('enabled', True):
            self.attack_methods['pgd'] = PGDAttack(config.get('pgd', {}))
        
        if config.get('cw', {}).get('enabled', True):
            self.attack_methods['cw'] = CWAttack(config.get('cw', {}))
        
        self.is_fitted = False
        self.attack_statistics = {}
        
        self.logger.info(f"Attack generator initialized with methods: {list(self.attack_methods.keys())}")
    
    def fit_scaler(self, X: np.ndarray) -> None:
        """
        Fit the scaler on training data.
        
        Args:
            X: Training feature matrix
        """
        self.scaler.fit(X)
        self.is_fitted = True
        self.logger.info("Scaler fitted on training data")
    
    def generate_adversarial_examples(self, 
                                    X: np.ndarray, 
                                    y: np.ndarray,
                                    model,
                                    attack_methods: Optional[List[str]] = None,
                                    ratio_per_method: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate adversarial examples using specified attack methods.
        
        Args:
            X: Clean input samples
            y: True labels
            model: Target model for attacks
            attack_methods: List of attack methods to use
            ratio_per_method: Fraction of samples to attack per method
            
        Returns:
            Tuple of (adversarial_examples, adversarial_labels)
        """
        if not self.is_fitted:
            self.fit_scaler(X)
        
        if attack_methods is None:
            attack_methods = list(self.attack_methods.keys())
        
        self.logger.info(f"Generating adversarial examples using: {attack_methods}")
        
        all_adversarial_x = []
        all_adversarial_y = []
        attack_stats = {}
        
        # Normalize input data
        X_normalized = self.scaler.transform(X)
        
        for attack_name in attack_methods:
            if attack_name not in self.attack_methods:
                self.logger.warning(f"Attack method {attack_name} not available")
                continue
            
            self.logger.info(f"Generating {attack_name.upper()} adversarial examples")
            
            # Select subset of samples for this attack
            n_samples = int(len(X_normalized) * ratio_per_method)
            if n_samples == 0:
                continue
            
            indices = np.random.choice(len(X_normalized), n_samples, replace=False)
            X_subset = X_normalized[indices]
            y_subset = y[indices]
            
            try:
                # Generate adversarial examples
                attack_method = self.attack_methods[attack_name]
                adv_examples = attack_method.generate(X_subset, y_subset, model)
                
                # Denormalize back to original scale
                adv_examples_denorm = self.scaler.inverse_transform(adv_examples)
                
                all_adversarial_x.append(adv_examples_denorm)
                all_adversarial_y.append(y_subset)
                
                # Calculate attack statistics
                success_rate = self._calculate_success_rate(
                    X_subset, adv_examples, y_subset, model
                )
                
                attack_stats[attack_name] = {
                    'samples_generated': len(adv_examples),
                    'success_rate': success_rate,
                    'perturbation_magnitude': np.mean(np.abs(adv_examples - X_subset))
                }
                
                self.logger.info(f"{attack_name.upper()}: Generated {len(adv_examples)} samples, "
                               f"Success rate: {success_rate:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error generating {attack_name} examples: {e}")
                continue
        
        # Combine all adversarial examples
        if all_adversarial_x:
            X_adversarial = np.vstack(all_adversarial_x)
            y_adversarial = np.hstack(all_adversarial_y)
        else:
            X_adversarial = np.empty((0, X.shape[1]))
            y_adversarial = np.empty((0,))
        
        self.attack_statistics = attack_stats
        
        self.logger.info(f"Total adversarial examples generated: {len(X_adversarial)}")
        
        return X_adversarial, y_adversarial
    
    def create_mixed_training_set(self, 
                                X_clean: np.ndarray, 
                                y_clean: np.ndarray,
                                model,
                                adversarial_ratio: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create mixed training set with clean and adversarial examples.
        
        Args:
            X_clean: Clean training samples
            y_clean: Clean training labels
            model: Target model
            adversarial_ratio: Fraction of adversarial examples
            
        Returns:
            Mixed training set (features, labels)
        """
        self.logger.info(f"Creating mixed training set with {adversarial_ratio:.1%} adversarial examples")
        
        # Calculate number of adversarial examples needed
        n_total_samples = len(X_clean)
        n_adversarial = int(n_total_samples * adversarial_ratio)
        n_clean = n_total_samples - n_adversarial
        
        # Sample clean examples
        clean_indices = np.random.choice(len(X_clean), n_clean, replace=False)
        X_clean_subset = X_clean[clean_indices]
        y_clean_subset = y_clean[clean_indices]
        
        # Generate adversarial examples
        ratio_per_method = adversarial_ratio / len(self.attack_methods)
        X_adversarial, y_adversarial = self.generate_adversarial_examples(
            X_clean, y_clean, model, ratio_per_method=ratio_per_method
        )
        
        # Combine clean and adversarial examples
        X_mixed = np.vstack([X_clean_subset, X_adversarial])
        y_mixed = np.hstack([y_clean_subset, y_adversarial])
        
        # Shuffle the mixed dataset
        shuffle_indices = np.random.permutation(len(X_mixed))
        X_mixed = X_mixed[shuffle_indices]
        y_mixed = y_mixed[shuffle_indices]
        
        self.logger.info(f"Mixed training set created: {len(X_mixed)} samples "
                        f"({n_clean} clean + {len(X_adversarial)} adversarial)")
        
        return X_mixed, y_mixed
    
    def _calculate_success_rate(self, 
                               X_clean: np.ndarray, 
                               X_adversarial: np.ndarray,
                               y_true: np.ndarray, 
                               model) -> float:
        """Calculate attack success rate."""
        try:
            # Get model predictions on clean and adversarial examples
            pred_clean = model.predict(X_clean)
            pred_adversarial = model.predict(X_adversarial)
            
            # For probabilistic predictions, get class predictions
            if pred_clean.ndim > 1:
                pred_clean = np.argmax(pred_clean, axis=1)
            if pred_adversarial.ndim > 1:
                pred_adversarial = np.argmax(pred_adversarial, axis=1)
            
            # Calculate success rate (percentage of changed predictions)
            correctly_classified = (pred_clean == y_true)
            attack_success = (pred_adversarial != pred_clean) & correctly_classified
            
            success_rate = np.mean(attack_success) if np.any(correctly_classified) else 0.0
            
            return success_rate
            
        except Exception as e:
            self.logger.warning(f"Could not calculate success rate: {e}")
            return 0.0
    
    def evaluate_robustness(self, 
                          X_test: np.ndarray, 
                          y_test: np.ndarray,
                          model,
                          attack_methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate model robustness against adversarial attacks.
        
        Args:
            X_test: Test samples
            y_test: Test labels
            model: Model to evaluate
            attack_methods: Attack methods to use
            
        Returns:
            Robustness evaluation results
        """
        self.logger.info("Evaluating model robustness")
        
        if attack_methods is None:
            attack_methods = list(self.attack_methods.keys())
        
        robustness_results = {
            'clean_accuracy': 0.0,
            'attack_results': {},
            'overall_robustness': 0.0
        }
        
        # Evaluate clean accuracy
        try:
            clean_predictions = model.predict(X_test)
            if clean_predictions.ndim > 1:
                clean_predictions = np.argmax(clean_predictions, axis=1)
            
            clean_accuracy = np.mean(clean_predictions == y_test)
            robustness_results['clean_accuracy'] = clean_accuracy
            
            self.logger.info(f"Clean accuracy: {clean_accuracy:.4f}")
        
        except Exception as e:
            self.logger.error(f"Error calculating clean accuracy: {e}")
            return robustness_results
        
        # Evaluate against each attack method
        total_robust_accuracy = 0.0
        valid_attacks = 0
        
        for attack_name in attack_methods:
            if attack_name not in self.attack_methods:
                continue
            
            try:
                # Generate adversarial examples
                X_adv, _ = self.generate_adversarial_examples(
                    X_test, y_test, model, [attack_name], ratio_per_method=1.0
                )
                
                if len(X_adv) == 0:
                    continue
                
                # Evaluate on adversarial examples
                adv_predictions = model.predict(X_adv)
                if adv_predictions.ndim > 1:
                    adv_predictions = np.argmax(adv_predictions, axis=1)
                
                adv_accuracy = np.mean(adv_predictions == y_test[:len(X_adv)])
                
                # Calculate attack success rate
                attack_success_rate = 1.0 - (adv_accuracy / clean_accuracy) if clean_accuracy > 0 else 1.0
                
                robustness_results['attack_results'][attack_name] = {
                    'adversarial_accuracy': adv_accuracy,
                    'attack_success_rate': attack_success_rate,
                    'samples_tested': len(X_adv)
                }
                
                total_robust_accuracy += adv_accuracy
                valid_attacks += 1
                
                self.logger.info(f"{attack_name.upper()}: Adversarial accuracy = {adv_accuracy:.4f}, "
                               f"Attack success rate = {attack_success_rate:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error evaluating {attack_name}: {e}")
                continue
        
        # Calculate overall robustness
        if valid_attacks > 0:
            overall_robustness = total_robust_accuracy / valid_attacks
            robustness_results['overall_robustness'] = overall_robustness
            
            self.logger.info(f"Overall robustness: {overall_robustness:.4f}")
        
        return robustness_results
    
    def get_attack_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated attacks."""
        return {
            'attack_methods_used': list(self.attack_methods.keys()),
            'attack_statistics': self.attack_statistics,
            'total_methods': len(self.attack_methods)
        }
    
    def visualize_adversarial_examples(self, 
                                     X_clean: np.ndarray, 
                                     X_adversarial: np.ndarray,
                                     feature_names: Optional[List[str]] = None,
                                     n_samples: int = 5) -> None:
        """
        Visualize adversarial perturbations (requires matplotlib).
        
        Args:
            X_clean: Clean examples
            X_adversarial: Adversarial examples
            feature_names: Names of features
            n_samples: Number of samples to visualize
        """
        try:
            import matplotlib.pyplot as plt
            
            n_samples = min(n_samples, len(X_clean), len(X_adversarial))
            
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(X_clean.shape[1])]
            
            # Calculate perturbations
            perturbations = X_adversarial[:n_samples] - X_clean[:n_samples]
            
            # Create visualization
            fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4*n_samples))
            
            for i in range(n_samples):
                # Clean example
                axes[i, 0].bar(range(len(X_clean[i])), X_clean[i])
                axes[i, 0].set_title(f'Clean Example {i+1}')
                axes[i, 0].set_xlabel('Features')
                
                # Adversarial example
                axes[i, 1].bar(range(len(X_adversarial[i])), X_adversarial[i])
                axes[i, 1].set_title(f'Adversarial Example {i+1}')
                axes[i, 1].set_xlabel('Features')
                
                # Perturbation
                axes[i, 2].bar(range(len(perturbations[i])), perturbations[i])
                axes[i, 2].set_title(f'Perturbation {i+1}')
                axes[i, 2].set_xlabel('Features')
                axes[i, 2].axhline(y=0, color='r', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            self.logger.warning("matplotlib not available for visualization")
        except Exception as e:
            self.logger.error(f"Error in visualization: {e}")
'''

with open('./enhanced_iot_botscan/src/core/adversarial/attack_generator.py', 'w') as f:
    f.write(attack_generator_content)

print("✅ Created attack_generator.py - Main adversarial attack orchestrator")

print("\n⚡ Continuing with complete implementation...")