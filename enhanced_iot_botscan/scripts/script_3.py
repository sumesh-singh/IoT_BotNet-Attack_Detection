# Now let's implement ALL the complete core components. Starting with meta-learner

# 1. Meta-Learner Implementation
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

class MetaLearner:
    """Meta-learner for stacking ensemble that learns to combine predictions from base models optimally."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.algorithm_name = config.get('algorithm', 'logistic_regression')
        self.meta_model = self._initialize_meta_model(self.algorithm_name, config)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.training_history = {}
    
    def _initialize_meta_model(self, algorithm: str, config: Dict[str, Any]):
        if algorithm == 'logistic_regression':
            return LogisticRegression(
                C=config.get('C', 1.0), penalty=config.get('penalty', 'l2'),
                solver=config.get('solver', 'lbfgs'), max_iter=config.get('max_iter', 1000),
                random_state=config.get('random_state', 42), n_jobs=-1
            )
        elif algorithm == 'random_forest':
            return RandomForestClassifier(
                n_estimators=config.get('n_estimators', 100), max_depth=config.get('max_depth', 5),
                random_state=config.get('random_state', 42), n_jobs=-1
            )
        elif algorithm == 'svm':
            return SVC(C=config.get('C', 1.0), kernel=config.get('kernel', 'rbf'),
                      probability=True, random_state=config.get('random_state', 42))
        elif algorithm == 'neural_network':
            return MLPClassifier(
                hidden_layer_sizes=config.get('hidden_layer_sizes', (100, 50)),
                activation=config.get('activation', 'relu'), solver=config.get('solver', 'adam'),
                max_iter=config.get('max_iter', 500), random_state=config.get('random_state', 42)
            )
        else:
            return LogisticRegression(random_state=42, n_jobs=-1)
    
    def get_sklearn_model(self):
        return self.meta_model
    
    def prepare_meta_features(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        feature_arrays = []
        feature_names = []
        
        for model_name, predictions in base_predictions.items():
            if predictions.ndim == 1:
                feature_arrays.append(predictions.reshape(-1, 1))
                feature_names.append(f"{model_name}_pred")
            else:
                feature_arrays.append(predictions)
                for i in range(predictions.shape[1]):
                    feature_names.append(f"{model_name}_prob_{i}")
        
        self.feature_names = feature_names
        return np.hstack(feature_arrays)
    
    def fit(self, base_predictions: Dict[str, np.ndarray], y: np.ndarray) -> Dict[str, Any]:
        start_time = datetime.now()
        
        X_meta = self.prepare_meta_features(base_predictions)
        
        if self.algorithm_name in ['svm', 'neural_network', 'logistic_regression']:
            X_meta_scaled = self.scaler.fit_transform(X_meta)
        else:
            X_meta_scaled = X_meta
        
        self.meta_model.fit(X_meta_scaled, y)
        self.is_trained = True
        
        train_pred = self.meta_model.predict(X_meta_scaled)
        train_accuracy = accuracy_score(y, train_pred)
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
        return training_results
    
    def predict(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Meta-learner must be trained first")
        
        X_meta = self.prepare_meta_features(base_predictions)
        
        if self.algorithm_name in ['svm', 'neural_network', 'logistic_regression']:
            X_meta_scaled = self.scaler.transform(X_meta)
        else:
            X_meta_scaled = X_meta
        
        return self.meta_model.predict(X_meta_scaled)
    
    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Meta-learner must be trained first")
        
        if not hasattr(self.meta_model, 'predict_proba'):
            raise ValueError(f"Meta-model {self.algorithm_name} doesn't support probability prediction")
        
        X_meta = self.prepare_meta_features(base_predictions)
        
        if self.algorithm_name in ['svm', 'neural_network', 'logistic_regression']:
            X_meta_scaled = self.scaler.transform(X_meta)
        else:
            X_meta_scaled = X_meta
        
        return self.meta_model.predict_proba(X_meta_scaled)
    
    def _cross_validate(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> np.ndarray:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        return cross_val_score(self.meta_model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_trained:
            return {}
        
        importance_dict = {}
        
        if hasattr(self.meta_model, 'feature_importances_'):
            importances = self.meta_model.feature_importances_
            importance_dict = dict(zip(self.feature_names, importances))
        elif hasattr(self.meta_model, 'coef_'):
            coef = self.meta_model.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            else:
                coef = np.abs(coef)
            importance_dict = dict(zip(self.feature_names, coef))
        
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def get_model_weights(self) -> Dict[str, float]:
        if not self.is_trained:
            return {}
        
        weights = {}
        importance = self.get_feature_importance()
        
        for feature_name, importance_value in importance.items():
            base_model = feature_name.split('_')[0]
            if base_model not in weights:
                weights[base_model] = 0.0
            weights[base_model] += importance_value
        
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
'''

with open('./enhanced_iot_botscan/src/core/ensemble/meta_learner.py', 'w') as f:
    f.write(meta_learner_content)

print("✅ Created meta_learner.py")

# 2. FGSM Attack Implementation
fgsm_attack_content = '''"""
Fast Gradient Sign Method (FGSM) Attack Implementation
Author: Kotiwale Sumesh Singh (160124862043)

FGSM attack for generating adversarial examples against IoT botnet detection models.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional, Union
from sklearn.base import BaseEstimator
import logging

class FGSMAttack:
    """Fast Gradient Sign Method adversarial attack implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.epsilon = config.get('epsilon', 0.1)
        self.clip_min = config.get('clip_min', 0.0)
        self.clip_max = config.get('clip_max', 1.0)
        self.targeted = config.get('targeted', False)
    
    def generate(self, X: np.ndarray, y: np.ndarray, model) -> np.ndarray:
        """Generate FGSM adversarial examples."""
        
        # Convert to TensorFlow tensors
        X_tf = tf.Variable(X.astype(np.float32), trainable=True)
        y_tf = tf.constant(y.astype(np.int32))
        
        # Create a simple TF model wrapper if needed
        if hasattr(model, 'predict_proba'):
            def model_fn(x):
                # For sklearn models, we need to convert back to numpy
                x_np = x.numpy()
                probs = model.predict_proba(x_np)
                return tf.constant(probs.astype(np.float32))
        else:
            def model_fn(x):
                x_np = x.numpy()
                pred = model.predict(x_np)
                # Convert to one-hot if needed
                if pred.ndim == 1:
                    n_classes = len(np.unique(y))
                    pred_onehot = np.eye(n_classes)[pred]
                    return tf.constant(pred_onehot.astype(np.float32))
                return tf.constant(pred.astype(np.float32))
        
        with tf.GradientTape() as tape:
            tape.watch(X_tf)
            predictions = model_fn(X_tf)
            
            # Calculate loss
            if self.targeted:
                # For targeted attacks, minimize loss for target class
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=y_tf, logits=predictions
                    )
                )
                loss = -loss  # Minimize loss = maximize negative loss
            else:
                # For untargeted attacks, maximize loss for true class
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=y_tf, logits=predictions
                    )
                )
        
        # Calculate gradients
        gradients = tape.gradient(loss, X_tf)
        
        # Generate adversarial examples using FGSM
        signed_gradients = tf.sign(gradients)
        adversarial_examples = X_tf + self.epsilon * signed_gradients
        
        # Clip to valid range
        adversarial_examples = tf.clip_by_value(
            adversarial_examples, self.clip_min, self.clip_max
        )
        
        return adversarial_examples.numpy()
    
    def generate_simple(self, X: np.ndarray, y: np.ndarray, model) -> np.ndarray:
        """Simple FGSM implementation for sklearn models."""
        adversarial_examples = []
        
        for i in range(len(X)):
            x_sample = X[i:i+1].copy()
            y_sample = y[i]
            
            # Calculate numerical gradient
            gradient = self._compute_numerical_gradient(x_sample, y_sample, model)
            
            # Apply FGSM perturbation
            perturbation = self.epsilon * np.sign(gradient)
            adv_example = x_sample + perturbation
            
            # Clip to valid range
            adv_example = np.clip(adv_example, self.clip_min, self.clip_max)
            adversarial_examples.append(adv_example[0])
        
        return np.array(adversarial_examples)
    
    def _compute_numerical_gradient(self, x: np.ndarray, y_true: int, model, epsilon: float = 1e-7) -> np.ndarray:
        """Compute numerical gradient for sklearn models."""
        gradient = np.zeros_like(x)
        
        for i in range(x.shape[1]):
            # Forward difference
            x_plus = x.copy()
            x_plus[0, i] += epsilon
            
            x_minus = x.copy() 
            x_minus[0, i] -= epsilon
            
            # Get predictions
            if hasattr(model, 'predict_proba'):
                prob_plus = model.predict_proba(x_plus)[0]
                prob_minus = model.predict_proba(x_minus)[0]
                
                # Use cross-entropy loss gradient
                prob_plus_log = np.log(prob_plus[y_true] + 1e-15)
                prob_minus_log = np.log(prob_minus[y_true] + 1e-15)
                
                gradient[0, i] = (prob_plus_log - prob_minus_log) / (2 * epsilon)
            else:
                pred_plus = model.predict(x_plus)[0]
                pred_minus = model.predict(x_minus)[0]
                
                # Simple difference for classification
                gradient[0, i] = (float(pred_plus == y_true) - float(pred_minus == y_true)) / (2 * epsilon)
        
        return -gradient if not self.targeted else gradient  # Negative for untargeted attacks
'''

with open('./enhanced_iot_botscan/src/core/adversarial/fgsm_attack.py', 'w') as f:
    f.write(fgsm_attack_content)

print("✅ Created fgsm_attack.py")

# 3. PGD Attack Implementation  
pgd_attack_content = '''"""
Projected Gradient Descent (PGD) Attack Implementation
Author: Kotiwale Sumesh Singh (160124862043)

PGD attack for generating stronger adversarial examples against IoT botnet detection models.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional
import logging

class PGDAttack:
    """Projected Gradient Descent adversarial attack implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.epsilon = config.get('epsilon', 0.1)
        self.alpha = config.get('alpha', 0.01)
        self.num_iter = config.get('num_iter', 10)
        self.clip_min = config.get('clip_min', 0.0)
        self.clip_max = config.get('clip_max', 1.0)
        self.targeted = config.get('targeted', False)
        self.norm = config.get('norm', 'inf')
    
    def generate(self, X: np.ndarray, y: np.ndarray, model) -> np.ndarray:
        """Generate PGD adversarial examples."""
        adversarial_examples = []
        
        for i in range(len(X)):
            x_orig = X[i:i+1].copy()
            y_sample = y[i]
            
            # Generate single adversarial example
            adv_example = self._generate_single_example(x_orig, y_sample, model)
            adversarial_examples.append(adv_example[0])
        
        return np.array(adversarial_examples)
    
    def _generate_single_example(self, x_orig: np.ndarray, y_true: int, model) -> np.ndarray:
        """Generate single PGD adversarial example."""
        
        # Initialize with random perturbation
        if self.norm == 'inf':
            delta = np.random.uniform(-self.epsilon, self.epsilon, x_orig.shape)
        else:  # L2 norm
            delta = np.random.normal(0, 1, x_orig.shape)
            delta = delta / np.linalg.norm(delta) * self.epsilon
        
        x_adv = x_orig + delta
        x_adv = np.clip(x_adv, self.clip_min, self.clip_max)
        
        # Iterative optimization
        for iteration in range(self.num_iter):
            # Compute gradient
            gradient = self._compute_gradient(x_adv, y_true, model)
            
            # Update adversarial example
            if self.norm == 'inf':
                # L-infinity PGD step
                x_adv = x_adv + self.alpha * np.sign(gradient)
                
                # Project back to L-infinity ball
                delta = x_adv - x_orig
                delta = np.clip(delta, -self.epsilon, self.epsilon)
                x_adv = x_orig + delta
                
            else:  # L2 norm
                # L2 PGD step
                x_adv = x_adv + self.alpha * gradient / (np.linalg.norm(gradient) + 1e-15)
                
                # Project back to L2 ball
                delta = x_adv - x_orig
                delta_norm = np.linalg.norm(delta)
                if delta_norm > self.epsilon:
                    delta = delta / delta_norm * self.epsilon
                x_adv = x_orig + delta
            
            # Clip to valid range
            x_adv = np.clip(x_adv, self.clip_min, self.clip_max)
        
        return x_adv
    
    def _compute_gradient(self, x: np.ndarray, y_true: int, model, epsilon: float = 1e-7) -> np.ndarray:
        """Compute numerical gradient for the loss function."""
        gradient = np.zeros_like(x)
        
        for i in range(x.shape[1]):
            # Compute partial derivative using finite differences
            x_plus = x.copy()
            x_plus[0, i] += epsilon
            
            x_minus = x.copy()
            x_minus[0, i] -= epsilon
            
            # Calculate loss difference
            loss_plus = self._compute_loss(x_plus, y_true, model)
            loss_minus = self._compute_loss(x_minus, y_true, model)
            
            # Numerical gradient
            gradient[0, i] = (loss_plus - loss_minus) / (2 * epsilon)
        
        # For untargeted attacks, we want to maximize loss (gradient ascent)
        return gradient if not self.targeted else -gradient
    
    def _compute_loss(self, x: np.ndarray, y_true: int, model) -> float:
        """Compute loss for a single sample."""
        try:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(x)[0]
                # Cross-entropy loss
                loss = -np.log(probs[y_true] + 1e-15)
            else:
                pred = model.predict(x)[0]
                # Simple 0-1 loss
                loss = float(pred != y_true)
            
            return loss
        except:
            return 0.0
    
    def generate_batch(self, X: np.ndarray, y: np.ndarray, model, batch_size: int = 32) -> np.ndarray:
        """Generate adversarial examples in batches for efficiency."""
        adversarial_examples = []
        
        for i in range(0, len(X), batch_size):
            batch_end = min(i + batch_size, len(X))
            X_batch = X[i:batch_end]
            y_batch = y[i:batch_end]
            
            batch_adv = self.generate(X_batch, y_batch, model)
            adversarial_examples.append(batch_adv)
        
        return np.vstack(adversarial_examples) if adversarial_examples else np.array([])
    
    def evaluate_attack_success(self, X_orig: np.ndarray, X_adv: np.ndarray, 
                               y: np.ndarray, model) -> Dict[str, float]:
        """Evaluate the success rate of the PGD attack."""
        
        # Get predictions on original and adversarial examples
        pred_orig = model.predict(X_orig)
        pred_adv = model.predict(X_adv)
        
        # Calculate metrics
        orig_accuracy = np.mean(pred_orig == y)
        adv_accuracy = np.mean(pred_adv == y)
        
        # Attack success rate (for untargeted attacks)
        attack_success = np.mean(pred_adv != pred_orig)
        
        # Perturbation statistics
        if self.norm == 'inf':
            avg_perturbation = np.mean(np.max(np.abs(X_adv - X_orig), axis=1))
        else:  # L2 norm
            avg_perturbation = np.mean(np.linalg.norm(X_adv - X_orig, axis=1))
        
        return {
            'original_accuracy': orig_accuracy,
            'adversarial_accuracy': adv_accuracy,
            'attack_success_rate': attack_success,
            'average_perturbation': avg_perturbation,
            'samples_evaluated': len(X_orig)
        }
'''

with open('./enhanced_iot_botscan/src/core/adversarial/pgd_attack.py', 'w') as f:
    f.write(pgd_attack_content)

print("✅ Created pgd_attack.py")

print("\n⚡ Core adversarial components created! Continuing with more implementations...")