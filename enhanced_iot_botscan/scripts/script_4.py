# Continue implementing the remaining core components

# 4. C&W Attack Implementation
cw_attack_content = '''"""
Carlini & Wagner (C&W) Attack Implementation
Author: Kotiwale Sumesh Singh (160124862043)

C&W attack for generating sophisticated adversarial examples against IoT botnet detection models.
"""

import numpy as np
from typing import Dict, Any, Optional
import logging
from scipy.optimize import minimize

class CWAttack:
    """Carlini & Wagner adversarial attack implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.c = config.get('c', 1.0)
        self.kappa = config.get('kappa', 0.0)
        self.max_iter = config.get('max_iter', 1000)
        self.learning_rate = config.get('learning_rate', 0.01)
        self.binary_search_steps = config.get('binary_search_steps', 10)
        self.targeted = config.get('targeted', False)
        self.clip_min = config.get('clip_min', 0.0)
        self.clip_max = config.get('clip_max', 1.0)
    
    def generate(self, X: np.ndarray, y: np.ndarray, model) -> np.ndarray:
        """Generate C&W adversarial examples."""
        adversarial_examples = []
        
        for i in range(len(X)):
            x_orig = X[i:i+1].copy()
            y_sample = y[i]
            
            # Generate single adversarial example
            adv_example = self._generate_single_example(x_orig, y_sample, model)
            adversarial_examples.append(adv_example[0])
        
        return np.array(adversarial_examples)
    
    def _generate_single_example(self, x_orig: np.ndarray, y_true: int, model) -> np.ndarray:
        """Generate single C&W adversarial example using binary search."""
        
        # Binary search bounds
        c_low = 0.0
        c_high = 1000.0
        best_adv = x_orig.copy()
        best_distance = float('inf')
        
        # Binary search for optimal c
        for search_step in range(self.binary_search_steps):
            c_current = (c_low + c_high) / 2.0
            
            # Optimize with current c value
            adv_example, success = self._optimize_cw(x_orig, y_true, model, c_current)
            
            if success:
                # Attack succeeded, try smaller c
                distance = np.linalg.norm(adv_example - x_orig)
                if distance < best_distance:
                    best_distance = distance
                    best_adv = adv_example.copy()
                c_high = c_current
            else:
                # Attack failed, try larger c
                c_low = c_current
        
        return best_adv
    
    def _optimize_cw(self, x_orig: np.ndarray, y_true: int, model, c: float) -> tuple:
        """Optimize C&W objective function."""
        
        # Initialize optimization variable (w in arctanh space)
        w = np.arctanh(2 * (x_orig - self.clip_min) / (self.clip_max - self.clip_min) - 1)
        
        def objective_function(w_flat):
            # Reshape w
            w_reshaped = w_flat.reshape(x_orig.shape)
            
            # Transform back to input space
            x_adv = (self.clip_max - self.clip_min) * (np.tanh(w_reshaped) + 1) / 2 + self.clip_min
            
            # Distance term (L2 norm)
            distance = np.sum((x_adv - x_orig) ** 2)
            
            # Classification loss term
            classification_loss = self._classification_loss(x_adv, y_true, model)
            
            # Combined objective
            return distance + c * classification_loss
        
        def objective_gradient(w_flat):
            # Numerical gradient computation
            epsilon = 1e-7
            grad = np.zeros_like(w_flat)
            
            for i in range(len(w_flat)):
                w_plus = w_flat.copy()
                w_minus = w_flat.copy()
                w_plus[i] += epsilon
                w_minus[i] -= epsilon
                
                grad[i] = (objective_function(w_plus) - objective_function(w_minus)) / (2 * epsilon)
            
            return grad
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                objective_function,
                w.flatten(),
                method='L-BFGS-B',
                jac=objective_gradient,
                options={'maxiter': self.max_iter}
            )
            
            # Get final adversarial example
            w_final = result.x.reshape(x_orig.shape)
            x_adv = (self.clip_max - self.clip_min) * (np.tanh(w_final) + 1) / 2 + self.clip_min
            
            # Check if attack succeeded
            success = self._check_attack_success(x_adv, y_true, model)
            
            return x_adv, success
            
        except Exception as e:
            return x_orig.copy(), False
    
    def _classification_loss(self, x: np.ndarray, y_true: int, model) -> float:
        """Compute classification loss for C&W attack."""
        try:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(x)[0]
                
                if self.targeted:
                    # For targeted attacks, maximize target class probability
                    return -np.log(probs[y_true] + 1e-15)
                else:
                    # For untargeted attacks, use C&W formulation
                    # max(max(Z_i) - Z_t, -kappa) where Z_t is true class logit
                    logits = np.log(probs + 1e-15)
                    max_other = np.max([logits[i] for i in range(len(logits)) if i != y_true])
                    loss = max(max_other - logits[y_true], -self.kappa)
                    return loss
            else:
                # For models without probability prediction
                pred = model.predict(x)[0]
                if self.targeted:
                    return float(pred != y_true)
                else:
                    return float(pred == y_true)
        except:
            return 0.0
    
    def _check_attack_success(self, x_adv: np.ndarray, y_true: int, model) -> bool:
        """Check if the attack succeeded."""
        try:
            pred = model.predict(x_adv)[0]
            
            if self.targeted:
                return pred == y_true
            else:
                return pred != y_true
        except:
            return False
    
    def generate_with_confidence(self, X: np.ndarray, y: np.ndarray, model, 
                                confidence_threshold: float = 0.5) -> np.ndarray:
        """Generate C&W adversarial examples with confidence threshold."""
        adversarial_examples = []
        
        for i in range(len(X)):
            x_orig = X[i:i+1].copy()
            y_sample = y[i]
            
            # Check if original prediction is confident enough
            if hasattr(model, 'predict_proba'):
                orig_probs = model.predict_proba(x_orig)[0]
                if np.max(orig_probs) < confidence_threshold:
                    # Skip if original prediction is not confident
                    adversarial_examples.append(x_orig[0])
                    continue
            
            # Generate adversarial example
            adv_example = self._generate_single_example(x_orig, y_sample, model)
            adversarial_examples.append(adv_example[0])
        
        return np.array(adversarial_examples)
    
    def evaluate_perturbation_quality(self, X_orig: np.ndarray, X_adv: np.ndarray) -> Dict[str, float]:
        """Evaluate the quality of perturbations generated by C&W attack."""
        
        # L2 distance statistics
        l2_distances = np.linalg.norm(X_adv - X_orig, axis=1)
        
        # L-infinity distance statistics
        linf_distances = np.max(np.abs(X_adv - X_orig), axis=1)
        
        # Feature-wise perturbation analysis
        feature_perturbations = np.abs(X_adv - X_orig)
        avg_feature_perturbation = np.mean(feature_perturbations, axis=0)
        
        return {
            'l2_distance_mean': np.mean(l2_distances),
            'l2_distance_std': np.std(l2_distances),
            'l2_distance_max': np.max(l2_distances),
            'linf_distance_mean': np.mean(linf_distances),
            'linf_distance_std': np.std(linf_distances),
            'linf_distance_max': np.max(linf_distances),
            'avg_feature_perturbation': avg_feature_perturbation.tolist(),
            'samples_analyzed': len(X_orig)
        }
'''

with open('./enhanced_iot_botscan/src/core/adversarial/cw_attack.py', 'w') as f:
    f.write(cw_attack_content)

print("✅ Created cw_attack.py")

# 5. Adversarial Trainer Implementation
adversarial_trainer_content = '''"""
Adversarial Trainer for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Implements adversarial training to improve model robustness against adversarial attacks.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging
from datetime import datetime
import joblib
import os

from .attack_generator import AdversarialAttackGenerator

class AdversarialTrainer:
    """Adversarial training system for robust model development."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adversarial_ratio = config.get('adversarial_ratio', 0.3)
        self.training_epochs = config.get('training_epochs', 10)
        self.batch_size = config.get('batch_size', 1000)
        
        # Initialize attack generator
        self.attack_generator = AdversarialAttackGenerator(config.get('attacks', {}))
        
        self.training_history = []
        self.current_epoch = 0
        
    def train_robust_model(self, 
                          base_model,
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_val: Optional[np.ndarray] = None,
                          y_val: Optional[np.ndarray] = None,
                          validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train model with adversarial examples for improved robustness.
        
        Args:
            base_model: Model to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            validation_split: Fraction for validation if X_val not provided
            
        Returns:
            Training results and metrics
        """
        start_time = datetime.now()
        
        # Split validation set if not provided
        if X_val is None or y_val is None:
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=validation_split, 
                random_state=42, stratify=y_train
            )
        else:
            X_train_split = X_train
            y_train_split = y_train
        
        # Fit attack generator scaler
        self.attack_generator.fit_scaler(X_train_split)
        
        # Training loop
        training_results = {
            'epochs': [],
            'clean_accuracy': [],
            'adversarial_accuracy': [], 
            'validation_accuracy': [],
            'robust_accuracy': []
        }
        
        for epoch in range(self.training_epochs):
            self.current_epoch = epoch + 1
            
            print(f"\\nEpoch {self.current_epoch}/{self.training_epochs}")
            print("-" * 50)
            
            # Create mixed training set
            X_mixed, y_mixed = self._create_adversarial_training_set(
                X_train_split, y_train_split, base_model
            )
            
            # Train model on mixed dataset
            epoch_results = self._train_single_epoch(
                base_model, X_mixed, y_mixed, X_val, y_val
            )
            
            # Store results
            for key, value in epoch_results.items():
                training_results[key].append(value)
            
            # Print epoch results
            print(f"Clean Accuracy: {epoch_results['clean_accuracy']:.4f}")
            print(f"Adversarial Accuracy: {epoch_results['adversarial_accuracy']:.4f}")
            print(f"Validation Accuracy: {epoch_results['validation_accuracy']:.4f}")
            print(f"Robust Accuracy: {epoch_results['robust_accuracy']:.4f}")
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Final evaluation
        final_results = self._evaluate_final_robustness(base_model, X_val, y_val)
        
        complete_results = {
            'training_time': training_time,
            'epochs_completed': self.training_epochs,
            'training_history': training_results,
            'final_evaluation': final_results,
            'adversarial_ratio': self.adversarial_ratio
        }
        
        self.training_history.append(complete_results)
        
        print(f"\\n✅ Adversarial training completed in {training_time:.2f} seconds")
        print(f"Final robust accuracy: {final_results.get('overall_robustness', 0.0):.4f}")
        
        return complete_results
    
    def _create_adversarial_training_set(self, 
                                       X_clean: np.ndarray, 
                                       y_clean: np.ndarray,
                                       model) -> Tuple[np.ndarray, np.ndarray]:
        """Create mixed training set with adversarial examples."""
        
        # Create mixed training set
        X_mixed, y_mixed = self.attack_generator.create_mixed_training_set(
            X_clean, y_clean, model, self.adversarial_ratio
        )
        
        print(f"Created mixed training set: {len(X_mixed)} samples "
              f"({len(X_clean)} clean + {len(X_mixed) - len(X_clean)} adversarial)")
        
        return X_mixed, y_mixed
    
    def _train_single_epoch(self, 
                           model,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_val: np.ndarray,
                           y_val: np.ndarray) -> Dict[str, float]:
        """Train model for single epoch and evaluate performance."""
        
        # Train model
        if hasattr(model, 'partial_fit'):
            # For online learning models
            model.partial_fit(X_train, y_train)
        else:
            # For batch learning models
            model.fit(X_train, y_train)
        
        # Evaluate on clean validation data
        clean_pred = model.predict(X_val)
        clean_accuracy = accuracy_score(y_val, clean_pred)
        
        # Generate adversarial validation examples
        X_val_adv, y_val_adv = self.attack_generator.generate_adversarial_examples(
            X_val, y_val, model, ratio_per_method=0.2
        )
        
        # Evaluate on adversarial examples
        if len(X_val_adv) > 0:
            adv_pred = model.predict(X_val_adv)
            adversarial_accuracy = accuracy_score(y_val_adv, adv_pred)
        else:
            adversarial_accuracy = 0.0
        
        # Combined validation accuracy
        validation_accuracy = clean_accuracy
        
        # Robust accuracy (harmonic mean of clean and adversarial)
        if adversarial_accuracy > 0:
            robust_accuracy = 2 * (clean_accuracy * adversarial_accuracy) / (clean_accuracy + adversarial_accuracy)
        else:
            robust_accuracy = clean_accuracy
        
        return {
            'clean_accuracy': clean_accuracy,
            'adversarial_accuracy': adversarial_accuracy,
            'validation_accuracy': validation_accuracy,
            'robust_accuracy': robust_accuracy
        }
    
    def _evaluate_final_robustness(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Comprehensive robustness evaluation."""
        
        # Evaluate robustness against all attack methods
        robustness_results = self.attack_generator.evaluate_robustness(
            X_test, y_test, model
        )
        
        return robustness_results
    
    def incremental_adversarial_training(self,
                                       model,
                                       X_new: np.ndarray,
                                       y_new: np.ndarray,
                                       adaptation_rate: float = 0.1) -> Dict[str, Any]:
        """
        Perform incremental adversarial training with new data.
        
        Args:
            model: Pre-trained model
            X_new: New training data
            y_new: New training labels
            adaptation_rate: Learning rate for adaptation
            
        Returns:
            Adaptation results
        """
        
        print(f"Performing incremental adversarial training on {len(X_new)} new samples")
        
        # Generate adversarial examples for new data
        X_new_adv, y_new_adv = self.attack_generator.generate_adversarial_examples(
            X_new, y_new, model, ratio_per_method=self.adversarial_ratio
        )
        
        # Combine new clean and adversarial data
        X_combined = np.vstack([X_new, X_new_adv]) if len(X_new_adv) > 0 else X_new
        y_combined = np.hstack([y_new, y_new_adv]) if len(X_new_adv) > 0 else y_new
        
        # Incremental training
        if hasattr(model, 'partial_fit'):
            model.partial_fit(X_combined, y_combined)
        else:
            # For models without incremental learning, retrain with sample
            sample_size = min(1000, len(X_combined))
            sample_indices = np.random.choice(len(X_combined), sample_size, replace=False)
            model.fit(X_combined[sample_indices], y_combined[sample_indices])
        
        # Evaluate adaptation performance
        new_pred = model.predict(X_new)
        adaptation_accuracy = accuracy_score(y_new, new_pred)
        
        results = {
            'samples_processed': len(X_new),
            'adversarial_samples_generated': len(X_new_adv),
            'adaptation_accuracy': adaptation_accuracy,
            'total_training_samples': len(X_combined)
        }
        
        print(f"Incremental training completed. Adaptation accuracy: {adaptation_accuracy:.4f}")
        
        return results
    
    def save_training_state(self, filepath: str) -> None:
        """Save adversarial training state."""
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        training_state = {
            'config': self.config,
            'training_history': self.training_history,
            'current_epoch': self.current_epoch,
            'adversarial_ratio': self.adversarial_ratio,
            'attack_generator_stats': self.attack_generator.get_attack_statistics()
        }
        
        joblib.dump(training_state, filepath)
        print(f"Training state saved to {filepath}")
    
    def load_training_state(self, filepath: str) -> None:
        """Load adversarial training state."""
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Training state file not found: {filepath}")
        
        training_state = joblib.load(filepath)
        
        self.training_history = training_state['training_history']
        self.current_epoch = training_state['current_epoch']
        self.adversarial_ratio = training_state['adversarial_ratio']
        
        print(f"Training state loaded from {filepath}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        
        if not self.training_history:
            return {'message': 'No training history available'}
        
        latest_training = self.training_history[-1]
        
        return {
            'total_training_sessions': len(self.training_history),
            'latest_training_time': latest_training['training_time'],
            'epochs_completed': latest_training['epochs_completed'],
            'final_robust_accuracy': latest_training['final_evaluation'].get('overall_robustness', 0.0),
            'adversarial_ratio_used': latest_training['adversarial_ratio'],
            'attack_methods_used': list(self.attack_generator.attack_methods.keys())
        }
'''

with open('./enhanced_iot_botscan/src/core/adversarial/adversarial_trainer.py', 'w') as f:
    f.write(adversarial_trainer_content)

print("✅ Created adversarial_trainer.py")

print("\n⚡ Adversarial training system completed! Continuing with concept drift detection...")