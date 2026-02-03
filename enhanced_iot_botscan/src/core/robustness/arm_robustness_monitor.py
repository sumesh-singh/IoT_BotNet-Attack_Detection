"""
Adaptive Robustness Monitor (ARM) - Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Unified robustness and drift monitoring system for tree ensemble models.
Designed for practical IoT threat scenarios.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List, Tuple
from sklearn.base import BaseEstimator
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)


class AdaptiveRobustnessMonitor:
    """
    Unified system for monitoring model robustness under adversarial conditions
    and concept drift, optimized for tree ensemble architectures.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize ARM with configuration."""
        
        self.config = config or {}
        
        # Component initialization (lazy loading)
        self._noise_injector = None
        self._feature_masker = None
        self._burst_generator = None
        self._confidence_monitor = None
        self._accuracy_monitor = None
        self._stability_analyzer = None
        
        # Configuration
        self.noise_levels = self.config.get('noise_levels', [0.0, 0.05, 0.1, 0.2])
        self.masking_rates = self.config.get('masking_rates', [0.0, 0.1, 0.2, 0.3])
        self.burst_intensities = self.config.get('burst_intensities', [1.0, 1.5, 2.0])
        
        # Thresholds
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.accuracy_drop_threshold = self.config.get('accuracy_drop_threshold', 0.05)
        self.stability_threshold = self.config.get('stability_threshold', 0.1)
        
        # State tracking
        self.baseline_metrics = {}
        self.evaluation_history = []
        self.threat_detections = []
        
        logger.info("AdaptiveRobustnessMonitor initialized")

    # ==================== THREAT GENERATORS ====================
    
    def _get_noise_injector(self):
        """Lazy load noise injector."""
        if self._noise_injector is None:
            from .threat_generators.noise_injector import NoiseInjector
            self._noise_injector = NoiseInjector(self.config)
        return self._noise_injector
    
    def _get_feature_masker(self):
        """Lazy load feature masker."""
        if self._feature_masker is None:
            from .threat_generators.feature_masker import FeatureMasker
            self._feature_masker = FeatureMasker(self.config)
        return self._feature_masker
    
    def _get_burst_generator(self):
        """Lazy load burst generator."""
        if self._burst_generator is None:
            from .threat_generators.burst_generator import BurstGenerator
            self._burst_generator = BurstGenerator(self.config)
        return self._burst_generator
    
    # ==================== DETECTORS ====================
    
    def _get_confidence_monitor(self):
        """Lazy load confidence monitor."""
        if self._confidence_monitor is None:
            from .detectors.confidence_monitor import ConfidenceMonitor
            self._confidence_monitor = ConfidenceMonitor(self.config)
        return self._confidence_monitor
    
    def _get_accuracy_monitor(self):
        """Lazy load accuracy monitor."""
        if self._accuracy_monitor is None:
            from .detectors.accuracy_monitor import AccuracyMonitor
            self._accuracy_monitor = AccuracyMonitor(self.config)
        return self._accuracy_monitor
    
    def _get_stability_analyzer(self):
        """Lazy load stability analyzer."""
        if self._stability_analyzer is None:
            from .detectors.stability_analyzer import StabilityAnalyzer
            self._stability_analyzer = StabilityAnalyzer(self.config)
        return self._stability_analyzer

    # ==================== CORE EVALUATION ====================

    def establish_baseline(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Establish baseline performance metrics for the model.
        
        Args:
            model: Trained model
            X: Clean validation data
            y: True labels
            
        Returns:
            Baseline metrics
        """
        logger.info(f"Establishing baseline metrics on {len(X)} samples")
        
        # Predictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
        # Baseline metrics
        accuracy = np.mean(y_pred == y)
        confidence = np.mean(np.max(y_proba, axis=1))
        prediction_entropy = -np.mean(np.sum(y_proba * np.log(y_proba + 1e-10), axis=1))
        
        self.baseline_metrics = {
            'accuracy': accuracy,
            'confidence': confidence,
            'entropy': prediction_entropy,
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(X)
        }
        
        logger.info(f"Baseline established: accuracy={accuracy:.4f}, confidence={confidence:.4f}")
        return self.baseline_metrics

    def evaluate_comprehensive_robustness(self, model: BaseEstimator, X: np.ndarray, 
                                         y: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive robustness evaluation under multiple threat scenarios.
        
        Args:
            model: Model to evaluate
            X: Test data
            y: True labels
            
        Returns:
            Complete robustness evaluation results
        """
        logger.info("Starting comprehensive robustness evaluation")
        
        if not self.baseline_metrics:
            logger.warning("No baseline established, establishing now...")
            self.establish_baseline(model, X, y)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'baseline': self.baseline_metrics,
            'threat_scenarios': {}
        }
        
        # 1. Noise Robustness
        logger.info("Evaluating noise robustness...")
        results['threat_scenarios']['noise'] = self._evaluate_noise_robustness(model, X, y)
        
        # 2. Feature Masking Robustness
        logger.info("Evaluating feature masking robustness...")
        results['threat_scenarios']['masking'] = self._evaluate_masking_robustness(model, X, y)
        
        # 3. Burst Traffic Robustness
        logger.info("Evaluating burst traffic robustness...")
        results['threat_scenarios']['burst'] = self._evaluate_burst_robustness(model, X, y)
        
        # 4. Confidence Stability
        logger.info("Analyzing confidence stability...")
        results['confidence_analysis'] = self._analyze_confidence_stability(model, X, y)
        
        # 5. Feature Importance Stability
        logger.info("Analyzing feature stability...")
        results['feature_stability'] = self._analyze_feature_stability(model, X)
        
        # 6. Aggregate Robustness Score
        results['aggregate_scores'] = self._compute_aggregate_scores(results)
        
        # Store in history
        self.evaluation_history.append(results)
        
        logger.info(f"Comprehensive evaluation complete. Overall robustness: {results['aggregate_scores']['overall_robustness']:.4f}")
        
        return results

    # ==================== INDIVIDUAL THREAT EVALUATIONS ====================

    def _evaluate_noise_robustness(self, model: BaseEstimator, X: np.ndarray, 
                                   y: np.ndarray) -> Dict[str, Any]:
        """Evaluate robustness under various noise levels."""
        
        noise_injector = self._get_noise_injector()
        results = {}
        
        for noise_level in self.noise_levels:
            logger.info(f"  Testing noise level: {noise_level}")
            
            # Generate noisy data
            X_noisy = noise_injector.add_gaussian_noise(X, noise_level)
            
            # Evaluate
            y_pred = model.predict(X_noisy)
            y_proba = model.predict_proba(X_noisy)
            
            accuracy = np.mean(y_pred == y)
            confidence = np.mean(np.max(y_proba, axis=1))
            
            accuracy_drop = self.baseline_metrics['accuracy'] - accuracy
            confidence_drop = self.baseline_metrics['confidence'] - confidence
            
            results[f'noise_{noise_level}'] = {
                'accuracy': accuracy,
                'confidence': confidence,
                'accuracy_drop': accuracy_drop,
                'confidence_drop': confidence_drop,
                'robust': accuracy_drop < self.accuracy_drop_threshold
            }
        
        return results

    def _evaluate_masking_robustness(self, model: BaseEstimator, X: np.ndarray,
                                     y: np.ndarray) -> Dict[str, Any]:
        """Evaluate robustness under feature masking (sensor failures)."""
        
        masker = self._get_feature_masker()
        results = {}
        
        for masking_rate in self.masking_rates:
            logger.info(f"  Testing masking rate: {masking_rate}")
            
            # Generate masked data
            X_masked = masker.random_feature_masking(X, masking_rate)
            
            # Evaluate
            y_pred = model.predict(X_masked)
            y_proba = model.predict_proba(X_masked)
            
            accuracy = np.mean(y_pred == y)
            confidence = np.mean(np.max(y_proba, axis=1))
            
            accuracy_drop = self.baseline_metrics['accuracy'] - accuracy
            confidence_drop = self.baseline_metrics['confidence'] - confidence
            
            results[f'masking_{masking_rate}'] = {
                'accuracy': accuracy,
                'confidence': confidence,
                'accuracy_drop': accuracy_drop,
                'confidence_drop': confidence_drop,
                'robust': accuracy_drop < self.accuracy_drop_threshold
            }
        
        return results

    def _evaluate_burst_robustness(self, model: BaseEstimator, X: np.ndarray,
                                   y: np.ndarray) -> Dict[str, Any]:
        """Evaluate robustness under burst traffic conditions."""
        
        burst_gen = self._get_burst_generator()
        results = {}
        
        for intensity in self.burst_intensities:
            logger.info(f"  Testing burst intensity: {intensity}")
            
            # Generate burst traffic
            X_burst = burst_gen.simulate_burst_traffic(X, intensity)
            
            # Evaluate
            y_pred = model.predict(X_burst)
            y_proba = model.predict_proba(X_burst)
            
            accuracy = np.mean(y_pred == y)
            confidence = np.mean(np.max(y_proba, axis=1))
            
            accuracy_drop = self.baseline_metrics['accuracy'] - accuracy
            confidence_drop = self.baseline_metrics['confidence'] - confidence
            
            results[f'burst_{intensity}'] = {
                'accuracy': accuracy,
                'confidence': confidence,
                'accuracy_drop': accuracy_drop,
                'confidence_drop': confidence_drop,
                'robust': accuracy_drop < self.accuracy_drop_threshold
            }
        
        return results

    def _analyze_confidence_stability(self, model: BaseEstimator, X: np.ndarray,
                                      y: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction confidence stability."""
        
        confidence_mon = self._get_confidence_monitor()
        
        # Get baseline predictions
        y_proba = model.predict_proba(X)
        confidences = np.max(y_proba, axis=1)
        
        # Analyze under perturbations
        noise_injector = self._get_noise_injector()
        perturbed_confidences = []
        
        for _ in range(10):  # 10 random perturbations
            X_perturbed = noise_injector.add_gaussian_noise(X, 0.05)
            y_proba_perturbed = model.predict_proba(X_perturbed)
            perturbed_confidences.append(np.max(y_proba_perturbed, axis=1))
        
        perturbed_confidences = np.array(perturbed_confidences)
        
        # Compute stability metrics
        confidence_variance = np.var(perturbed_confidences, axis=0)
        mean_confidence_shift = np.mean(np.abs(perturbed_confidences - confidences), axis=0)
        
        return {
            'mean_confidence': np.mean(confidences),
            'mean_variance': np.mean(confidence_variance),
            'mean_shift': np.mean(mean_confidence_shift),
            'stable': np.mean(mean_confidence_shift) < self.stability_threshold
        }

    def _analyze_feature_stability(self, model: BaseEstimator, X: np.ndarray) -> Dict[str, Any]:
        """Analyze feature importance stability under perturbations.
        
        FIXED: Instead of retraining models (which requires y), we analyze how 
        prediction-based feature importance varies under noise. This is done by
        comparing feature importances from the trained model itself.
        """
        
        if not hasattr(model, 'feature_importances_'):
            return {'stable': True, 'note': 'Model does not expose feature importances'}
        
        # Baseline importance from the trained model
        baseline_importance = model.feature_importances_.copy()
        
        # For tree-based models, feature importances are fixed after training
        # We analyze stability by checking how consistent the importance ranking is
        # under different noise conditions using permutation importance
        
        noise_injector = self._get_noise_injector()
        perturbed_importances = []
        
        # Instead of retraining, we measure prediction variance per feature
        # under noise to estimate which features are stable contributors
        for _ in range(5):
            X_noisy = noise_injector.add_gaussian_noise(X, 0.05)
            
            # Compute prediction variance per feature using a simple permutation approach
            # Shuffle each feature and measure prediction change
            feature_sensitivity = []
            y_pred_orig = model.predict_proba(X_noisy)
            
            for feat_idx in range(min(X_noisy.shape[1], 50)):  # Limit to first 50 features
                X_permuted = X_noisy.copy()
                np.random.shuffle(X_permuted[:, feat_idx])
                y_pred_permuted = model.predict_proba(X_permuted)
                
                # Measure prediction change
                pred_change = np.mean(np.abs(y_pred_orig - y_pred_permuted))
                feature_sensitivity.append(pred_change)
            
            # Pad with zeros for remaining features if any
            while len(feature_sensitivity) < len(baseline_importance):
                feature_sensitivity.append(0.0)
            
            perturbed_importances.append(np.array(feature_sensitivity[:len(baseline_importance)]))
        
        if perturbed_importances:
            perturbed_importances = np.array(perturbed_importances)
            importance_variance = np.var(perturbed_importances, axis=0)
            mean_importance_shift = np.mean(np.abs(perturbed_importances - np.mean(perturbed_importances, axis=0)), axis=0)
        else:
            importance_variance = np.zeros_like(baseline_importance)
            mean_importance_shift = np.zeros_like(baseline_importance)
        
        return {
            'mean_variance': float(np.mean(importance_variance)),
            'mean_shift': float(np.mean(mean_importance_shift)),
            'stable': np.mean(mean_importance_shift) < self.stability_threshold,
            'top_stable_features': np.argsort(importance_variance)[:10].tolist(),
            'top_unstable_features': np.argsort(importance_variance)[-10:].tolist()
        }

    def _compute_aggregate_scores(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute aggregate robustness scores."""
        
        scores = {}
        
        # Noise robustness score
        noise_results = results['threat_scenarios']['noise']
        noise_accuracies = [v['accuracy'] for v in noise_results.values()]
        scores['noise_robustness'] = np.mean(noise_accuracies)
        
        # Masking robustness score
        masking_results = results['threat_scenarios']['masking']
        masking_accuracies = [v['accuracy'] for v in masking_results.values()]
        scores['masking_robustness'] = np.mean(masking_accuracies)
        
        # Burst robustness score
        burst_results = results['threat_scenarios']['burst']
        burst_accuracies = [v['accuracy'] for v in burst_results.values()]
        scores['burst_robustness'] = np.mean(burst_accuracies)
        
        # Confidence stability score
        confidence_analysis = results['confidence_analysis']
        scores['confidence_stability'] = 1.0 - confidence_analysis['mean_shift']
        
        # Overall robustness score (weighted average)
        weights = {
            'noise': 0.3,
            'masking': 0.3,
            'burst': 0.2,
            'confidence': 0.2
        }
        
        scores['overall_robustness'] = (
            weights['noise'] * scores['noise_robustness'] +
            weights['masking'] * scores['masking_robustness'] +
            weights['burst'] * scores['burst_robustness'] +
            weights['confidence'] * scores['confidence_stability']
        )
        
        return scores

    # ==================== THREAT DETECTION ====================

    def detect_threats(self, model: BaseEstimator, X_new: np.ndarray,
                      y_new: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Detect potential threats in new data.
        
        Args:
            model: Trained model
            X_new: New data to analyze
            y_new: Optional true labels
            
        Returns:
            Threat detection results
        """
        logger.info("Analyzing new data for potential threats")
        
        threats_detected = []
        
        # 1. Confidence drop detection
        y_proba = model.predict_proba(X_new)
        mean_confidence = np.mean(np.max(y_proba, axis=1))
        
        if mean_confidence < self.baseline_metrics.get('confidence', 0.8) - self.confidence_threshold:
            threats_detected.append({
                'type': 'confidence_drop',
                'severity': 'high',
                'description': f'Mean confidence dropped to {mean_confidence:.4f}'
            })
        
        # 2. Accuracy drop detection (if labels available)
        if y_new is not None:
            y_pred = model.predict(X_new)
            accuracy = np.mean(y_pred == y_new)
            
            if accuracy < self.baseline_metrics.get('accuracy', 0.9) - self.accuracy_drop_threshold:
                threats_detected.append({
                    'type': 'accuracy_drop',
                    'severity': 'critical',
                    'description': f'Accuracy dropped to {accuracy:.4f}'
                })
        
        # 3. Distribution shift detection
        # Compare feature distributions with baseline (using simple statistical test)
        # This would integrate with your existing drift detection
        
        detection_result = {
            'timestamp': datetime.now().isoformat(),
            'threats_detected': threats_detected,
            'threat_level': 'high' if len(threats_detected) > 0 else 'normal',
            'n_samples_analyzed': len(X_new)
        }
        
        self.threat_detections.append(detection_result)
        
        if threats_detected:
            logger.warning(f"Detected {len(threats_detected)} potential threats")
        else:
            logger.info("No threats detected")
        
        return detection_result

    # ==================== ADAPTIVE RESPONSE ====================

    def recommend_response(self, threat_detection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend appropriate response to detected threats.
        
        Args:
            threat_detection: Results from detect_threats()
            
        Returns:
            Recommended response actions
        """
        
        recommendations = []
        
        for threat in threat_detection.get('threats_detected', []):
            if threat['type'] == 'confidence_drop':
                recommendations.append({
                    'action': 'increase_monitoring',
                    'priority': 'high',
                    'description': 'Increase monitoring frequency and confidence thresholds'
                })
            
            elif threat['type'] == 'accuracy_drop':
                recommendations.append({
                    'action': 'trigger_retraining',
                    'priority': 'critical',
                    'description': 'Model retraining recommended with new data'
                })
            
            elif threat['type'] == 'distribution_shift':
                recommendations.append({
                    'action': 'incremental_update',
                    'priority': 'medium',
                    'description': 'Incremental model update to adapt to drift'
                })
        
        if not recommendations:
            recommendations.append({
                'action': 'continue_monitoring',
                'priority': 'normal',
                'description': 'System operating normally'
            })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'recommendations': recommendations,
            'auto_response_enabled': self.config.get('auto_response', False)
        }

    # ==================== REPORTING ====================

    def get_robustness_report(self) -> Dict[str, Any]:
        """Generate comprehensive robustness report."""
        
        if not self.evaluation_history:
            return {'status': 'No evaluations performed yet'}
        
        latest_eval = self.evaluation_history[-1]
        
        report = {
            'summary': {
                'overall_robustness': latest_eval['aggregate_scores']['overall_robustness'],
                'noise_robustness': latest_eval['aggregate_scores']['noise_robustness'],
                'masking_robustness': latest_eval['aggregate_scores']['masking_robustness'],
                'burst_robustness': latest_eval['aggregate_scores']['burst_robustness'],
                'confidence_stability': latest_eval['aggregate_scores']['confidence_stability']
            },
            'baseline_metrics': self.baseline_metrics,
            'latest_evaluation': latest_eval,
            'threat_history': self.threat_detections[-10:] if self.threat_detections else [],
            'total_evaluations': len(self.evaluation_history),
            'timestamp': datetime.now().isoformat()
        }
        
        return report

    def export_results(self, filepath: str) -> None:
        """Export evaluation results to file."""
        import json
        
        report = self.get_robustness_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Results exported to {filepath}")


# ==================== UTILITY FUNCTIONS ====================

def compare_models_robustness(arm: AdaptiveRobustnessMonitor,
                              models: Dict[str, BaseEstimator],
                              X_test: np.ndarray,
                              y_test: np.ndarray) -> pd.DataFrame:
    """
    Compare robustness of multiple models.
    
    Args:
        arm: AdaptiveRobustnessMonitor instance
        models: Dictionary of {model_name: model}
        X_test: Test data
        y_test: Test labels
        
    Returns:
        DataFrame with comparison results
    """
    
    results = []
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}...")
        
        # Establish baseline
        arm.establish_baseline(model, X_test, y_test)
        
        # Evaluate robustness
        eval_results = arm.evaluate_comprehensive_robustness(model, X_test, y_test)
        
        # Extract key metrics
        results.append({
            'model': model_name,
            'overall_robustness': eval_results['aggregate_scores']['overall_robustness'],
            'noise_robustness': eval_results['aggregate_scores']['noise_robustness'],
            'masking_robustness': eval_results['aggregate_scores']['masking_robustness'],
            'burst_robustness': eval_results['aggregate_scores']['burst_robustness'],
            'confidence_stability': eval_results['aggregate_scores']['confidence_stability']
        })
    
    return pd.DataFrame(results)
