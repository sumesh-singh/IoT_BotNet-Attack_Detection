"""
Adaptive Robustness Monitor (ARM) - Main Module
Author: Enhanced IoT BotScan Team

Provides tree-ensemble friendly robustness testing using realistic IoT threat simulations.
"""

import numpy as np
import logging
import warnings
from typing import Dict, Any, Optional
from sklearn.base import BaseEstimator
from datetime import datetime

from .threat_generators.noise_injector import NoiseInjector
from .threat_generators.feature_masker import FeatureMasker
from .threat_generators.burst_generator import BurstGenerator
from .detectors.confidence_monitor import ConfidenceMonitor
from .detectors.accuracy_monitor import AccuracyMonitor
from .detectors.stability_analyzer import StabilityAnalyzer
from .evaluation.robustness_metrics import RobustnessMetrics

logger = logging.getLogger(__name__)


class AdaptiveRobustnessMonitor:
    """
    Adaptive Robustness Monitor (ARM) for tree-based ensemble models.
    
    Unlike gradient-based attacks (FGSM/PGD), ARM uses realistic IoT threat
    simulations that work properly with non-differentiable models.
    
    Threat types tested:
    - Noise injection (sensor noise, environmental interference)
    - Feature masking (sensor failures, missing data)
    - Traffic bursts (DDoS patterns, flash crowds)
    - Evasion attempts (simple adversarial perturbations)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize ARM with configuration.
        
        Args:
            config: Configuration dictionary with:
                - noise_levels: List of noise scales to test
                - masking_rates: List of masking rates to test
                - burst_intensities: List of burst intensities to test
        """
        self.config = config or {}
        
        # Default test configurations
        self.noise_levels = self.config.get('noise_levels', [0.0, 0.05, 0.1, 0.2])
        self.masking_rates = self.config.get('masking_rates', [0.0, 0.1, 0.2, 0.3])
        self.burst_intensities = self.config.get('burst_intensities', [1.0, 1.5, 2.0, 3.0])
        
        # Initialize components
        self.noise_injector = NoiseInjector(config)
        self.feature_masker = FeatureMasker(config)
        self.burst_generator = BurstGenerator(config)
        self.confidence_monitor = ConfidenceMonitor(config)
        self.accuracy_monitor = AccuracyMonitor(config)
        self.stability_analyzer = StabilityAnalyzer(config)
        self.metrics = RobustnessMetrics(config)
        
        # Results storage
        self.baseline = None
        self.results_history = []
        
        logger.info("AdaptiveRobustnessMonitor initialized")
    
    def establish_baseline(self, model: BaseEstimator, X: np.ndarray, 
                          y: np.ndarray) -> Dict[str, Any]:
        """Establish baseline performance metrics.
        
        Args:
            model: Trained model
            X: Test features
            y: True labels
            
        Returns:
            Baseline metrics
        """
        logger.info("Establishing baseline performance...")
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            # Get predictions and probabilities
            y_pred = model.predict(X)
            
            try:
                y_proba = model.predict_proba(X)
                confidences = np.max(y_proba, axis=1)
            except Exception:
                confidences = np.ones(len(y_pred))
        
        # Set baselines in monitors
        self.accuracy_monitor.set_baseline(y, y_pred)
        self.confidence_monitor.set_baseline(confidences)
        
        self.baseline = {
            'accuracy': float(np.mean(y_pred == y)),
            'confidence': float(np.mean(confidences)),
            'predictions': y_pred.copy(),
            'confidences': confidences.copy(),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Baseline: accuracy={self.baseline['accuracy']:.4f}, "
                   f"confidence={self.baseline['confidence']:.4f}")
        
        return self.baseline
    
    def evaluate_noise_robustness(self, model: BaseEstimator, X: np.ndarray,
                                  y: np.ndarray) -> Dict[str, Dict]:
        """Evaluate robustness against noise injection.
        
        Args:
            model: Trained model
            X: Test features
            y: True labels
            
        Returns:
            Results for each noise level
        """
        results = {}
        
        for level in self.noise_levels:
            if level == 0:
                X_noisy = X.copy()
            else:
                X_noisy = self.noise_injector.inject_gaussian_noise(X, scale=level)
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                y_pred = model.predict(X_noisy)
                try:
                    confidences = np.max(model.predict_proba(X_noisy), axis=1)
                except:
                    confidences = np.ones(len(y_pred))
            
            accuracy = float(np.mean(y_pred == y))
            acc_drop = self.baseline['accuracy'] - accuracy if self.baseline else 0
            
            results[f'noise_{level}'] = {
                'accuracy': accuracy,
                'accuracy_drop': acc_drop,
                'mean_confidence': float(np.mean(confidences)),
                'flip_rate': float(np.mean(y_pred != self.baseline['predictions'])) if self.baseline else 0,
                'robustness_score': max(0, 1.0 - acc_drop * 2)
            }
        
        return results
    
    def evaluate_masking_robustness(self, model: BaseEstimator, X: np.ndarray,
                                    y: np.ndarray) -> Dict[str, Dict]:
        """Evaluate robustness against feature masking.
        
        Args:
            model: Trained model
            X: Test features
            y: True labels
            
        Returns:
            Results for each masking rate
        """
        results = {}
        
        for rate in self.masking_rates:
            if rate == 0:
                X_masked = X.copy()
            else:
                X_masked = self.feature_masker.mask_random_features(X, mask_rate=rate)
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                y_pred = model.predict(X_masked)
                try:
                    confidences = np.max(model.predict_proba(X_masked), axis=1)
                except:
                    confidences = np.ones(len(y_pred))
            
            accuracy = float(np.mean(y_pred == y))
            acc_drop = self.baseline['accuracy'] - accuracy if self.baseline else 0
            
            results[f'mask_{rate}'] = {
                'accuracy': accuracy,
                'accuracy_drop': acc_drop,
                'mean_confidence': float(np.mean(confidences)),
                'flip_rate': float(np.mean(y_pred != self.baseline['predictions'])) if self.baseline else 0,
                'robustness_score': max(0, 1.0 - acc_drop * 2)
            }
        
        return results
    
    def evaluate_burst_robustness(self, model: BaseEstimator, X: np.ndarray,
                                  y: np.ndarray) -> Dict[str, Dict]:
        """Evaluate robustness against traffic bursts.
        
        Args:
            model: Trained model
            X: Test features
            y: True labels
            
        Returns:
            Results for each burst intensity
        """
        results = {}
        
        for intensity in self.burst_intensities:
            if intensity == 1.0:
                X_burst = X.copy()
            else:
                X_burst = self.burst_generator.simulate_burst_traffic(X, intensity=intensity)
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                y_pred = model.predict(X_burst)
                try:
                    confidences = np.max(model.predict_proba(X_burst), axis=1)
                except:
                    confidences = np.ones(len(y_pred))
            
            accuracy = float(np.mean(y_pred == y))
            acc_drop = self.baseline['accuracy'] - accuracy if self.baseline else 0
            
            results[f'burst_{intensity}'] = {
                'accuracy': accuracy,
                'accuracy_drop': acc_drop,
                'mean_confidence': float(np.mean(confidences)),
                'flip_rate': float(np.mean(y_pred != self.baseline['predictions'])) if self.baseline else 0,
                'robustness_score': max(0, 1.0 - acc_drop * 2)
            }
        
        return results
    
    def evaluate_comprehensive_robustness(self, model: BaseEstimator, 
                                          X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Run comprehensive robustness evaluation.
        
        Args:
            model: Trained model
            X: Test features
            y: True labels
            
        Returns:
            Comprehensive results with aggregate scores
        """
        logger.info("Starting comprehensive robustness evaluation...")
        
        # Ensure baseline is set
        if self.baseline is None:
            self.establish_baseline(model, X, y)
        
        # Run all evaluations
        noise_results = self.evaluate_noise_robustness(model, X, y)
        masking_results = self.evaluate_masking_robustness(model, X, y)
        burst_results = self.evaluate_burst_robustness(model, X, y)
        
        # Aggregate results
        scenario_results = {
            'noise': noise_results,
            'masking': masking_results,
            'burst': burst_results
        }
        
        # Compute aggregate scores
        aggregate_scores = {
            'noise_robustness': np.mean([r['robustness_score'] for r in noise_results.values()]),
            'masking_robustness': np.mean([r['robustness_score'] for r in masking_results.values()]),
            'burst_robustness': np.mean([r['robustness_score'] for r in burst_results.values()]),
            'confidence_stability': self.confidence_monitor.get_stability_score()
        }
        
        aggregate_scores['overall_robustness'] = np.mean(list(aggregate_scores.values()))
        
        results = {
            'baseline': self.baseline,
            'threat_scenarios': scenario_results,
            'detailed_results': scenario_results, # Alias for frontend compatibility
            'aggregate_scores': aggregate_scores,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results_history.append(results)
        
        logger.info(f"Comprehensive evaluation complete. Overall robustness: "
                   f"{aggregate_scores['overall_robustness']:.4f}")
        
        return results
    
    def get_robustness_report(self) -> Dict[str, Any]:
        """Generate a summary robustness report.
        
        Returns:
            Report with key metrics and recommendations
        """
        if not self.results_history:
            return {'error': 'No evaluations performed yet'}
        
        latest = self.results_history[-1]
        scores = latest['aggregate_scores']
        
        # Determine weakest area
        areas = {k: v for k, v in scores.items() if k != 'overall_robustness'}
        weakest = min(areas, key=areas.get)
        
        # Generate recommendations
        recommendations = []
        if scores['noise_robustness'] < 0.9:
            recommendations.append("Consider adding noise augmentation during training")
        if scores['masking_robustness'] < 0.9:
            recommendations.append("Model is sensitive to missing features - add dropout/masking regularization")
        if scores['burst_robustness'] < 0.9:
            recommendations.append("Model may struggle with traffic spikes - add burst data to training")
        
        return {
            'summary': scores,
            'baseline_accuracy': self.baseline['accuracy'] if self.baseline else None,
            'weakest_area': weakest,
            'recommendations': recommendations,
            'evaluation_count': len(self.results_history),
            'timestamp': latest['timestamp']
        }
