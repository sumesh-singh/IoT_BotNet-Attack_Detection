"""
Robustness Metrics - Evaluation framework for ARM.
Author: Enhanced IoT BotScan Team
"""

import numpy as np
import logging
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, confusion_matrix

logger = logging.getLogger(__name__)


class RobustnessMetrics:
    """Compute and aggregate robustness metrics."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.weights = {
            'noise': config.get('noise_weight', 0.25) if config else 0.25,
            'masking': config.get('masking_weight', 0.25) if config else 0.25,
            'burst': config.get('burst_weight', 0.25) if config else 0.25,
            'confidence': config.get('confidence_weight', 0.25) if config else 0.25
        }
        logger.info("RobustnessMetrics initialized")
    
    def compute_accuracy_under_perturbation(self, y_true: np.ndarray, 
                                             y_pred_clean: np.ndarray,
                                             y_pred_perturbed: np.ndarray) -> Dict[str, float]:
        """Compute accuracy metrics under perturbation.
        
        Args:
            y_true: True labels
            y_pred_clean: Clean predictions
            y_pred_perturbed: Predictions on perturbed data
            
        Returns:
            Accuracy metrics
        """
        clean_accuracy = accuracy_score(y_true, y_pred_clean)
        perturbed_accuracy = accuracy_score(y_true, y_pred_perturbed)
        accuracy_drop = clean_accuracy - perturbed_accuracy
        
        # Flip rate: predictions that changed
        flip_rate = (y_pred_clean != y_pred_perturbed).mean()
        
        return {
            'clean_accuracy': clean_accuracy,
            'perturbed_accuracy': perturbed_accuracy,
            'accuracy_drop': accuracy_drop,
            'accuracy_retention': perturbed_accuracy / max(clean_accuracy, 0.01),
            'flip_rate': flip_rate,
            'robustness_score': max(0, 1.0 - accuracy_drop * 2)
        }
    
    def compute_confidence_stability(self, confidences_clean: np.ndarray,
                                     confidences_perturbed: np.ndarray) -> Dict[str, float]:
        """Compute confidence stability metrics.
        
        Args:
            confidences_clean: Confidences on clean data
            confidences_perturbed: Confidences on perturbed data
            
        Returns:
            Confidence stability metrics
        """
        mean_shift = np.abs(np.mean(confidences_clean) - np.mean(confidences_perturbed))
        max_shift = np.max(np.abs(confidences_clean - confidences_perturbed))
        correlation = np.corrcoef(confidences_clean, confidences_perturbed)[0, 1]
        
        # Handle NaN correlation
        if np.isnan(correlation):
            correlation = 0.0
        
        return {
            'mean_shift': mean_shift,
            'max_shift': max_shift,
            'correlation': correlation,
            'stability_score': max(0, correlation)
        }
    
    def compute_aggregate_robustness(self, scenario_results: Dict[str, Dict]) -> float:
        """Compute weighted aggregate robustness score.
        
        Args:
            scenario_results: Results from each threat scenario
            
        Returns:
            Aggregate robustness score (0-1)
        """
        scores = []
        
        for scenario, weight in self.weights.items():
            if scenario in scenario_results:
                # Average robustness across all settings for this scenario
                scenario_scores = []
                for setting, metrics in scenario_results[scenario].items():
                    if 'robustness_score' in metrics:
                        scenario_scores.append(metrics['robustness_score'])
                    elif 'accuracy_retention' in metrics:
                        scenario_scores.append(metrics['accuracy_retention'])
                
                if scenario_scores:
                    weighted_score = np.mean(scenario_scores) * weight
                    scores.append(weighted_score)
        
        if not scores:
            return 1.0  # No threats = fully robust (baseline)
        
        # Normalize by sum of used weights
        total_weight = sum(self.weights.values())
        return sum(scores) / total_weight * len(scores) / len(self.weights)
    
    def generate_summary(self, scenario_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate a summary of all robustness metrics.
        
        Args:
            scenario_results: Results from all scenarios
            
        Returns:
            Summary dictionary
        """
        summary = {
            'overall_robustness': self.compute_aggregate_robustness(scenario_results)
        }
        
        # Per-scenario summaries
        for scenario, results in scenario_results.items():
            if results:
                robustness_scores = [
                    m.get('robustness_score', m.get('accuracy_retention', 1.0))
                    for m in results.values() if isinstance(m, dict)
                ]
                if robustness_scores:
                    summary[f'{scenario}_robustness'] = np.mean(robustness_scores)
        
        return summary
