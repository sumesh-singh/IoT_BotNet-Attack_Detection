"""
Stability Analyzer - Analyzes prediction stability under perturbations.
Author: Enhanced IoT BotScan Team
"""

import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class StabilityAnalyzer:
    """Analyze prediction stability across multiple scenarios."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.prediction_history = []
        logger.info("StabilityAnalyzer initialized")
    
    def analyze_prediction_stability(self, predictions_list: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze stability across multiple prediction runs.
        
        Args:
            predictions_list: List of prediction arrays from different scenarios
            
        Returns:
            Stability analysis results
        """
        if len(predictions_list) < 2:
            return {'stability_score': 1.0, 'variance': 0.0, 'message': 'Need more predictions'}
        
        # Convert to 2D array (scenarios x samples)
        predictions_array = np.array(predictions_list)
        
        # Calculate per-sample variance (how much each sample varies across scenarios)
        sample_variance = np.var(predictions_array, axis=0)
        mean_variance = np.mean(sample_variance)
        
        # Flip rate: how often predictions change between scenarios
        flip_count = 0
        for i in range(1, len(predictions_list)):
            flips = (predictions_list[i] != predictions_list[i-1]).sum()
            flip_count += flips
        
        total_comparisons = (len(predictions_list) - 1) * len(predictions_list[0])
        flip_rate = flip_count / max(total_comparisons, 1)
        
        # Stability score (higher is better)
        stability_score = max(0, 1.0 - flip_rate * 2)
        
        return {
            'stability_score': stability_score,
            'mean_variance': float(mean_variance),
            'flip_rate': float(flip_rate),
            'total_flips': int(flip_count),
            'is_stable': stability_score > 0.8
        }
    
    def analyze_feature_sensitivity(self, model, X: np.ndarray, 
                                    perturbation_scale: float = 0.1) -> Dict[str, float]:
        """Analyze sensitivity to each feature.
        
        Args:
            model: Trained model
            X: Input features
            perturbation_scale: Scale of perturbation as fraction of feature std
            
        Returns:
            Sensitivity score for each feature
        """
        n_features = X.shape[1]
        sensitivity = {}
        
        # Baseline predictions
        baseline_pred = model.predict(X)
        
        for i in range(n_features):
            X_perturbed = X.copy()
            std = np.std(X[:, i])
            std = 1.0 if std == 0 else std
            
            # Perturb single feature
            X_perturbed[:, i] += np.random.randn(X.shape[0]) * std * perturbation_scale
            
            # Check how many predictions change
            new_pred = model.predict(X_perturbed)
            flip_rate = (new_pred != baseline_pred).mean()
            
            sensitivity[f'feature_{i}'] = float(flip_rate)
        
        return sensitivity
    
    def get_most_sensitive_features(self, sensitivity: Dict[str, float], 
                                    top_k: int = 10) -> List[str]:
        """Get most sensitive features.
        
        Args:
            sensitivity: Feature sensitivity scores
            top_k: Number of top features to return
            
        Returns:
            List of most sensitive feature names
        """
        sorted_features = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
        return [f[0] for f in sorted_features[:top_k]]
