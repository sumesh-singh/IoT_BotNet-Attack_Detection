"""
Confidence Monitor - Tracks prediction confidence stability.
Author: Enhanced IoT BotScan Team
"""

import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConfidenceMonitor:
    """Monitor prediction confidence and detect instability."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.baseline_confidence = None
        self.confidence_history = []
        logger.info("ConfidenceMonitor initialized")
    
    def set_baseline(self, confidences: np.ndarray):
        """Set baseline confidence distribution.
        
        Args:
            confidences: Array of prediction confidences
        """
        self.baseline_confidence = {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'median': np.median(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences)
        }
        logger.info(f"Baseline confidence set: mean={self.baseline_confidence['mean']:.4f}")
    
    def analyze_confidence_shift(self, new_confidences: np.ndarray) -> Dict[str, Any]:
        """Detect confidence shifts from baseline.
        
        Args:
            new_confidences: Current prediction confidences
            
        Returns:
            Analysis results with shift detection
        """
        current = {
            'mean': np.mean(new_confidences),
            'std': np.std(new_confidences),
            'median': np.median(new_confidences)
        }
        
        if self.baseline_confidence is None:
            return {
                'shift_detected': False,
                'current': current,
                'message': 'No baseline set'
            }
        
        mean_shift = abs(current['mean'] - self.baseline_confidence['mean'])
        std_shift = abs(current['std'] - self.baseline_confidence['std'])
        
        # Detect significant shift (>10% change)
        threshold = self.config.get('shift_threshold', 0.1)
        significant = mean_shift > threshold or std_shift > threshold
        
        result = {
            'shift_detected': significant,
            'mean_shift': mean_shift,
            'std_shift': std_shift,
            'current': current,
            'baseline': self.baseline_confidence,
            'stability_score': 1.0 - min(mean_shift * 2, 1.0)
        }
        
        self.confidence_history.append(current['mean'])
        
        return result
    
    def get_stability_score(self) -> float:
        """Get overall confidence stability score.
        
        Returns:
            Score from 0 (unstable) to 1 (stable)
        """
        if len(self.confidence_history) < 2:
            return 1.0
        
        # Variance in confidence over time
        variance = np.var(self.confidence_history)
        return max(0, 1.0 - variance * 10)
