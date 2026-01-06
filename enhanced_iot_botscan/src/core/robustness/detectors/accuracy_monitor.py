"""
Accuracy Monitor - Tracks model performance degradation.
Author: Enhanced IoT BotScan Team
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class AccuracyMonitor:
    """Monitor model accuracy and detect performance degradation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.baseline_accuracy = None
        self.baseline_metrics = None
        self.accuracy_history = []
        self.degradation_threshold = config.get('degradation_threshold', 0.05) if config else 0.05
        logger.info("AccuracyMonitor initialized")
    
    def set_baseline(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Set baseline accuracy metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        self.baseline_accuracy = accuracy_score(y_true, y_pred)
        self.baseline_metrics = {
            'accuracy': self.baseline_accuracy,
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        logger.info(f"Baseline accuracy set: {self.baseline_accuracy:.4f}")
    
    def detect_degradation(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Detect performance degradation from baseline.
        
        Args:
            y_true: True labels
            y_pred: Current predictions
            
        Returns:
            Degradation analysis results
        """
        current_accuracy = accuracy_score(y_true, y_pred)
        current_metrics = {
            'accuracy': current_accuracy,
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        self.accuracy_history.append(current_accuracy)
        
        if self.baseline_accuracy is None:
            return {
                'degradation_detected': False,
                'current': current_metrics,
                'message': 'No baseline set'
            }
        
        accuracy_drop = self.baseline_accuracy - current_accuracy
        degraded = accuracy_drop > self.degradation_threshold
        
        return {
            'degradation_detected': degraded,
            'accuracy_drop': accuracy_drop,
            'drop_percentage': accuracy_drop / max(self.baseline_accuracy, 0.01) * 100,
            'current': current_metrics,
            'baseline': self.baseline_metrics,
            'robustness_score': max(0, 1.0 - accuracy_drop * 2)
        }
    
    def get_degradation_rate(self) -> float:
        """Get rate of degradation over time.
        
        Returns:
            Degradation rate (negative = improving)
        """
        if len(self.accuracy_history) < 2:
            return 0.0
        
        # Linear regression slope
        x = np.arange(len(self.accuracy_history))
        slope = np.polyfit(x, self.accuracy_history, 1)[0]
        return -slope  # Negative slope means degradation
