"""
Performance Monitoring Service for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Tracks model performance over time and detects significant degradation (concept drift).
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitors model performance for degradation signals."""

    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self.history = []
        self.window_size = config.get('window_size', 5)
        self.degradation_threshold = config.get('degradation_threshold', 0.05) # 5% drop
        self.baseline_accuracy = None

    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update monitor with new evaluation metrics.
        
        Args:
            metrics: Dictionary containing 'ensemble_validation_accuracy' or similar.
        """
        timestamp = datetime.now().isoformat()
        accuracy = metrics.get('ensemble_validation_accuracy') or metrics.get('accuracy')
        
        if accuracy is None:
            logger.warning("No accuracy metric found in update.")
            return

        record = {
            'timestamp': timestamp,
            'accuracy': float(accuracy),
            'metrics': metrics
        }
        
        self.history.append(record)
        
        # Update baseline if it's the first record or explicitly reset
        if self.baseline_accuracy is None:
            self.baseline_accuracy = accuracy
            logger.info(f"Baseline accuracy set to {accuracy:.4f}")
            
        logger.info(f"Performance metric recorded: Accuracy={accuracy:.4f}")

    def check_degradation(self) -> Dict[str, Any]:
        """
        Check for performance degradation.
        
        Returns:
            Dictionary with status and details.
        """
        if not self.history:
            return {'status': 'ok', 'drift_detected': False, 'message': 'No history available.'}
            
        current_acc = self.history[-1]['accuracy']
        
        # Method 1: Drop from baseline
        if self.baseline_accuracy:
            drop = self.baseline_accuracy - current_acc
            if drop > self.degradation_threshold:
                return {
                    'status': 'warning',
                    'drift_detected': True,
                    'type': 'baseline_drop',
                    'message': f"Accuracy dropped by {drop:.2%} from baseline ({self.baseline_accuracy:.2%}).",
                    'current_accuracy': current_acc,
                    'baseline_accuracy': self.baseline_accuracy
                }

        # Method 2: Moving average drop (if enough history)
        if len(self.history) >= self.window_size:
            recent_accs = [h['accuracy'] for h in self.history[-self.window_size:]]
            avg_acc = np.mean(recent_accs)
            # If current is significantly below moving average? 
            # Or if moving average is declining? 
            # Simple check: compare current to average of previous N
            prev_avg = np.mean(recent_accs[:-1])
            if (prev_avg - current_acc) > self.degradation_threshold:
                 return {
                    'status': 'warning',
                    'drift_detected': True,
                    'type': 'moving_avg_drop',
                    'message': f"Sudden accuracy drop detected ({prev_avg:.2%} -> {current_acc:.2%}).",
                    'current_accuracy': current_acc
                }

        return {
            'status': 'ok', 
            'drift_detected': False, 
            'message': 'Performance stable.',
            'current_accuracy': current_acc
        }

    def reset_baseline(self):
        """Reset baseline to current latest accuracy."""
        if self.history:
            self.baseline_accuracy = self.history[-1]['accuracy']
            logger.info(f"Baseline reset to {self.baseline_accuracy:.4f}")

    def get_history_df(self) -> pd.DataFrame:
        """Get history as DataFrame."""
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history)
