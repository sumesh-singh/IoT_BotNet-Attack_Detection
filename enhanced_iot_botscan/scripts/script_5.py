# Now implement the concept drift detection components

# 6. Kolmogorov-Smirnov Drift Detection
ks_drift_content = '''"""
Kolmogorov-Smirnov Test for Concept Drift Detection
Author: Kotiwale Sumesh Singh (160124862043)

Implements K-S test to detect distributional changes in IoT network traffic patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
import logging
from datetime import datetime
from collections import deque

class KolmogorovSmirnovDriftDetector:
    """Kolmogorov-Smirnov test-based concept drift detector."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alpha = config.get('alpha', 0.05)  # Significance level
        self.window_size = config.get('window_size', 1000)
        self.alternative = config.get('alternative', 'two-sided')
        self.min_samples = config.get('min_samples', 100)
        
        # Storage for reference and current windows
        self.reference_window = deque(maxlen=self.window_size)
        self.current_window = deque(maxlen=self.window_size)
        
        # Drift detection state
        self.drift_detected = False
        self.p_values_history = []
        self.drift_history = []
        self.feature_drift_scores = {}
        
        # Statistics
        self.total_samples_processed = 0
        self.drift_detections = 0
        self.last_drift_timestamp = None
        
    def add_reference_data(self, X: np.ndarray) -> None:
        """
        Add reference data (training distribution).
        
        Args:
            X: Reference feature matrix
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Store reference data
        for sample in X:
            self.reference_window.append(sample.copy())
            
        print(f"Added {len(X)} samples to reference window (total: {len(self.reference_window)})")
    
    def detect_drift(self, X_new: np.ndarray) -> Dict[str, Any]:
        """
        Detect concept drift in new data batch.
        
        Args:
            X_new: New data batch
            
        Returns:
            Drift detection results
        """
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        
        # Add new samples to current window
        for sample in X_new:
            self.current_window.append(sample.copy())
        
        self.total_samples_processed += len(X_new)
        
        # Check if we have enough samples for testing
        if len(self.reference_window) < self.min_samples or len(self.current_window) < self.min_samples:
            return {
                'drift_detected': False,
                'reason': 'Insufficient samples',
                'reference_samples': len(self.reference_window),
                'current_samples': len(self.current_window)
            }
        
        # Perform K-S test
        drift_results = self._perform_ks_test()
        
        # Update drift history
        self.drift_history.append({
            'timestamp': datetime.now().isoformat(),
            'samples_processed': self.total_samples_processed,
            'drift_detected': drift_results['drift_detected'],
            'p_value': drift_results['p_value'],
            'ks_statistic': drift_results['ks_statistic']
        })
        
        # Update state if drift detected
        if drift_results['drift_detected']:
            self.drift_detected = True
            self.drift_detections += 1
            self.last_drift_timestamp = datetime.now()
            print(f"🚨 DRIFT DETECTED! P-value: {drift_results['p_value']:.6f}")
        
        return drift_results
    
    def _perform_ks_test(self) -> Dict[str, Any]:
        """Perform Kolmogorov-Smirnov test for each feature."""
        
        # Convert windows to arrays
        ref_data = np.array(list(self.reference_window))
        cur_data = np.array(list(self.current_window))
        
        n_features = ref_data.shape[1]
        
        # Test each feature separately
        feature_results = {}
        p_values = []
        ks_statistics = []
        
        for feature_idx in range(n_features):
            ref_feature = ref_data[:, feature_idx]
            cur_feature = cur_data[:, feature_idx]
            
            # Perform K-S test
            ks_stat, p_value = stats.ks_2samp(
                ref_feature, cur_feature, 
                alternative=self.alternative
            )
            
            feature_results[f'feature_{feature_idx}'] = {
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'drift_detected': p_value < self.alpha
            }
            
            p_values.append(p_value)
            ks_statistics.append(ks_stat)
        
        # Overall drift decision
        min_p_value = min(p_values)
        max_ks_stat = max(ks_statistics)
        
        # Apply Bonferroni correction for multiple testing
        corrected_alpha = self.alpha / n_features
        overall_drift = min_p_value < corrected_alpha
        
        self.p_values_history.append(min_p_value)
        self.feature_drift_scores = feature_results
        
        return {
            'drift_detected': overall_drift,
            'p_value': min_p_value,
            'ks_statistic': max_ks_stat,
            'corrected_alpha': corrected_alpha,
            'feature_results': feature_results,
            'n_features_tested': n_features,
            'samples_compared': (len(self.reference_window), len(self.current_window))
        }
    
    def reset_current_window(self) -> None:
        """Reset current window after drift detection."""
        self.current_window.clear()
        self.drift_detected = False
        print("Current window reset after drift detection")
    
    def update_reference_window(self, X_new: Optional[np.ndarray] = None) -> None:
        """
        Update reference window with new data.
        
        Args:
            X_new: New reference data, if None uses current window
        """
        if X_new is not None:
            if X_new.ndim == 1:
                X_new = X_new.reshape(-1, 1)
            
            for sample in X_new:
                self.reference_window.append(sample.copy())
        else:
            # Move current window to reference
            current_data = list(self.current_window)
            for sample in current_data:
                self.reference_window.append(sample.copy())
        
        print(f"Reference window updated (size: {len(self.reference_window)})")
    
    def get_drift_statistics(self) -> Dict[str, Any]:
        """Get comprehensive drift detection statistics."""
        
        return {
            'total_samples_processed': self.total_samples_processed,
            'drift_detections': self.drift_detections,
            'drift_rate': self.drift_detections / max(len(self.drift_history), 1),
            'last_drift_timestamp': self.last_drift_timestamp.isoformat() if self.last_drift_timestamp else None,
            'current_drift_status': self.drift_detected,
            'reference_window_size': len(self.reference_window),
            'current_window_size': len(self.current_window),
            'alpha_threshold': self.alpha,
            'recent_p_values': self.p_values_history[-10:],  # Last 10 p-values
            'feature_drift_scores': self.feature_drift_scores
        }
    
    def plot_drift_history(self, save_path: Optional[str] = None) -> None:
        """Plot drift detection history."""
        try:
            import matplotlib.pyplot as plt
            
            if not self.drift_history:
                print("No drift history to plot")
                return
            
            # Extract data for plotting
            timestamps = [entry['timestamp'] for entry in self.drift_history]
            p_values = [entry['p_value'] for entry in self.drift_history]
            drift_points = [i for i, entry in enumerate(self.drift_history) if entry['drift_detected']]
            
            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(p_values)), p_values, 'b-', label='P-values', alpha=0.7)
            plt.axhline(y=self.alpha, color='r', linestyle='--', label=f'Significance level (α={self.alpha})')
            
            # Mark drift detections
            if drift_points:
                plt.scatter(drift_points, [p_values[i] for i in drift_points], 
                           color='red', s=100, label='Drift detected', zorder=5)
            
            plt.xlabel('Detection Event')
            plt.ylabel('P-value')
            plt.title('Concept Drift Detection History (K-S Test)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Drift history plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("matplotlib not available for plotting")
        except Exception as e:
            print(f"Error plotting drift history: {e}")
    
    def export_drift_history(self, filepath: str) -> None:
        """Export drift detection history to CSV."""
        
        if not self.drift_history:
            print("No drift history to export")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.drift_history)
        df.to_csv(filepath, index=False)
        
        print(f"Drift history exported to {filepath}")
    
    def get_feature_drift_analysis(self) -> Dict[str, Any]:
        """Analyze which features are most prone to drift."""
        
        if not self.feature_drift_scores:
            return {}
        
        # Analyze feature-wise drift patterns
        feature_analysis = {}
        
        for feature_name, results in self.feature_drift_scores.items():
            feature_analysis[feature_name] = {
                'current_p_value': results['p_value'],
                'current_ks_statistic': results['ks_statistic'],
                'drift_detected': results['drift_detected'],
                'drift_severity': 'High' if results['p_value'] < 0.01 else 'Medium' if results['p_value'] < 0.05 else 'Low'
            }
        
        return feature_analysis
'''

with open('./enhanced_iot_botscan/src/core/drift_detection/kolmogorov_smirnov.py', 'w') as f:
    f.write(ks_drift_content)

print("✅ Created kolmogorov_smirnov.py")

# 7. Page-Hinkley Drift Detection
ph_drift_content = '''"""
Page-Hinkley Test for Concept Drift Detection
Author: Kotiwale Sumesh Singh (160124862043)

Implements Page-Hinkley test to detect gradual concept drift in IoT environments.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
from collections import deque

class PageHinkleyDriftDetector:
    """Page-Hinkley test-based concept drift detector for gradual drift detection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.threshold = config.get('threshold', 50)
        self.alpha = config.get('alpha', 0.005)
        self.min_instances = config.get('min_instances', 30)
        
        # Page-Hinkley statistics
        self.sum = 0.0
        self.x_mean = 0.0
        self.sample_count = 0
        self.ph_sum = 0.0
        self.ph_min = 0.0
        
        # Drift detection state
        self.drift_detected = False
        self.warning_detected = False
        self.drift_history = []
        self.ph_values_history = deque(maxlen=1000)
        
        # Statistics tracking
        self.total_samples_processed = 0
        self.drift_detections = 0
        self.warning_detections = 0
        self.last_drift_timestamp = None
        self.last_warning_timestamp = None
    
    def add_element(self, x: float) -> Dict[str, Any]:
        """
        Add new element and check for drift.
        
        Args:
            x: New data point (typically model accuracy or error)
            
        Returns:
            Drift detection result
        """
        self.sample_count += 1
        self.total_samples_processed += 1
        
        # Update mean
        if self.sample_count == 1:
            self.x_mean = x
            self.sum = 0.0
        else:
            self.x_mean = self.x_mean + (x - self.x_mean) / self.sample_count
        
        # Update Page-Hinkley statistics
        self.sum += x - self.x_mean - self.alpha
        
        if self.sum < self.ph_min:
            self.ph_min = self.sum
        
        self.ph_sum = self.sum - self.ph_min
        
        # Store PH value for history
        self.ph_values_history.append(self.ph_sum)
        
        # Check for drift
        drift_result = self._check_drift()
        
        # Update history
        self.drift_history.append({
            'timestamp': datetime.now().isoformat(),
            'sample_count': self.sample_count,
            'value': x,
            'ph_sum': self.ph_sum,
            'drift_detected': drift_result['drift_detected'],
            'warning_detected': drift_result['warning_detected']
        })
        
        return drift_result
    
    def _check_drift(self) -> Dict[str, Any]:
        """Check if drift has been detected."""
        
        # Not enough samples yet
        if self.sample_count < self.min_instances:
            return {
                'drift_detected': False,
                'warning_detected': False,
                'ph_sum': self.ph_sum,
                'threshold': self.threshold,
                'reason': 'Insufficient samples'
            }
        
        # Check for drift
        drift_detected = self.ph_sum > self.threshold
        warning_detected = self.ph_sum > self.threshold * 0.5  # Warning at 50% of threshold
        
        # Update state
        if drift_detected and not self.drift_detected:
            self.drift_detected = True
            self.drift_detections += 1
            self.last_drift_timestamp = datetime.now()
            print(f"🚨 PAGE-HINKLEY DRIFT DETECTED! PH Sum: {self.ph_sum:.4f}")
        
        if warning_detected and not self.warning_detected:
            self.warning_detected = True
            self.warning_detections += 1
            self.last_warning_timestamp = datetime.now()
            print(f"⚠️ Page-Hinkley warning: PH Sum: {self.ph_sum:.4f}")
        
        return {
            'drift_detected': drift_detected,
            'warning_detected': warning_detected,
            'ph_sum': self.ph_sum,
            'threshold': self.threshold,
            'samples_processed': self.sample_count,
            'x_mean': self.x_mean
        }
    
    def reset(self) -> None:
        """Reset the detector after drift is handled."""
        self.sum = 0.0
        self.x_mean = 0.0
        self.sample_count = 0
        self.ph_sum = 0.0
        self.ph_min = 0.0
        self.drift_detected = False
        self.warning_detected = False
        
        print("Page-Hinkley detector reset")
    
    def batch_detect(self, X: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process batch of data points and detect drift.
        
        Args:
            X: Array of data points
            
        Returns:
            List of detection results for each point
        """
        results = []
        
        for x in X:
            result = self.add_element(float(x))
            results.append(result)
        
        return results
    
    def detect_performance_drift(self, 
                                y_true: np.ndarray, 
                                y_pred: np.ndarray, 
                                metric: str = 'accuracy') -> Dict[str, Any]:
        """
        Detect drift based on model performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            metric: Performance metric to monitor
            
        Returns:
            Drift detection result
        """
        
        # Calculate performance metric
        if metric == 'accuracy':
            performance = np.mean(y_true == y_pred)
        elif metric == 'error':
            performance = np.mean(y_true != y_pred)
        elif metric == 'precision':
            from sklearn.metrics import precision_score
            performance = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        elif metric == 'recall':
            from sklearn.metrics import recall_score
            performance = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Add performance value to detector
        result = self.add_element(performance)
        result['performance_value'] = performance
        result['metric_used'] = metric
        
        return result
    
    def get_drift_statistics(self) -> Dict[str, Any]:
        """Get comprehensive drift detection statistics."""
        
        return {
            'total_samples_processed': self.total_samples_processed,
            'drift_detections': self.drift_detections,
            'warning_detections': self.warning_detections,
            'current_ph_sum': self.ph_sum,
            'threshold': self.threshold,
            'current_mean': self.x_mean,
            'drift_detected': self.drift_detected,
            'warning_detected': self.warning_detected,
            'last_drift_timestamp': self.last_drift_timestamp.isoformat() if self.last_drift_timestamp else None,
            'last_warning_timestamp': self.last_warning_timestamp.isoformat() if self.last_warning_timestamp else None,
            'sample_count': self.sample_count
        }
    
    def plot_ph_evolution(self, save_path: Optional[str] = None) -> None:
        """Plot Page-Hinkley sum evolution over time."""
        try:
            import matplotlib.pyplot as plt
            
            if not self.ph_values_history:
                print("No PH history to plot")
                return
            
            ph_values = list(self.ph_values_history)
            
            plt.figure(figsize=(12, 6))
            plt.plot(ph_values, 'b-', label='PH Sum', alpha=0.7)
            plt.axhline(y=self.threshold, color='r', linestyle='--', label=f'Drift threshold ({self.threshold})')
            plt.axhline(y=self.threshold * 0.5, color='orange', linestyle=':', label='Warning threshold')
            
            # Mark drift points
            drift_points = [i for i, entry in enumerate(self.drift_history[-len(ph_values):]) 
                          if entry['drift_detected']]
            if drift_points:
                plt.scatter(drift_points, [ph_values[i] for i in drift_points], 
                           color='red', s=100, label='Drift detected', zorder=5)
            
            # Mark warning points
            warning_points = [i for i, entry in enumerate(self.drift_history[-len(ph_values):]) 
                            if entry['warning_detected'] and not entry['drift_detected']]
            if warning_points:
                plt.scatter(warning_points, [ph_values[i] for i in warning_points], 
                           color='orange', s=50, label='Warning', zorder=5)
            
            plt.xlabel('Sample Index')
            plt.ylabel('Page-Hinkley Sum')
            plt.title('Page-Hinkley Drift Detection')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"PH evolution plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("matplotlib not available for plotting")
        except Exception as e:
            print(f"Error plotting PH evolution: {e}")
    
    def export_history(self, filepath: str) -> None:
        """Export drift detection history to CSV."""
        
        if not self.drift_history:
            print("No drift history to export")
            return
        
        df = pd.DataFrame(self.drift_history)
        df.to_csv(filepath, index=False)
        
        print(f"Page-Hinkley history exported to {filepath}")
    
    def get_recent_trend(self, window_size: int = 50) -> Dict[str, Any]:
        """Analyze recent trend in PH values."""
        
        if len(self.ph_values_history) < window_size:
            return {'message': 'Insufficient data for trend analysis'}
        
        recent_values = list(self.ph_values_history)[-window_size:]
        
        # Calculate trend
        x = np.arange(len(recent_values))
        slope, intercept = np.polyfit(x, recent_values, 1)
        
        # Trend direction
        if slope > 0.1:
            trend = 'Increasing (Potential drift approaching)'
        elif slope < -0.1:
            trend = 'Decreasing (Drift subsiding)'
        else:
            trend = 'Stable'
        
        return {
            'trend_direction': trend,
            'slope': slope,
            'recent_mean': np.mean(recent_values),
            'recent_std': np.std(recent_values),
            'current_ph_sum': self.ph_sum,
            'distance_to_threshold': max(0, self.threshold - self.ph_sum)
        }
'''

with open('./enhanced_iot_botscan/src/core/drift_detection/page_hinkley.py', 'w') as f:
    f.write(ph_drift_content)

print("✅ Created page_hinkley.py")

print("\n🔄 Concept drift detection components implemented! Continuing with more core modules...")