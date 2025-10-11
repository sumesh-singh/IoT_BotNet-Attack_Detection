"""
Kolmogorov-Smirnov Drift Detection Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Implements Kolmogorov-Smirnov test for detecting concept drift in data distributions.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from scipy import stats
from scipy.stats import ks_2samp
import warnings

logger = logging.getLogger(__name__)


class KolmogorovSmirnovDriftDetector:
    """Kolmogorov-Smirnov test for concept drift detection."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize KS drift detector with configuration."""

        self.config = config or {}
        self.reference_data = None
        self.drift_history = []

        # KS test configuration
        self.alpha = self.config.get('alpha', 0.05)
        self.min_samples = self.config.get('min_samples', 100)
        self.window_size = self.config.get('window_size', 1000)
        self.feature_threshold = self.config.get('feature_threshold', 0.5)
        self.adaptive_threshold = self.config.get('adaptive_threshold', True)

        logger.info(
            f"KolmogorovSmirnovDriftDetector initialized with alpha={self.alpha}")

    def set_reference_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Set reference data for drift detection.

        Args:
            X: Reference features
            y: Reference labels (optional)
        """

        if len(X) < self.min_samples:
            raise ValueError(
                f"Reference data must have at least {self.min_samples} samples")

        self.reference_data = {
            'X': X.copy(),
            'y': y.copy() if y is not None else None,
            'n_samples': len(X),
            'n_features': X.shape[1] if len(X.shape) > 1 else 1
        }

        logger.info(
            f"Reference data set: {len(X)} samples, {X.shape[1]} features")

    def detect_drift(self, X_new: np.ndarray, y_new: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Detect concept drift in new data.

        Args:
            X_new: New data to test for drift
            y_new: New labels (optional)

        Returns:
            Drift detection results
        """

        if self.reference_data is None:
            raise ValueError(
                "Reference data must be set before drift detection")

        if len(X_new) < self.min_samples:
            logger.warning(
                f"New data has only {len(X_new)} samples, minimum required: {self.min_samples}")
            return {
                'drift_detected': False,
                'reason': 'insufficient_samples',
                'n_samples': len(X_new)
            }

        logger.info(f"Detecting drift in {len(X_new)} new samples")

        # Detect drift in features
        feature_drift_results = self._detect_feature_drift(X_new)

        # Detect drift in labels if available
        label_drift_results = None
        if y_new is not None and self.reference_data['y'] is not None:
            label_drift_results = self._detect_label_drift(y_new)

        # Determine overall drift
        drift_detected = self._determine_overall_drift(
            feature_drift_results, label_drift_results)

        # Create results
        results = {
            'drift_detected': drift_detected,
            'feature_drift': feature_drift_results,
            'label_drift': label_drift_results,
            'n_samples': len(X_new),
            'n_features': X_new.shape[1] if len(X_new.shape) > 1 else 1,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        # Store in history
        self.drift_history.append(results)

        logger.info(
            f"Drift detection completed: drift_detected={drift_detected}")

        return results

    def _detect_feature_drift(self, X_new: np.ndarray) -> Dict[str, Any]:
        """Detect drift in feature distributions."""

        X_ref = self.reference_data['X']
        n_features = X_ref.shape[1] if len(X_ref.shape) > 1 else 1

        # Ensure X_new has the same shape
        if len(X_new.shape) == 1:
            X_new = X_new.reshape(-1, 1)
        if len(X_ref.shape) == 1:
            X_ref = X_ref.reshape(-1, 1)

        feature_results = []
        drift_count = 0

        for i in range(n_features):
            try:
                # Perform KS test
                ks_statistic, p_value = ks_2samp(X_ref[:, i], X_new[:, i])

                # Determine if drift is detected
                drift_detected = p_value < self.alpha

                feature_result = {
                    'feature_index': i,
                    'ks_statistic': ks_statistic,
                    'p_value': p_value,
                    'drift_detected': drift_detected,
                    'reference_mean': np.mean(X_ref[:, i]),
                    'reference_std': np.std(X_ref[:, i]),
                    'new_mean': np.mean(X_new[:, i]),
                    'new_std': np.std(X_new[:, i])
                }

                feature_results.append(feature_result)

                if drift_detected:
                    drift_count += 1

            except Exception as e:
                logger.error(f"Error in KS test for feature {i}: {e}")
                feature_results.append({
                    'feature_index': i,
                    'error': str(e),
                    'drift_detected': False
                })

        # Calculate overall feature drift
        drift_ratio = drift_count / n_features
        overall_feature_drift = drift_ratio >= self.feature_threshold

        return {
            'overall_drift': overall_feature_drift,
            'drift_ratio': drift_ratio,
            'drift_count': drift_count,
            'total_features': n_features,
            'feature_results': feature_results,
            'threshold': self.feature_threshold
        }

    def _detect_label_drift(self, y_new: np.ndarray) -> Dict[str, Any]:
        """Detect drift in label distributions."""

        y_ref = self.reference_data['y']

        try:
            # Perform KS test on labels
            ks_statistic, p_value = ks_2samp(y_ref, y_new)

            # Determine if drift is detected
            drift_detected = p_value < self.alpha

            # Calculate label distribution statistics
            ref_unique, ref_counts = np.unique(y_ref, return_counts=True)
            new_unique, new_counts = np.unique(y_new, return_counts=True)

            ref_distribution = dict(zip(ref_unique, ref_counts / len(y_ref)))
            new_distribution = dict(zip(new_unique, new_counts / len(y_new)))

            return {
                'drift_detected': drift_detected,
                'ks_statistic': ks_statistic,
                'p_value': p_value,
                'reference_distribution': ref_distribution,
                'new_distribution': new_distribution,
                'reference_mean': np.mean(y_ref),
                'new_mean': np.mean(y_new)
            }

        except Exception as e:
            logger.error(f"Error in label drift detection: {e}")
            return {
                'drift_detected': False,
                'error': str(e)
            }

    def _determine_overall_drift(self, feature_drift_results: Dict[str, Any],
                                 label_drift_results: Optional[Dict[str, Any]]) -> bool:
        """Determine overall drift based on feature and label drift results."""

        # Check feature drift
        feature_drift = feature_drift_results['overall_drift']

        # Check label drift if available
        label_drift = False
        if label_drift_results is not None and 'error' not in label_drift_results:
            label_drift = label_drift_results['drift_detected']

        # Overall drift is detected if either feature or label drift is detected
        overall_drift = feature_drift or label_drift

        return overall_drift

    def get_drift_statistics(self) -> Dict[str, Any]:
        """Get comprehensive drift detection statistics."""

        if not self.drift_history:
            return {'n_detections': 0}

        # Calculate statistics
        drift_detections = [result['drift_detected']
                            for result in self.drift_history]
        drift_rate = np.mean(drift_detections)

        # Feature drift statistics
        feature_drift_rates = []
        for result in self.drift_history:
            if 'feature_drift' in result:
                feature_drift_rates.append(
                    result['feature_drift']['drift_ratio'])

        # Label drift statistics
        label_drift_detections = []
        for result in self.drift_history:
            if result['label_drift'] is not None and 'error' not in result['label_drift']:
                label_drift_detections.append(
                    result['label_drift']['drift_detected'])

        statistics = {
            'n_detections': len(self.drift_history),
            'overall_drift_rate': drift_rate,
            'feature_drift_rate': np.mean(feature_drift_rates) if feature_drift_rates else 0,
            'label_drift_rate': np.mean(label_drift_detections) if label_drift_detections else 0,
            'last_detection': self.drift_history[-1]['timestamp'] if self.drift_history else None
        }

        return statistics

    def get_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance based on drift detection frequency."""

        if not self.drift_history:
            return {}

        # Count drift detections per feature
        feature_drift_counts = {}

        for result in self.drift_history:
            if 'feature_drift' in result and 'feature_results' in result['feature_drift']:
                for feature_result in result['feature_drift']['feature_results']:
                    if 'drift_detected' in feature_result and feature_result['drift_detected']:
                        feature_idx = feature_result['feature_index']
                        feature_drift_counts[feature_idx] = feature_drift_counts.get(
                            feature_idx, 0) + 1

        # Calculate importance scores
        total_detections = len(self.drift_history)
        feature_importance = {}

        for feature_idx, count in feature_drift_counts.items():
            feature_importance[f'feature_{feature_idx}'] = count / \
                total_detections

        return feature_importance

    def update_reference_data(self, X_new: np.ndarray, y_new: Optional[np.ndarray] = None) -> None:
        """Update reference data with new data."""

        if self.reference_data is None:
            self.set_reference_data(X_new, y_new)
            return

        # Combine reference and new data
        X_combined = np.vstack([self.reference_data['X'], X_new])
        y_combined = None

        if y_new is not None and self.reference_data['y'] is not None:
            y_combined = np.hstack([self.reference_data['y'], y_new])

        # Update reference data
        self.reference_data = {
            'X': X_combined,
            'y': y_combined,
            'n_samples': len(X_combined),
            'n_features': X_combined.shape[1] if len(X_combined.shape) > 1 else 1
        }

        logger.info(f"Reference data updated: {len(X_combined)} samples")

    def reset_detector(self) -> None:
        """Reset the drift detector."""

        self.reference_data = None
        self.drift_history = []

        logger.info("Drift detector reset")


# Example usage and testing
if __name__ == '__main__':
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # Reference data (normal distribution)
    X_ref = np.random.normal(0, 1, (n_samples, n_features))
    y_ref = np.random.randint(0, 2, n_samples)

    # New data (same distribution)
    X_new_same = np.random.normal(0, 1, (n_samples, n_features))
    y_new_same = np.random.randint(0, 2, n_samples)

    # New data (different distribution - drift)
    X_new_drift = np.random.normal(2, 1, (n_samples, n_features))
    y_new_drift = np.random.randint(0, 2, n_samples)

    # Initialize detector
    detector = KolmogorovSmirnovDriftDetector({
        'alpha': 0.05,
        'feature_threshold': 0.3
    })

    # Set reference data
    detector.set_reference_data(X_ref, y_ref)

    # Test with same distribution
    print("Testing with same distribution:")
    results_same = detector.detect_drift(X_new_same, y_new_same)
    print(f"Drift detected: {results_same['drift_detected']}")
    print(
        f"Feature drift ratio: {results_same['feature_drift']['drift_ratio']:.4f}")

    # Test with different distribution
    print("\nTesting with different distribution:")
    results_drift = detector.detect_drift(X_new_drift, y_new_drift)
    print(f"Drift detected: {results_drift['drift_detected']}")
    print(
        f"Feature drift ratio: {results_drift['feature_drift']['drift_ratio']:.4f}")

    # Get statistics
    stats = detector.get_drift_statistics()
    print(f"\nDrift Statistics:")
    print(f"Total detections: {stats['n_detections']}")
    print(f"Overall drift rate: {stats['overall_drift_rate']:.4f}")

    # Get feature importance
    importance = detector.get_feature_importance()
    print(f"\nFeature Importance (drift frequency):")
    for feature, score in importance.items():
        print(f"  {feature}: {score:.4f}")

    # Test with streaming data
    print("\nTesting with streaming data:")
    for i in range(5):
        # Gradually introduce drift
        drift_amount = i * 0.5
        X_stream = np.random.normal(drift_amount, 1, (200, n_features))
        y_stream = np.random.randint(0, 2, 200)

        results = detector.detect_drift(X_stream, y_stream)
        print(f"Stream {i+1}: drift_detected={results['drift_detected']}, "
              f"drift_ratio={results['feature_drift']['drift_ratio']:.4f}")

        # Update reference data
        detector.update_reference_data(X_stream, y_stream)
