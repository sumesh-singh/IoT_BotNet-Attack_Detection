"""
Page-Hinkley Drift Detection Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Implements Page-Hinkley test for detecting concept drift in streaming data.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
import warnings

logger = logging.getLogger(__name__)


class PageHinkleyDriftDetector:
    """Page-Hinkley test for concept drift detection in streaming data."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Page-Hinkley drift detector with configuration."""

        self.config = config or {}
        self.drift_history = []

        # Page-Hinkley configuration
        self.delta = self.config.get('delta', 0.005)
        self.min_samples = self.config.get('min_samples', 30)
        self.window_size = self.config.get('window_size', 100)
        self.threshold = self.config.get('threshold', 50)
        self.alpha = self.config.get('alpha', 0.05)

        # State variables
        self.cumulative_sum = 0
        self.min_cumulative_sum = 0
        self.max_cumulative_sum = 0
        self.sample_count = 0
        self.reference_mean = None
        self.reference_std = None

        logger.info(
            f"PageHinkleyDriftDetector initialized with delta={self.delta}, threshold={self.threshold}")

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

        # Calculate reference statistics
        self.reference_mean = np.mean(X, axis=0)
        self.reference_std = np.std(X, axis=0)

        # Reset state
        self.cumulative_sum = 0
        self.min_cumulative_sum = 0
        self.max_cumulative_sum = 0
        self.sample_count = 0

        logger.info(
            f"Reference data set: {len(X)} samples, {X.shape[1]} features")

    def detect_drift(self, X_new: np.ndarray, y_new: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Detect concept drift in new data using Page-Hinkley test.

        Args:
            X_new: New data to test for drift
            y_new: New labels (optional)

        Returns:
            Drift detection results
        """

        if self.reference_mean is None:
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
        if y_new is not None:
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
        """Detect drift in feature distributions using Page-Hinkley test."""

        n_features = X_new.shape[1] if len(X_new.shape) > 1 else 1

        # Ensure X_new has the same shape as reference
        if len(X_new.shape) == 1:
            X_new = X_new.reshape(-1, 1)

        feature_results = []
        drift_count = 0

        for i in range(n_features):
            try:
                # Calculate Page-Hinkley statistic for this feature
                ph_statistic, ph_detected = self._calculate_page_hinkley_statistic(
                    X_new[:, i], self.reference_mean[i], self.reference_std[i]
                )

                feature_result = {
                    'feature_index': i,
                    'ph_statistic': ph_statistic,
                    'drift_detected': ph_detected,
                    'reference_mean': self.reference_mean[i],
                    'reference_std': self.reference_std[i],
                    'new_mean': np.mean(X_new[:, i]),
                    'new_std': np.std(X_new[:, i])
                }

                feature_results.append(feature_result)

                if ph_detected:
                    drift_count += 1

            except Exception as e:
                logger.error(
                    f"Error in Page-Hinkley test for feature {i}: {e}")
                feature_results.append({
                    'feature_index': i,
                    'error': str(e),
                    'drift_detected': False
                })

        # Calculate overall feature drift
        drift_ratio = drift_count / n_features
        overall_feature_drift = drift_ratio >= 0.5  # Threshold for overall drift

        return {
            'overall_drift': overall_feature_drift,
            'drift_ratio': drift_ratio,
            'drift_count': drift_count,
            'total_features': n_features,
            'feature_results': feature_results
        }

    def _calculate_page_hinkley_statistic(self, data: np.ndarray, ref_mean: float, ref_std: float) -> Tuple[float, bool]:
        """Calculate Page-Hinkley statistic for a single feature."""

        # Normalize data using reference statistics
        normalized_data = (data - ref_mean) / (ref_std + 1e-8)

        # Calculate Page-Hinkley statistic
        ph_statistic = 0
        min_ph = 0
        max_ph = 0

        for value in normalized_data:
            # Update cumulative sum
            ph_statistic += value - self.delta

            # Update min and max
            min_ph = min(min_ph, ph_statistic)
            max_ph = max(max_ph, ph_statistic)

            # Check for drift
            if ph_statistic - min_ph > self.threshold:
                return max_ph - min_ph, True

        return max_ph - min_ph, False

    def _detect_label_drift(self, y_new: np.ndarray) -> Dict[str, Any]:
        """Detect drift in label distributions using Page-Hinkley test."""

        try:
            # Calculate label statistics
            new_mean = np.mean(y_new)
            new_std = np.std(y_new)

            # For binary classification, use proportion of positive class
            if len(np.unique(y_new)) == 2:
                positive_proportion = np.mean(y_new)

                # Calculate Page-Hinkley statistic for label proportion
                ph_statistic, ph_detected = self._calculate_page_hinkley_statistic(
                    y_new, 0.5, 0.5  # Assume balanced reference
                )

                return {
                    'drift_detected': ph_detected,
                    'ph_statistic': ph_statistic,
                    'positive_proportion': positive_proportion,
                    'new_mean': new_mean,
                    'new_std': new_std
                }
            else:
                # For multi-class, use mean
                ph_statistic, ph_detected = self._calculate_page_hinkley_statistic(
                    y_new, 0, 1  # Assume zero-mean, unit-variance reference
                )

                return {
                    'drift_detected': ph_detected,
                    'ph_statistic': ph_statistic,
                    'new_mean': new_mean,
                    'new_std': new_std
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

    def detect_drift_streaming(self, X_new: np.ndarray, y_new: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Detect drift in streaming data (single sample or small batch).

        Args:
            X_new: New data sample(s)
            y_new: New labels (optional)

        Returns:
            Drift detection results
        """

        if self.reference_mean is None:
            raise ValueError(
                "Reference data must be set before drift detection")

        # Ensure X_new has the same shape as reference
        if len(X_new.shape) == 1:
            X_new = X_new.reshape(-1, 1)

        n_features = X_new.shape[1]
        feature_results = []
        drift_count = 0

        for i in range(n_features):
            try:
                # Calculate Page-Hinkley statistic for this feature
                ph_statistic, ph_detected = self._calculate_page_hinkley_statistic(
                    X_new[:, i], self.reference_mean[i], self.reference_std[i]
                )

                feature_result = {
                    'feature_index': i,
                    'ph_statistic': ph_statistic,
                    'drift_detected': ph_detected,
                    'reference_mean': self.reference_mean[i],
                    'reference_std': self.reference_std[i],
                    'new_mean': np.mean(X_new[:, i]),
                    'new_std': np.std(X_new[:, i])
                }

                feature_results.append(feature_result)

                if ph_detected:
                    drift_count += 1

            except Exception as e:
                logger.error(
                    f"Error in Page-Hinkley test for feature {i}: {e}")
                feature_results.append({
                    'feature_index': i,
                    'error': str(e),
                    'drift_detected': False
                })

        # Calculate overall feature drift
        drift_ratio = drift_count / n_features
        overall_feature_drift = drift_ratio >= 0.5

        # Detect label drift if available
        label_drift_results = None
        if y_new is not None:
            label_drift_results = self._detect_label_drift(y_new)

        # Determine overall drift
        drift_detected = self._determine_overall_drift(
            {'overall_drift': overall_feature_drift, 'drift_ratio': drift_ratio},
            label_drift_results
        )

        # Create results
        results = {
            'drift_detected': drift_detected,
            'feature_drift': {
                'overall_drift': overall_feature_drift,
                'drift_ratio': drift_ratio,
                'drift_count': drift_count,
                'total_features': n_features,
                'feature_results': feature_results
            },
            'label_drift': label_drift_results,
            'n_samples': len(X_new),
            'n_features': n_features,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        # Store in history
        self.drift_history.append(results)

        return results

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

    def update_reference_data(self, X_new: np.ndarray, y_new: Optional[np.ndarray] = None) -> None:
        """Update reference data with new data."""

        if self.reference_mean is None:
            self.set_reference_data(X_new, y_new)
            return

        # Update reference statistics using exponential moving average
        alpha = 0.1  # Learning rate
        new_mean = np.mean(X_new, axis=0)
        new_std = np.std(X_new, axis=0)

        self.reference_mean = (1 - alpha) * \
            self.reference_mean + alpha * new_mean
        self.reference_std = (1 - alpha) * self.reference_std + alpha * new_std

        logger.info("Reference data updated with exponential moving average")

    def reset_detector(self) -> None:
        """Reset the drift detector."""

        self.reference_mean = None
        self.reference_std = None
        self.drift_history = []
        self.cumulative_sum = 0
        self.min_cumulative_sum = 0
        self.max_cumulative_sum = 0
        self.sample_count = 0

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
    detector = PageHinkleyDriftDetector({
        'delta': 0.005,
        'threshold': 50,
        'min_samples': 30
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

    # Test streaming detection
    print("\nTesting streaming detection:")
    for i in range(5):
        # Gradually introduce drift
        drift_amount = i * 0.5
        X_stream = np.random.normal(drift_amount, 1, (50, n_features))
        y_stream = np.random.randint(0, 2, 50)

        results = detector.detect_drift_streaming(X_stream, y_stream)
        print(f"Stream {i+1}: drift_detected={results['drift_detected']}, "
              f"drift_ratio={results['feature_drift']['drift_ratio']:.4f}")

        # Update reference data
        detector.update_reference_data(X_stream, y_stream)

    # Get statistics
    stats = detector.get_drift_statistics()
    print(f"\nDrift Statistics:")
    print(f"Total detections: {stats['n_detections']}")
    print(f"Overall drift rate: {stats['overall_drift_rate']:.4f}")

    # Reset detector
    detector.reset_detector()
    print("\nDetector reset completed")
