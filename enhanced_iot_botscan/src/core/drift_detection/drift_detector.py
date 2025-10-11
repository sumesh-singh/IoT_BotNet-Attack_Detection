"""
Drift Detector Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Main orchestrator for concept drift detection combining multiple methods.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
import warnings

# Import drift detection methods
from .kolmogorov_smirnov import KolmogorovSmirnovDriftDetector
from .page_hinkley import PageHinkleyDriftDetector

logger = logging.getLogger(__name__)


class DriftDetector:
    """Main orchestrator for concept drift detection."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize drift detector with configuration."""

        self.config = config or {}
        self.detection_history = []

        # Detection configuration
        self.enabled_methods = self.config.get('enabled_methods', ['ks', 'ph'])
        self.consensus_threshold = self.config.get('consensus_threshold', 0.5)
        self.adaptive_threshold = self.config.get('adaptive_threshold', True)

        # Initialize detection methods
        self.detectors = {}

        if 'ks' in self.enabled_methods:
            self.detectors['ks'] = KolmogorovSmirnovDriftDetector(
                self.config.get('ks_config', {})
            )

        if 'ph' in self.enabled_methods:
            self.detectors['ph'] = PageHinkleyDriftDetector(
                self.config.get('ph_config', {})
            )

        logger.info(
            f"DriftDetector initialized with methods: {self.enabled_methods}")

    def set_reference_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Set reference data for all drift detection methods.

        Args:
            X: Reference features
            y: Reference labels (optional)
        """

        for method_name, detector in self.detectors.items():
            try:
                detector.set_reference_data(X, y)
                logger.info(f"Reference data set for {method_name} detector")
            except Exception as e:
                logger.error(
                    f"Failed to set reference data for {method_name}: {e}")

    def detect_drift(self, X_new: np.ndarray, y_new: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Detect concept drift using all enabled methods.

        Args:
            X_new: New data to test for drift
            y_new: New labels (optional)

        Returns:
            Comprehensive drift detection results
        """

        logger.info(
            f"Detecting drift in {len(X_new)} new samples using {len(self.detectors)} methods")

        # Run all detection methods
        method_results = {}
        drift_detections = []

        for method_name, detector in self.detectors.items():
            try:
                results = detector.detect_drift(X_new, y_new)
                method_results[method_name] = results
                drift_detections.append(results['drift_detected'])

                logger.info(
                    f"{method_name} detector: drift_detected={results['drift_detected']}")

            except Exception as e:
                logger.error(f"Failed to run {method_name} detector: {e}")
                method_results[method_name] = {
                    'error': str(e), 'drift_detected': False}
                drift_detections.append(False)

        # Determine consensus
        consensus_drift = self._determine_consensus(drift_detections)

        # Create comprehensive results
        comprehensive_results = {
            'drift_detected': consensus_drift,
            'consensus_threshold': self.consensus_threshold,
            'method_results': method_results,
            'individual_detections': drift_detections,
            'n_methods': len(self.detectors),
            'n_samples': len(X_new),
            'timestamp': pd.Timestamp.now().isoformat()
        }

        # Store in history
        self.detection_history.append(comprehensive_results)

        logger.info(
            f"Drift detection completed: consensus_drift={consensus_drift}")

        return comprehensive_results

    def _determine_consensus(self, drift_detections: List[bool]) -> bool:
        """Determine consensus drift detection from multiple methods."""

        if not drift_detections:
            return False

        # Calculate consensus ratio
        consensus_ratio = np.mean(drift_detections)

        # Determine consensus based on threshold
        consensus_drift = consensus_ratio >= self.consensus_threshold

        return consensus_drift

    def get_drift_statistics(self) -> Dict[str, Any]:
        """Get comprehensive drift detection statistics."""

        if not self.detection_history:
            return {'n_detections': 0}

        # Calculate overall statistics
        drift_detections = [result['drift_detected']
                            for result in self.detection_history]
        overall_drift_rate = np.mean(drift_detections)

        # Calculate method-specific statistics
        method_stats = {}
        for method_name in self.detectors.keys():
            method_detections = []
            for result in self.detection_history:
                if method_name in result['method_results']:
                    method_detections.append(
                        result['method_results'][method_name]['drift_detected'])

            if method_detections:
                method_stats[method_name] = {
                    'drift_rate': np.mean(method_detections),
                    'n_detections': len(method_detections)
                }

        # Calculate consensus statistics
        consensus_ratios = []
        for result in self.detection_history:
            if 'individual_detections' in result:
                consensus_ratios.append(
                    np.mean(result['individual_detections']))

        statistics = {
            'n_detections': len(self.detection_history),
            'overall_drift_rate': overall_drift_rate,
            'method_statistics': method_stats,
            'mean_consensus_ratio': np.mean(consensus_ratios) if consensus_ratios else 0,
            'last_detection': self.detection_history[-1]['timestamp'] if self.detection_history else None
        }

        return statistics

    def update_reference_data(self, X_new: np.ndarray, y_new: Optional[np.ndarray] = None) -> None:
        """Update reference data for all detection methods."""

        for method_name, detector in self.detectors.items():
            try:
                detector.update_reference_data(X_new, y_new)
                logger.info(
                    f"Reference data updated for {method_name} detector")
            except Exception as e:
                logger.error(
                    f"Failed to update reference data for {method_name}: {e}")

    def reset_detector(self) -> None:
        """Reset all drift detectors."""

        for method_name, detector in self.detectors.items():
            try:
                detector.reset_detector()
                logger.info(f"{method_name} detector reset")
            except Exception as e:
                logger.error(f"Failed to reset {method_name} detector: {e}")

        self.detection_history = []
        logger.info("All drift detectors reset")

    def get_detection_report(self) -> Dict[str, Any]:
        """Get comprehensive detection report."""

        return {
            'enabled_methods': self.enabled_methods,
            'consensus_threshold': self.consensus_threshold,
            'adaptive_threshold': self.adaptive_threshold,
            'detection_history': self.detection_history,
            'statistics': self.get_drift_statistics()
        }


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
    detector = DriftDetector({
        'enabled_methods': ['ks', 'ph'],
        'consensus_threshold': 0.5,
        'ks_config': {'alpha': 0.05, 'feature_threshold': 0.3},
        'ph_config': {'delta': 0.005, 'threshold': 50}
    })

    # Set reference data
    detector.set_reference_data(X_ref, y_ref)

    # Test with same distribution
    print("Testing with same distribution:")
    results_same = detector.detect_drift(X_new_same, y_new_same)
    print(f"Consensus drift detected: {results_same['drift_detected']}")
    print(f"Individual detections: {results_same['individual_detections']}")

    # Test with different distribution
    print("\nTesting with different distribution:")
    results_drift = detector.detect_drift(X_new_drift, y_new_drift)
    print(f"Consensus drift detected: {results_drift['drift_detected']}")
    print(f"Individual detections: {results_drift['individual_detections']}")

    # Get statistics
    stats = detector.get_drift_statistics()
    print(f"\nDrift Statistics:")
    print(f"Total detections: {stats['n_detections']}")
    print(f"Overall drift rate: {stats['overall_drift_rate']:.4f}")
    print(f"Mean consensus ratio: {stats['mean_consensus_ratio']:.4f}")

    # Get detection report
    report = detector.get_detection_report()
    print(f"\nDetection Report:")
    print(f"Enabled methods: {report['enabled_methods']}")
    print(f"Consensus threshold: {report['consensus_threshold']}")

    # Reset detector
    detector.reset_detector()
    print("\nDetector reset completed")
