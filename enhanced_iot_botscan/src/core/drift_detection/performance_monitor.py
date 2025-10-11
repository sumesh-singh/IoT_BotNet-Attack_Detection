"""
Performance Monitor Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Monitors model performance and detects performance degradation.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.base import BaseEstimator
import warnings

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Performance monitoring system for detecting model degradation."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize performance monitor with configuration."""

        self.config = config or {}
        self.monitoring_history = []

        # Monitoring configuration
        self.performance_threshold = self.config.get(
            'performance_threshold', 0.8)
        self.degradation_threshold = self.config.get(
            'degradation_threshold', 0.1)
        self.window_size = self.config.get('window_size', 100)
        self.min_samples = self.config.get('min_samples', 50)
        self.alert_threshold = self.config.get('alert_threshold', 0.05)

        # State variables
        self.baseline_performance = None
        self.performance_window = []
        self.degradation_detected = False
        self.alert_count = 0

        logger.info(
            f"PerformanceMonitor initialized with threshold={self.performance_threshold}")

    def set_baseline(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> None:
        """
        Set baseline performance for monitoring.

        Args:
            model: Model to monitor
            X: Baseline data
            y: Baseline labels
        """

        if len(X) < self.min_samples:
            raise ValueError(
                f"Baseline data must have at least {self.min_samples} samples")

        # Calculate baseline performance
        baseline_accuracy = model.score(X, y)
        self.baseline_performance = baseline_accuracy

        # Initialize performance window
        self.performance_window = [baseline_accuracy]

        logger.info(f"Baseline performance set: {baseline_accuracy:.4f}")

    def monitor_performance(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Monitor model performance on new data.

        Args:
            model: Model to monitor
            X: New data
            y: New labels
            context: Additional context information

        Returns:
            Performance monitoring results
        """

        if self.baseline_performance is None:
            raise ValueError(
                "Baseline performance must be set before monitoring")

        if len(X) < self.min_samples:
            logger.warning(
                f"New data has only {len(X)} samples, minimum required: {self.min_samples}")
            return {
                'performance': 0,
                'degradation_detected': False,
                'reason': 'insufficient_samples',
                'n_samples': len(X)
            }

        logger.info(f"Monitoring performance on {len(X)} new samples")

        # Calculate current performance
        current_performance = model.score(X, y)

        # Update performance window
        self.performance_window.append(current_performance)
        if len(self.performance_window) > self.window_size:
            self.performance_window.pop(0)

        # Detect performance degradation
        degradation_results = self._detect_degradation(current_performance)

        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            current_performance)

        # Create monitoring results
        results = {
            'performance': current_performance,
            'baseline_performance': self.baseline_performance,
            'performance_drop': self.baseline_performance - current_performance,
            'degradation_detected': degradation_results['degradation_detected'],
            'degradation_severity': degradation_results['severity'],
            'performance_metrics': performance_metrics,
            'performance_window': self.performance_window.copy(),
            'n_samples': len(X),
            'context': context,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        # Store in history
        self.monitoring_history.append(results)

        # Update state
        self.degradation_detected = degradation_results['degradation_detected']
        if degradation_results['degradation_detected']:
            self.alert_count += 1

        logger.info(f"Performance monitoring completed: performance={current_performance:.4f}, "
                    f"degradation_detected={degradation_results['degradation_detected']}")

        return results

    def _detect_degradation(self, current_performance: float) -> Dict[str, Any]:
        """Detect performance degradation."""

        # Calculate performance drop from baseline
        performance_drop = self.baseline_performance - current_performance

        # Determine degradation severity
        if performance_drop >= self.degradation_threshold:
            severity = 'severe'
            degradation_detected = True
        elif performance_drop >= self.alert_threshold:
            severity = 'moderate'
            degradation_detected = True
        else:
            severity = 'none'
            degradation_detected = False

        # Check for trend-based degradation
        if len(self.performance_window) >= 5:
            # Calculate performance trend
            trend = np.polyfit(
                range(len(self.performance_window)), self.performance_window, 1)[0]

            # If trend is significantly negative, consider it degradation
            if trend < -0.01:  # Performance decreasing by more than 1% per sample
                if not degradation_detected:
                    severity = 'trend'
                    degradation_detected = True

        return {
            'degradation_detected': degradation_detected,
            'severity': severity,
            'performance_drop': performance_drop,
            'trend': trend if len(self.performance_window) >= 5 else 0
        }

    def _calculate_performance_metrics(self, current_performance: float) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""

        # Basic metrics
        metrics = {
            'current_performance': current_performance,
            'baseline_performance': self.baseline_performance,
            'performance_drop': self.baseline_performance - current_performance,
            'performance_ratio': current_performance / self.baseline_performance
        }

        # Window-based metrics
        if len(self.performance_window) > 1:
            metrics.update({
                'window_mean': np.mean(self.performance_window),
                'window_std': np.std(self.performance_window),
                'window_min': np.min(self.performance_window),
                'window_max': np.max(self.performance_window),
                'window_trend': np.polyfit(range(len(self.performance_window)), self.performance_window, 1)[0]
            })

        # Stability metrics
        if len(self.performance_window) > 2:
            # Calculate coefficient of variation
            cv = np.std(self.performance_window) / \
                np.mean(self.performance_window)
            metrics['coefficient_of_variation'] = cv

            # Calculate performance stability
            stability = 1 - cv  # Higher stability = lower variation
            metrics['stability'] = max(0, stability)

        return metrics

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""

        if not self.monitoring_history:
            return {'n_monitoring_sessions': 0}

        # Calculate summary statistics
        performances = [result['performance']
                        for result in self.monitoring_history]
        degradations = [result['degradation_detected']
                        for result in self.monitoring_history]

        # Calculate performance trends
        if len(performances) > 1:
            performance_trend = np.polyfit(
                range(len(performances)), performances, 1)[0]
        else:
            performance_trend = 0

        # Calculate degradation statistics
        degradation_rate = np.mean(degradations)

        # Calculate alert statistics
        alert_rate = self.alert_count / len(self.monitoring_history)

        summary = {
            'n_monitoring_sessions': len(self.monitoring_history),
            'baseline_performance': self.baseline_performance,
            'current_performance': performances[-1] if performances else 0,
            'mean_performance': np.mean(performances),
            'std_performance': np.std(performances),
            'min_performance': np.min(performances),
            'max_performance': np.max(performances),
            'performance_trend': performance_trend,
            'degradation_rate': degradation_rate,
            'alert_rate': alert_rate,
            'alert_count': self.alert_count,
            'last_monitoring': self.monitoring_history[-1]['timestamp'] if self.monitoring_history else None
        }

        return summary

    def get_performance_alerts(self) -> List[Dict[str, Any]]:
        """Get performance alerts from monitoring history."""

        alerts = []

        for result in self.monitoring_history:
            if result['degradation_detected']:
                alert = {
                    'timestamp': result['timestamp'],
                    'performance': result['performance'],
                    'performance_drop': result['performance_drop'],
                    'severity': result['degradation_severity'],
                    'n_samples': result['n_samples'],
                    'context': result.get('context', {})
                }
                alerts.append(alert)

        return alerts

    def reset_monitor(self) -> None:
        """Reset the performance monitor."""

        self.baseline_performance = None
        self.performance_window = []
        self.degradation_detected = False
        self.alert_count = 0
        self.monitoring_history = []

        logger.info("Performance monitor reset")

    def get_monitoring_report(self) -> Dict[str, Any]:
        """Get comprehensive monitoring report."""

        return {
            'baseline_performance': self.baseline_performance,
            'degradation_threshold': self.degradation_threshold,
            'alert_threshold': self.alert_threshold,
            'window_size': self.window_size,
            'monitoring_history': self.monitoring_history,
            'summary': self.get_performance_summary(),
            'alerts': self.get_performance_alerts()
        }


# Example usage and testing
if __name__ == '__main__':
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # Baseline data
    X_baseline = np.random.normal(0, 1, (500, n_features))
    y_baseline = np.random.randint(0, 2, 500)

    # Test data with varying performance
    X_test_good = np.random.normal(0, 1, (100, n_features))
    y_test_good = np.random.randint(0, 2, 100)

    X_test_degraded = np.random.normal(2, 1, (100, n_features))
    y_test_degraded = np.random.randint(0, 2, 100)

    # Create base model
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=42)
    model.fit(X_baseline, y_baseline)

    # Initialize performance monitor
    monitor = PerformanceMonitor({
        'performance_threshold': 0.8,
        'degradation_threshold': 0.1,
        'alert_threshold': 0.05,
        'window_size': 10
    })

    # Set baseline
    monitor.set_baseline(model, X_baseline, y_baseline)

    print("Baseline performance:", monitor.baseline_performance)

    # Monitor performance on good data
    print("\nMonitoring performance on good data:")
    results_good = monitor.monitor_performance(model, X_test_good, y_test_good)
    print(f"Performance: {results_good['performance']:.4f}")
    print(f"Degradation detected: {results_good['degradation_detected']}")
    print(f"Severity: {results_good['degradation_severity']}")

    # Monitor performance on degraded data
    print("\nMonitoring performance on degraded data:")
    results_degraded = monitor.monitor_performance(
        model, X_test_degraded, y_test_degraded)
    print(f"Performance: {results_degraded['performance']:.4f}")
    print(f"Degradation detected: {results_degraded['degradation_detected']}")
    print(f"Severity: {results_degraded['degradation_severity']}")

    # Get performance summary
    summary = monitor.get_performance_summary()
    print(f"\nPerformance Summary:")
    print(f"Monitoring sessions: {summary['n_monitoring_sessions']}")
    print(f"Mean performance: {summary['mean_performance']:.4f}")
    print(f"Degradation rate: {summary['degradation_rate']:.4f}")
    print(f"Alert count: {summary['alert_count']}")

    # Get performance alerts
    alerts = monitor.get_performance_alerts()
    print(f"\nPerformance Alerts: {len(alerts)}")
    for alert in alerts:
        print(f"  {alert['timestamp']}: {alert['severity']} degradation "
              f"(performance={alert['performance']:.4f})")

    # Get monitoring report
    report = monitor.get_monitoring_report()
    print(f"\nMonitoring Report:")
    print(f"Baseline performance: {report['baseline_performance']:.4f}")
    print(f"Degradation threshold: {report['degradation_threshold']}")
    print(f"Alert threshold: {report['alert_threshold']}")

    # Reset monitor
    monitor.reset_monitor()
    print("\nMonitor reset completed")
