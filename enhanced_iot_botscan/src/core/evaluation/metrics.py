"""
Metrics Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Comprehensive evaluation metrics for IoT botnet detection including classification,
adversarial robustness, and concept drift metrics.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.metrics import (
    matthews_corrcoef, cohen_kappa_score, log_loss,
    brier_score_loss, balanced_accuracy_score
)
import warnings

logger = logging.getLogger(__name__)


class ClassificationMetrics:
    """Comprehensive classification metrics for IoT botnet detection."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize metrics calculator with configuration."""

        self.config = config or {}
        self.metrics_history = []
        self.average_method = self.config.get('average_method', 'weighted')
        self.pos_label = self.config.get('pos_label', 1)

        logger.info("ClassificationMetrics initialized")

    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                              y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)

        Returns:
            Dictionary of all metrics
        """

        logger.info(f"Calculating metrics for {len(y_true)} samples")

        metrics = {}

        # Basic classification metrics
        metrics.update(self._calculate_basic_metrics(y_true, y_pred))

        # Confusion matrix metrics
        metrics.update(
            self._calculate_confusion_matrix_metrics(y_true, y_pred))

        # Probability-based metrics
        if y_proba is not None:
            metrics.update(
                self._calculate_probability_metrics(y_true, y_proba))

        # Advanced metrics
        metrics.update(self._calculate_advanced_metrics(y_true, y_pred))

        # Store in history
        self.metrics_history.append(metrics)

        logger.info("All metrics calculated successfully")
        return metrics

    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate basic classification metrics."""

        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=self.average_method, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=self.average_method, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=self.average_method, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred)
        }

    def _calculate_confusion_matrix_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate confusion matrix based metrics."""

        cm = confusion_matrix(y_true, y_pred)

        if cm.shape == (2, 2):  # Binary classification
            tn, fp, fn, tp = cm.ravel()

            return {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
                'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0
            }
        else:
            # Multi-class classification
            return {
                'confusion_matrix': cm.tolist(),
                'n_classes': cm.shape[0]
            }

    def _calculate_probability_metrics(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate probability-based metrics."""

        metrics = {}

        try:
            # ROC AUC
            if len(np.unique(y_true)) == 2:  # Binary classification
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
                metrics['average_precision'] = average_precision_score(
                    y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
            else:  # Multi-class
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_proba, multi_class='ovr', average=self.average_method)
                metrics['average_precision'] = average_precision_score(
                    y_true, y_proba, average=self.average_method)

            # Log loss
            metrics['log_loss'] = log_loss(y_true, y_proba)

            # Brier score
            if len(np.unique(y_true)) == 2:
                metrics['brier_score'] = brier_score_loss(
                    y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)

        except Exception as e:
            logger.warning(f"Error calculating probability metrics: {e}")
            metrics['probability_metrics_error'] = str(e)

        return metrics

    def _calculate_advanced_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate advanced classification metrics."""

        return {
            'matthews_correlation_coefficient': matthews_corrcoef(y_true, y_pred),
            'cohen_kappa_score': cohen_kappa_score(y_true, y_pred),
            'hamming_loss': 1 - accuracy_score(y_true, y_pred),
            'jaccard_score': self._calculate_jaccard_score(y_true, y_pred)
        }

    def _calculate_jaccard_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Jaccard score."""

        if len(np.unique(y_true)) == 2:  # Binary classification
            intersection = np.sum(y_true & y_pred)
            union = np.sum(y_true | y_pred)
            return intersection / union if union > 0 else 0
        else:
            # Multi-class Jaccard score
            intersection = np.sum(y_true == y_pred)
            union = len(y_true)
            return intersection / union

    def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  target_names: Optional[List[str]] = None) -> str:
        """Get detailed classification report."""

        return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)

    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Get confusion matrix."""

        return confusion_matrix(y_true, y_pred)

    def calculate_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate ROC curve."""

        if len(np.unique(y_true)) == 2:
            fpr, tpr, thresholds = roc_curve(
                y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
        else:
            # Multi-class ROC curve
            fpr, tpr, thresholds = roc_curve(
                y_true, y_proba, multi_class='ovr')

        return fpr, tpr, thresholds

    def calculate_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate precision-recall curve."""

        precision, recall, thresholds = precision_recall_curve(
            y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba
        )
        return precision, recall, thresholds

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all calculated metrics."""

        if not self.metrics_history:
            return {'message': 'No metrics calculated yet'}

        # Calculate average metrics across all evaluations
        avg_metrics = {}
        for key in self.metrics_history[0].keys():
            if isinstance(self.metrics_history[0][key], (int, float)):
                values = [metrics[key]
                          for metrics in self.metrics_history if key in metrics]
                avg_metrics[f'avg_{key}'] = np.mean(values)
                avg_metrics[f'std_{key}'] = np.std(values)

        return {
            'n_evaluations': len(self.metrics_history),
            'average_metrics': avg_metrics,
            'latest_metrics': self.metrics_history[-1]
        }


class AdversarialRobustnessMetrics:
    """Metrics for evaluating adversarial robustness."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize adversarial robustness metrics."""

        self.config = config or {}
        self.robustness_history = []

        logger.info("AdversarialRobustnessMetrics initialized")

    def calculate_robustness_metrics(self, original_accuracy: float, adversarial_accuracy: float,
                                     attack_success_rate: float, perturbation_norm: float) -> Dict[str, Any]:
        """
        Calculate adversarial robustness metrics.

        Args:
            original_accuracy: Accuracy on clean data
            adversarial_accuracy: Accuracy on adversarial data
            attack_success_rate: Rate of successful attacks
            perturbation_norm: Average perturbation magnitude

        Returns:
            Robustness metrics dictionary
        """

        accuracy_drop = original_accuracy - adversarial_accuracy
        robustness_score = 1 - attack_success_rate
        relative_accuracy_drop = accuracy_drop / \
            original_accuracy if original_accuracy > 0 else 0

        metrics = {
            'original_accuracy': original_accuracy,
            'adversarial_accuracy': adversarial_accuracy,
            'accuracy_drop': accuracy_drop,
            'relative_accuracy_drop': relative_accuracy_drop,
            'attack_success_rate': attack_success_rate,
            'robustness_score': robustness_score,
            'perturbation_norm': perturbation_norm,
            'robustness_ratio': adversarial_accuracy / original_accuracy if original_accuracy > 0 else 0
        }

        self.robustness_history.append(metrics)
        logger.info(
            f"Robustness metrics calculated: robustness_score={robustness_score:.4f}")

        return metrics

    def compare_robustness(self, metrics_list: List[Dict[str, Any]],
                           model_names: List[str]) -> Dict[str, Any]:
        """Compare robustness across multiple models."""

        comparison = {}

        for i, (metrics, name) in enumerate(zip(metrics_list, model_names)):
            comparison[name] = {
                'robustness_score': metrics.get('robustness_score', 0),
                'accuracy_drop': metrics.get('accuracy_drop', 0),
                'attack_success_rate': metrics.get('attack_success_rate', 0)
            }

        # Find best model
        best_model = max(comparison.keys(),
                         key=lambda x: comparison[x]['robustness_score'])

        comparison['best_model'] = best_model
        comparison['best_robustness_score'] = comparison[best_model]['robustness_score']

        return comparison


class ConceptDriftMetrics:
    """Metrics for evaluating concept drift detection."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize concept drift metrics."""

        self.config = config or {}
        self.drift_history = []

        logger.info("ConceptDriftMetrics initialized")

    def calculate_drift_metrics(self, drift_detected: bool, drift_confidence: float,
                                performance_drop: float, detection_time: float) -> Dict[str, Any]:
        """
        Calculate concept drift detection metrics.

        Args:
            drift_detected: Whether drift was detected
            drift_confidence: Confidence in drift detection
            performance_drop: Performance drop due to drift
            detection_time: Time taken to detect drift

        Returns:
            Drift metrics dictionary
        """

        metrics = {
            'drift_detected': drift_detected,
            'drift_confidence': drift_confidence,
            'performance_drop': performance_drop,
            'detection_time': detection_time,
            'detection_efficiency': 1 / (detection_time + 1e-8),
            'drift_severity': 'high' if performance_drop > 0.2 else 'medium' if performance_drop > 0.1 else 'low'
        }

        self.drift_history.append(metrics)
        logger.info(
            f"Drift metrics calculated: drift_detected={drift_detected}")

        return metrics

    def get_drift_statistics(self) -> Dict[str, Any]:
        """Get comprehensive drift detection statistics."""

        if not self.drift_history:
            return {'message': 'No drift detections recorded'}

        drift_detections = [metrics['drift_detected']
                            for metrics in self.drift_history]
        drift_rate = np.mean(drift_detections)

        performance_drops = [metrics['performance_drop']
                             for metrics in self.drift_history]
        avg_performance_drop = np.mean(performance_drops)

        detection_times = [metrics['detection_time']
                           for metrics in self.drift_history]
        avg_detection_time = np.mean(detection_times)

        return {
            'total_detections': len(self.drift_history),
            'drift_detection_rate': drift_rate,
            'average_performance_drop': avg_performance_drop,
            'average_detection_time': avg_detection_time,
            'drift_frequency': drift_rate * 100  # Percentage
        }


class ComprehensiveEvaluator:
    """Comprehensive evaluation combining all metric types."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize comprehensive evaluator."""

        self.config = config or {}
        self.classification_metrics = ClassificationMetrics(config)
        self.robustness_metrics = AdversarialRobustnessMetrics(config)
        self.drift_metrics = ConceptDriftMetrics(config)

        logger.info("ComprehensiveEvaluator initialized")

    def evaluate_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Evaluate comprehensive model performance."""

        # Classification metrics
        classification_results = self.classification_metrics.calculate_all_metrics(
            y_true, y_pred, y_proba
        )

        return {
            'classification_metrics': classification_results,
            'evaluation_timestamp': pd.Timestamp.now().isoformat()
        }

    def evaluate_adversarial_robustness(self, original_accuracy: float, adversarial_accuracy: float,
                                        attack_success_rate: float, perturbation_norm: float) -> Dict[str, Any]:
        """Evaluate adversarial robustness."""

        robustness_results = self.robustness_metrics.calculate_robustness_metrics(
            original_accuracy, adversarial_accuracy, attack_success_rate, perturbation_norm
        )

        return {
            'robustness_metrics': robustness_results,
            'evaluation_timestamp': pd.Timestamp.now().isoformat()
        }

    def evaluate_concept_drift(self, drift_detected: bool, drift_confidence: float,
                               performance_drop: float, detection_time: float) -> Dict[str, Any]:
        """Evaluate concept drift detection."""

        drift_results = self.drift_metrics.calculate_drift_metrics(
            drift_detected, drift_confidence, performance_drop, detection_time
        )

        return {
            'drift_metrics': drift_results,
            'evaluation_timestamp': pd.Timestamp.now().isoformat()
        }

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive evaluation report."""

        return {
            'classification_summary': self.classification_metrics.get_metrics_summary(),
            'robustness_summary': self.robustness_metrics.robustness_history[-1] if self.robustness_metrics.robustness_history else {},
            'drift_summary': self.drift_metrics.get_drift_statistics(),
            'report_timestamp': pd.Timestamp.now().isoformat()
        }


# Example usage and testing
if __name__ == '__main__':
    # Create sample data
    np.random.seed(42)
    n_samples = 1000

    # Generate binary classification data
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.random.randint(0, 2, n_samples)
    y_proba = np.random.rand(n_samples, 2)
    # Normalize probabilities
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

    print("Testing Classification Metrics:")

    # Test classification metrics
    metrics = ClassificationMetrics()
    results = metrics.calculate_all_metrics(y_true, y_pred, y_proba)

    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"ROC AUC: {results['roc_auc']:.4f}")

    # Test adversarial robustness metrics
    print("\nTesting Adversarial Robustness Metrics:")
    robustness_metrics = AdversarialRobustnessMetrics()
    robustness_results = robustness_metrics.calculate_robustness_metrics(
        original_accuracy=0.85,
        adversarial_accuracy=0.65,
        attack_success_rate=0.3,
        perturbation_norm=0.1
    )

    print(f"Robustness Score: {robustness_results['robustness_score']:.4f}")
    print(f"Accuracy Drop: {robustness_results['accuracy_drop']:.4f}")

    # Test concept drift metrics
    print("\nTesting Concept Drift Metrics:")
    drift_metrics = ConceptDriftMetrics()
    drift_results = drift_metrics.calculate_drift_metrics(
        drift_detected=True,
        drift_confidence=0.85,
        performance_drop=0.15,
        detection_time=2.5
    )

    print(f"Drift Detected: {drift_results['drift_detected']}")
    print(f"Drift Confidence: {drift_results['drift_confidence']:.4f}")
    print(f"Performance Drop: {drift_results['performance_drop']:.4f}")

    # Test comprehensive evaluator
    print("\nTesting Comprehensive Evaluator:")
    evaluator = ComprehensiveEvaluator()
    comprehensive_results = evaluator.evaluate_model_performance(
        y_true, y_pred, y_proba)

    print("Comprehensive evaluation completed successfully")
    print(
        f"Classification metrics keys: {list(comprehensive_results['classification_metrics'].keys())}")
