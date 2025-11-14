"""
Performance Evaluator for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Comprehensive evaluation of model performance including adversarial robustness.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, auc
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import logging
from datetime import datetime
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except Exception:
    sns = None

class PerformanceEvaluator:
    """Comprehensive performance evaluation for IoT botnet detection models."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.evaluation_history = []

    def comprehensive_evaluation(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Perform comprehensive model evaluation."""

        # Make predictions
        y_pred = model.predict(X_test)

        # Get prediction probabilities if available
        try:
            y_proba = model.predict_proba(X_test)
        except:
            y_proba = None

        # Basic classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Per-class metrics
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # ROC-AUC if probabilities available
        roc_auc = None
        if y_proba is not None:
            try:
                if len(np.unique(y_test)) == 2:  # Binary classification
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                else:  # Multi-class
                    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            except:
                pass

        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(X_test),
            'n_features': len(X_test.columns),
            'n_classes': len(np.unique(y_test)),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': y_pred.tolist(),
            'true_labels': y_test.tolist()
        }

        if y_proba is not None:
            evaluation_results['prediction_probabilities'] = y_proba.tolist()

        self.evaluation_history.append(evaluation_results)
        return evaluation_results

    def evaluate_adversarial_robustness(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate model robustness against adversarial attacks."""

        from ..core.adversarial.attack_generator import AdversarialAttackGenerator

        # Initialize attack generator
        attack_config = self.config.get('adversarial_attacks', {
            'fgsm': {'enabled': True, 'epsilon': 0.1},
            'pgd': {'enabled': True, 'epsilon': 0.1, 'alpha': 0.01, 'num_iter': 10},
            'cw': {'enabled': True, 'c': 1.0}
        })

        attack_generator = AdversarialAttackGenerator(attack_config)

        # Evaluate robustness
        robustness_results = attack_generator.evaluate_robustness(
            X_test.values, y_test.values, model
        )

        return robustness_results

    def cross_dataset_evaluation(self, model, datasets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate model performance across different datasets."""

        cross_results = {}

        for dataset_name, dataset in datasets.items():
            X_test = pd.DataFrame(dataset['features'])
            y_test = pd.Series(dataset['labels'])

            results = self.comprehensive_evaluation(model, X_test, y_test)
            cross_results[dataset_name] = results

        return cross_results

    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, save_path=None):
        """Plot confusion matrix."""

        try:
            cm = confusion_matrix(y_true, y_pred)

            plt.figure(figsize=(8, 6))
            if sns is not None:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=class_names, yticklabels=class_names)
            else:
                plt.imshow(cm, cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()

        except Exception as e:
            print(f"Error plotting confusion matrix: {e}")

    def plot_roc_curve(self, y_true, y_proba, save_path=None):
        """Plot ROC curve for binary classification."""

        try:
            if len(np.unique(y_true)) != 2:
                print("ROC curve only available for binary classification")
                return

            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()

        except Exception as e:
            print(f"Error plotting ROC curve: {e}")

    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive evaluation report."""

        report = f"""
ENHANCED IOT BOTSCAN - MODEL EVALUATION REPORT
=============================================

Evaluation Timestamp: {results['timestamp']}
Dataset Size: {results['n_samples']} samples, {results['n_features']} features
Number of Classes: {results['n_classes']}

OVERALL PERFORMANCE METRICS:
----------------------------
Accuracy:  {results['accuracy']:.4f}
Precision: {results['precision']:.4f}
Recall:    {results['recall']:.4f}
F1-Score:  {results['f1_score']:.4f}"""

        if results.get('roc_auc'):
            report += f"\nROC-AUC:   {results['roc_auc']:.4f}"

        report += "\n\nPER-CLASS PERFORMANCE:"
        report += "\n" + "-" * 22

        class_report = results['classification_report']
        for class_name, metrics in class_report.items():
            if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                report += f"\nClass {class_name}:"
                report += f"\n  Precision: {metrics['precision']:.4f}"
                report += f"\n  Recall:    {metrics['recall']:.4f}"
                report += f"\n  F1-Score:  {metrics['f1-score']:.4f}"
                report += f"\n  Support:   {metrics['support']}"

        return report

def create_evaluation_summary(evaluation_results: Dict[str, Any]) -> pd.DataFrame:
    """Create summary DataFrame from evaluation results."""

    summary_data = []

    for eval_name, results in evaluation_results.items():
        summary_row = {
            'Evaluation': eval_name,
            'Accuracy': results.get('accuracy', 0),
            'Precision': results.get('precision', 0),
            'Recall': results.get('recall', 0),
            'F1-Score': results.get('f1_score', 0),
            'ROC-AUC': results.get('roc_auc', 'N/A'),
            'Samples': results.get('n_samples', 0)
        }
        summary_data.append(summary_row)

    return pd.DataFrame(summary_data)
