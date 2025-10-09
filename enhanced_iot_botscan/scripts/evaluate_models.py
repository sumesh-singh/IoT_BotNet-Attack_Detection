#!/usr/bin/env python3
"""
Model Evaluation Script for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Evaluates trained models on multiple datasets and generates comprehensive reports.
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import joblib
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.ensemble.hybrid_ensemble import HybridEnsemble
from core.adversarial.attack_generator import AdversarialAttackGenerator
from core.drift_detection.drift_detector import DriftDetector
from data.data_loader import DataLoader
from evaluation.performance_evaluator import PerformanceEvaluator
from utils.config_manager import ConfigManager
from utils.logger import setup_logging

class ModelEvaluator:
    """Comprehensive model evaluation system."""

    def __init__(self, config_path: str = None):
        # Setup configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config

        # Setup logging
        setup_logging(self.config.get('logging', {}))
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.data_loader = DataLoader(self.config.get('data', {}))
        self.evaluator = PerformanceEvaluator(self.config.get('evaluation', {}))

        self.logger.info("ModelEvaluator initialized successfully")

    def load_model(self, model_path: str) -> HybridEnsemble:
        """Load trained model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = HybridEnsemble(self.config_manager.config_path)
        model.load_model(model_path)

        self.logger.info(f"Model loaded from {model_path}")
        return model

    def evaluate_single_dataset(self, model, dataset_name: str) -> Dict[str, Any]:
        """Evaluate model on single dataset."""

        # Load dataset
        dataset = self.data_loader.load_dataset(dataset_name)
        X = pd.DataFrame(dataset['features'])
        y = pd.Series(dataset['labels'])

        # Basic evaluation
        results = self.evaluator.comprehensive_evaluation(model, X, y)

        # Adversarial robustness evaluation
        try:
            robustness_results = self.evaluator.evaluate_adversarial_robustness(model, X, y)
            results['robustness'] = robustness_results
        except Exception as e:
            self.logger.warning(f"Adversarial evaluation failed: {e}")
            results['robustness'] = None

        results['dataset_name'] = dataset_name
        return results

    def cross_dataset_evaluation(self, model, dataset_names: List[str]) -> Dict[str, Any]:
        """Perform cross-dataset evaluation."""

        datasets = {}
        for name in dataset_names:
            try:
                datasets[name] = self.data_loader.load_dataset(name)
            except Exception as e:
                self.logger.error(f"Failed to load {name}: {e}")

        return self.evaluator.cross_dataset_evaluation(model, datasets)

    def generate_evaluation_report(self, results: Dict[str, Any], output_path: str) -> None:
        """Generate comprehensive evaluation report."""

        report = f"""
ENHANCED IOT BOTSCAN - MODEL EVALUATION REPORT
=============================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY:
"""

        for dataset_name, result in results.items():
            if isinstance(result, dict) and 'accuracy' in result:
                report += f"""
{dataset_name.upper()}:
- Accuracy: {result['accuracy']:.4f}
- Precision: {result.get('precision', 0):.4f}
- Recall: {result.get('recall', 0):.4f}
- F1-Score: {result.get('f1_score', 0):.4f}
- ROC-AUC: {result.get('roc_auc', 'N/A')}
- Samples: {result.get('n_samples', 'N/A')}
"""

        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)

        self.logger.info(f"Evaluation report saved to {output_path}")

    def run_comprehensive_evaluation(self, model_path: str, datasets: List[str]) -> Dict[str, Any]:
        """Run comprehensive evaluation on all datasets."""

        # Load model
        model = self.load_model(model_path)

        # Evaluate on each dataset
        results = {}
        for dataset_name in datasets:
            try:
                results[dataset_name] = self.evaluate_single_dataset(model, dataset_name)
                self.logger.info(f"Completed evaluation on {dataset_name}")
            except Exception as e:
                self.logger.error(f"Evaluation failed for {dataset_name}: {e}")

        # Cross-dataset evaluation
        if len(datasets) > 1:
            try:
                results['cross_dataset'] = self.cross_dataset_evaluation(model, datasets)
                self.logger.info("Completed cross-dataset evaluation")
            except Exception as e:
                self.logger.error(f"Cross-dataset evaluation failed: {e}")

        return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Enhanced IoT BotScan models")
    parser.add_argument('--config', default='./config/config.yaml', help='Configuration file path')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--datasets', nargs='+', default=['n_baiot'], choices=['n_baiot', 'iot_23', 'bot_iot'])
    parser.add_argument('--output-dir', default='./data/results', help='Output directory for results')

    args = parser.parse_args()

    try:
        evaluator = ModelEvaluator(args.config)
        results = evaluator.run_comprehensive_evaluation(args.model, args.datasets)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save detailed results as JSON
        import json
        results_file = os.path.join(args.output_dir, f'evaluation_results_{timestamp}.json')
        os.makedirs(args.output_dir, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Generate report
        report_file = os.path.join(args.output_dir, f'evaluation_report_{timestamp}.txt')
        evaluator.generate_evaluation_report(results, report_file)

        print("\n🎉 Evaluation completed successfully!")
        print(f"Results saved to: {results_file}")
        print(f"Report saved to: {report_file}")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
