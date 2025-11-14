#!/usr/bin/env python3
"""
Model Training Script for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Trains hybrid ensemble models with adversarial robustness and concept drift adaptation.
"""

import sys
import os
import argparse
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import random
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.ensemble.hybrid_ensemble import HybridEnsemble
from core.adversarial.adversarial_trainer import AdversarialTrainer
from core.drift_detection.drift_detector import DriftDetector
from data.data_loader import DataLoader
from evaluation.performance_evaluator import PerformanceEvaluator
from utils.config_manager import ConfigManager
from utils.logger import setup_logging
from core.preprocessing.feature_engineer import FeatureEngineer
from core.preprocessing.scaler import Scaler

class ModelTrainer:
    """Main training orchestrator for Enhanced IoT BotScan."""

    def __init__(self, config_path: str = None):
        """Initialize trainer with configuration."""

        # Setup configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config

        # Setup logging
        setup_logging(self.config.get('logging', {}))
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.data_loader = DataLoader(self.config.get('data', {}))
        self.ensemble = HybridEnsemble(self.config)
        self.adversarial_trainer = AdversarialTrainer(self.config.get('adversarial_training', {}))
        self.drift_detector = DriftDetector(self.config.get('concept_drift', {}))
        self.evaluator = PerformanceEvaluator(self.config.get('evaluation', {}))

        # Global seeds for reproducibility
        seed = 42
        random.seed(seed)
        np.random.seed(seed)

        # Training state
        self.datasets = {}
        self.training_results = {}

        self.logger.info("ModelTrainer initialized successfully")

    def load_datasets(self, dataset_names: list = None) -> None:
        """Load specified datasets."""

        if dataset_names is None:
            dataset_names = ['n_baiot', 'iot_23', 'bot_iot']

        self.logger.info(f"Loading datasets: {dataset_names}")

        for dataset_name in dataset_names:
            try:
                dataset = self.data_loader.load_dataset(dataset_name)
                self.datasets[dataset_name] = dataset

                # Log dataset info
                self.logger.info(f"Loaded {dataset_name}: {dataset['total_samples']} samples, "
                               f"{dataset['n_features']} features, {dataset['n_classes']} classes")

            except Exception as e:
                self.logger.error(f"Failed to load {dataset_name}: {e}")

    def prepare_training_data(self, dataset_name: str = 'n_baiot') -> tuple:
        """Prepare training and validation data."""

        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")

        dataset = self.datasets[dataset_name]
        X = dataset['features']
        y = dataset['labels']

        # Split data: 70/15/15
        from sklearn.model_selection import train_test_split
        X_df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])
        y_sr = pd.Series(y)

        X_train_full, X_temp, y_train_full, y_temp = train_test_split(
            X_df, y_sr, test_size=0.30, random_state=42, stratify=y_sr
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
        )

        # Feature engineering fitted on train, applied to val/test
        fe_config = self.config.get('feature_engineering', {
            'create_statistical_features': True,
            'create_interaction_features': True,
            'create_polynomial_features': False,
            'feature_selection_method': 'mutual_info',
            'n_features_select': min(50, X_train_full.shape[1])
        })
        feature_engineer = FeatureEngineer(fe_config)
        X_train_eng = feature_engineer.engineer_features(X_train_full, y_train_full)
        X_val_eng = feature_engineer.transform_new_data(X_val)
        X_test_eng = feature_engineer.transform_new_data(X_test)

        # Scaling fitted on train, applied to val/test
        scaler_config = self.config.get('scaling', {'method': 'standard', 'scale_features': True})
        scaler = Scaler(scaler_config)
        X_train_scaled, _ = scaler.fit_transform(X_train_eng)
        X_val_scaled, _ = scaler.transform(X_val_eng)
        X_test_scaled, _ = scaler.transform(X_test_eng)

        self.logger.info(f"Data split - Train: {len(X_train_scaled)}, Val: {len(X_val_scaled)}, Test: {len(X_test_scaled)}")

        return X_train_scaled.values.astype(np.float32), X_val_scaled.values.astype(np.float32), X_test_scaled.values.astype(np.float32), y_train_full.values, y_val.values, y_test.values

    def train_baseline_models(self, dataset_name: str = 'n_baiot') -> dict:
        """Train baseline ensemble models without adversarial training."""

        self.logger.info(f"Training baseline models on {dataset_name}")

        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_training_data(dataset_name)

        self.drift_detector.set_reference_data(X_train)

        # Class balance via random oversampling
        X_train, y_train = self._balance_classes(X_train, y_train)

        # Train ensemble
        # Enable hyperparameter optimization in base models
        self.ensemble.optimize_base_models = True

        training_results = self.ensemble.train(
            X=pd.DataFrame(X_train),
            y=pd.Series(y_train),
            validation_data=(pd.DataFrame(X_val), pd.Series(y_val))
        )

        # Evaluate on test set
        test_results = self.evaluator.comprehensive_evaluation(
            self.ensemble, 
            pd.DataFrame(X_test), 
            pd.Series(y_test)
        )

        # Save evaluation artifacts
        self._save_evaluation_artifacts(dataset_name, test_results)

        baseline_results = {
            'dataset': dataset_name,
            'training_results': training_results,
            'test_results': test_results,
            'model_info': self.ensemble.get_model_info()
        }

        self.training_results[f'{dataset_name}_baseline'] = baseline_results

        self.logger.info(f"Baseline training completed. Test accuracy: {test_results['accuracy']:.4f}")

        return baseline_results

    def train_adversarial_robust_models(self, dataset_name: str = 'n_baiot') -> dict:
        """Train models with adversarial robustness."""

        self.logger.info(f"Training adversarially robust models on {dataset_name}")

        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_training_data(dataset_name)

        # Create a fresh ensemble for adversarial training
        robust_ensemble = HybridEnsemble(self.config)
        robust_ensemble.optimize_base_models = True

        # Train with adversarial examples
        # Class balance via random oversampling
        X_train, y_train = self._balance_classes(X_train, y_train)

        adversarial_results = self.adversarial_trainer.train_robust_model(
            robust_ensemble,
            X_train, y_train,
            X_val, y_val
        )

        # Evaluate robustness
        robustness_results = self.evaluator.evaluate_adversarial_robustness(
            robust_ensemble,
            pd.DataFrame(X_test),
            pd.Series(y_test)
        )

        # Regular evaluation
        test_results = self.evaluator.comprehensive_evaluation(
            robust_ensemble,
            pd.DataFrame(X_test),
            pd.Series(y_test)
        )

        # Save evaluation artifacts
        self._save_evaluation_artifacts(dataset_name + "_robust", test_results)

        robust_model_results = {
            'dataset': dataset_name,
            'adversarial_training_results': adversarial_results,
            'robustness_evaluation': robustness_results,
            'test_results': test_results,
            'model_info': robust_ensemble.get_model_info()
        }

        self.training_results[f'{dataset_name}_robust'] = robust_model_results

        self.logger.info(f"Adversarial training completed. "
                        f"Clean accuracy: {test_results['accuracy']:.4f}, "
                        f"Robust accuracy: {robustness_results.get('overall_robustness', 0):.4f}")

        return robust_model_results

    def validate_cross_dataset(self) -> dict:
        """Perform cross-dataset validation."""

        self.logger.info("Starting cross-dataset validation")

        cross_validation_results = {}
        dataset_names = list(self.datasets.keys())

        for train_dataset in dataset_names:
            for test_dataset in dataset_names:
                if train_dataset == test_dataset:
                    continue

                self.logger.info(f"Training on {train_dataset}, testing on {test_dataset}")

                # Prepare training data
                train_data = self.datasets[train_dataset]
                X_train, y_train = train_data['features'], train_data['labels']

                # Prepare test data
                test_data = self.datasets[test_dataset]
                X_test, y_test = test_data['features'], test_data['labels']

                # Train model
                cross_ensemble = HybridEnsemble(self.config)
                cross_ensemble.train(pd.DataFrame(X_train), pd.Series(y_train))

                # Evaluate
                results = self.evaluator.comprehensive_evaluation(
                    cross_ensemble,
                    pd.DataFrame(X_test),
                    pd.Series(y_test)
                )

                cross_validation_results[f'{train_dataset}_to_{test_dataset}'] = results

                self.logger.info(f"Cross-validation {train_dataset}->{test_dataset}: "
                               f"Accuracy = {results['accuracy']:.4f}")

        self.training_results['cross_dataset_validation'] = cross_validation_results
        return cross_validation_results

    def _balance_classes(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Randomly oversample minority classes to match majority count."""
        unique, counts = np.unique(y, return_counts=True)
        max_count = counts.max()
        X_balanced = []
        y_balanced = []
        rng = np.random.default_rng(42)
        for cls, cnt in zip(unique, counts):
            idx = np.where(y == cls)[0]
            X_cls = X[idx]
            y_cls = y[idx]
            if cnt < max_count:
                add_n = max_count - cnt
                add_idx = rng.choice(idx, size=add_n, replace=True)
                X_balanced.append(np.concatenate([X_cls, X[add_idx]], axis=0))
                y_balanced.append(np.concatenate([y_cls, y[add_idx]], axis=0))
            else:
                X_balanced.append(X_cls)
                y_balanced.append(y_cls)
        X_new = np.concatenate(X_balanced, axis=0)
        y_new = np.concatenate(y_balanced, axis=0)
        # Shuffle
        perm = rng.permutation(len(y_new))
        return X_new[perm], y_new[perm]

    def _save_evaluation_artifacts(self, eval_name: str, results: dict) -> None:
        """Save evaluation report and plots to results directory."""
        output_dir = Path('./data/results')
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        # Save textual report
        report_str = self.evaluator.generate_evaluation_report(results)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = output_dir / f'{eval_name}_report_{ts}.txt'
        with open(report_file, 'w') as f:
            f.write(report_str)
        # Save confusion matrix
        y_true = results.get('true_labels', [])
        y_pred = results.get('predictions', [])
        if y_true and y_pred:
            cm_path = plots_dir / f'{eval_name}_confusion_matrix_{ts}.png'
            self.evaluator.plot_confusion_matrix(y_true, y_pred, save_path=str(cm_path))
        # Save ROC curve for binary if available
        y_proba = results.get('prediction_probabilities')
        if y_proba:
            roc_path = plots_dir / f'{eval_name}_roc_{ts}.png'
            self.evaluator.plot_roc_curve(y_true, np.array(y_proba), save_path=str(roc_path))

    def test_concept_drift_detection(self, dataset_name: str = 'n_baiot') -> dict:
        """Test concept drift detection capabilities."""

        self.logger.info(f"Testing concept drift detection on {dataset_name}")

        # Get dataset
        dataset = self.datasets[dataset_name]
        X, y = dataset['features'], dataset['labels']

        # Split data into batches to simulate streaming
        n_batches = 10
        batch_size = len(X) // n_batches

        drift_results = []

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X))

            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]

            # Simulate concept drift by modifying later batches
            if i > 6:  # Introduce drift after batch 6
                # Add noise to simulate drift
                noise = np.random.normal(0, 0.5, X_batch.shape)
                X_batch = X_batch + noise

            # Detect drift
            drift_result = self.drift_detector.detect_drift(X_new=X_batch)
            drift_results.append({
                'batch': i,
                'drift_detected': drift_result['drift_detected'],
                'drift_methods': drift_result.get('individual_results', {})
            })

            if drift_result['drift_detected']:
                self.logger.info(f"Drift detected in batch {i}")

        # Get comprehensive drift statistics
        drift_stats = self.drift_detector.get_comprehensive_statistics()

        concept_drift_results = {
            'dataset': dataset_name,
            'batch_results': drift_results,
            'drift_statistics': drift_stats,
            'total_batches': n_batches,
            'drift_detections': sum(1 for r in drift_results if r['drift_detected'])
        }

        self.training_results['concept_drift_test'] = concept_drift_results

        self.logger.info(f"Concept drift testing completed. "
                        f"Drift detected in {concept_drift_results['drift_detections']} batches")

        return concept_drift_results

    def save_results(self, output_dir: str = './data/results') -> None:
        """Save all training results."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save results as YAML
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = output_path / f'training_results_{timestamp}.yaml'

        with open(results_file, 'w') as f:
            yaml.dump(self.training_results, f, default_flow_style=False)

        # Save models
        models_dir = output_path / 'models'
        models_dir.mkdir(exist_ok=True)

        if hasattr(self.ensemble, 'is_trained') and self.ensemble.is_trained:
            model_file = models_dir / f'ensemble_model_{timestamp}.pkl'
            self.ensemble.save_model(str(model_file))

        self.logger.info(f"Results saved to {results_file}")

    def run_full_training_pipeline(self, datasets: list = None) -> dict:
        """Run the complete training pipeline."""

        self.logger.info("Starting full training pipeline")

        # Load datasets
        self.load_datasets(datasets)

        # Train baseline models on primary dataset
        primary_dataset = 'n_baiot'
        if primary_dataset in self.datasets:
            self.train_baseline_models(primary_dataset)
            self.train_adversarial_robust_models(primary_dataset)

        # Cross-dataset validation
        if len(self.datasets) > 1:
            self.validate_cross_dataset()

        # Test concept drift detection
        if primary_dataset in self.datasets:
            self.test_concept_drift_detection(primary_dataset)

        # Save results
        self.save_results()

        self.logger.info("Full training pipeline completed")

        return self.training_results

def main():
    """Main function for command-line usage."""

    parser = argparse.ArgumentParser(
        description="Train Enhanced IoT BotScan models"
    )
    parser.add_argument(
        '--config',
        default='./config/config.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['n_baiot'],
        choices=['n_baiot', 'iot_23', 'bot_iot'],
        help='Datasets to use for training'
    )
    parser.add_argument(
        '--mode',
        choices=['baseline', 'adversarial', 'full'],
        default='full',
        help='Training mode'
    )
    parser.add_argument(
        '--output-dir',
        default='./data/results',
        help='Output directory for results'
    )

    args = parser.parse_args()

    try:
        # Initialize trainer
        trainer = ModelTrainer(args.config)

        # Load datasets
        trainer.load_datasets(args.datasets)

        # Run training based on mode
        if args.mode == 'baseline':
            trainer.train_baseline_models(args.datasets[0])
        elif args.mode == 'adversarial':
            trainer.train_adversarial_robust_models(args.datasets[0])
        elif args.mode == 'full':
            trainer.run_full_training_pipeline(args.datasets)

        # Save results
        trainer.save_results(args.output_dir)

        print("\n🎉 Training completed successfully!")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        logging.exception("Training failed")
        sys.exit(1)

if __name__ == '__main__':
    main()