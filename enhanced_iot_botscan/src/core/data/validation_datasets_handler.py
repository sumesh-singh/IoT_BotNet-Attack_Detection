"""
Validation Datasets Handler Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Specialized handler for validation datasets with cross-dataset evaluation,
domain adaptation, and transfer learning capabilities.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import os
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)


class ValidationDatasetsHandler:
    """Handler for validation datasets with cross-dataset evaluation capabilities."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize validation datasets handler with configuration."""

        self.config = config or {}
        self.validation_datasets = {}
        self.evaluation_results = {}
        self.cross_dataset_results = {}

        # Validation configuration
        self.validation_metrics = self.config.get(
            'validation_metrics', ['accuracy', 'precision', 'recall', 'f1'])
        self.domain_adaptation_enabled = self.config.get(
            'domain_adaptation_enabled', True)
        self.transfer_learning_enabled = self.config.get(
            'transfer_learning_enabled', True)

        logger.info("ValidationDatasetsHandler initialized")

    def load_validation_dataset(self, dataset_name: str, file_path: str,
                                dataset_type: str = 'validation') -> pd.DataFrame:
        """
        Load a validation dataset.

        Args:
            dataset_name: Name of the dataset
            file_path: Path to dataset file
            dataset_type: Type of dataset ('validation', 'test', 'external')

        Returns:
            Loaded validation dataset
        """

        logger.info(f"Loading {dataset_type} dataset: {dataset_name}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Validation dataset file not found: {file_path}")

        # Load dataset based on file extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        # Store dataset metadata
        self.validation_datasets[dataset_name] = {
            'data': df,
            'file_path': file_path,
            'dataset_type': dataset_type,
            'load_timestamp': datetime.now().isoformat(),
            'shape': df.shape,
            'columns': df.columns.tolist()
        }

        logger.info(
            f"Validation dataset {dataset_name} loaded successfully. Shape: {df.shape}")

        return df

    def evaluate_model_on_validation_datasets(self, model, model_name: str,
                                              target_column: str = 'label') -> Dict[str, Any]:
        """
        Evaluate model on all validation datasets.

        Args:
            model: Trained model to evaluate
            model_name: Name of the model
            target_column: Name of target column

        Returns:
            Evaluation results across all validation datasets
        """

        logger.info(f"Evaluating model {model_name} on validation datasets")

        evaluation_results = {
            'model_name': model_name,
            'evaluation_timestamp': datetime.now().isoformat(),
            'dataset_results': {},
            'overall_summary': {}
        }

        for dataset_name, dataset_info in self.validation_datasets.items():
            logger.info(f"Evaluating on dataset: {dataset_name}")

            df = dataset_info['data']

            # Prepare features and target
            if target_column not in df.columns:
                logger.warning(
                    f"Target column {target_column} not found in {dataset_name}")
                continue

            X = df.drop(columns=[target_column])
            y = df[target_column]

            try:
                # Make predictions
                y_pred = model.predict(X)
                y_proba = model.predict_proba(X) if hasattr(
                    model, 'predict_proba') else None

                # Calculate metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

                metrics = {
                    'accuracy': accuracy_score(y, y_pred),
                    'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
                    'f1_score': f1_score(y, y_pred, average='weighted', zero_division=0)
                }

                # Add additional metrics if probabilities available
                if y_proba is not None:
                    from sklearn.metrics import roc_auc_score, log_loss
                    try:
                        metrics['roc_auc'] = roc_auc_score(
                            y, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
                        metrics['log_loss'] = log_loss(y, y_proba)
                    except Exception as e:
                        logger.warning(
                            f"Could not calculate probability metrics: {e}")

                evaluation_results['dataset_results'][dataset_name] = {
                    'metrics': metrics,
                    'n_samples': len(df),
                    'n_features': len(X.columns),
                    'dataset_type': dataset_info['dataset_type']
                }

            except Exception as e:
                logger.error(f"Error evaluating on {dataset_name}: {e}")
                evaluation_results['dataset_results'][dataset_name] = {
                    'error': str(e),
                    'n_samples': len(df),
                    'n_features': len(X.columns)
                }

        # Calculate overall summary
        evaluation_results['overall_summary'] = self._calculate_overall_summary(
            evaluation_results['dataset_results'])

        # Store results
        self.evaluation_results[model_name] = evaluation_results

        logger.info(f"Model evaluation completed for {model_name}")

        return evaluation_results

    def perform_cross_dataset_evaluation(self, models: Dict[str, Any],
                                         target_column: str = 'label') -> Dict[str, Any]:
        """
        Perform cross-dataset evaluation across multiple models and datasets.

        Args:
            models: Dictionary of trained models {model_name: model}
            target_column: Name of target column

        Returns:
            Cross-dataset evaluation results
        """

        logger.info(
            f"Performing cross-dataset evaluation on {len(models)} models")

        cross_evaluation_results = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_results': {},
            'dataset_performance': {},
            'best_model_per_dataset': {},
            'overall_best_model': None
        }

        # Evaluate each model on all datasets
        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")

            model_results = self.evaluate_model_on_validation_datasets(
                model, model_name, target_column
            )

            cross_evaluation_results['model_results'][model_name] = model_results

        # Analyze performance across datasets
        cross_evaluation_results['dataset_performance'] = self._analyze_dataset_performance(
            cross_evaluation_results['model_results']
        )

        # Find best model per dataset
        cross_evaluation_results['best_model_per_dataset'] = self._find_best_model_per_dataset(
            cross_evaluation_results['model_results']
        )

        # Find overall best model
        cross_evaluation_results['overall_best_model'] = self._find_overall_best_model(
            cross_evaluation_results['model_results']
        )

        # Store results
        self.cross_dataset_results = cross_evaluation_results

        logger.info("Cross-dataset evaluation completed")

        return cross_evaluation_results

    def perform_domain_adaptation_evaluation(self, source_model, target_dataset_name: str,
                                             adaptation_method: str = 'fine_tuning',
                                             target_column: str = 'label') -> Dict[str, Any]:
        """
        Perform domain adaptation evaluation.

        Args:
            source_model: Source domain model
            target_dataset_name: Name of target dataset
            adaptation_method: Adaptation method ('fine_tuning', 'feature_adaptation')
            target_column: Name of target column

        Returns:
            Domain adaptation evaluation results
        """

        logger.info(
            f"Performing domain adaptation evaluation on {target_dataset_name}")

        if target_dataset_name not in self.validation_datasets:
            raise ValueError(f"Target dataset {target_dataset_name} not found")

        target_df = self.validation_datasets[target_dataset_name]['data']

        if target_column not in target_df.columns:
            raise ValueError(
                f"Target column {target_column} not found in {target_dataset_name}")

        X_target = target_df.drop(columns=[target_column])
        y_target = target_df[target_column]

        adaptation_results = {
            'target_dataset': target_dataset_name,
            'adaptation_method': adaptation_method,
            'evaluation_timestamp': datetime.now().isoformat(),
            'baseline_performance': {},
            'adapted_performance': {},
            'adaptation_improvement': {}
        }

        # Baseline performance (source model on target data)
        try:
            y_pred_baseline = source_model.predict(X_target)
            baseline_accuracy = accuracy_score(y_target, y_pred_baseline)

            adaptation_results['baseline_performance'] = {
                'accuracy': baseline_accuracy,
                'n_samples': len(X_target)
            }
        except Exception as e:
            logger.error(f"Baseline evaluation failed: {e}")
            adaptation_results['baseline_performance'] = {'error': str(e)}

        # Domain adaptation
        if adaptation_method == 'fine_tuning':
            adapted_model = self._fine_tune_model(
                source_model, X_target, y_target)
        elif adaptation_method == 'feature_adaptation':
            adapted_model = self._adapt_features(
                source_model, X_target, y_target)
        else:
            raise ValueError(f"Unknown adaptation method: {adaptation_method}")

        # Adapted model performance
        try:
            y_pred_adapted = adapted_model.predict(X_target)
            adapted_accuracy = accuracy_score(y_target, y_pred_adapted)

            adaptation_results['adapted_performance'] = {
                'accuracy': adapted_accuracy,
                'n_samples': len(X_target)
            }
        except Exception as e:
            logger.error(f"Adapted model evaluation failed: {e}")
            adaptation_results['adapted_performance'] = {'error': str(e)}

        # Calculate improvement
        if 'error' not in adaptation_results['baseline_performance'] and 'error' not in adaptation_results['adapted_performance']:
            baseline_acc = adaptation_results['baseline_performance']['accuracy']
            adapted_acc = adaptation_results['adapted_performance']['accuracy']

            adaptation_results['adaptation_improvement'] = {
                'accuracy_improvement': adapted_acc - baseline_acc,
                'relative_improvement': (adapted_acc - baseline_acc) / baseline_acc if baseline_acc > 0 else 0
            }

        logger.info(
            f"Domain adaptation evaluation completed for {target_dataset_name}")

        return adaptation_results

    def perform_transfer_learning_evaluation(self, source_model, target_datasets: List[str],
                                             transfer_method: str = 'feature_extraction',
                                             target_column: str = 'label') -> Dict[str, Any]:
        """
        Perform transfer learning evaluation.

        Args:
            source_model: Source model for transfer learning
            target_datasets: List of target dataset names
            transfer_method: Transfer learning method
            target_column: Name of target column

        Returns:
            Transfer learning evaluation results
        """

        logger.info(
            f"Performing transfer learning evaluation on {len(target_datasets)} datasets")

        transfer_results = {
            'transfer_method': transfer_method,
            'evaluation_timestamp': datetime.now().isoformat(),
            'target_datasets': {},
            'overall_summary': {}
        }

        for target_dataset in target_datasets:
            if target_dataset not in self.validation_datasets:
                logger.warning(
                    f"Target dataset {target_dataset} not found, skipping")
                continue

            logger.info(f"Transfer learning on dataset: {target_dataset}")

            target_df = self.validation_datasets[target_dataset]['data']

            if target_column not in target_df.columns:
                logger.warning(
                    f"Target column {target_column} not found in {target_dataset}")
                continue

            X_target = target_df.drop(columns=[target_column])
            y_target = target_df[target_column]

            try:
                # Apply transfer learning
                if transfer_method == 'feature_extraction':
                    transferred_model = self._extract_features(
                        source_model, X_target, y_target)
                elif transfer_method == 'fine_tuning':
                    transferred_model = self._fine_tune_model(
                        source_model, X_target, y_target)
                else:
                    raise ValueError(
                        f"Unknown transfer method: {transfer_method}")

                # Evaluate transferred model
                y_pred = transferred_model.predict(X_target)
                accuracy = accuracy_score(y_target, y_pred)

                transfer_results['target_datasets'][target_dataset] = {
                    'accuracy': accuracy,
                    'n_samples': len(X_target),
                    'transfer_successful': True
                }

            except Exception as e:
                logger.error(
                    f"Transfer learning failed on {target_dataset}: {e}")
                transfer_results['target_datasets'][target_dataset] = {
                    'error': str(e),
                    'transfer_successful': False
                }

        # Calculate overall summary
        successful_transfers = [result for result in transfer_results['target_datasets'].values()
                                if result.get('transfer_successful', False)]

        if successful_transfers:
            transfer_results['overall_summary'] = {
                'successful_transfers': len(successful_transfers),
                'total_datasets': len(target_datasets),
                'success_rate': len(successful_transfers) / len(target_datasets),
                'mean_accuracy': np.mean([result['accuracy'] for result in successful_transfers])
            }

        logger.info("Transfer learning evaluation completed")

        return transfer_results

    def _calculate_overall_summary(self, dataset_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall summary across datasets."""

        valid_results = {k: v for k,
                         v in dataset_results.items() if 'error' not in v}

        if not valid_results:
            return {'error': 'No valid results found'}

        # Calculate average metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        summary = {}

        for metric in metrics:
            values = [result['metrics'][metric] for result in valid_results.values()
                      if metric in result['metrics']]
            if values:
                summary[f'mean_{metric}'] = np.mean(values)
                summary[f'std_{metric}'] = np.std(values)
                summary[f'min_{metric}'] = np.min(values)
                summary[f'max_{metric}'] = np.max(values)

        summary['total_datasets'] = len(valid_results)
        summary['total_samples'] = sum(result['n_samples']
                                       for result in valid_results.values())

        return summary

    def _analyze_dataset_performance(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance across datasets."""

        dataset_performance = {}

        # Get all unique datasets
        all_datasets = set()
        for model_name, results in model_results.items():
            all_datasets.update(results['dataset_results'].keys())

        # Analyze each dataset
        for dataset_name in all_datasets:
            dataset_metrics = {}

            for model_name, results in model_results.items():
                if dataset_name in results['dataset_results']:
                    dataset_result = results['dataset_results'][dataset_name]
                    if 'error' not in dataset_result:
                        dataset_metrics[model_name] = dataset_result['metrics']

            if dataset_metrics:
                # Calculate statistics for this dataset
                accuracy_values = [metrics['accuracy']
                                   for metrics in dataset_metrics.values()]
                dataset_performance[dataset_name] = {
                    'mean_accuracy': np.mean(accuracy_values),
                    'std_accuracy': np.std(accuracy_values),
                    'best_model': max(dataset_metrics.keys(), key=lambda x: dataset_metrics[x]['accuracy']),
                    'worst_model': min(dataset_metrics.keys(), key=lambda x: dataset_metrics[x]['accuracy']),
                    'n_models_tested': len(dataset_metrics)
                }

        return dataset_performance

    def _find_best_model_per_dataset(self, model_results: Dict[str, Any]) -> Dict[str, str]:
        """Find best model for each dataset."""

        best_models = {}

        # Get all unique datasets
        all_datasets = set()
        for results in model_results.values():
            all_datasets.update(results['dataset_results'].keys())

        # Find best model for each dataset
        for dataset_name in all_datasets:
            best_accuracy = -1
            best_model = None

            for model_name, results in model_results.items():
                if dataset_name in results['dataset_results']:
                    dataset_result = results['dataset_results'][dataset_name]
                    if 'error' not in dataset_result and 'accuracy' in dataset_result['metrics']:
                        accuracy = dataset_result['metrics']['accuracy']
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model = model_name

            if best_model:
                best_models[dataset_name] = best_model

        return best_models

    def _find_overall_best_model(self, model_results: Dict[str, Any]) -> str:
        """Find overall best model across all datasets."""

        model_scores = {}

        for model_name, results in model_results.items():
            valid_results = [
                r for r in results['dataset_results'].values() if 'error' not in r]
            if valid_results:
                # Calculate average accuracy across all datasets
                accuracies = [r['metrics']['accuracy'] for r in valid_results]
                model_scores[model_name] = np.mean(accuracies)

        if model_scores:
            return max(model_scores.keys(), key=lambda x: model_scores[x])
        else:
            return None

    def _fine_tune_model(self, source_model, X_target: pd.DataFrame, y_target: pd.Series):
        """Fine-tune model on target data."""

        # Create a copy of the source model
        import copy
        fine_tuned_model = copy.deepcopy(source_model)

        # Fine-tune on target data
        fine_tuned_model.fit(X_target, y_target)

        return fine_tuned_model

    def _adapt_features(self, source_model, X_target: pd.DataFrame, y_target: pd.Series):
        """Adapt features for domain adaptation."""

        # Simple feature adaptation - could be enhanced with more sophisticated methods
        from sklearn.preprocessing import StandardScaler

        # Normalize target features
        scaler = StandardScaler()
        X_target_scaled = scaler.fit_transform(X_target)
        X_target_scaled = pd.DataFrame(
            X_target_scaled, columns=X_target.columns)

        # Create adapted model
        import copy
        adapted_model = copy.deepcopy(source_model)
        adapted_model.fit(X_target_scaled, y_target)

        return adapted_model

    def _extract_features(self, source_model, X_target: pd.DataFrame, y_target: pd.Series):
        """Extract features using source model."""

        # Simple feature extraction - could be enhanced with more sophisticated methods
        try:
            # Try to get feature importance or use the model directly
            if hasattr(source_model, 'feature_importances_'):
                # Use feature importance for feature selection
                feature_importance = source_model.feature_importances_
                top_features = np.argsort(
                    feature_importance)[-10:]  # Top 10 features
                X_target_selected = X_target.iloc[:, top_features]
            else:
                X_target_selected = X_target

            # Create new model with selected features
            from sklearn.linear_model import LogisticRegression
            extracted_model = LogisticRegression(random_state=42)
            extracted_model.fit(X_target_selected, y_target)

            return extracted_model

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            # Fallback to fine-tuning
            return self._fine_tune_model(source_model, X_target, y_target)

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary."""

        return {
            'validation_datasets': list(self.validation_datasets.keys()),
            'evaluation_results': list(self.evaluation_results.keys()),
            'cross_dataset_results_available': bool(self.cross_dataset_results),
            'total_validation_datasets': len(self.validation_datasets),
            'total_evaluations': len(self.evaluation_results)
        }

    def export_evaluation_results(self, output_path: str) -> None:
        """Export evaluation results to file."""

        import json

        export_data = {
            'evaluation_results': self.evaluation_results,
            'cross_dataset_results': self.cross_dataset_results,
            'validation_summary': self.get_validation_summary(),
            'export_timestamp': datetime.now().isoformat()
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Evaluation results exported to {output_path}")


# Example usage and testing
if __name__ == '__main__':
    # Create sample validation datasets
    np.random.seed(42)
    n_samples = 500

    # Create sample validation data
    validation_data = {
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(5, 2, n_samples),
        'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
        'label': np.random.randint(0, 2, n_samples)
    }

    df_validation = pd.DataFrame(validation_data)

    # Create sample test data
    test_data = {
        'feature_1': np.random.normal(1, 1.5, n_samples),
        'feature_2': np.random.normal(4, 2.5, n_samples),
        'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
        'label': np.random.randint(0, 2, n_samples)
    }

    df_test = pd.DataFrame(test_data)

    print("Testing Validation Datasets Handler:")

    # Initialize handler
    handler = ValidationDatasetsHandler()

    # Save sample datasets
    os.makedirs('test_validation_data', exist_ok=True)
    df_validation.to_csv(
        'test_validation_data/validation_dataset.csv', index=False)
    df_test.to_csv('test_validation_data/test_dataset.csv', index=False)

    # Load validation datasets
    handler.load_validation_dataset(
        'validation_dataset', 'test_validation_data/validation_dataset.csv', 'validation')
    handler.load_validation_dataset(
        'test_dataset', 'test_validation_data/test_dataset.csv', 'test')

    # Create sample models
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    # Train models on sample data
    X_train = df_validation.drop('label', axis=1)
    y_train = df_validation['label']

    model1 = LogisticRegression(random_state=42)
    model1.fit(X_train, y_train)

    model2 = RandomForestClassifier(n_estimators=10, random_state=42)
    model2.fit(X_train, y_train)

    models = {
        'LogisticRegression': model1,
        'RandomForest': model2
    }

    # Test model evaluation on validation datasets
    print("\n1. Testing Model Evaluation:")
    for model_name, model in models.items():
        results = handler.evaluate_model_on_validation_datasets(
            model, model_name)
        print(
            f"{model_name} - Overall accuracy: {results['overall_summary'].get('mean_accuracy', 'N/A')}")

    # Test cross-dataset evaluation
    print("\n2. Testing Cross-Dataset Evaluation:")
    cross_results = handler.perform_cross_dataset_evaluation(models)
    print(f"Overall best model: {cross_results['overall_best_model']}")
    print(f"Best model per dataset: {cross_results['best_model_per_dataset']}")

    # Test domain adaptation
    print("\n3. Testing Domain Adaptation:")
    adaptation_results = handler.perform_domain_adaptation_evaluation(
        model1, 'test_dataset', 'fine_tuning'
    )
    print(
        f"Domain adaptation improvement: {adaptation_results.get('adaptation_improvement', {})}")

    # Test transfer learning
    print("\n4. Testing Transfer Learning:")
    transfer_results = handler.perform_transfer_learning_evaluation(
        model1, ['test_dataset'], 'feature_extraction'
    )
    print(
        f"Transfer learning success rate: {transfer_results['overall_summary'].get('success_rate', 'N/A')}")

    # Get validation summary
    summary = handler.get_validation_summary()
    print(f"\nValidation Summary:")
    print(f"Total validation datasets: {summary['total_validation_datasets']}")
    print(f"Total evaluations: {summary['total_evaluations']}")

    print("Validation datasets handler testing completed")
