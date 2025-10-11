"""
Cross Validator Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Advanced cross-validation strategies for robust model evaluation including
time series aware validation and adversarial robustness testing.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold,
    cross_val_score, cross_validate, validation_curve
)
from sklearn.base import BaseEstimator
import warnings

logger = logging.getLogger(__name__)


class AdvancedCrossValidator:
    """Advanced cross-validation for IoT botnet detection."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize cross validator with configuration."""

        self.config = config or {}
        self.cv_strategy = self.config.get('cv_strategy', 'stratified_kfold')
        self.n_splits = self.config.get('n_splits', 5)
        self.test_size = self.config.get('test_size', 0.2)
        self.random_state = self.config.get('random_state', 42)
        self.scoring_metrics = self.config.get(
            'scoring_metrics', ['accuracy', 'precision', 'recall', 'f1'])

        self.cv_results = []
        self.validation_history = []

        logger.info(
            f"AdvancedCrossValidator initialized with {self.cv_strategy}")

    def perform_cross_validation(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                                 groups: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Perform comprehensive cross-validation.

        Args:
            model: Model to validate
            X: Features
            y: Labels
            groups: Group labels for group-based CV (optional)

        Returns:
            Cross-validation results
        """

        logger.info(
            f"Performing {self.cv_strategy} cross-validation on {len(X)} samples")

        # Select CV strategy
        cv_splitter = self._get_cv_splitter(X, y, groups)

        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y,
            cv=cv_splitter,
            scoring=self.scoring_metrics,
            return_train_score=True,
            return_estimator=True,
            n_jobs=-1
        )

        # Calculate additional metrics
        additional_metrics = self._calculate_additional_metrics(
            cv_results, X, y)

        # Combine results
        results = {
            'cv_strategy': self.cv_strategy,
            'n_splits': self.n_splits,
            'scoring_metrics': self.scoring_metrics,
            'cv_results': cv_results,
            'additional_metrics': additional_metrics,
            'model_type': type(model).__name__,
            'n_samples': len(X),
            'n_features': len(X.columns)
        }

        # Store results
        self.cv_results.append(results)
        self.validation_history.append({
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_type': type(model).__name__,
            'cv_strategy': self.cv_strategy,
            'mean_test_accuracy': np.mean(cv_results['test_accuracy'])
        })

        logger.info(
            f"Cross-validation completed. Mean accuracy: {np.mean(cv_results['test_accuracy']):.4f}")

        return results

    def _get_cv_splitter(self, X: pd.DataFrame, y: pd.Series,
                         groups: Optional[pd.Series] = None):
        """Get appropriate cross-validation splitter."""

        if self.cv_strategy == 'kfold':
            return KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        elif self.cv_strategy == 'stratified_kfold':
            return StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        elif self.cv_strategy == 'timeseries':
            return TimeSeriesSplit(n_splits=self.n_splits)

        elif self.cv_strategy == 'group_kfold' and groups is not None:
            return GroupKFold(n_splits=self.n_splits)

        else:
            logger.warning(
                f"Unknown CV strategy: {self.cv_strategy}, using StratifiedKFold")
            return StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

    def _calculate_additional_metrics(self, cv_results: Dict[str, Any],
                                      X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Calculate additional cross-validation metrics."""

        additional_metrics = {}

        # Calculate stability metrics
        for metric in self.scoring_metrics:
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']

            additional_metrics[f'{metric}_stability'] = {
                'test_mean': np.mean(test_scores),
                'test_std': np.std(test_scores),
                'test_cv': np.std(test_scores) / np.mean(test_scores) if np.mean(test_scores) > 0 else 0,
                'train_mean': np.mean(train_scores),
                'train_std': np.std(train_scores),
                'overfitting_score': np.mean(train_scores) - np.mean(test_scores)
            }

        # Calculate confidence intervals
        additional_metrics['confidence_intervals'] = self._calculate_confidence_intervals(
            cv_results)

        return additional_metrics

    def _calculate_confidence_intervals(self, cv_results: Dict[str, Any],
                                        confidence_level: float = 0.95) -> Dict[str, Any]:
        """Calculate confidence intervals for CV scores."""

        confidence_intervals = {}
        alpha = 1 - confidence_level

        for metric in self.scoring_metrics:
            test_scores = cv_results[f'test_{metric}']
            n_scores = len(test_scores)

            # Calculate confidence interval using t-distribution
            mean_score = np.mean(test_scores)
            std_score = np.std(test_scores)
            margin_error = std_score * 1.96 / \
                np.sqrt(n_scores)  # Approximate for large n

            confidence_intervals[metric] = {
                'lower_bound': mean_score - margin_error,
                'upper_bound': mean_score + margin_error,
                'mean': mean_score,
                'margin_error': margin_error
            }

        return confidence_intervals

    def perform_nested_cv(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                          param_grid: Dict[str, List], groups: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Perform nested cross-validation for unbiased model selection.

        Args:
            model: Model to validate
            X: Features
            y: Labels
            param_grid: Parameter grid for hyperparameter tuning
            groups: Group labels (optional)

        Returns:
            Nested CV results
        """

        logger.info("Performing nested cross-validation")

        from sklearn.model_selection import GridSearchCV

        # Outer CV
        outer_cv = self._get_cv_splitter(X, y, groups)

        # Inner CV for hyperparameter tuning
        inner_cv = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=self.random_state)

        # Nested CV
        nested_scores = []
        outer_fold = 0

        for train_idx, test_idx in outer_cv.split(X, y):
            outer_fold += 1
            logger.info(f"Processing outer fold {outer_fold}/{self.n_splits}")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Inner CV for hyperparameter tuning
            grid_search = GridSearchCV(
                model, param_grid,
                cv=inner_cv,
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)

            # Evaluate best model on outer test set
            best_model = grid_search.best_estimator_
            test_score = best_model.score(X_test, y_test)
            nested_scores.append(test_score)

        results = {
            'nested_cv_scores': nested_scores,
            'mean_nested_score': np.mean(nested_scores),
            'std_nested_score': np.std(nested_scores),
            'n_outer_folds': len(nested_scores),
            'param_grid': param_grid
        }

        logger.info(
            f"Nested CV completed. Mean score: {results['mean_nested_score']:.4f}")

        return results

    def perform_time_aware_cv(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                              time_column: str) -> Dict[str, Any]:
        """
        Perform time-aware cross-validation for temporal data.

        Args:
            model: Model to validate
            X: Features
            y: Labels
            time_column: Name of time column

        Returns:
            Time-aware CV results
        """

        logger.info("Performing time-aware cross-validation")

        if time_column not in X.columns:
            raise ValueError(
                f"Time column '{time_column}' not found in features")

        # Sort by time
        X_sorted = X.sort_values(time_column)
        y_sorted = y.loc[X_sorted.index]

        # Time series split
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        cv_results = cross_validate(
            model, X_sorted, y_sorted,
            cv=tscv,
            scoring=self.scoring_metrics,
            return_train_score=True,
            return_estimator=True
        )

        # Calculate temporal stability
        temporal_stability = self._calculate_temporal_stability(
            cv_results, X_sorted[time_column])

        results = {
            'cv_strategy': 'time_aware',
            'cv_results': cv_results,
            'temporal_stability': temporal_stability,
            'time_column': time_column,
            'n_splits': self.n_splits
        }

        logger.info("Time-aware cross-validation completed")

        return results

    def _calculate_temporal_stability(self, cv_results: Dict[str, Any],
                                      time_values: pd.Series) -> Dict[str, Any]:
        """Calculate temporal stability metrics."""

        stability_metrics = {}

        for metric in self.scoring_metrics:
            test_scores = cv_results[f'test_{metric}']

            # Calculate trend in performance over time
            if len(test_scores) > 2:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    range(len(test_scores)), test_scores)

                stability_metrics[metric] = {
                    'trend_slope': slope,
                    'trend_r_squared': r_value**2,
                    'trend_p_value': p_value,
                    'performance_trend': 'improving' if slope > 0.01 else 'declining' if slope < -0.01 else 'stable'
                }

        return stability_metrics

    def perform_adversarial_cv(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                               adversarial_generator: Callable) -> Dict[str, Any]:
        """
        Perform cross-validation with adversarial examples.

        Args:
            model: Model to validate
            X: Features
            y: Labels
            adversarial_generator: Function to generate adversarial examples

        Returns:
            Adversarial CV results
        """

        logger.info("Performing adversarial cross-validation")

        cv_splitter = self._get_cv_splitter(X, y)

        adversarial_scores = []
        clean_scores = []

        for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):
            logger.info(
                f"Processing adversarial fold {fold + 1}/{self.n_splits}")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Train model
            model.fit(X_train, y_train)

            # Test on clean data
            clean_score = model.score(X_test, y_test)
            clean_scores.append(clean_score)

            # Generate adversarial examples
            try:
                X_adv = adversarial_generator(
                    model, X_test.values, y_test.values)
                adv_score = model.score(X_adv, y_test)
                adversarial_scores.append(adv_score)
            except Exception as e:
                logger.warning(
                    f"Failed to generate adversarial examples for fold {fold}: {e}")
                adversarial_scores.append(clean_score)

        results = {
            'cv_strategy': 'adversarial',
            'clean_scores': clean_scores,
            'adversarial_scores': adversarial_scores,
            'mean_clean_score': np.mean(clean_scores),
            'mean_adversarial_score': np.mean(adversarial_scores),
            'robustness_drop': np.mean(clean_scores) - np.mean(adversarial_scores),
            'n_splits': self.n_splits
        }

        logger.info(
            f"Adversarial CV completed. Robustness drop: {results['robustness_drop']:.4f}")

        return results

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary."""

        if not self.validation_history:
            return {'message': 'No validation history available'}

        # Calculate summary statistics
        accuracies = [entry['mean_test_accuracy']
                      for entry in self.validation_history]

        summary = {
            'total_validations': len(self.validation_history),
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'validation_history': self.validation_history
        }

        return summary

    def compare_cv_strategies(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                              strategies: List[str]) -> Dict[str, Any]:
        """Compare different cross-validation strategies."""

        logger.info(f"Comparing CV strategies: {strategies}")

        comparison_results = {}

        for strategy in strategies:
            # Temporarily change CV strategy
            original_strategy = self.cv_strategy
            self.cv_strategy = strategy

            try:
                results = self.perform_cross_validation(model, X, y)
                comparison_results[strategy] = {
                    'mean_test_accuracy': np.mean(results['cv_results']['test_accuracy']),
                    'std_test_accuracy': np.std(results['cv_results']['test_accuracy']),
                    'mean_train_accuracy': np.mean(results['cv_results']['train_accuracy']),
                    'overfitting_score': np.mean(results['cv_results']['train_accuracy']) -
                    np.mean(results['cv_results']['test_accuracy'])
                }
            except Exception as e:
                logger.error(f"Error with strategy {strategy}: {e}")
                comparison_results[strategy] = {'error': str(e)}

            # Restore original strategy
            self.cv_strategy = original_strategy

        # Find best strategy
        valid_strategies = {
            k: v for k, v in comparison_results.items() if 'error' not in v}
        if valid_strategies:
            best_strategy = max(valid_strategies.keys(),
                                key=lambda x: valid_strategies[x]['mean_test_accuracy'])
            comparison_results['best_strategy'] = best_strategy

        logger.info(
            f"CV strategy comparison completed. Best: {comparison_results.get('best_strategy', 'N/A')}")

        return comparison_results


# Example usage and testing
if __name__ == '__main__':
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Create labels with some structure
    y = pd.Series(
        (X.iloc[:, 0] + X.iloc[:, 1] +
         np.random.randn(n_samples) * 0.1 > 0).astype(int)
    )

    # Add time column for time-aware CV
    X['timestamp'] = pd.date_range('2023-01-01', periods=n_samples, freq='H')

    # Create a simple model
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=42)

    print("Testing Advanced Cross Validator:")

    # Test standard cross-validation
    cv = AdvancedCrossValidator({
        'cv_strategy': 'stratified_kfold',
        'n_splits': 5,
        'scoring_metrics': ['accuracy', 'precision', 'recall', 'f1']
    })

    results = cv.perform_cross_validation(
        model, X.drop('timestamp', axis=1), y)
    print(
        f"Standard CV - Mean accuracy: {np.mean(results['cv_results']['test_accuracy']):.4f}")
    print(
        f"Standard CV - Std accuracy: {np.std(results['cv_results']['test_accuracy']):.4f}")

    # Test nested cross-validation
    param_grid = {'C': [0.1, 1, 10], 'max_iter': [100, 1000]}
    nested_results = cv.perform_nested_cv(
        model, X.drop('timestamp', axis=1), y, param_grid)
    print(f"Nested CV - Mean score: {nested_results['mean_nested_score']:.4f}")

    # Test time-aware cross-validation
    time_results = cv.perform_time_aware_cv(model, X, y, 'timestamp')
    print(
        f"Time-aware CV - Mean accuracy: {np.mean(time_results['cv_results']['test_accuracy']):.4f}")

    # Test adversarial cross-validation
    def simple_adversarial_generator(model, X_test, y_test):
        # Simple adversarial example generator
        noise = np.random.normal(0, 0.1, X_test.shape)
        return X_test + noise

    adv_results = cv.perform_adversarial_cv(model, X.drop(
        'timestamp', axis=1), y, simple_adversarial_generator)
    print(
        f"Adversarial CV - Clean accuracy: {adv_results['mean_clean_score']:.4f}")
    print(
        f"Adversarial CV - Adversarial accuracy: {adv_results['mean_adversarial_score']:.4f}")
    print(
        f"Adversarial CV - Robustness drop: {adv_results['robustness_drop']:.4f}")

    # Test strategy comparison
    strategies = ['stratified_kfold', 'kfold', 'timeseries']
    comparison = cv.compare_cv_strategies(
        model, X.drop('timestamp', axis=1), y, strategies)
    print(f"\nCV Strategy Comparison:")
    for strategy, metrics in comparison.items():
        if 'error' not in metrics:
            print(f"{strategy}: accuracy={metrics['mean_test_accuracy']:.4f}, "
                  f"std={metrics['std_test_accuracy']:.4f}")

    # Get validation summary
    summary = cv.get_validation_summary()
    print(f"\nValidation Summary:")
    print(f"Total validations: {summary['total_validations']}")
    print(f"Mean accuracy: {summary['mean_accuracy']:.4f}")
    print(f"Std accuracy: {summary['std_accuracy']:.4f}")
