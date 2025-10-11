"""
Robustness Tester Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Comprehensive robustness testing framework for evaluating model resilience
against various types of attacks and data perturbations.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
import warnings

logger = logging.getLogger(__name__)


class RobustnessTester:
    """Comprehensive robustness testing framework."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize robustness tester with configuration."""

        self.config = config or {}
        self.test_results = []
        self.robustness_history = []

        # Test configuration
        self.noise_levels = self.config.get(
            'noise_levels', [0.01, 0.05, 0.1, 0.2])
        self.outlier_percentages = self.config.get(
            'outlier_percentages', [0.01, 0.05, 0.1])
        self.missing_value_percentages = self.config.get(
            'missing_value_percentages', [0.01, 0.05, 0.1])
        self.feature_corruption_levels = self.config.get(
            'feature_corruption_levels', [0.01, 0.05, 0.1])

        logger.info("RobustnessTester initialized")

    def test_noise_robustness(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Test model robustness against noise.

        Args:
            model: Model to test
            X: Features
            y: Labels

        Returns:
            Noise robustness test results
        """

        logger.info("Testing noise robustness")

        results = {
            'test_type': 'noise_robustness',
            'noise_levels': self.noise_levels,
            'baseline_accuracy': model.score(X, y),
            'noise_results': {}
        }

        for noise_level in self.noise_levels:
            logger.info(f"Testing noise level: {noise_level}")

            # Generate noisy data
            X_noisy = self._add_gaussian_noise(X, noise_level)

            # Test model performance
            accuracy = model.score(X_noisy, y)
            accuracy_drop = results['baseline_accuracy'] - accuracy

            results['noise_results'][noise_level] = {
                'accuracy': accuracy,
                'accuracy_drop': accuracy_drop,
                'relative_drop': accuracy_drop / results['baseline_accuracy'] if results['baseline_accuracy'] > 0 else 0
            }

        # Calculate robustness score
        results['robustness_score'] = self._calculate_noise_robustness_score(
            results)

        self.test_results.append(results)
        logger.info(
            f"Noise robustness test completed. Score: {results['robustness_score']:.4f}")

        return results

    def test_outlier_robustness(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Test model robustness against outliers.

        Args:
            model: Model to test
            X: Features
            y: Labels

        Returns:
            Outlier robustness test results
        """

        logger.info("Testing outlier robustness")

        results = {
            'test_type': 'outlier_robustness',
            'outlier_percentages': self.outlier_percentages,
            'baseline_accuracy': model.score(X, y),
            'outlier_results': {}
        }

        for outlier_pct in self.outlier_percentages:
            logger.info(f"Testing outlier percentage: {outlier_pct}")

            # Generate data with outliers
            X_outliers = self._add_outliers(X, outlier_pct)

            # Test model performance
            accuracy = model.score(X_outliers, y)
            accuracy_drop = results['baseline_accuracy'] - accuracy

            results['outlier_results'][outlier_pct] = {
                'accuracy': accuracy,
                'accuracy_drop': accuracy_drop,
                'relative_drop': accuracy_drop / results['baseline_accuracy'] if results['baseline_accuracy'] > 0 else 0
            }

        # Calculate robustness score
        results['robustness_score'] = self._calculate_outlier_robustness_score(
            results)

        self.test_results.append(results)
        logger.info(
            f"Outlier robustness test completed. Score: {results['robustness_score']:.4f}")

        return results

    def test_missing_value_robustness(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Test model robustness against missing values.

        Args:
            model: Model to test
            X: Features
            y: Labels

        Returns:
            Missing value robustness test results
        """

        logger.info("Testing missing value robustness")

        results = {
            'test_type': 'missing_value_robustness',
            'missing_percentages': self.missing_value_percentages,
            'baseline_accuracy': model.score(X, y),
            'missing_results': {}
        }

        for missing_pct in self.missing_value_percentages:
            logger.info(f"Testing missing value percentage: {missing_pct}")

            # Generate data with missing values
            X_missing = self._add_missing_values(X, missing_pct)

            # Test model performance
            try:
                accuracy = model.score(X_missing, y)
                accuracy_drop = results['baseline_accuracy'] - accuracy
            except Exception as e:
                logger.warning(
                    f"Model failed with {missing_pct} missing values: {e}")
                accuracy = 0
                accuracy_drop = results['baseline_accuracy']

            results['missing_results'][missing_pct] = {
                'accuracy': accuracy,
                'accuracy_drop': accuracy_drop,
                'relative_drop': accuracy_drop / results['baseline_accuracy'] if results['baseline_accuracy'] > 0 else 0,
                'model_failed': accuracy == 0
            }

        # Calculate robustness score
        results['robustness_score'] = self._calculate_missing_value_robustness_score(
            results)

        self.test_results.append(results)
        logger.info(
            f"Missing value robustness test completed. Score: {results['robustness_score']:.4f}")

        return results

    def test_feature_corruption_robustness(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Test model robustness against feature corruption.

        Args:
            model: Model to test
            X: Features
            y: Labels

        Returns:
            Feature corruption robustness test results
        """

        logger.info("Testing feature corruption robustness")

        results = {
            'test_type': 'feature_corruption_robustness',
            'corruption_levels': self.feature_corruption_levels,
            'baseline_accuracy': model.score(X, y),
            'corruption_results': {}
        }

        for corruption_level in self.feature_corruption_levels:
            logger.info(f"Testing corruption level: {corruption_level}")

            # Generate corrupted data
            X_corrupted = self._corrupt_features(X, corruption_level)

            # Test model performance
            accuracy = model.score(X_corrupted, y)
            accuracy_drop = results['baseline_accuracy'] - accuracy

            results['corruption_results'][corruption_level] = {
                'accuracy': accuracy,
                'accuracy_drop': accuracy_drop,
                'relative_drop': accuracy_drop / results['baseline_accuracy'] if results['baseline_accuracy'] > 0 else 0
            }

        # Calculate robustness score
        results['robustness_score'] = self._calculate_corruption_robustness_score(
            results)

        self.test_results.append(results)
        logger.info(
            f"Feature corruption robustness test completed. Score: {results['robustness_score']:.4f}")

        return results

    def test_adversarial_robustness(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                                    attack_generators: List[Callable]) -> Dict[str, Any]:
        """
        Test model robustness against adversarial attacks.

        Args:
            model: Model to test
            X: Features
            y: Labels
            attack_generators: List of attack generator functions

        Returns:
            Adversarial robustness test results
        """

        logger.info("Testing adversarial robustness")

        results = {
            'test_type': 'adversarial_robustness',
            'baseline_accuracy': model.score(X, y),
            'attack_results': {}
        }

        for i, attack_generator in enumerate(attack_generators):
            attack_name = f"attack_{i}"
            logger.info(f"Testing adversarial attack: {attack_name}")

            try:
                # Generate adversarial examples
                X_adv = attack_generator(model, X.values, y.values)

                # Test model performance
                accuracy = model.score(X_adv, y)
                accuracy_drop = results['baseline_accuracy'] - accuracy

                results['attack_results'][attack_name] = {
                    'accuracy': accuracy,
                    'accuracy_drop': accuracy_drop,
                    'relative_drop': accuracy_drop / results['baseline_accuracy'] if results['baseline_accuracy'] > 0 else 0,
                    'attack_successful': accuracy_drop > 0.05  # Threshold for successful attack
                }

            except Exception as e:
                logger.error(f"Attack {attack_name} failed: {e}")
                results['attack_results'][attack_name] = {
                    'error': str(e),
                    'attack_successful': False
                }

        # Calculate robustness score
        results['robustness_score'] = self._calculate_adversarial_robustness_score(
            results)

        self.test_results.append(results)
        logger.info(
            f"Adversarial robustness test completed. Score: {results['robustness_score']:.4f}")

        return results

    def test_comprehensive_robustness(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                                      attack_generators: Optional[List[Callable]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive robustness testing.

        Args:
            model: Model to test
            X: Features
            y: Labels
            attack_generators: Optional list of attack generators

        Returns:
            Comprehensive robustness test results
        """

        logger.info("Performing comprehensive robustness testing")

        comprehensive_results = {
            'model_type': type(model).__name__,
            'n_samples': len(X),
            'n_features': len(X.columns),
            'baseline_accuracy': model.score(X, y),
            'test_timestamp': pd.Timestamp.now().isoformat()
        }

        # Test noise robustness
        noise_results = self.test_noise_robustness(model, X, y)
        comprehensive_results['noise_robustness'] = noise_results

        # Test outlier robustness
        outlier_results = self.test_outlier_robustness(model, X, y)
        comprehensive_results['outlier_robustness'] = outlier_results

        # Test missing value robustness
        missing_results = self.test_missing_value_robustness(model, X, y)
        comprehensive_results['missing_value_robustness'] = missing_results

        # Test feature corruption robustness
        corruption_results = self.test_feature_corruption_robustness(
            model, X, y)
        comprehensive_results['feature_corruption_robustness'] = corruption_results

        # Test adversarial robustness if generators provided
        if attack_generators:
            adv_results = self.test_adversarial_robustness(
                model, X, y, attack_generators)
            comprehensive_results['adversarial_robustness'] = adv_results

        # Calculate overall robustness score
        comprehensive_results['overall_robustness_score'] = self._calculate_overall_robustness_score(
            comprehensive_results)

        # Store in history
        self.robustness_history.append(comprehensive_results)

        logger.info(
            f"Comprehensive robustness testing completed. Overall score: {comprehensive_results['overall_robustness_score']:.4f}")

        return comprehensive_results

    def _add_gaussian_noise(self, X: pd.DataFrame, noise_level: float) -> pd.DataFrame:
        """Add Gaussian noise to features."""

        X_noisy = X.copy()
        for column in X.columns:
            if X[column].dtype in ['float64', 'int64']:
                noise = np.random.normal(
                    0, noise_level * X[column].std(), len(X))
                X_noisy[column] = X[column] + noise

        return X_noisy

    def _add_outliers(self, X: pd.DataFrame, outlier_pct: float) -> pd.DataFrame:
        """Add outliers to features."""

        X_outliers = X.copy()
        n_outliers = int(len(X) * outlier_pct)

        for column in X.columns:
            if X[column].dtype in ['float64', 'int64']:
                # Select random samples to make outliers
                outlier_indices = np.random.choice(
                    len(X), n_outliers, replace=False)

                # Create outliers by multiplying by large factor
                outlier_factor = np.random.choice([-10, 10], n_outliers)
                X_outliers.loc[outlier_indices,
                               column] = X.loc[outlier_indices, column] * outlier_factor

        return X_outliers

    def _add_missing_values(self, X: pd.DataFrame, missing_pct: float) -> pd.DataFrame:
        """Add missing values to features."""

        X_missing = X.copy()
        n_missing = int(len(X) * missing_pct)

        for column in X.columns:
            # Select random samples to make missing
            missing_indices = np.random.choice(
                len(X), n_missing, replace=False)
            X_missing.loc[missing_indices, column] = np.nan

        return X_missing

    def _corrupt_features(self, X: pd.DataFrame, corruption_level: float) -> pd.DataFrame:
        """Corrupt features by swapping values."""

        X_corrupted = X.copy()
        n_corruptions = int(len(X) * corruption_level)

        for column in X.columns:
            # Select random samples to corrupt
            corruption_indices = np.random.choice(
                len(X), n_corruptions, replace=False)

            # Swap values randomly
            for idx in corruption_indices:
                swap_idx = np.random.choice(len(X))
                X_corrupted.loc[idx, column], X_corrupted.loc[swap_idx, column] = \
                    X_corrupted.loc[swap_idx,
                                    column], X_corrupted.loc[idx, column]

        return X_corrupted

    def _calculate_noise_robustness_score(self, results: Dict[str, Any]) -> float:
        """Calculate noise robustness score."""

        noise_results = results['noise_results']
        scores = []

        for noise_level, metrics in noise_results.items():
            # Higher noise level should have lower impact for robust models
            robustness = 1 - metrics['relative_drop']
            # Weight by noise level (higher noise = more important)
            weighted_score = robustness * noise_level
            scores.append(weighted_score)

        return np.mean(scores) if scores else 0

    def _calculate_outlier_robustness_score(self, results: Dict[str, Any]) -> float:
        """Calculate outlier robustness score."""

        outlier_results = results['outlier_results']
        scores = []

        for outlier_pct, metrics in outlier_results.items():
            robustness = 1 - metrics['relative_drop']
            # Weight by outlier percentage
            weighted_score = robustness * outlier_pct
            scores.append(weighted_score)

        return np.mean(scores) if scores else 0

    def _calculate_missing_value_robustness_score(self, results: Dict[str, Any]) -> float:
        """Calculate missing value robustness score."""

        missing_results = results['missing_results']
        scores = []

        for missing_pct, metrics in missing_results.items():
            if not metrics['model_failed']:
                robustness = 1 - metrics['relative_drop']
                weighted_score = robustness * missing_pct
                scores.append(weighted_score)

        return np.mean(scores) if scores else 0

    def _calculate_corruption_robustness_score(self, results: Dict[str, Any]) -> float:
        """Calculate corruption robustness score."""

        corruption_results = results['corruption_results']
        scores = []

        for corruption_level, metrics in corruption_results.items():
            robustness = 1 - metrics['relative_drop']
            weighted_score = robustness * corruption_level
            scores.append(weighted_score)

        return np.mean(scores) if scores else 0

    def _calculate_adversarial_robustness_score(self, results: Dict[str, Any]) -> float:
        """Calculate adversarial robustness score."""

        attack_results = results['attack_results']
        scores = []

        for attack_name, metrics in attack_results.items():
            if 'error' not in metrics:
                robustness = 1 - metrics['relative_drop']
                scores.append(robustness)

        return np.mean(scores) if scores else 0

    def _calculate_overall_robustness_score(self, comprehensive_results: Dict[str, Any]) -> float:
        """Calculate overall robustness score."""

        scores = []

        # Collect individual robustness scores
        for test_type in ['noise_robustness', 'outlier_robustness', 'missing_value_robustness',
                          'feature_corruption_robustness', 'adversarial_robustness']:
            if test_type in comprehensive_results:
                score = comprehensive_results[test_type]['robustness_score']
                scores.append(score)

        return np.mean(scores) if scores else 0

    def get_robustness_summary(self) -> Dict[str, Any]:
        """Get comprehensive robustness testing summary."""

        if not self.robustness_history:
            return {'message': 'No robustness tests performed yet'}

        # Calculate summary statistics
        overall_scores = [result['overall_robustness_score']
                          for result in self.robustness_history]

        summary = {
            'total_tests': len(self.robustness_history),
            'mean_overall_score': np.mean(overall_scores),
            'std_overall_score': np.std(overall_scores),
            'min_overall_score': np.min(overall_scores),
            'max_overall_score': np.max(overall_scores),
            'test_history': self.robustness_history
        }

        return summary

    def compare_model_robustness(self, models: Dict[str, BaseEstimator], X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Compare robustness across multiple models."""

        logger.info(f"Comparing robustness across {len(models)} models")

        comparison_results = {}

        for model_name, model in models.items():
            logger.info(f"Testing robustness for {model_name}")

            try:
                results = self.test_comprehensive_robustness(model, X, y)
                comparison_results[model_name] = {
                    'overall_robustness_score': results['overall_robustness_score'],
                    'baseline_accuracy': results['baseline_accuracy'],
                    'noise_robustness': results['noise_robustness']['robustness_score'],
                    'outlier_robustness': results['outlier_robustness']['robustness_score'],
                    'missing_value_robustness': results['missing_value_robustness']['robustness_score'],
                    'feature_corruption_robustness': results['feature_corruption_robustness']['robustness_score']
                }
            except Exception as e:
                logger.error(f"Error testing {model_name}: {e}")
                comparison_results[model_name] = {'error': str(e)}

        # Find most robust model
        valid_models = {k: v for k,
                        v in comparison_results.items() if 'error' not in v}
        if valid_models:
            most_robust = max(valid_models.keys(),
                              key=lambda x: valid_models[x]['overall_robustness_score'])
            comparison_results['most_robust_model'] = most_robust

        logger.info(
            f"Model robustness comparison completed. Most robust: {comparison_results.get('most_robust_model', 'N/A')}")

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

    # Create a simple model
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    print("Testing Robustness Tester:")

    # Initialize robustness tester
    tester = RobustnessTester({
        'noise_levels': [0.01, 0.05, 0.1],
        'outlier_percentages': [0.01, 0.05],
        'missing_value_percentages': [0.01, 0.05],
        'feature_corruption_levels': [0.01, 0.05]
    })

    # Test individual robustness types
    print("\n1. Testing Noise Robustness:")
    noise_results = tester.test_noise_robustness(model, X, y)
    print(f"Noise robustness score: {noise_results['robustness_score']:.4f}")

    print("\n2. Testing Outlier Robustness:")
    outlier_results = tester.test_outlier_robustness(model, X, y)
    print(
        f"Outlier robustness score: {outlier_results['robustness_score']:.4f}")

    print("\n3. Testing Missing Value Robustness:")
    missing_results = tester.test_missing_value_robustness(model, X, y)
    print(
        f"Missing value robustness score: {missing_results['robustness_score']:.4f}")

    print("\n4. Testing Feature Corruption Robustness:")
    corruption_results = tester.test_feature_corruption_robustness(model, X, y)
    print(
        f"Feature corruption robustness score: {corruption_results['robustness_score']:.4f}")

    # Test comprehensive robustness
    print("\n5. Testing Comprehensive Robustness:")
    comprehensive_results = tester.test_comprehensive_robustness(model, X, y)
    print(
        f"Overall robustness score: {comprehensive_results['overall_robustness_score']:.4f}")

    # Test adversarial robustness
    def simple_attack_generator(model, X_test, y_test):
        # Simple adversarial example generator
        noise = np.random.normal(0, 0.1, X_test.shape)
        return X_test + noise

    print("\n6. Testing Adversarial Robustness:")
    adv_results = tester.test_adversarial_robustness(
        model, X, y, [simple_attack_generator])
    print(
        f"Adversarial robustness score: {adv_results['robustness_score']:.4f}")

    # Test model comparison
    print("\n7. Testing Model Comparison:")
    from sklearn.ensemble import RandomForestClassifier
    models = {
        'LogisticRegression': LogisticRegression(random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=10, random_state=42)
    }

    for model_name, model in models.items():
        model.fit(X, y)

    comparison = tester.compare_model_robustness(models, X, y)
    print(f"Most robust model: {comparison.get('most_robust_model', 'N/A')}")

    # Get robustness summary
    summary = tester.get_robustness_summary()
    print(f"\nRobustness Summary:")
    print(f"Total tests: {summary['total_tests']}")
    print(f"Mean overall score: {summary['mean_overall_score']:.4f}")
