"""
Attack Generator Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Unified interface for generating various types of adversarial attacks.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.base import BaseEstimator
import warnings

# Import attack implementations
from .fgsm_attack import FGSMAttack, FGSMAttackGenerator
from .pgd_attack import PGDAttack, PGDAttackGenerator
from .cw_attack import CWAttack, CWAttackGenerator

logger = logging.getLogger(__name__)


class AdversarialAttackGenerator:
    """Unified interface for generating adversarial attacks."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize attack generator with configuration."""

        self.config = config or {}
        self.attack_configs = self.config.get('attacks', {})
        self.enabled_attacks = self.config.get(
            'enabled_attacks', ['fgsm', 'pgd', 'cw'])

        # Initialize attack generators
        self.attack_generators = {}

        if 'fgsm' in self.enabled_attacks:
            self.attack_generators['fgsm'] = FGSMAttackGenerator(
                self.attack_configs.get('fgsm', {})
            )

        if 'pgd' in self.enabled_attacks:
            self.attack_generators['pgd'] = PGDAttackGenerator(
                self.attack_configs.get('pgd', {})
            )

        if 'cw' in self.enabled_attacks:
            self.attack_generators['cw'] = CWAttackGenerator(
                self.attack_configs.get('cw', {})
            )

        logger.info(
            f"AdversarialAttackGenerator initialized with attacks: {self.enabled_attacks}")

    def generate_all_attacks(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Generate all enabled types of adversarial attacks.

        Args:
            model: Target model
            X: Input features
            y: True labels

        Returns:
            Dictionary of attack results
        """

        logger.info(f"Generating all adversarial attacks on {len(X)} samples")

        all_results = {}

        for attack_type, generator in self.attack_generators.items():
            try:
                logger.info(f"Generating {attack_type.upper()} attacks...")

                if attack_type == 'fgsm':
                    results = generator.generate_multiple_attacks(model, X, y)
                elif attack_type == 'pgd':
                    results = generator.generate_multiple_attacks(model, X, y)
                elif attack_type == 'cw':
                    results = generator.generate_multiple_attacks(model, X, y)
                else:
                    logger.warning(f"Unknown attack type: {attack_type}")
                    continue

                all_results[attack_type] = results
                logger.info(
                    f"Generated {len(results)} {attack_type.upper()} attack variants")

            except Exception as e:
                logger.error(f"Failed to generate {attack_type} attacks: {e}")
                all_results[attack_type] = {'error': str(e)}

        return all_results

    def generate_single_attack(self, attack_type: str, model: BaseEstimator,
                               X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Generate a single type of adversarial attack.

        Args:
            attack_type: Type of attack ('fgsm', 'pgd', 'cw')
            model: Target model
            X: Input features
            y: True labels
            **kwargs: Additional parameters for the attack

        Returns:
            Attack results
        """

        if attack_type not in self.attack_generators:
            raise ValueError(f"Attack type {attack_type} not enabled")

        logger.info(f"Generating {attack_type.upper()} attack")

        try:
            if attack_type == 'fgsm':
                attack = FGSMAttack(kwargs)
                X_adv = attack.generate_attack(model, X, y)
                results = attack.evaluate_attack(model, X, y, X_adv)

            elif attack_type == 'pgd':
                attack = PGDAttack(kwargs)
                X_adv = attack.generate_attack(model, X, y)
                results = attack.evaluate_attack(model, X, y, X_adv)

            elif attack_type == 'cw':
                attack = CWAttack(kwargs)
                X_adv = attack.generate_attack(model, X, y)
                results = attack.evaluate_attack(model, X, y, X_adv)

            else:
                raise ValueError(f"Unknown attack type: {attack_type}")

            return {
                'adversarial_examples': X_adv,
                'evaluation': results
            }

        except Exception as e:
            logger.error(f"Failed to generate {attack_type} attack: {e}")
            return {'error': str(e)}

    def evaluate_robustness(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model robustness against all attack types.

        Args:
            model: Target model
            X: Input features
            y: True labels

        Returns:
            Comprehensive robustness evaluation
        """

        logger.info("Evaluating model robustness against adversarial attacks")

        # Generate all attacks
        attack_results = self.generate_all_attacks(model, X, y)

        # Calculate robustness metrics
        robustness_metrics = {}

        for attack_type, results in attack_results.items():
            if 'error' in results:
                continue

            # Aggregate metrics across all variants
            success_rates = []
            accuracy_drops = []
            perturbation_norms = []

            for variant_name, variant_result in results.items():
                if 'error' in variant_result:
                    continue

                eval_results = variant_result['evaluation']
                success_rates.append(eval_results['success_rate'])
                accuracy_drops.append(eval_results['accuracy_drop'])
                perturbation_norms.append(
                    eval_results['mean_perturbation_norm'])

            if success_rates:
                robustness_metrics[attack_type] = {
                    'mean_success_rate': np.mean(success_rates),
                    'max_success_rate': np.max(success_rates),
                    'mean_accuracy_drop': np.mean(accuracy_drops),
                    'max_accuracy_drop': np.max(accuracy_drops),
                    'mean_perturbation_norm': np.mean(perturbation_norms),
                    'n_variants': len(success_rates)
                }

        # Overall robustness score
        if robustness_metrics:
            overall_robustness = 1.0 - np.mean([
                metrics['mean_success_rate'] for metrics in robustness_metrics.values()
            ])
        else:
            overall_robustness = 0.0

        evaluation_results = {
            'overall_robustness': overall_robustness,
            'attack_metrics': robustness_metrics,
            'detailed_results': attack_results,
            'n_attack_types': len(robustness_metrics),
            'n_samples': len(X)
        }

        logger.info(
            f"Robustness evaluation completed. Overall robustness: {overall_robustness:.4f}")

        return evaluation_results

    def find_weakest_attack(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Find the weakest attack that still achieves significant success.

        Args:
            model: Target model
            X: Input features
            y: True labels

        Returns:
            Information about the weakest effective attack
        """

        logger.info("Finding weakest effective attack")

        # Generate all attacks
        attack_results = self.generate_all_attacks(model, X, y)

        weakest_attack = None
        min_perturbation = float('inf')
        min_success_rate = 0.3  # Minimum success rate to consider

        for attack_type, results in attack_results.items():
            if 'error' in results:
                continue

            for variant_name, variant_result in results.items():
                if 'error' in variant_result:
                    continue

                eval_results = variant_result['evaluation']

                if (eval_results['success_rate'] >= min_success_rate and
                        eval_results['mean_perturbation_norm'] < min_perturbation):

                    min_perturbation = eval_results['mean_perturbation_norm']
                    weakest_attack = {
                        'attack_type': attack_type,
                        'variant_name': variant_name,
                        'success_rate': eval_results['success_rate'],
                        'perturbation_norm': eval_results['mean_perturbation_norm'],
                        'accuracy_drop': eval_results['accuracy_drop'],
                        'parameters': eval_results
                    }

        if weakest_attack is None:
            logger.warning("No effective attack found")
            return {}

        logger.info(f"Weakest attack: {weakest_attack['attack_type']} "
                    f"(success_rate={weakest_attack['success_rate']:.4f}, "
                    f"perturbation={weakest_attack['perturbation_norm']:.4f})")

        return weakest_attack

    def compare_attack_methods(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Compare different attack methods.

        Args:
            model: Target model
            X: Input features
            y: True labels

        Returns:
            Comparison results
        """

        logger.info("Comparing attack methods")

        # Generate single attacks of each type with standard parameters
        comparison_results = {}

        # FGSM
        if 'fgsm' in self.enabled_attacks:
            fgsm_result = self.generate_single_attack(
                'fgsm', model, X, y, epsilon=0.1, norm='inf'
            )
            if 'error' not in fgsm_result:
                comparison_results['fgsm'] = fgsm_result['evaluation']

        # PGD
        if 'pgd' in self.enabled_attacks:
            pgd_result = self.generate_single_attack(
                'pgd', model, X, y, epsilon=0.1, alpha=0.01, num_iter=10, norm='inf'
            )
            if 'error' not in pgd_result:
                comparison_results['pgd'] = pgd_result['evaluation']

        # C&W
        if 'cw' in self.enabled_attacks:
            cw_result = self.generate_single_attack(
                'cw', model, X, y, c=1.0, max_iter=500, norm='2'
            )
            if 'error' not in cw_result:
                comparison_results['cw'] = cw_result['evaluation']

        # Rank attacks by effectiveness
        if comparison_results:
            ranking = {
                'success_rate': sorted(
                    comparison_results.items(),
                    key=lambda x: x[1]['success_rate'],
                    reverse=True
                ),
                'accuracy_drop': sorted(
                    comparison_results.items(),
                    key=lambda x: x[1]['accuracy_drop'],
                    reverse=True
                ),
                'perturbation_efficiency': sorted(
                    comparison_results.items(),
                    key=lambda x: x[1]['success_rate'] /
                    (x[1]['mean_perturbation_norm'] + 1e-8),
                    reverse=True
                )
            }
        else:
            ranking = {}

        comparison = {
            'attack_results': comparison_results,
            'ranking': ranking,
            'summary': {
                'best_success_rate': max(
                    [r['success_rate'] for r in comparison_results.values()]
                ) if comparison_results else 0,
                'worst_success_rate': min(
                    [r['success_rate'] for r in comparison_results.values()]
                ) if comparison_results else 0,
                'n_methods': len(comparison_results)
            }
        }

        logger.info(
            f"Attack comparison completed. {len(comparison_results)} methods compared")

        return comparison

    def get_attack_statistics(self, attack_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive statistics from attack results.

        Args:
            attack_results: Results from generate_all_attacks

        Returns:
            Attack statistics
        """

        statistics = {
            'total_attacks': 0,
            'successful_attacks': 0,
            'failed_attacks': 0,
            'attack_types': {},
            'overall_success_rate': 0,
            'overall_accuracy_drop': 0
        }

        all_success_rates = []
        all_accuracy_drops = []

        for attack_type, results in attack_results.items():
            if 'error' in results:
                statistics['attack_types'][attack_type] = {
                    'error': results['error']}
                continue

            type_stats = {
                'variants': len(results),
                'successful_variants': 0,
                'failed_variants': 0,
                'success_rates': [],
                'accuracy_drops': []
            }

            for variant_name, variant_result in results.items():
                statistics['total_attacks'] += 1

                if 'error' in variant_result:
                    type_stats['failed_variants'] += 1
                    statistics['failed_attacks'] += 1
                else:
                    type_stats['successful_variants'] += 1
                    statistics['successful_attacks'] += 1

                    eval_results = variant_result['evaluation']
                    type_stats['success_rates'].append(
                        eval_results['success_rate'])
                    type_stats['accuracy_drops'].append(
                        eval_results['accuracy_drop'])

                    all_success_rates.append(eval_results['success_rate'])
                    all_accuracy_drops.append(eval_results['accuracy_drop'])

            if type_stats['success_rates']:
                type_stats['mean_success_rate'] = np.mean(
                    type_stats['success_rates'])
                type_stats['mean_accuracy_drop'] = np.mean(
                    type_stats['accuracy_drops'])

            statistics['attack_types'][attack_type] = type_stats

        if all_success_rates:
            statistics['overall_success_rate'] = np.mean(all_success_rates)
            statistics['overall_accuracy_drop'] = np.mean(all_accuracy_drops)

        return statistics


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

    # Create labels
    y = pd.Series(
        (X.iloc[:, 0] + X.iloc[:, 1] +
         np.random.randn(n_samples) * 0.1 > 0).astype(int)
    )

    # Create a simple model
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    print("Original model accuracy:", model.score(X, y))

    # Initialize attack generator
    generator = AdversarialAttackGenerator({
        'enabled_attacks': ['fgsm', 'pgd', 'cw'],
        'attacks': {
            'fgsm': {'epsilon_range': [0.05, 0.1, 0.2]},
            'pgd': {'epsilon_range': [0.05, 0.1], 'num_iter_range': [5, 10]},
            'cw': {'c_range': [0.5, 1.0], 'max_iter_range': [100, 500]}
        }
    })

    # Generate all attacks
    all_results = generator.generate_all_attacks(model, X.values, y.values)
    print(f"\nGenerated attacks for {len(all_results)} attack types")

    # Evaluate robustness
    robustness_eval = generator.evaluate_robustness(model, X.values, y.values)
    print(f"\nOverall robustness: {robustness_eval['overall_robustness']:.4f}")

    # Find weakest attack
    weakest = generator.find_weakest_attack(model, X.values, y.values)
    if weakest:
        print(f"\nWeakest attack: {weakest['attack_type']} "
              f"(success_rate={weakest['success_rate']:.4f})")

    # Compare attack methods
    comparison = generator.compare_attack_methods(model, X.values, y.values)
    print("\nAttack Method Comparison:")
    for method, results in comparison['attack_results'].items():
        print(f"  {method.upper()}: success_rate={results['success_rate']:.4f}, "
              f"accuracy_drop={results['accuracy_drop']:.4f}")

    # Get attack statistics
    stats = generator.get_attack_statistics(all_results)
    print(f"\nAttack Statistics:")
    print(f"  Total attacks: {stats['total_attacks']}")
    print(f"  Successful: {stats['successful_attacks']}")
    print(f"  Failed: {stats['failed_attacks']}")
    print(f"  Overall success rate: {stats['overall_success_rate']:.4f}")
