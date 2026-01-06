"""
Attack Generator Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

OPTIMIZED VERSION with reduced attack combinations for faster evaluation.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.base import BaseEstimator
import warnings

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
        self.enabled_attacks = self.config.get('enabled_attacks', ['fgsm', 'pgd'])  # Removed 'cw' for speed
        
        # OPTIMIZED: Reduced configurations for faster evaluation
        default_fgsm_config = {
            'epsilon_range': [0.1, 0.2],  # Reduced from 5 to 2 values
            'norms': ['inf']  # Only L-inf for speed
        }
        
        default_pgd_config = {
            'epsilon_range': [0.1],  # Only 1 value
            'alpha_range': [0.01],  # Only 1 value
            'num_iter_range': [10],  # Only 1 value
            'norms': ['inf']  # Only L-inf for speed
        }
        
        default_cw_config = {
            'c_range': [1.0],  # Only 1 value
            'max_iter_range': [100],  # Reduced iterations
            'norms': ['2']  # Only L2 for C&W
        }

        self.attack_generators = {}

        if 'fgsm' in self.enabled_attacks:
            fgsm_config = self.attack_configs.get('fgsm', default_fgsm_config)
            self.attack_generators['fgsm'] = FGSMAttackGenerator(fgsm_config)

        if 'pgd' in self.enabled_attacks:
            pgd_config = self.attack_configs.get('pgd', default_pgd_config)
            self.attack_generators['pgd'] = PGDAttackGenerator(pgd_config)

        if 'cw' in self.enabled_attacks:
            cw_config = self.attack_configs.get('cw', default_cw_config)
            self.attack_generators['cw'] = CWAttackGenerator(cw_config)

        logger.info(f"AdversarialAttackGenerator initialized with attacks: {self.enabled_attacks}")
        logger.info(f"OPTIMIZED MODE: Using reduced attack configurations for speed")

    def generate_all_attacks(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Generate all enabled types of adversarial attacks."""
        logger.info(f"Generating all adversarial attacks on {len(X)} samples")
        
        # Adaptive sampling based on dataset size
        if len(X) > 10000:
            max_samples = 3000  # Large dataset - use 3000
        elif len(X) > 5000:
            max_samples = 2000  # Medium dataset - use 2000
        else:
            max_samples = len(X)  # Small dataset - use all
        
        if len(X) > max_samples:
            logger.info(f"Sampling {max_samples} from {len(X)} for adversarial attack generation")
            indices = np.random.choice(len(X), max_samples, replace=False)
            # Handle both DataFrame and numpy array inputs
            if hasattr(X, 'iloc'):
                X_sample = X.iloc[indices].values  # Convert to numpy for consistency
                y_sample = y.iloc[indices].values if hasattr(y, 'iloc') else y[indices]
            else:
                X_sample = X[indices]
                y_sample = y[indices]
        else:
            # Ensure numpy arrays for consistency
            X_sample = X.values if hasattr(X, 'values') else X
            y_sample = y.values if hasattr(y, 'values') else y

        all_results = {}

        for attack_type, generator in self.attack_generators.items():
            try:
                logger.info(f"Generating {attack_type.upper()} attacks...")

                if attack_type == 'fgsm':
                    results = generator.generate_multiple_attacks(model, X_sample, y_sample)
                elif attack_type == 'pgd':
                    results = generator.generate_multiple_attacks(model, X_sample, y_sample)
                elif attack_type == 'cw':
                    results = generator.generate_multiple_attacks(model, X_sample, y_sample)
                else:
                    logger.warning(f"Unknown attack type: {attack_type}")
                    continue

                all_results[attack_type] = results
                
                # Count successful attacks
                successful_count = sum(1 for r in results.values() if 'error' not in r)
                logger.info(f"Generated {successful_count}/{len(results)} successful {attack_type.upper()} attack variants")

            except Exception as e:
                logger.error(f"Failed to generate {attack_type} attacks: {e}")
                all_results[attack_type] = {'error': str(e)}

        return all_results

    def generate_single_attack(self, attack_type: str, model: BaseEstimator,
                               X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Generate a single type of adversarial attack."""
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
                'adversarial_examples': X_adv,  # Full array for adversarial training
                'adversarial_examples_sample': X_adv[:50],  # Sample for display
                'evaluation': results
            }

        except Exception as e:
            logger.error(f"Failed to generate {attack_type} attack: {e}")
            return {'error': str(e)}

    def evaluate_robustness(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate model robustness against all attack types."""
        logger.info("Evaluating model robustness against adversarial attacks")

        # Generate all attacks
        attack_results = self.generate_all_attacks(model, X, y)

        # Calculate robustness metrics
        robustness_metrics = {}

        for attack_type, results in attack_results.items():
            if 'error' in results:
                logger.warning(f"Skipping {attack_type} due to error: {results['error']}")
                continue

            success_rates = []
            accuracy_drops = []
            perturbation_norms = []

            for variant_name, variant_result in results.items():
                if 'error' in variant_result:
                    continue

                eval_results = variant_result['evaluation']
                success_rates.append(eval_results['success_rate'])
                accuracy_drops.append(eval_results['accuracy_drop'])
                perturbation_norms.append(eval_results['mean_perturbation_norm'])

            if success_rates:
                robustness_metrics[attack_type] = {
                    'mean_success_rate': np.mean(success_rates),
                    'max_success_rate': np.max(success_rates),
                    'mean_accuracy_drop': np.mean(accuracy_drops),
                    'max_accuracy_drop': np.max(accuracy_drops),
                    'mean_perturbation_norm': np.mean(perturbation_norms),
                    'n_variants': len(success_rates)
                }

        # Overall robustness score (1.0 = perfectly robust, 0.0 = completely vulnerable)
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

        logger.info(f"Robustness evaluation completed. Overall robustness: {overall_robustness:.4f}")
        
        # Log per-attack robustness
        for attack_type, metrics in robustness_metrics.items():
            logger.info(f"  {attack_type.upper()}: mean_success_rate={metrics['mean_success_rate']:.4f}, "
                       f"mean_accuracy_drop={metrics['mean_accuracy_drop']:.4f}")

        return evaluation_results

    def find_weakest_attack(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Find the weakest attack that still achieves significant success."""
        logger.info("Finding weakest effective attack")

        attack_results = self.generate_all_attacks(model, X, y)

        weakest_attack = None
        min_perturbation = float('inf')
        min_success_rate = 0.3

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
        """Compare different attack methods."""
        logger.info("Comparing attack methods")

        comparison_results = {}

        # FGSM
        if 'fgsm' in self.enabled_attacks:
            fgsm_result = self.generate_single_attack('fgsm', model, X, y, epsilon=0.1, norm='inf')
            if 'error' not in fgsm_result:
                comparison_results['fgsm'] = fgsm_result['evaluation']

        # PGD
        if 'pgd' in self.enabled_attacks:
            pgd_result = self.generate_single_attack('pgd', model, X, y, epsilon=0.1, alpha=0.01, 
                                                      num_iter=10, norm='inf')
            if 'error' not in pgd_result:
                comparison_results['pgd'] = pgd_result['evaluation']

        # C&W
        if 'cw' in self.enabled_attacks:
            cw_result = self.generate_single_attack('cw', model, X, y, c=1.0, max_iter=100, norm='2')
            if 'error' not in cw_result:
                comparison_results['cw'] = cw_result['evaluation']

        # Rank attacks
        if comparison_results:
            ranking = {
                'success_rate': sorted(comparison_results.items(), 
                                      key=lambda x: x[1]['success_rate'], reverse=True),
                'accuracy_drop': sorted(comparison_results.items(), 
                                       key=lambda x: x[1]['accuracy_drop'], reverse=True),
                'perturbation_efficiency': sorted(comparison_results.items(),
                    key=lambda x: x[1]['success_rate'] / (x[1]['mean_perturbation_norm'] + 1e-8),
                    reverse=True)
            }
        else:
            ranking = {}

        comparison = {
            'attack_results': comparison_results,
            'ranking': ranking,
            'summary': {
                'best_success_rate': max([r['success_rate'] for r in comparison_results.values()]) 
                                    if comparison_results else 0,
                'worst_success_rate': min([r['success_rate'] for r in comparison_results.values()]) 
                                     if comparison_results else 0,
                'n_methods': len(comparison_results)
            }
        }

        logger.info(f"Attack comparison completed. {len(comparison_results)} methods compared")

        return comparison

    def get_attack_statistics(self, attack_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive statistics from attack results."""
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
                statistics['attack_types'][attack_type] = {'error': results['error']}
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
                    type_stats['success_rates'].append(eval_results['success_rate'])
                    type_stats['accuracy_drops'].append(eval_results['accuracy_drop'])

                    all_success_rates.append(eval_results['success_rate'])
                    all_accuracy_drops.append(eval_results['accuracy_drop'])

            if type_stats['success_rates']:
                type_stats['mean_success_rate'] = np.mean(type_stats['success_rates'])
                type_stats['mean_accuracy_drop'] = np.mean(type_stats['accuracy_drops'])

            statistics['attack_types'][attack_type] = type_stats

        if all_success_rates:
            statistics['overall_success_rate'] = np.mean(all_success_rates)
            statistics['overall_accuracy_drop'] = np.mean(all_accuracy_drops)

        return statistics