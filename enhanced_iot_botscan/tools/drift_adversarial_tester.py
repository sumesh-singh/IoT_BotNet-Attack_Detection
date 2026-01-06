"""
Comprehensive Testing Framework for Drift Detection and Adversarial Robustness
Works without PyTorch - uses the trained hybrid ensemble model
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

logger = logging.getLogger(__name__)


class DriftAdversarialTester:
    """Comprehensive testing for drift and adversarial scenarios"""
    
    def __init__(self, model_path: str = "models/hybrid_ensemble.joblib"):
        """
        Args:
            model_path: Path to trained model
        """
        self.model_path = Path(model_path)
        self.model = None
        self.drift_detector = None
        self.baseline_metrics = {}
        self.test_results = {}
        
    def load_model(self):
        """Load trained hybrid ensemble model"""
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")
    
    def set_baseline(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Establish baseline performance on clean data
        
        Args:
            X_test: Clean test features
            y_test: True labels
        """
        logger.info("Establishing baseline performance...")
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.baseline_metrics = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'true_labels': y_test,
            'X_test': X_test
        }
        
        logger.info(f"Baseline accuracy: {accuracy:.4f}")
        print(f"\n{'='*60}")
        print(f"BASELINE PERFORMANCE")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
    def test_covariate_drift(self, drift_magnitudes: List[float] = [0.1, 0.2, 0.3, 0.5]):
        """
        Test model under covariate shift
        
        Args:
            drift_magnitudes: List of drift strengths to test
        """
        logger.info("Testing covariate drift scenarios...")
        
        X_clean = self.baseline_metrics['X_test']
        y_true = self.baseline_metrics['true_labels']
        
        results = []
        
        for magnitude in drift_magnitudes:
            # Generate drifted data
            X_drift = self._generate_covariate_drift(X_clean, magnitude)
            
            # Predict on drifted data
            y_pred = self.model.predict(X_drift)
            accuracy = accuracy_score(y_true, y_pred)
            
            # Calculate drift metrics
            drift_score = self._calculate_drift_score(X_clean, X_drift)
            
            results.append({
                'magnitude': magnitude,
                'accuracy': accuracy,
                'accuracy_drop': self.baseline_metrics['accuracy'] - accuracy,
                'drift_score': drift_score
            })
            
            logger.info(f"Drift magnitude {magnitude}: Accuracy={accuracy:.4f}, Drop={self.baseline_metrics['accuracy'] - accuracy:.4f}")
        
        self.test_results['covariate_drift'] = pd.DataFrame(results)
        
        print(f"\n{'='*60}")
        print(f"COVARIATE DRIFT ANALYSIS")
        print(f"{'='*60}")
        print(self.test_results['covariate_drift'].to_string(index=False))
        
    def test_label_drift(self, shift_ratios: List[float] = [0.1, 0.2, 0.3, 0.5]):
        """
        Test model under label shift
        
        Args:
            shift_ratios: Ratios of labels to shift
        """
        logger.info("Testing label drift scenarios...")
        
        X_clean = self.baseline_metrics['X_test']
        y_true = self.baseline_metrics['true_labels']
        
        results = []
        
        for ratio in shift_ratios:
            # Generate label shift
            y_shifted = self._generate_label_drift(y_true, ratio)
            
            # Predict on clean data but evaluate against shifted labels
            y_pred = self.model.predict(X_clean)
            accuracy = accuracy_score(y_shifted, y_pred)
            
            results.append({
                'shift_ratio': ratio,
                'accuracy': accuracy,
                'accuracy_drop': self.baseline_metrics['accuracy'] - accuracy
            })
            
            logger.info(f"Label shift {ratio}: Accuracy={accuracy:.4f}")
        
        self.test_results['label_drift'] = pd.DataFrame(results)
        
        print(f"\n{'='*60}")
        print(f"LABEL DRIFT ANALYSIS")
        print(f"{'='*60}")
        print(self.test_results['label_drift'].to_string(index=False))
    
    def test_adversarial_robustness(self, attack_types: List[str] = ['fgsm', 'noise', 'combined']):
        """
        Test model robustness against adversarial attacks
        
        Args:
            attack_types: Types of attacks to simulate
        """
        logger.info("Testing adversarial robustness...")
        
        X_clean = self.baseline_metrics['X_test']
        y_true = self.baseline_metrics['true_labels']
        
        results = []
        
        for attack in attack_types:
            # Generate adversarial examples
            if attack == 'fgsm':
                X_adv = self._fgsm_attack(X_clean, epsilon=0.1)
            elif attack == 'noise':
                X_adv = self._noise_attack(X_clean, noise_level=0.2)
            elif attack == 'combined':
                X_adv = self._combined_attack(X_clean)
            else:
                continue
            
            # Predict on adversarial examples
            y_pred = self.model.predict(X_adv)
            accuracy = accuracy_score(y_true, y_pred)
            
            # Calculate perturbation magnitude
            perturbation = np.mean(np.abs(X_adv.values - X_clean.values))
            
            results.append({
                'attack_type': attack,
                'accuracy': accuracy,
                'accuracy_drop': self.baseline_metrics['accuracy'] - accuracy,
                'avg_perturbation': perturbation
            })
            
            logger.info(f"{attack.upper()} attack: Accuracy={accuracy:.4f}, Drop={self.baseline_metrics['accuracy'] - accuracy:.4f}")
        
        self.test_results['adversarial'] = pd.DataFrame(results)
        
        print(f"\n{'='*60}")
        print(f"ADVERSARIAL ROBUSTNESS ANALYSIS")
        print(f"{'='*60}")
        print(self.test_results['adversarial'].to_string(index=False))
    
    def test_combined_scenarios(self):
        """Test combined drift + adversarial scenarios"""
        logger.info("Testing combined scenarios...")
        
        X_clean = self.baseline_metrics['X_test']
        y_true = self.baseline_metrics['true_labels']
        
        results = []
        
        # Scenario 1: Drift + FGSM
        X_drift = self._generate_covariate_drift(X_clean, magnitude=0.2)
        X_drift_adv = self._fgsm_attack(X_drift, epsilon=0.1)
        y_pred = self.model.predict(X_drift_adv)
        acc1 = accuracy_score(y_true, y_pred)
        
        results.append({
            'scenario': 'Covariate Drift + FGSM',
            'accuracy': acc1,
            'accuracy_drop': self.baseline_metrics['accuracy'] - acc1
        })
        
        # Scenario 2: Drift + Noise
        X_drift_noise = self._noise_attack(X_drift, noise_level=0.15)
        y_pred = self.model.predict(X_drift_noise)
        acc2 = accuracy_score(y_true, y_pred)
        
        results.append({
            'scenario': 'Covariate Drift + Noise',
            'accuracy': acc2,
            'accuracy_drop': self.baseline_metrics['accuracy'] - acc2
        })
        
        self.test_results['combined'] = pd.DataFrame(results)
        
        print(f"\n{'='*60}")
        print(f"COMBINED SCENARIOS ANALYSIS")
        print(f"{'='*60}")
        print(self.test_results['combined'].to_string(index=False))
    
    def visualize_results(self, save_path: str = "reports/robustness_analysis.png"):
        """Generate comprehensive visualization of all test results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Covariate Drift
        if 'covariate_drift' in self.test_results:
            df = self.test_results['covariate_drift']
            axes[0, 0].plot(df['magnitude'], df['accuracy'], marker='o', linewidth=2)
            axes[0, 0].axhline(y=self.baseline_metrics['accuracy'], color='r', linestyle='--', label='Baseline')
            axes[0, 0].set_title('Covariate Drift Impact', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Drift Magnitude')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Label Drift
        if 'label_drift' in self.test_results:
            df = self.test_results['label_drift']
            axes[0, 1].plot(df['shift_ratio'], df['accuracy'], marker='s', linewidth=2, color='orange')
            axes[0, 1].axhline(y=self.baseline_metrics['accuracy'], color='r', linestyle='--', label='Baseline')
            axes[0, 1].set_title('Label Drift Impact', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Shift Ratio')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Adversarial Robustness
        if 'adversarial' in self.test_results:
            df = self.test_results['adversarial']
            axes[1, 0].bar(df['attack_type'], df['accuracy'], color='steelblue', alpha=0.7)
            axes[1, 0].axhline(y=self.baseline_metrics['accuracy'], color='r', linestyle='--', label='Baseline')
            axes[1, 0].set_title('Adversarial Robustness', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Attack Type')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Combined Scenarios
        if 'combined' in self.test_results:
            df = self.test_results['combined']
            axes[1, 1].barh(df['scenario'], df['accuracy'], color='green', alpha=0.7)
            axes[1, 1].axvline(x=self.baseline_metrics['accuracy'], color='r', linestyle='--', label='Baseline')
            axes[1, 1].set_title('Combined Scenarios', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
        plt.close()
    
    # Helper methods for generating attacks/drift
    def _generate_covariate_drift(self, X: pd.DataFrame, magnitude: float) -> pd.DataFrame:
        """Generate covariate shift"""
        X_drift = X.copy()
        n_drift_features = int(len(X.columns) * 0.3)
        drift_features = np.random.choice(X.columns, n_drift_features, replace=False)
        
        for feature in drift_features:
            shift = X[feature].std() * magnitude
            X_drift[feature] = X[feature] + shift
            noise = np.random.normal(0, X[feature].std() * 0.1, len(X))
            X_drift[feature] += noise
        
        return X_drift
    
    def _generate_label_drift(self, y: pd.Series, shift_ratio: float) -> pd.Series:
        """Generate label shift"""
        y_shift = y.copy()
        classes = y.unique()
        n_shift = int(len(y) * shift_ratio)
        shift_indices = np.random.choice(len(y), n_shift, replace=False)
        
        for idx in shift_indices:
            current_class = y_shift.iloc[idx]
            new_class = np.random.choice([c for c in classes if c != current_class])
            y_shift.iloc[idx] = new_class
        
        return y_shift
    
    def _fgsm_attack(self, X: pd.DataFrame, epsilon: float) -> pd.DataFrame:
        """Simulate FGSM attack"""
        X_adv = X.copy()
        for col in X.columns:
            sign = np.random.choice([-1, 1], size=len(X))
            perturbation = epsilon * X[col].std() * sign
            X_adv[col] = X[col] + perturbation
        return X_adv.clip(X.min().min(), X.max().max())
    
    def _noise_attack(self, X: pd.DataFrame, noise_level: float) -> pd.DataFrame:
        """Add random noise"""
        X_noisy = X.copy()
        for col in X.columns:
            noise = np.random.normal(0, X[col].std() * noise_level, len(X))
            X_noisy[col] = X[col] + noise
        return X_noisy
    
    def _combined_attack(self, X: pd.DataFrame) -> pd.DataFrame:
        """Combined FGSM + noise attack"""
        X_adv = self._fgsm_attack(X, epsilon=0.08)
        X_adv = self._noise_attack(X_adv, noise_level=0.15)
        return X_adv
    
    def _calculate_drift_score(self, X_ref: pd.DataFrame, X_new: pd.DataFrame) -> float:
        """Calculate drift score using KS test"""
        from scipy.stats import ks_2samp
        
        drift_scores = []
        for col in X_ref.columns:
            statistic, _ = ks_2samp(X_ref[col], X_new[col])
            drift_scores.append(statistic)
        
        return np.mean(drift_scores)
    
    def generate_report(self, output_path: str = "reports/robustness_report.txt"):
        """Generate comprehensive text report"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ENHANCED IoT BOTSCAN - ROBUSTNESS ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Baseline Accuracy: {self.baseline_metrics['accuracy']:.4f}\n\n")
            
            if 'covariate_drift' in self.test_results:
                f.write("-"*80 + "\n")
                f.write("COVARIATE DRIFT ANALYSIS\n")
                f.write("-"*80 + "\n")
                f.write(self.test_results['covariate_drift'].to_string(index=False))
                f.write("\n\n")
            
            if 'label_drift' in self.test_results:
                f.write("-"*80 + "\n")
                f.write("LABEL DRIFT ANALYSIS\n")
                f.write("-"*80 + "\n")
                f.write(self.test_results['label_drift'].to_string(index=False))
                f.write("\n\n")
            
            if 'adversarial' in self.test_results:
                f.write("-"*80 + "\n")
                f.write("ADVERSARIAL ROBUSTNESS ANALYSIS\n")
                f.write("-"*80 + "\n")
                f.write(self.test_results['adversarial'].to_string(index=False))
                f.write("\n\n")
            
            if 'combined' in self.test_results:
                f.write("-"*80 + "\n")
                f.write("COMBINED SCENARIOS ANALYSIS\n")
                f.write("-"*80 + "\n")
                f.write(self.test_results['combined'].to_string(index=False))
                f.write("\n\n")
        
        logger.info(f"Report saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize tester
    tester = DriftAdversarialTester(model_path="models/hybrid_ensemble.joblib")
    
    # Load model
    # tester.load_model()
    
    # Load test data (you need to provide this)
    # X_test, y_test = load_your_test_data()
    
    # Set baseline
    # tester.set_baseline(X_test, y_test)
    
    # Run all tests
    # tester.test_covariate_drift()
    # tester.test_label_drift()
    # tester.test_adversarial_robustness()
    # tester.test_combined_scenarios()
    
    # Generate visualizations and report
    # tester.visualize_results()
    # tester.generate_report()
    
    print("Testing framework ready!")
