"""
Test script for Adaptive Robustness Monitor (ARM)
Run: python test_arm.py
"""

import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import RandomForestClassifier
from src.core.robustness.robustness_monitor import AdaptiveRobustnessMonitor

print("=" * 60)
print("ADAPTIVE ROBUSTNESS MONITOR (ARM) TEST")
print("=" * 60)

# Generate sample data
np.random.seed(42)
n_samples = 1000
n_features = 20

print(f"\n1. Generating synthetic data: {n_samples} samples, {n_features} features")
X = np.random.randn(n_samples, n_features)
y = (X[:, 0] + X[:, 1] > 0).astype(int)
print(f"   Class distribution: {np.bincount(y)}")

# Train a simple model
print("\n2. Training RandomForest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
print(f"   Training accuracy: {model.score(X, y):.4f}")

# Initialize ARM
print("\n3. Initializing ARM...")
arm = AdaptiveRobustnessMonitor({
    'noise_levels': [0.0, 0.05, 0.1, 0.2],
    'masking_rates': [0.0, 0.1, 0.2, 0.3],
    'burst_intensities': [1.0, 1.5, 2.0, 3.0]
})

# Establish baseline
print("\n4. Establishing baseline...")
baseline = arm.establish_baseline(model, X, y)
print(f"   Baseline accuracy: {baseline['accuracy']:.4f}")
print(f"   Baseline confidence: {baseline['confidence']:.4f}")

# Comprehensive evaluation
print("\n" + "=" * 60)
print("5. COMPREHENSIVE ROBUSTNESS EVALUATION")
print("=" * 60)

results = arm.evaluate_comprehensive_robustness(model, X, y)

print(f"\n--- Noise Robustness ---")
for scenario, metrics in results['threat_scenarios']['noise'].items():
    print(f"  {scenario}: accuracy={metrics['accuracy']:.4f}, "
          f"drop={metrics['accuracy_drop']:.4f}, "
          f"robustness={metrics['robustness_score']:.4f}")

print(f"\n--- Masking Robustness ---")
for scenario, metrics in results['threat_scenarios']['masking'].items():
    print(f"  {scenario}: accuracy={metrics['accuracy']:.4f}, "
          f"drop={metrics['accuracy_drop']:.4f}, "
          f"robustness={metrics['robustness_score']:.4f}")

print(f"\n--- Burst Robustness ---")
for scenario, metrics in results['threat_scenarios']['burst'].items():
    print(f"  {scenario}: accuracy={metrics['accuracy']:.4f}, "
          f"drop={metrics['accuracy_drop']:.4f}, "
          f"robustness={metrics['robustness_score']:.4f}")

# Aggregate scores
print("\n" + "=" * 60)
print("6. AGGREGATE ROBUSTNESS SCORES")
print("=" * 60)
for metric, score in results['aggregate_scores'].items():
    print(f"  {metric}: {score:.4f}")

# Get report
report = arm.get_robustness_report()
print("\n" + "=" * 60)
print("7. ROBUSTNESS REPORT")
print("=" * 60)
print(f"  Overall Robustness: {report['summary']['overall_robustness']:.4f}")
print(f"  Weakest Area: {report['weakest_area']}")
print(f"  Recommendations:")
for rec in report['recommendations']:
    print(f"    - {rec}")

print("\n" + "=" * 60)
print("✅ ARM TEST COMPLETE!")
print("=" * 60)
