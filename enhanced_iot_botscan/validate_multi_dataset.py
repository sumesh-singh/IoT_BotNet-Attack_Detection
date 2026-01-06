"""
Multi-Dataset Validation Script for ARM
Tests model robustness across N-BaIoT, IoT-23, and BoT-IoT datasets.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.robustness.robustness_monitor import AdaptiveRobustnessMonitor
from src.core.ensemble.hybrid_ensemble import HybridEnsemble
from src.core.preprocessing.data_cleaner import DataCleaner
from src.core.preprocessing.feature_engineer import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_on_dataset(model, dataset_path: str, dataset_name: str, 
                        feature_engineer: FeatureEngineer) -> dict:
    """Validate model robustness on a specific dataset."""
    
    print(f"\n{'='*60}")
    print(f"VALIDATING ON: {dataset_name}")
    print(f"{'='*60}")
    
    try:
        # Load data
        df = pd.read_csv(dataset_path)
        print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
        
        # Find target column
        target_candidates = [c for c in df.columns if any(x in c.lower() for x in ['label', 'class', 'attack'])]
        target_col = target_candidates[0] if target_candidates else df.columns[-1]
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Clean data
        cleaner = DataCleaner({})
        X_clean = cleaner.clean_dataset(X)
        y = y.loc[X_clean.index]
        
        # Engineer features (using existing engineer to match training)
        X_eng = feature_engineer.transform_new_data(X_clean)
        print(f"After engineering: {X_eng.shape[0]} samples, {X_eng.shape[1]} features")
        
        # Run ARM evaluation
        arm = AdaptiveRobustnessMonitor({
            'noise_levels': [0.0, 0.05, 0.1, 0.2],
            'masking_rates': [0.0, 0.1, 0.2, 0.3],
            'burst_intensities': [1.0, 1.5, 2.0, 3.0]
        })
        
        X_np = X_eng.values
        y_np = y.values
        
        # Establish baseline
        baseline = arm.establish_baseline(model, X_np, y_np)
        print(f"Baseline Accuracy: {baseline['accuracy']:.2%}")
        print(f"Baseline Confidence: {baseline['confidence']:.2%}")
        
        # Comprehensive evaluation
        results = arm.evaluate_comprehensive_robustness(model, X_np, y_np)
        scores = results['aggregate_scores']
        
        print(f"\n--- Robustness Scores ---")
        print(f"Overall:  {scores['overall_robustness']:.2%}")
        print(f"Noise:    {scores['noise_robustness']:.2%}")
        print(f"Masking:  {scores['masking_robustness']:.2%}")
        print(f"Burst:    {scores['burst_robustness']:.2%}")
        
        return {
            'dataset': dataset_name,
            'samples': len(df),
            'baseline_accuracy': baseline['accuracy'],
            'baseline_confidence': baseline['confidence'],
            'overall_robustness': scores['overall_robustness'],
            'noise_robustness': scores['noise_robustness'],
            'masking_robustness': scores['masking_robustness'],
            'burst_robustness': scores['burst_robustness'],
            'status': 'success'
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            'dataset': dataset_name,
            'status': 'error',
            'error': str(e)
        }


def run_multi_dataset_validation():
    """Run validation across all available datasets."""
    
    print("="*60)
    print("MULTI-DATASET VALIDATION")
    print("="*60)
    
    # Load model
    model_path = "models/hybrid_ensemble.joblib"
    if not os.path.exists(model_path):
        print("❌ Model not found. Please train a model first.")
        return
    
    model = HybridEnsemble({})
    model.load_model(model_path)
    
    # Get feature engineer from model
    engineer = FeatureEngineer({})
    if hasattr(model, '_feature_engineer_state'):
        engineer.restore_state(model._feature_engineer_state)
        print("✓ Feature engineer restored from model")
    
    # Find available datasets
    data_dirs = [
        'data/processed',
        'data/raw'
    ]
    
    results = []
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith('.csv'):
                    dataset_path = os.path.join(data_dir, file)
                    result = validate_on_dataset(model, dataset_path, file, engineer)
                    results.append(result)
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    successful = [r for r in results if r.get('status') == 'success']
    
    if successful:
        print(f"\n{'Dataset':<30} {'Accuracy':<12} {'Robustness':<12}")
        print("-"*54)
        for r in successful:
            print(f"{r['dataset']:<30} {r['baseline_accuracy']:.2%}        {r['overall_robustness']:.2%}")
        
        # Average robustness
        avg_robustness = np.mean([r['overall_robustness'] for r in successful])
        print(f"\n{'AVERAGE':<30} {'':<12} {avg_robustness:.2%}")
    
    return results


if __name__ == "__main__":
    run_multi_dataset_validation()
