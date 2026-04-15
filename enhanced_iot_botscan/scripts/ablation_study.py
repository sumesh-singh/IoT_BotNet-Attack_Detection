"""
Ablation Study — Enhanced IoT BotScan
======================================
4-condition experiment demonstrating incremental value of each component:

  Condition A: RF only (baseline, reproduces base paper)
  Condition B: Stacking ensemble (RF+XGB+LGB) — N-BaIoT only
  Condition C: Stacking ensemble — multi-dataset (N-BaIoT + IoT-23 + BoT-IoT)
  Condition D: Full system (C + ARM adversarial augmentation during training)

Fixed dataset: N-BaIoT, 80/20 split, random_state=42.
All conditions are EVALUATED on the same N-BaIoT test set.

Reports: Accuracy, F1 (weighted), ARM robustness score.

Memory-safe: designed for 8GB RAM.
"""

import sys
import os
import json
import time
import gc
import warnings
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from core.ensemble.random_forest_model import RandomForestModel
from core.ensemble.xgboost_model import XGBoostModel
from core.ensemble.lightgbm_model import LightGBMModel
from core.ensemble.hybrid_ensemble import HybridEnsemble
from core.preprocessing.data_cleaner import ConservativeDataCleaner
from core.preprocessing.feature_engineer import FeatureEngineer
from core.robustness.arm_robustness_monitor import AdaptiveRobustnessMonitor
from core.robustness.threat_generators.noise_injector import NoiseInjector
from core.robustness.threat_generators.feature_masker import FeatureMasker
from core.robustness.threat_generators.burst_generator import BurstGenerator
from data.data_loader import DataLoader

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# =====================================================================
# Configuration (shared across all conditions)
# =====================================================================

CONFIG = {
    'data_paths': {
        'n_baiot': os.path.join(PROJECT_ROOT, 'data', 'raw', 'n_baiot'),
        'iot_23': os.path.join(PROJECT_ROOT, 'data', 'raw', 'iot_23'),
        'bot_iot': os.path.join(PROJECT_ROOT, 'data', 'raw', 'bot_iot'),
    },
    'chunk_size': 5000,
    'max_samples_per_dataset': 50000,

    # Model hyperparameters
    'random_forest': {
        'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 5,
        'min_samples_leaf': 2, 'max_features': 'sqrt', 'random_state': 42,
        'n_jobs': 2, 'class_weight': 'balanced',
    },
    'xgboost': {
        'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1,
        'reg_lambda': 1.0, 'random_state': 42, 'n_jobs': 2,
    },
    'lightgbm': {
        'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1,
        'num_leaves': 31, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'random_state': 42, 'n_jobs': 2,
    },

    # Ensemble
    'use_stacking': True,
    'stacking_cv_folds': 3,
    'optimize_base_models': False,

    # Feature engineering
    'create_polynomial_features': False,
    'create_interaction_features': False,
    'create_statistical_features': True,
    'feature_selection_method': 'mutual_info',
    'n_features_select': 50,

    # Training
    'test_size': 0.2,
    'random_state': 42,
}


# =====================================================================
# Data Loaders
# =====================================================================

def load_nbaiot_memorysafe(data_path='./data/raw/n_baiot', max_per_file=5000):
    """Load N-BaIoT with per-file sampling. ~450K samples, 115 features."""
    import glob
    from data.data_loader import parse_nbaiot_attack_label

    csv_files = sorted(glob.glob(os.path.join(data_path, '*.csv')))
    csv_files = [f for f in csv_files if os.path.basename(f).split('.')[0].isdigit()]
    print(f"    Found {len(csv_files)} N-BaIoT CSV files, sampling {max_per_file} rows each...")

    all_data, all_labels = [], []
    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath, nrows=max_per_file, dtype=np.float32)
            fname = os.path.basename(fpath).lower()
            label = parse_nbaiot_attack_label(fname, fine_grained=False)
            all_data.append(df)
            all_labels.extend([label] * len(df))
        except Exception as e:
            print(f"      Skipping {os.path.basename(fpath)}: {e}")

    X = pd.concat(all_data, ignore_index=True).replace([np.inf, -np.inf], np.nan).fillna(0)
    y = np.array(all_labels)
    print(f"    N-BaIoT loaded: {len(X):,} samples, {X.shape[1]} features")
    return X, pd.Series(y)


def load_iot23(config):
    """Load IoT-23 dataset via DataLoader."""
    loader = DataLoader(config)
    dataset = loader.load_iot_23_dataset()
    X = pd.DataFrame(dataset['features'], columns=dataset['feature_names'])
    y = pd.Series(dataset['labels'])
    if len(np.unique(y)) > 2:
        y = (y != 0).astype(int)
    return X, y


def load_botiot(config):
    """Load BoT-IoT dataset via DataLoader."""
    loader = DataLoader(config)
    dataset = loader.load_bot_iot_dataset()
    X = pd.DataFrame(dataset['features'], columns=dataset['feature_names'])
    y = pd.Series(dataset['labels'])
    if len(np.unique(y)) > 2:
        y = (y != 0).astype(int)
    return X, y


def clean_and_binarize(X, y):
    """Conservative cleaning + binary label conversion."""
    y = (y != 0).astype(int)
    cleaner = ConservativeDataCleaner(
        remove_exact_duplicates=True, handle_missing=True, remove_outliers=False
    )
    df = pd.concat([X, y.rename('label')], axis=1)
    df_clean = cleaner.clean(df, target_col='label')
    X_c = df_clean.drop(columns=['label']).replace([np.inf, -np.inf], np.nan).fillna(0)
    y_c = df_clean['label']
    return X_c, y_c


# =====================================================================
# ARM Robustness Score (lightweight version for ablation)
# =====================================================================

def compute_arm_robustness(model, X_test, y_test):
    """
    Compute the ARM overall robustness score for a given model.
    Uses the full AdaptiveRobustnessMonitor pipeline.
    """
    arm = AdaptiveRobustnessMonitor({})

    # Convert to numpy if needed
    X_np = X_test.values if hasattr(X_test, 'values') else X_test
    y_np = y_test.values if hasattr(y_test, 'values') else y_test

    # Use a subset for ARM to keep memory and time reasonable
    max_arm_samples = 5000
    if len(X_np) > max_arm_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_np), max_arm_samples, replace=False)
        X_arm = X_np[idx]
        y_arm = y_np[idx]
    else:
        X_arm = X_np
        y_arm = y_np

    # Establish baseline and evaluate
    arm.establish_baseline(model, X_arm, y_arm)
    results = arm.evaluate_comprehensive_robustness(model, X_arm, y_arm)

    return results['aggregate_scores']


# =====================================================================
# ARM Augmentation: generate adversarial training data
# =====================================================================

def generate_arm_augmented_data(X_train, y_train, augment_fraction=0.15):
    """
    Create ARM-augmented training data by appending adversarially perturbed
    versions of a subset of the training data.

    This simulates noise, feature masking, and burst traffic conditions,
    training the model to be robust under real-world IoT perturbations.

    augment_fraction: fraction of training set to augment (default 15%)
    """
    X_np = X_train.values if hasattr(X_train, 'values') else X_train
    y_np = y_train.values if hasattr(y_train, 'values') else y_train

    n_aug = int(len(X_np) * augment_fraction)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_np), n_aug, replace=False)
    X_sub = X_np[idx]
    y_sub = y_np[idx]

    noise_inj = NoiseInjector({})
    masker = FeatureMasker({})
    burst_gen = BurstGenerator({})

    augmented_X = []
    augmented_y = []

    # 1. Gaussian noise (σ = 10%)
    X_noisy = noise_inj.add_gaussian_noise(X_sub, scale=0.10)
    augmented_X.append(X_noisy)
    augmented_y.append(y_sub)

    # 2. Feature masking (20%)
    X_masked = masker.random_feature_masking(X_sub, mask_rate=0.20)
    augmented_X.append(X_masked)
    augmented_y.append(y_sub)

    # 3. Burst traffic (1.5x)
    X_burst = burst_gen.simulate_burst_traffic(X_sub, intensity=1.5)
    augmented_X.append(X_burst)
    augmented_y.append(y_sub)

    # Combine original + augmented
    X_combined = np.vstack([X_np] + augmented_X)
    y_combined = np.concatenate([y_np] + augmented_y)

    # Shuffle
    shuffle_idx = rng.permutation(len(X_combined))
    X_combined = X_combined[shuffle_idx]
    y_combined = y_combined[shuffle_idx]

    print(f"    ARM augmentation: {len(X_np):,} → {len(X_combined):,} samples "
          f"(+{len(X_combined)-len(X_np):,} augmented: noise+mask+burst)")

    return X_combined, y_combined


# =====================================================================
# Main Ablation Pipeline
# =====================================================================

def main():
    print("=" * 100)
    print("  ABLATION STUDY — ENHANCED IoT BOTSCAN")
    print("  4-Condition Component Analysis")
    print("=" * 100)

    global_start = time.time()
    results = {}

    # ==================================================================
    # STEP 1: Load and preprocess N-BaIoT (fixed dataset for all conditions)
    # ==================================================================
    print("\n[STEP 1] Loading N-BaIoT (primary evaluation dataset)...")
    X_nbaiot, y_nbaiot = load_nbaiot_memorysafe(CONFIG['data_paths']['n_baiot'])
    X_nbaiot_clean, y_nbaiot_clean = clean_and_binarize(X_nbaiot, y_nbaiot)
    del X_nbaiot, y_nbaiot
    gc.collect()

    # Feature engineering on N-BaIoT
    print("    Feature engineering...")
    fe_nbaiot = FeatureEngineer(CONFIG)
    X_nbaiot_eng = fe_nbaiot.engineer_features(X_nbaiot_clean, y_nbaiot_clean)
    print(f"    N-BaIoT after FE: {X_nbaiot_eng.shape[0]:,} × {X_nbaiot_eng.shape[1]} features")

    # Fixed 80/20 split (seed=42) — same for ALL conditions
    X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(
        X_nbaiot_eng, y_nbaiot_clean,
        test_size=0.2, random_state=42, stratify=y_nbaiot_clean
    )
    print(f"    Fixed split: Train={len(X_train_nb):,} | Test={len(X_test_nb):,}")
    del X_nbaiot_clean, X_nbaiot_eng
    gc.collect()

    # ==================================================================
    # CONDITION A: RF Only (Baseline — reproduces base paper)
    # ==================================================================
    print("\n" + "#" * 100)
    print("  CONDITION A: Random Forest Only (Baseline)")
    print("#" * 100)

    t0 = time.time()
    rf_model = RandomForestModel(CONFIG)
    rf_model.train(X_train_nb, y_train_nb, validation_data=(X_test_nb, y_test_nb))
    y_pred_a = rf_model.predict(X_test_nb)
    acc_a = accuracy_score(y_test_nb, y_pred_a)
    f1_a = f1_score(y_test_nb, y_pred_a, average='weighted', zero_division=0)
    print(f"    Accuracy={acc_a:.6f}  F1(W)={f1_a:.6f}")

    print("    Computing ARM robustness...")
    arm_a = compute_arm_robustness(rf_model, X_test_nb, y_test_nb)
    print(f"    ARM overall={arm_a['overall_robustness']:.4f}")

    results['A'] = {
        'condition': 'A: RF Only (Baseline)',
        'accuracy': acc_a, 'f1_weighted': f1_a,
        'arm_overall': arm_a['overall_robustness'],
        'arm_noise': arm_a['noise_robustness'],
        'arm_masking': arm_a['masking_robustness'],
        'arm_burst': arm_a['burst_robustness'],
        'arm_confidence': arm_a['confidence_stability'],
        'time_s': time.time() - t0,
    }
    del rf_model
    gc.collect()

    # ==================================================================
    # CONDITION B: Stacking Ensemble (N-BaIoT only)
    # ==================================================================
    print("\n" + "#" * 100)
    print("  CONDITION B: Stacking Ensemble (RF+XGB+LGBM) — N-BaIoT Only")
    print("#" * 100)

    t0 = time.time()
    ensemble_b = HybridEnsemble(CONFIG)
    ensemble_b.train(X_train_nb, y_train_nb, validation_data=(X_test_nb, y_test_nb))
    y_pred_b = ensemble_b.predict(X_test_nb)
    acc_b = accuracy_score(y_test_nb, y_pred_b)
    f1_b = f1_score(y_test_nb, y_pred_b, average='weighted', zero_division=0)
    print(f"    Accuracy={acc_b:.6f}  F1(W)={f1_b:.6f}")

    print("    Computing ARM robustness...")
    arm_b = compute_arm_robustness(ensemble_b, X_test_nb, y_test_nb)
    print(f"    ARM overall={arm_b['overall_robustness']:.4f}")

    results['B'] = {
        'condition': 'B: Stacking Ensemble (N-BaIoT)',
        'accuracy': acc_b, 'f1_weighted': f1_b,
        'arm_overall': arm_b['overall_robustness'],
        'arm_noise': arm_b['noise_robustness'],
        'arm_masking': arm_b['masking_robustness'],
        'arm_burst': arm_b['burst_robustness'],
        'arm_confidence': arm_b['confidence_stability'],
        'time_s': time.time() - t0,
    }
    del ensemble_b
    gc.collect()

    # ==================================================================
    # STEP 2: Load IoT-23 + BoT-IoT for multi-dataset conditions (C & D)
    # ==================================================================
    print("\n[STEP 2] Loading auxiliary datasets for multi-dataset training...")

    # IoT-23
    print("    Loading IoT-23...")
    X_iot23, y_iot23 = load_iot23(CONFIG)
    X_iot23_clean, y_iot23_clean = clean_and_binarize(X_iot23, y_iot23)
    del X_iot23, y_iot23
    gc.collect()

    # BoT-IoT
    print("    Loading BoT-IoT...")
    X_botiot, y_botiot = load_botiot(CONFIG)
    X_botiot_clean, y_botiot_clean = clean_and_binarize(X_botiot, y_botiot)
    del X_botiot, y_botiot
    gc.collect()

    # Align features: use only columns common with N-BaIoT train set
    # Since datasets have different raw features, we need to align via FE
    # Strategy: run FeatureEngineer independently, then use shared statistical features
    print("    Feature engineering on IoT-23...")
    fe_iot23 = FeatureEngineer(CONFIG)
    X_iot23_eng = fe_iot23.engineer_features(X_iot23_clean, y_iot23_clean)
    del X_iot23_clean
    gc.collect()

    print("    Feature engineering on BoT-IoT...")
    fe_botiot = FeatureEngineer(CONFIG)
    X_botiot_eng = fe_botiot.engineer_features(X_botiot_clean, y_botiot_clean)
    del X_botiot_clean
    gc.collect()

    # Align columns: use intersection of features across all datasets
    nbaiot_cols = set(X_train_nb.columns if hasattr(X_train_nb, 'columns') else
                     [f'f_{i}' for i in range(X_train_nb.shape[1])])
    iot23_cols = set(X_iot23_eng.columns if hasattr(X_iot23_eng, 'columns') else
                    [f'f_{i}' for i in range(X_iot23_eng.shape[1])])
    botiot_cols = set(X_botiot_eng.columns if hasattr(X_botiot_eng, 'columns') else
                     [f'f_{i}' for i in range(X_botiot_eng.shape[1])])

    # Find shared feature columns
    shared_cols = sorted(nbaiot_cols & iot23_cols & botiot_cols)
    if len(shared_cols) < 5:
        # If very few shared columns (different feature spaces), use statistical features only
        print(f"    Only {len(shared_cols)} shared columns — using statistical feature alignment")
        stat_cols = [c for c in nbaiot_cols if c.startswith('row_')]
        iot23_stat = [c for c in iot23_cols if c.startswith('row_')]
        botiot_stat = [c for c in botiot_cols if c.startswith('row_')]
        shared_cols = sorted(set(stat_cols) & set(iot23_stat) & set(botiot_stat))
        if len(shared_cols) < 3:
            # Fallback: pad auxiliary data to match N-BaIoT feature count
            print(f"    Fallback: padding auxiliary datasets to {X_train_nb.shape[1]} features")
            n_feat = X_train_nb.shape[1]

            def pad_or_trim(X_df, target_n):
                X_np = X_df.values if hasattr(X_df, 'values') else X_df
                if X_np.shape[1] >= target_n:
                    return X_np[:, :target_n]
                else:
                    pad = np.zeros((X_np.shape[0], target_n - X_np.shape[1]), dtype=np.float32)
                    return np.hstack([X_np, pad])

            X_iot23_aligned = pad_or_trim(X_iot23_eng, n_feat)
            X_botiot_aligned = pad_or_trim(X_botiot_eng, n_feat)
            X_nbaiot_train_np = X_train_nb.values if hasattr(X_train_nb, 'values') else X_train_nb
        else:
            X_nbaiot_train_np = (X_train_nb[shared_cols].values
                                if hasattr(X_train_nb, 'columns') else X_train_nb)
            X_iot23_aligned = (X_iot23_eng[shared_cols].values
                              if hasattr(X_iot23_eng, 'columns') else X_iot23_eng)
            X_botiot_aligned = (X_botiot_eng[shared_cols].values
                               if hasattr(X_botiot_eng, 'columns') else X_botiot_eng)
    else:
        print(f"    {len(shared_cols)} shared feature columns found")
        X_nbaiot_train_np = (X_train_nb[shared_cols].values
                            if hasattr(X_train_nb, 'columns') else X_train_nb)
        X_iot23_aligned = (X_iot23_eng[shared_cols].values
                          if hasattr(X_iot23_eng, 'columns') else X_iot23_eng)
        X_botiot_aligned = (X_botiot_eng[shared_cols].values
                           if hasattr(X_botiot_eng, 'columns') else X_botiot_eng)

    del X_iot23_eng, X_botiot_eng
    gc.collect()

    # Subsample auxiliary datasets to prevent them from overwhelming N-BaIoT
    rng = np.random.RandomState(42)
    max_aux = 20000
    if len(X_iot23_aligned) > max_aux:
        idx = rng.choice(len(X_iot23_aligned), max_aux, replace=False)
        X_iot23_aligned = X_iot23_aligned[idx]
        y_iot23_clean = y_iot23_clean.iloc[idx].values if hasattr(y_iot23_clean, 'iloc') else y_iot23_clean[idx]
    else:
        y_iot23_clean = y_iot23_clean.values if hasattr(y_iot23_clean, 'values') else y_iot23_clean

    if len(X_botiot_aligned) > max_aux:
        idx = rng.choice(len(X_botiot_aligned), max_aux, replace=False)
        X_botiot_aligned = X_botiot_aligned[idx]
        y_botiot_clean = y_botiot_clean.iloc[idx].values if hasattr(y_botiot_clean, 'iloc') else y_botiot_clean[idx]
    else:
        y_botiot_clean = y_botiot_clean.values if hasattr(y_botiot_clean, 'values') else y_botiot_clean

    # Build unified multi-dataset training set
    X_nbaiot_np = X_train_nb.values if hasattr(X_train_nb, 'values') else X_train_nb
    if X_nbaiot_np.shape[1] != X_iot23_aligned.shape[1]:
        # Ensure all have same width
        target_w = X_nbaiot_np.shape[1]
        def match_width(arr, w):
            if arr.shape[1] >= w:
                return arr[:, :w]
            pad = np.zeros((arr.shape[0], w - arr.shape[1]), dtype=np.float32)
            return np.hstack([arr, pad])
        X_iot23_aligned = match_width(X_iot23_aligned, target_w)
        X_botiot_aligned = match_width(X_botiot_aligned, target_w)

    y_nbaiot_np = y_train_nb.values if hasattr(y_train_nb, 'values') else y_train_nb

    X_multi = np.vstack([X_nbaiot_np, X_iot23_aligned, X_botiot_aligned]).astype(np.float32)
    y_multi = np.concatenate([y_nbaiot_np, y_iot23_clean, y_botiot_clean]).astype(int)

    # Shuffle
    shuffle_idx = rng.permutation(len(X_multi))
    X_multi = X_multi[shuffle_idx]
    y_multi = y_multi[shuffle_idx]

    print(f"    Multi-dataset training set: {len(X_multi):,} samples "
          f"(N-BaIoT: {len(X_nbaiot_np):,}, IoT-23: {len(X_iot23_aligned):,}, "
          f"BoT-IoT: {len(X_botiot_aligned):,})")
    print(f"    Multi-dataset label distribution: {dict(zip(*np.unique(y_multi, return_counts=True)))}")

    del X_iot23_aligned, X_botiot_aligned, y_iot23_clean, y_botiot_clean
    gc.collect()

    # Prepare test set in same format
    X_test_np = X_test_nb.values if hasattr(X_test_nb, 'values') else X_test_nb
    y_test_np = y_test_nb.values if hasattr(y_test_nb, 'values') else y_test_nb
    if X_test_np.shape[1] != X_multi.shape[1]:
        X_test_eval = X_test_np[:, :X_multi.shape[1]]
    else:
        X_test_eval = X_test_np

    # Convert back to DataFrame for ensemble compatibility (it expects .columns)
    feat_cols = list(X_train_nb.columns[:X_multi.shape[1]]) if hasattr(X_train_nb, 'columns') else [f'f_{i}' for i in range(X_multi.shape[1])]
    X_multi = pd.DataFrame(X_multi, columns=feat_cols)
    X_test_eval = pd.DataFrame(X_test_eval, columns=feat_cols)
    y_multi = pd.Series(y_multi)
    y_test_np = pd.Series(y_test_np)

    # ==================================================================
    # CONDITION C: Stacking Ensemble + Multi-Dataset Training
    # ==================================================================
    print("\n" + "#" * 100)
    print("  CONDITION C: Stacking Ensemble + Multi-Dataset Training")
    print("#" * 100)

    t0 = time.time()
    ensemble_c = HybridEnsemble(CONFIG)
    ensemble_c.train(X_multi, y_multi, validation_data=(X_test_eval, y_test_np))
    y_pred_c = ensemble_c.predict(X_test_eval)
    acc_c = accuracy_score(y_test_np, y_pred_c)
    f1_c = f1_score(y_test_np, y_pred_c, average='weighted', zero_division=0)
    print(f"    Accuracy={acc_c:.6f}  F1(W)={f1_c:.6f}")

    print("    Computing ARM robustness...")
    arm_c = compute_arm_robustness(ensemble_c, X_test_eval, y_test_np)
    print(f"    ARM overall={arm_c['overall_robustness']:.4f}")

    results['C'] = {
        'condition': 'C: Stacking + Multi-Dataset',
        'accuracy': acc_c, 'f1_weighted': f1_c,
        'arm_overall': arm_c['overall_robustness'],
        'arm_noise': arm_c['noise_robustness'],
        'arm_masking': arm_c['masking_robustness'],
        'arm_burst': arm_c['burst_robustness'],
        'arm_confidence': arm_c['confidence_stability'],
        'time_s': time.time() - t0,
    }
    del ensemble_c
    gc.collect()

    # ==================================================================
    # CONDITION D: Full System (Multi-Dataset + ARM Augmentation)
    # ==================================================================
    print("\n" + "#" * 100)
    print("  CONDITION D: Full System (Multi-Dataset + ARM Augmentation)")
    print("#" * 100)

    t0 = time.time()

    # ARM augmentation: add adversarially perturbed samples to training data
    print("    Generating ARM-augmented training data...")
    X_multi_aug, y_multi_aug = generate_arm_augmented_data(X_multi, y_multi, augment_fraction=0.15)
    
    # Convert augmented data to DataFrame and Series
    X_multi_aug = pd.DataFrame(X_multi_aug, columns=feat_cols)
    # Clean up any inf/-inf/nan generated by threat generators
    X_multi_aug = X_multi_aug.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    y_multi_aug = pd.Series(y_multi_aug)

    # Train on augmented data
    ensemble_d = HybridEnsemble(CONFIG)
    ensemble_d.train(X_multi_aug, y_multi_aug, validation_data=(X_test_eval, y_test_np))
    y_pred_d = ensemble_d.predict(X_test_eval)
    acc_d = accuracy_score(y_test_np, y_pred_d)
    f1_d = f1_score(y_test_np, y_pred_d, average='weighted', zero_division=0)
    print(f"    Accuracy={acc_d:.6f}  F1(W)={f1_d:.6f}")

    print("    Computing ARM robustness...")
    arm_d = compute_arm_robustness(ensemble_d, X_test_eval, y_test_np)
    print(f"    ARM overall={arm_d['overall_robustness']:.4f}")

    results['D'] = {
        'condition': 'D: Full System (Multi-DS + ARM Aug)',
        'accuracy': acc_d, 'f1_weighted': f1_d,
        'arm_overall': arm_d['overall_robustness'],
        'arm_noise': arm_d['noise_robustness'],
        'arm_masking': arm_d['masking_robustness'],
        'arm_burst': arm_d['burst_robustness'],
        'arm_confidence': arm_d['confidence_stability'],
        'time_s': time.time() - t0,
    }
    del ensemble_d, X_multi, X_multi_aug, y_multi, y_multi_aug
    gc.collect()

    # ==================================================================
    # RESULTS TABLE
    # ==================================================================
    global_elapsed = time.time() - global_start

    print("\n" + "=" * 100)
    print("  ABLATION STUDY — RESULTS")
    print("=" * 100)
    print(f"\n  Evaluation dataset: N-BaIoT (test split) | {len(y_test_np):,} samples | "
          f"Seed=42 | 80/20 split")
    print()

    header = (f"  {'Cond':<6} {'Description':<40} {'Accuracy':>10} {'F1(W)':>10} "
              f"{'ARM Score':>10} {'Noise':>8} {'Mask':>8} {'Burst':>8} {'Conf':>8} {'Time':>8}")
    print(header)
    print("  " + "-" * 116)

    for key in ['A', 'B', 'C', 'D']:
        r = results[key]
        desc = r['condition'].split(': ', 1)[1][:38]
        print(f"  {key:<6} {desc:<40} {r['accuracy']:>10.6f} {r['f1_weighted']:>10.6f} "
              f"{r['arm_overall']:>10.4f} {r['arm_noise']:>8.4f} {r['arm_masking']:>8.4f} "
              f"{r['arm_burst']:>8.4f} {r['arm_confidence']:>8.4f} {r['time_s']:>7.0f}s")

    print("  " + "-" * 116)
    print(f"\n  Total ablation time: {global_elapsed:.0f}s ({global_elapsed/60:.1f} min)")

    # Incremental contribution analysis
    print("\n" + "=" * 100)
    print("  INCREMENTAL CONTRIBUTION ANALYSIS")
    print("=" * 100)

    contrib_acc_b = results['B']['accuracy'] - results['A']['accuracy']
    contrib_acc_c = results['C']['accuracy'] - results['B']['accuracy']
    contrib_acc_d = results['D']['accuracy'] - results['C']['accuracy']

    contrib_arm_b = results['B']['arm_overall'] - results['A']['arm_overall']
    contrib_arm_c = results['C']['arm_overall'] - results['B']['arm_overall']
    contrib_arm_d = results['D']['arm_overall'] - results['C']['arm_overall']

    print(f"\n  A→B (+ Stacking Ensemble):      Acc Δ = {contrib_acc_b:+.6f}   ARM Δ = {contrib_arm_b:+.4f}")
    print(f"  B→C (+ Multi-Dataset Training):  Acc Δ = {contrib_acc_c:+.6f}   ARM Δ = {contrib_arm_c:+.4f}")
    print(f"  C→D (+ ARM Augmentation):        Acc Δ = {contrib_acc_d:+.6f}   ARM Δ = {contrib_arm_d:+.4f}")

    # ==================================================================
    # Save results
    # ==================================================================
    os.makedirs('results', exist_ok=True)

    # JSON
    with open('results/ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Raw metrics saved to: results/ablation_results.json")

    # Markdown report
    md = generate_ablation_report(results, global_elapsed, len(y_test_np))
    with open('results/ablation_results.md', 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"  Report saved to: results/ablation_results.md")

    print(f"\n{'=' * 100}")


def generate_ablation_report(results, elapsed, n_test):
    """Generate publication-ready markdown ablation report."""
    lines = []
    lines.append("# Ablation Study Results")
    lines.append("**Enhanced IoT BotScan — Component Contribution Analysis**")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append(f"*Evaluation dataset: N-BaIoT ({n_test:,} test samples) | Train/Test: 80/20 | Seed: 42*")
    lines.append(f"*Total evaluation time: {elapsed:.0f}s ({elapsed/60:.1f} min)*")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Ablation Conditions")
    lines.append("")
    lines.append("| Condition | Description | Training Data | Components |")
    lines.append("|---|---|---|---|")
    lines.append("| **A** | RF Only (Baseline) | N-BaIoT only | Single Random Forest classifier |")
    lines.append("| **B** | Stacking Ensemble | N-BaIoT only | RF + XGBoost + LightGBM + LR Meta-Learner |")
    lines.append("| **C** | Stacking + Multi-Dataset | N-BaIoT + IoT-23 + BoT-IoT | Same as B, multi-dataset training |")
    lines.append("| **D** | Full System | N-BaIoT + IoT-23 + BoT-IoT + ARM Aug | Same as C + adversarial data augmentation |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Results Table")
    lines.append("")
    lines.append("| Condition | Accuracy | F1-Score (W) | ARM Robustness | ARM Noise | ARM Masking | ARM Burst | ARM Confidence |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for key in ['A', 'B', 'C', 'D']:
        r = results[key]
        bold = "**" if key == 'D' else ""
        lines.append(
            f"| {bold}{key}: {r['condition'].split(': ', 1)[1]}{bold} | "
            f"{bold}{r['accuracy']:.6f}{bold} | "
            f"{bold}{r['f1_weighted']:.6f}{bold} | "
            f"{bold}{r['arm_overall']:.4f}{bold} | "
            f"{r['arm_noise']:.4f} | {r['arm_masking']:.4f} | "
            f"{r['arm_burst']:.4f} | {r['arm_confidence']:.4f} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Incremental Contribution Analysis")
    lines.append("")

    # Compute deltas
    d_ab_acc = results['B']['accuracy'] - results['A']['accuracy']
    d_bc_acc = results['C']['accuracy'] - results['B']['accuracy']
    d_cd_acc = results['D']['accuracy'] - results['C']['accuracy']

    d_ab_arm = results['B']['arm_overall'] - results['A']['arm_overall']
    d_bc_arm = results['C']['arm_overall'] - results['B']['arm_overall']
    d_cd_arm = results['D']['arm_overall'] - results['C']['arm_overall']

    d_ab_f1 = results['B']['f1_weighted'] - results['A']['f1_weighted']
    d_bc_f1 = results['C']['f1_weighted'] - results['B']['f1_weighted']
    d_cd_f1 = results['D']['f1_weighted'] - results['C']['f1_weighted']

    lines.append("| Transition | Component Added | Accuracy Δ | F1 Δ | ARM Δ |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| A → B | + Stacking Ensemble | {d_ab_acc:+.6f} | {d_ab_f1:+.6f} | {d_ab_arm:+.4f} |")
    lines.append(f"| B → C | + Multi-Dataset Training | {d_bc_acc:+.6f} | {d_bc_f1:+.6f} | {d_bc_arm:+.4f} |")
    lines.append(f"| C → D | + ARM Augmentation | {d_cd_acc:+.6f} | {d_cd_f1:+.6f} | {d_cd_arm:+.4f} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Discussion")
    lines.append("")
    lines.append("### A → B: Adding Stacking Ensemble")
    lines.append("")
    lines.append("The transition from a single Random Forest to a stacking ensemble (RF + XGBoost + LightGBM "
                 "with Logistic Regression meta-learner) measures the contribution of model diversity and "
                 "intelligent prediction combination. When multiple complementary tree-based architectures "
                 "agree on a classification, confidence increases; where they disagree, the meta-learner "
                 "learns the optimal weighting strategy from out-of-fold cross-validation predictions.")
    lines.append("")
    lines.append("### B → C: Adding Multi-Dataset Training")
    lines.append("")
    lines.append("Training on a unified corpus of N-BaIoT, IoT-23, and BoT-IoT exposes the model to "
                 "diverse network environments, device types, attack families, and feature representations. "
                 "This cross-domain exposure is designed to improve generalization robustness — the model "
                 "learns invariant patterns of botnet behavior rather than dataset-specific artifacts. "
                 "The ARM robustness score change reflects whether this broader training distribution "
                 "makes the model more or less resilient to perturbations.")
    lines.append("")
    lines.append("### C → D: Adding ARM Adversarial Augmentation")
    lines.append("")
    lines.append("ARM augmentation adds adversarially perturbed training samples (Gaussian noise at 10%, "
                 "feature masking at 20%, burst traffic at 1.5×) to the training set, explicitly teaching "
                 "the model to classify correctly even under degraded input conditions. This directly "
                 "targets the ARM robustness score, as the model has been exposed to the same types of "
                 "perturbations used during robustness evaluation. The expected effect is a measurable "
                 "improvement in the ARM composite score, particularly in noise and masking robustness.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Methodology Notes")
    lines.append("")
    lines.append("- **Fixed evaluation set**: All conditions evaluated on the same N-BaIoT 20% test split (seed=42)")
    lines.append("- **ARM evaluation**: 5,000-sample subset with Gaussian noise (5/10/20%), feature masking (10/20/30%), burst traffic (1.5×/2×)")
    lines.append("- **ARM augmentation (Condition D)**: 15% of training data duplicated with perturbations (noise σ=10%, mask=20%, burst=1.5×)")
    lines.append("- **Multi-dataset alignment (Conditions C/D)**: Feature padding/truncation to match N-BaIoT feature dimensionality")
    lines.append("- **Auxiliary dataset cap**: 20,000 samples each from IoT-23 and BoT-IoT to prevent class imbalance dominance")
    lines.append("")

    return "\n".join(lines)


if __name__ == '__main__':
    main()
