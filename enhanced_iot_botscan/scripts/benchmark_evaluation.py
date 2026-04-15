"""
Multi-Dataset Benchmark Evaluation for Enhanced IoT BotScan
============================================================
Trains individual base models (RF, XGBoost, LightGBM) and the full Stacking Ensemble
on N-BaIoT, IoT-23, and BoT-IoT datasets independently.

Reports: Accuracy, Precision, Recall, F1-Score (macro & weighted), AUC-ROC.

Usage:
    python scripts/benchmark_evaluation.py

Memory-safe: designed for 8GB RAM systems.
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from sklearn.preprocessing import label_binarize

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
from data.data_loader import DataLoader

logging.basicConfig(level=logging.WARNING)  # Suppress verbose logs during benchmark
logger = logging.getLogger(__name__)


# ====================================================================================
# Memory-Safe N-BaIoT Loader (avoids 7M-row OOM from standard loader)
# ====================================================================================

def load_nbaiot_memorysafe(data_path='./data/raw/n_baiot', max_per_file=5000):
    """
    Load N-BaIoT with strict per-file sampling to stay within 8GB RAM.
    Correctly parses attack labels from filenames.
    
    With ~90 files × 5000 rows = ~450K samples × 115 cols × 4 bytes ≈ 200MB.
    """
    import glob
    from data.data_loader import parse_nbaiot_attack_label

    csv_files = sorted(glob.glob(os.path.join(data_path, '*.csv')))
    # Filter to numbered device files only (e.g., '1.benign.csv')
    csv_files = [f for f in csv_files if os.path.basename(f).split('.')[0].isdigit()]

    print(f"  Found {len(csv_files)} N-BaIoT CSV files, sampling {max_per_file} rows each...")

    all_data = []
    all_labels = []

    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath, nrows=max_per_file, dtype=np.float32)
            fname = os.path.basename(fpath).lower()
            label = parse_nbaiot_attack_label(fname, fine_grained=False)
            all_data.append(df)
            all_labels.extend([label] * len(df))
        except Exception as e:
            print(f"    Skipping {os.path.basename(fpath)}: {e}")

    X = pd.concat(all_data, ignore_index=True)
    y = np.array(all_labels)

    # Clean inf/nan
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"  N-BaIoT loaded: {len(X):,} samples, {X.shape[1]} features")
    print(f"  Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, pd.Series(y)


# ====================================================================================
# Configuration
# ====================================================================================

CONFIG = {
    'data_paths': {
        'n_baiot': './data/raw/n_baiot',
        'iot_23': './data/raw/iot_23',
        'bot_iot': './data/raw/bot_iot',
    },
    'chunk_size': 5000,
    'memory_limit_gb': 8,
    'use_optimized_loader': False,    # DISABLED: optimized loader only loads benign class
    'max_samples_per_device': 5000,   # 9 devices x 5K = ~45K total for N-BaIoT
    'max_samples_per_dataset': 50000,
    'fine_grained_labels': False,     # Binary: Benign(0) vs Malicious(1)

    # Model hyperparameters (tuned for 8GB RAM)
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': 2,
        'class_weight': 'balanced',
    },
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': 2,
    },
    'lightgbm': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': 2,
    },

    # Ensemble
    'use_stacking': True,
    'stacking_cv_folds': 3,
    'optimize_base_models': False,

    # Feature engineering
    'create_polynomial_features': False,
    'create_interaction_features': False,  # Disabled to keep feature count manageable
    'create_statistical_features': True,
    'feature_selection_method': 'mutual_info',
    'n_features_select': 50,

    # Training
    'test_size': 0.2,
    'random_state': 42,
}


# ====================================================================================
# Evaluation Helpers
# ====================================================================================

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate a trained model and return comprehensive metrics."""
    print(f"    Evaluating {model_name}...", end=" ", flush=True)
    start = time.time()

    y_pred = model.predict(X_test)

    # Probabilities for AUC-ROC
    y_proba = None
    auc_roc = None
    try:
        y_proba = model.predict_proba(X_test)
        unique_classes = np.unique(y_test)

        if len(unique_classes) == 2:
            # Binary AUC-ROC
            auc_roc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            # Multi-class OVR AUC-ROC
            auc_roc = roc_auc_score(
                y_test, y_proba, multi_class='ovr', average='weighted'
            )
    except Exception as e:
        auc_roc = None
        logger.warning(f"AUC-ROC computation failed for {model_name}: {e}")

    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'auc_roc': auc_roc,
        'n_test_samples': len(y_test),
        'n_classes': len(np.unique(y_test)),
    }

    elapsed = time.time() - start
    print(f"done ({elapsed:.1f}s) | Acc={metrics['accuracy']:.4f}")
    return metrics


def print_metrics_table(results_list, dataset_name):
    """Print a formatted table of metrics."""
    print(f"\n{'='*100}")
    print(f"  RESULTS: {dataset_name}")
    print(f"{'='*100}")

    header = f"{'Model':<25} {'Accuracy':>10} {'Prec(W)':>10} {'Rec(W)':>10} {'F1(W)':>10} {'F1(M)':>10} {'AUC-ROC':>10}"
    print(header)
    print("-" * 100)

    for r in results_list:
        auc_str = f"{r['auc_roc']:.4f}" if r['auc_roc'] is not None else "N/A"
        row = (
            f"{r['model']:<25} "
            f"{r['accuracy']:>10.4f} "
            f"{r['precision_weighted']:>10.4f} "
            f"{r['recall_weighted']:>10.4f} "
            f"{r['f1_weighted']:>10.4f} "
            f"{r['f1_macro']:>10.4f} "
            f"{auc_str:>10}"
        )
        print(row)

    print("-" * 100)
    print(f"  Test samples: {results_list[0]['n_test_samples']:,} | Classes: {results_list[0]['n_classes']}")
    print()


# ====================================================================================
# Dataset Loading + Preprocessing
# ====================================================================================

def load_and_preprocess(dataset_name, config):
    """Load a single dataset, clean it, engineer features, and split for training."""
    print(f"\n  Loading {dataset_name} dataset...")

    # N-BaIoT: use custom memory-safe loader (standard loader OOMs on 8GB)
    if dataset_name == 'n_baiot':
        X, y = load_nbaiot_memorysafe(
            data_path=config['data_paths']['n_baiot'],
            max_per_file=5000
        )
        # Convert coarse labels (0=benign, 1=mirai, 2=gafgyt) to binary
        y = (y != 0).astype(int)

        print(f"  Raw: {X.shape[0]:,} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
        print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        # Clean
        cleaner = ConservativeDataCleaner(
            remove_exact_duplicates=True, handle_missing=True, remove_outliers=False
        )
        df = pd.concat([X, y.rename('label')], axis=1)
        df_clean = cleaner.clean(df, target_col='label')
        X_clean = df_clean.drop(columns=['label'])
        y_clean = df_clean['label']
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan).fillna(0)

        print(f"  After cleaning: {X_clean.shape[0]:,} samples, {X_clean.shape[1]} features")

        # Feature engineering
        fe = FeatureEngineer(config)
        X_eng = fe.engineer_features(X_clean, y_clean)
        print(f"  After feature engineering: {X_eng.shape[0]:,} samples, {X_eng.shape[1]} features")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_eng, y_clean, test_size=config.get('test_size', 0.2),
            random_state=config.get('random_state', 42), stratify=y_clean
        )
        print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
        return X_train, X_test, y_train, y_test, fe

    # Other datasets: use DataLoader as before
    loader = DataLoader(config)

    if dataset_name == 'iot_23':
        dataset = loader.load_iot_23_dataset()
    elif dataset_name == 'bot_iot':
        dataset = loader.load_bot_iot_dataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    X = pd.DataFrame(dataset['features'], columns=dataset['feature_names'])
    y = pd.Series(dataset['labels'])

    # Ensure binary labels: 0=benign, 1=malicious (for all datasets)
    unique_labels = np.unique(y)
    if len(unique_labels) > 2:
        print(f"  Converting {len(unique_labels)} classes to binary...")
        y = (y != 0).astype(int)
    elif len(unique_labels) == 1:
        raise ValueError(f"Only 1 class found in {dataset_name} — data loading issue!")

    print(f"  Raw: {X.shape[0]:,} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Clean
    cleaner = ConservativeDataCleaner(
        remove_exact_duplicates=True,
        handle_missing=True,
        remove_outliers=False,
    )
    df = pd.concat([X, y.rename('label')], axis=1)
    df_clean = cleaner.clean(df, target_col='label')

    X_clean = df_clean.drop(columns=['label'])
    y_clean = df_clean['label']

    # Handle inf/nan after cleaning
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"  After cleaning: {X_clean.shape[0]:,} samples, {X_clean.shape[1]} features")

    # Feature engineering
    fe = FeatureEngineer(config)
    X_eng = fe.engineer_features(X_clean, y_clean)
    print(f"  After feature engineering: {X_eng.shape[0]:,} samples, {X_eng.shape[1]} features")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_eng, y_clean,
        test_size=config.get('test_size', 0.2),
        random_state=config.get('random_state', 42),
        stratify=y_clean
    )

    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    return X_train, X_test, y_train, y_test, fe


# ====================================================================================
# Per-Dataset Evaluation Pipeline
# ====================================================================================

def run_benchmark_for_dataset(dataset_name, config):
    """Run the full benchmark for a single dataset."""
    print(f"\n{'#'*100}")
    print(f"  BENCHMARK: {dataset_name.upper()}")
    print(f"{'#'*100}")

    total_start = time.time()

    # Load and preprocess
    X_train, X_test, y_train, y_test, fe = load_and_preprocess(dataset_name, config)
    results = []

    # ---- 1. Random Forest (standalone) ----
    print(f"\n  [1/4] Training Random Forest...")
    rf = RandomForestModel(config)
    rf.train(X_train, y_train, validation_data=(X_test, y_test))
    metrics_rf = evaluate_model(rf, X_test, y_test, "Random Forest")
    results.append(metrics_rf)
    del rf
    gc.collect()

    # ---- 2. XGBoost (standalone) ----
    print(f"  [2/4] Training XGBoost...")
    xgb = XGBoostModel(config)
    xgb.train(X_train, y_train, validation_data=(X_test, y_test))
    metrics_xgb = evaluate_model(xgb, X_test, y_test, "XGBoost")
    results.append(metrics_xgb)
    del xgb
    gc.collect()

    # ---- 3. LightGBM (standalone) ----
    print(f"  [3/4] Training LightGBM...")
    lgbm = LightGBMModel(config)
    lgbm.train(X_train, y_train, validation_data=(X_test, y_test))
    metrics_lgbm = evaluate_model(lgbm, X_test, y_test, "LightGBM")
    results.append(metrics_lgbm)
    del lgbm
    gc.collect()

    # ---- 4. Stacking Ensemble ----
    print(f"  [4/4] Training Stacking Ensemble (RF+XGB+LGBM+Meta)...")
    ensemble = HybridEnsemble(config)
    ensemble.train(X_train, y_train, validation_data=(X_test, y_test))
    metrics_ens = evaluate_model(ensemble, X_test, y_test, "Stacking Ensemble")
    results.append(metrics_ens)
    del ensemble
    gc.collect()

    elapsed_total = time.time() - total_start
    print(f"\n  Total time for {dataset_name}: {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)")

    # Print table
    print_metrics_table(results, dataset_name.upper())

    return results


# ====================================================================================
# Report Generation
# ====================================================================================

def generate_markdown_report(all_results, elapsed_total):
    """Generate publication-ready markdown report."""

    lines = []
    lines.append("# Benchmark Evaluation Results")
    lines.append(f"**Enhanced IoT BotScan — Hybrid Stacking Ensemble**")
    lines.append(f"")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append(f"*Total evaluation time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)*")
    lines.append("")
    lines.append("---")
    lines.append("")

    dataset_display = {'n_baiot': 'N-BaIoT', 'iot_23': 'IoT-23', 'bot_iot': 'BoT-IoT'}

    # Per-dataset tables
    for i, (ds_name, results_list) in enumerate(all_results.items(), 1):
        ds_display = dataset_display.get(ds_name, ds_name)
        n_samples = results_list[0]['n_test_samples']
        n_classes = results_list[0]['n_classes']

        lines.append(f"## Table {i}: {ds_display} Dataset Results")
        lines.append(f"")
        lines.append(f"*Test samples: {n_samples:,} | Classes: {n_classes} (Binary: Benign vs Malicious)*")
        lines.append(f"")
        lines.append("| Model | Accuracy | Precision (W) | Recall (W) | F1-Score (W) | F1-Score (M) | AUC-ROC |")
        lines.append("|-------|----------|---------------|------------|-------------|-------------|---------|")

        for r in results_list:
            auc = f"{r['auc_roc']:.4f}" if r['auc_roc'] is not None else "N/A"
            bold = "**" if "Ensemble" in r['model'] else ""
            lines.append(
                f"| {bold}{r['model']}{bold} | "
                f"{r['accuracy']:.4f} | "
                f"{r['precision_weighted']:.4f} | "
                f"{r['recall_weighted']:.4f} | "
                f"{r['f1_weighted']:.4f} | "
                f"{r['f1_macro']:.4f} | "
                f"{auc} |"
            )

        lines.append("")

    # Cross-dataset comparison table
    table_num = len(all_results) + 1
    lines.append(f"## Table {table_num}: Cross-Dataset Ensemble Comparison")
    lines.append("")
    lines.append("| Dataset | Accuracy | Precision (W) | Recall (W) | F1-Score (W) | AUC-ROC |")
    lines.append("|---------|----------|---------------|------------|-------------|---------|")

    for ds_name, results_list in all_results.items():
        ds_display = dataset_display.get(ds_name, ds_name)
        ens_result = [r for r in results_list if "Ensemble" in r['model']][0]
        auc = f"{ens_result['auc_roc']:.4f}" if ens_result['auc_roc'] is not None else "N/A"
        lines.append(
            f"| **{ds_display}** | "
            f"{ens_result['accuracy']:.4f} | "
            f"{ens_result['precision_weighted']:.4f} | "
            f"{ens_result['recall_weighted']:.4f} | "
            f"{ens_result['f1_weighted']:.4f} | "
            f"{auc} |"
        )
    lines.append("")

    # Base paper comparison
    table_num += 1
    lines.append(f"## Table {table_num}: Base Paper Comparison (N-BaIoT)")
    lines.append("")

    if 'n_baiot' in all_results:
        nb_results = all_results['n_baiot']
        rf_result = [r for r in nb_results if r['model'] == 'Random Forest'][0]
        ens_result = [r for r in nb_results if 'Ensemble' in r['model']][0]

        lines.append("| Approach | Model | Accuracy | F1-Score (W) | AUC-ROC | Notes |")
        lines.append("|----------|-------|----------|-------------|---------|-------|")
        lines.append("| Base Paper | Random Forest | 99.55% | — | — | Single RF, binary classification |")
        auc_rf = f"{rf_result['auc_roc']:.4f}" if rf_result['auc_roc'] is not None else "N/A"
        lines.append(
            f"| **Ours** | Random Forest | {rf_result['accuracy']*100:.2f}% | "
            f"{rf_result['f1_weighted']:.4f} | "
            f"{auc_rf} | "
            f"Our RF config (n_est=200, depth=20) |"
        )
        auc_ens = f"{ens_result['auc_roc']:.4f}" if ens_result['auc_roc'] is not None else "N/A"
        lines.append(
            f"| **Ours** | **Stacking Ensemble** | **{ens_result['accuracy']*100:.2f}%** | "
            f"**{ens_result['f1_weighted']:.4f}** | "
            f"**{auc_ens}** | "
            f"RF+XGB+LGBM+LR Meta-Learner |"
        )
    else:
        lines.append("*N-BaIoT results not available.*")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Methodology Notes")
    lines.append("")
    lines.append("- **Preprocessing**: Conservative data cleaning (exact dedup, missing value imputation) + mutual information feature selection (top 50)")
    lines.append("- **Split**: 80/20 stratified train/test split (random_state=42)")
    lines.append("- **Stacking**: 3-fold cross-validation to generate out-of-fold predictions for meta-learner training")
    lines.append("- **Meta-Learner**: Logistic Regression (C=1.0, max_iter=1000)")
    lines.append("- **AUC-ROC**: Weighted One-vs-Rest for multi-class; direct for binary")
    lines.append("- **(W)** = weighted average, **(M)** = macro average")
    lines.append("")

    return "\n".join(lines)


# ====================================================================================
# Main
# ====================================================================================

def main():
    print("=" * 100)
    print("  ENHANCED IoT BOTSCAN — MULTI-DATASET BENCHMARK EVALUATION")
    print("  Models: Random Forest | XGBoost | LightGBM | Stacking Ensemble")
    print("  Datasets: N-BaIoT | IoT-23 | BoT-IoT")
    print("=" * 100)

    global_start = time.time()

    all_results = {}
    datasets_to_run = ['n_baiot', 'iot_23', 'bot_iot']

    for ds_name in datasets_to_run:
        try:
            results = run_benchmark_for_dataset(ds_name, CONFIG)
            all_results[ds_name] = results
        except Exception as e:
            print(f"\n  *** ERROR on {ds_name}: {e} ***")
            import traceback
            traceback.print_exc()
        finally:
            gc.collect()

    global_elapsed = time.time() - global_start

    # Print cross-dataset summary
    print("\n" + "=" * 100)
    print("  CROSS-DATASET ENSEMBLE SUMMARY")
    print("=" * 100)
    for ds_name, results_list in all_results.items():
        ens = [r for r in results_list if "Ensemble" in r['model']]
        if ens:
            e = ens[0]
            auc_str = f"{e['auc_roc']:.4f}" if e['auc_roc'] is not None else 'N/A'
            print(f"  {ds_name.upper():<12} Acc={e['accuracy']:.4f}  F1(W)={e['f1_weighted']:.4f}  AUC={auc_str}")
    print(f"\n  Total time: {global_elapsed:.0f}s ({global_elapsed/60:.1f} min)")
    print("=" * 100)

    # Save results
    os.makedirs('results', exist_ok=True)

    # JSON (raw)
    json_path = os.path.join('results', 'benchmark_results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Raw metrics saved to: {json_path}")

    # Markdown (publication-ready)
    md_content = generate_markdown_report(all_results, global_elapsed)
    md_path = os.path.join('results', 'benchmark_results.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"  Report saved to: {md_path}")


if __name__ == '__main__':
    main()
