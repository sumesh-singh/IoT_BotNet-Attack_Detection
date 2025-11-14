## Verification Summary
- New datasets `IoT-23` and `BoT-IoT` are referenced and supported end‑to‑end in loaders, training, and configs.
- Training pipeline loads, splits, trains baseline and adversarial models, evaluates, and saves results with error handling.
- Classical augmentation (SMOTE/oversampling) is not implemented; adversarial example generation is present.

## Findings
- Data loading
  - `src/data/data_loader.py:35-150` loads N‑BaIoT, consolidates device CSVs, fills NaN/inf, outputs `float32` features and numeric labels.
  - `src/data/data_loader.py:152-217` loads IoT‑23, reads CSVs in chunks, assumes last column is label, converts to numeric.
  - `src/data/data_loader.py:219-299` loads BoT‑IoT, detects `label`/`Label`/last column, filters numeric features, converts labels.
- Preprocessing utilities
  - `src/core/preprocessing/data_cleaner.py` (cleaning/outliers/type optimization), `feature_engineer.py` (stat/interaction/domain features), `scaler.py`, `dimensionality_reducer.py` are implemented but not wired into `ModelTrainer`.
  - `src/core/data/dataset_manager.py:499-533` applies missing/duplicates/normalize/binary target; supports splitting/validation (`:158-243`, `:343-421`).
- Training pipeline
  - `scripts/train_models.py:30-56` initializes components (ConfigManager, DataLoader, HybridEnsemble, AdversarialTrainer, DriftDetector, PerformanceEvaluator).
  - `scripts/train_models.py:57-76` loads datasets; defaults to `['n_baiot','iot_23','bot_iot']`.
  - `scripts/train_models.py:77-103` splits data (train/val/test, stratified).
  - Baseline training `:104-140`; adversarial training `:142-188`; cross‑dataset validation `:190-230`; concept drift test `:232-287`.
  - Results and checkpoints saved `:289-310` to `./data/results` and `./data/results/models`.
  - Error handling in CLI `:373-396` with logged exception and non‑zero exit.
- Configs
  - `config/config.yaml:1-5` includes data paths for `n_baiot`, `iot_23`, `bot_iot`.
  - `src/utils/config_manager.py:246-277` validates config and checks data path existence.
- Models
  - Hybrid ensemble consumes tabular numeric features (`src/core/ensemble/hybrid_ensemble.py:54-72`); stacking enabled; meta‑learner.
  - RandomForest supports class imbalance via `class_weight='balanced'` (`src/core/ensemble/random_forest_model.py:33-45`).
  - XGBoost defaults without explicit multi‑class objective (`src/core/ensemble/xgboost_model.py:33-49`).
  - LightGBM sets `objective='binary'` (`src/core/ensemble/lightgbm_model.py:33-48`), which may require adjustment for multi‑class datasets.

## Gaps & Risks
- Multi‑class labels: N‑BaIoT loader produces 5 classes; IoT‑23/BoT‑IoT samples can be >2 classes.
  - LightGBM uses `objective='binary'`; change to `objective='multiclass'` and set `num_class` when `n_classes>2`.
  - XGBoost may need `objective='multi:softprob'` when `n_classes>2`.
- Augmentation: No SMOTE/oversampling for class imbalance; adversarial augmentation exists but is for robustness, not class balancing.
- Preprocessing: Advanced modules exist but `ModelTrainer` currently uses minimal cleaning; consider wiring `DatasetManager` or preprocessing utilities for consistency.

## Smoke Test Plan
- Prepare small data
  - Generate sample CSVs: run `python scripts/download_datasets.py --data-dir ./data/raw --dataset all --create-samples`.
  - Verify paths: ensure `./data/raw/{n_baiot,iot_23,bot_iot}` contain sample CSVs.
- Run baseline training (small subset)
  - Command: `python scripts/train_models.py --config ./config/config.yaml --datasets iot_23 --mode baseline --output-dir ./data/results`.
  - Success criteria: console shows success; YAML `training_results_*.yaml` and `models/ensemble_model_*.pkl` created.
- Run full pipeline on N‑BaIoT sample
  - Command: `python scripts/train_models.py --config ./config/config.yaml --datasets n_baiot --mode full --output-dir ./data/results`.
  - Check logs in `./logs/` and results in `./data/results/`.
- Cross‑dataset quick check
  - Command: `python scripts/train_models.py --config ./config/config.yaml --datasets n_baiot iot_23 bot_iot --mode full`.
  - Confirm `cross_dataset_validation` entries in results YAML.

## Implementation Adjustments (before full training)
- Auto‑set objectives based on `n_classes` at runtime in model wrappers.
  - LightGBM: set `objective='multiclass'`, `num_class=n_classes` when `n_classes>2`.
  - XGBoost: set `objective='multi:softprob'` when `n_classes>2`.
- Optional: add class imbalance handling (e.g., SMOTE or class‑weighted sampling) and wire `DatasetManager`/preprocessing utilities into `ModelTrainer` for consistent transforms.

## Validation Checklist
- Load: All three datasets load without errors (see `data_loader.py` functions).
- Preprocess: No NaN/inf after loader; basic normalization available via `DatasetManager`.
- Train: Baseline and adversarial runs complete; metrics present in YAML; models saved.
- Errors: CLI catches and logs exceptions; non‑zero exit on failure.

Confirm to proceed with the smoke tests and the minor objective adjustments for multi‑class handling. 