# Benchmark Evaluation Results
**Enhanced IoT BotScan — Hybrid Stacking Ensemble**

*Generated: 2026-04-07 21:46:13*
*Total evaluation time: 2733s (45.5 min)*

---

## Table 1: N-BaIoT Dataset Results

*Test samples: 89,000 | Classes: 2 (Binary: Benign vs Malicious)*

| Model | Accuracy | Precision (W) | Recall (W) | F1-Score (W) | F1-Score (M) | AUC-ROC |
|-------|----------|---------------|------------|-------------|-------------|---------|
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9999 | 1.0000 |
| LightGBM | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Stacking Ensemble** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Table 2: IoT-23 Dataset Results

*Test samples: 10,000 | Classes: 2 (Binary: Benign vs Malicious)*

| Model | Accuracy | Precision (W) | Recall (W) | F1-Score (W) | F1-Score (M) | AUC-ROC |
|-------|----------|---------------|------------|-------------|-------------|---------|
| Random Forest | 0.9566 | 0.9600 | 0.9566 | 0.9565 | 0.9564 | 0.9554 |
| XGBoost | 0.9567 | 0.9601 | 0.9567 | 0.9566 | 0.9565 | 0.9563 |
| LightGBM | 0.9567 | 0.9601 | 0.9567 | 0.9566 | 0.9565 | 0.9565 |
| **Stacking Ensemble** | 0.9567 | 0.9601 | 0.9567 | 0.9566 | 0.9565 | 0.9553 |

## Table 3: BoT-IoT Dataset Results

*Test samples: 1,600 | Classes: 2 (Binary: Benign vs Malicious)*

| Model | Accuracy | Precision (W) | Recall (W) | F1-Score (W) | F1-Score (M) | AUC-ROC |
|-------|----------|---------------|------------|-------------|-------------|---------|
| Random Forest | 0.7544 | 0.8148 | 0.7544 | 0.6500 | 0.4349 | 0.5060 |
| XGBoost | 0.7531 | 0.5672 | 0.7531 | 0.6471 | 0.4296 | 0.5153 |
| LightGBM | 0.7531 | 0.5672 | 0.7531 | 0.6471 | 0.4296 | 0.5290 |
| **Stacking Ensemble** | 0.7531 | 0.5672 | 0.7531 | 0.6471 | 0.4296 | 0.5152 |

## Table 4: Cross-Dataset Ensemble Comparison

| Dataset | Accuracy | Precision (W) | Recall (W) | F1-Score (W) | AUC-ROC |
|---------|----------|---------------|------------|-------------|---------|
| **N-BaIoT** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **IoT-23** | 0.9567 | 0.9601 | 0.9567 | 0.9566 | 0.9553 |
| **BoT-IoT** | 0.7531 | 0.5672 | 0.7531 | 0.6471 | 0.5152 |

## Table 5: Base Paper Comparison (N-BaIoT)

| Approach | Model | Accuracy | F1-Score (W) | AUC-ROC | Notes |
|----------|-------|----------|-------------|---------|-------|
| Base Paper | Random Forest | 99.55% | — | — | Single RF, binary classification |
| **Ours** | Random Forest | 100.00% | 1.0000 | 1.0000 | Our RF config (n_est=200, depth=20) |
| **Ours** | **Stacking Ensemble** | **100.00%** | **1.0000** | **1.0000** | RF+XGB+LGBM+LR Meta-Learner |

---

## Methodology Notes

- **Preprocessing**: Conservative data cleaning (exact dedup, missing value imputation) + mutual information feature selection (top 50)
- **Split**: 80/20 stratified train/test split (random_state=42)
- **Stacking**: 3-fold cross-validation to generate out-of-fold predictions for meta-learner training
- **Meta-Learner**: Logistic Regression (C=1.0, max_iter=1000)
- **AUC-ROC**: Weighted One-vs-Rest for multi-class; direct for binary
- **(W)** = weighted average, **(M)** = macro average
