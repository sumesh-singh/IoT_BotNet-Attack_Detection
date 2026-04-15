# Ablation Study Results
**Enhanced IoT BotScan — Component Contribution Analysis**

*Generated: 2026-04-10 00:15:37*
*Evaluation dataset: N-BaIoT (89,000 test samples) | Train/Test: 80/20 | Seed: 42*
*Total evaluation time: 14077s (234.6 min)*

---

## Ablation Conditions

| Condition | Description | Training Data | Components |
|---|---|---|---|
| **A** | RF Only (Baseline) | N-BaIoT only | Single Random Forest classifier |
| **B** | Stacking Ensemble | N-BaIoT only | RF + XGBoost + LightGBM + LR Meta-Learner |
| **C** | Stacking + Multi-Dataset | N-BaIoT + IoT-23 + BoT-IoT | Same as B, multi-dataset training |
| **D** | Full System | N-BaIoT + IoT-23 + BoT-IoT + ARM Aug | Same as C + adversarial data augmentation |

---

## Results Table

| Condition | Accuracy | F1-Score (W) | ARM Robustness | ARM Noise | ARM Masking | ARM Burst | ARM Confidence |
|---|---|---|---|---|---|---|---|
| A: RF Only (Baseline) | 1.000000 | 1.000000 | 0.8624 | 0.8092 | 0.9298 | 0.9970 | 0.7066 |
| B: Stacking Ensemble (N-BaIoT) | 0.999989 | 0.999989 | 0.9208 | 0.8590 | 0.9155 | 0.9973 | 0.9451 |
| C: Stacking + Multi-Dataset | 0.999978 | 0.999978 | 0.9200 | 0.8354 | 0.9594 | 0.9976 | 0.9101 |
| **D: Full System (Multi-DS + ARM Aug)** | **0.999966** | **0.999966** | **0.9991** | 0.9985 | 0.9995 | 0.9991 | 0.9993 |

---

## Incremental Contribution Analysis

| Transition | Component Added | Accuracy Δ | F1 Δ | ARM Δ |
|---|---|---|---|---|
| A → B | + Stacking Ensemble | -0.000011 | -0.000011 | +0.0584 |
| B → C | + Multi-Dataset Training | -0.000011 | -0.000011 | -0.0009 |
| C → D | + ARM Augmentation | -0.000011 | -0.000011 | +0.0791 |

---

## Discussion

### A → B: Adding Stacking Ensemble

The transition from a single Random Forest to a stacking ensemble (RF + XGBoost + LightGBM with Logistic Regression meta-learner) measures the contribution of model diversity and intelligent prediction combination. When multiple complementary tree-based architectures agree on a classification, confidence increases; where they disagree, the meta-learner learns the optimal weighting strategy from out-of-fold cross-validation predictions.

### B → C: Adding Multi-Dataset Training

Training on a unified corpus of N-BaIoT, IoT-23, and BoT-IoT exposes the model to diverse network environments, device types, attack families, and feature representations. This cross-domain exposure is designed to improve generalization robustness — the model learns invariant patterns of botnet behavior rather than dataset-specific artifacts. The ARM robustness score change reflects whether this broader training distribution makes the model more or less resilient to perturbations.

### C → D: Adding ARM Adversarial Augmentation

ARM augmentation adds adversarially perturbed training samples (Gaussian noise at 10%, feature masking at 20%, burst traffic at 1.5×) to the training set, explicitly teaching the model to classify correctly even under degraded input conditions. This directly targets the ARM robustness score, as the model has been exposed to the same types of perturbations used during robustness evaluation. The expected effect is a measurable improvement in the ARM composite score, particularly in noise and masking robustness.

---

## Methodology Notes

- **Fixed evaluation set**: All conditions evaluated on the same N-BaIoT 20% test split (seed=42)
- **ARM evaluation**: 5,000-sample subset with Gaussian noise (5/10/20%), feature masking (10/20/30%), burst traffic (1.5×/2×)
- **ARM augmentation (Condition D)**: 15% of training data duplicated with perturbations (noise σ=10%, mask=20%, burst=1.5×)
- **Multi-dataset alignment (Conditions C/D)**: Feature padding/truncation to match N-BaIoT feature dimensionality
- **Auxiliary dataset cap**: 20,000 samples each from IoT-23 and BoT-IoT to prevent class imbalance dominance
