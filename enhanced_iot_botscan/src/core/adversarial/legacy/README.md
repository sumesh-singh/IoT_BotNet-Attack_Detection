# Legacy Adversarial Models

This folder contains the deprecated gradient-based adversarial attack files:
- `fgsm_attack.py` (Fast Gradient Sign Method)
- `pgd_attack.py` (Projected Gradient Descent) 
- `cw_attack.py` (Carlini-Wagner)

### Deprecation Notice

These methodologies have been deprecated in favor of the **Zeroth-Order Optimization (ZOO)** and perturbation evaluation matrices (as implemented in `zoo_attack.py` and `ARM` modules).

Tree-based ensemble models (Random Forest, XGBoost, and LightGBM) possess non-differentiable decision boundaries. Applying white-box gradient-dependent attacks like FGSM or PGD on these architectures is mathematically inconsistent and fails to evaluate true security bounds, often resulting in phantom robustness (gradient masking).

The framework now exclusively utilizes black-box operational perturbations (noise, feature masking, burst generation) and proxy boundary estimations to correctly validate structural resistance.
