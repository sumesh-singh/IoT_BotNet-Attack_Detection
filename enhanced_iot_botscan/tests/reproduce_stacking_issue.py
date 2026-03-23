
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.core.ensemble.hybrid_ensemble import HybridEnsemble
from src.core.ensemble.meta_learner import StackingEnsemble

def test_multiclass_stacking_logic():
    print("Testing Multi-class Stacking Logic...")
    
    # Simulate a 3-class problem
    n_classes = 3
    n_samples = 2
    
    # Mock base model probabilities
    # Model 1: Confident in Class 0 [0.9, 0.05, 0.05]
    # Model 2: Confident in Class 1 [0.05, 0.9, 0.05]
    # Model 3: Confident in Class 2 [0.05, 0.05, 0.9]
    
    # Sample 1: All models agree on Class 0
    # Sample 2: Models disagree (Model 1 -> C0, Model 2 -> C1, Model 3 -> C2)
    
    # But wait, let's look at the implementation flaw directly.
    # The implementation takes MAX probability.
    
    # Case: Two different situations resulting in SAME stacking input
    
    # Situation A: Model predicts Class 0 with 0.9 confidence
    prob_a = np.array([[0.9, 0.05, 0.05]])
    
    # Situation B: Model predicts Class 1 with 0.9 confidence
    prob_b = np.array([[0.05, 0.9, 0.05]])
    
    # Current logic:
    stack_a = np.max(prob_a, axis=1)
    stack_b = np.max(prob_b, axis=1)
    
    print(f"Prob A (Class 0): {prob_a}")
    print(f"Prob B (Class 1): {prob_b}")
    print(f"Stacked A (np.max): {stack_a}")
    print(f"Stacked B (np.max): {stack_b}")
    
    if np.array_equal(stack_a, stack_b):
        print("\n[CRITICAL FAIL] Stacking features are IDENTICAL for different class predictions!")
        print("The meta-learner cannot distinguish between a confident prediction for Class 0 and Class 1.")
    else:
        print("\n[PASS] Stacking features are different.")

if __name__ == "__main__":
    test_multiclass_stacking_logic()
