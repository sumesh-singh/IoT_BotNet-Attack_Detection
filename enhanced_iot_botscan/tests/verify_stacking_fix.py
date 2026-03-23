
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.ensemble.meta_learner import StackingEnsemble
from src.core.ensemble.hybrid_ensemble import HybridEnsemble

# Mock class for base model to control probabilities
class MockModel:
    def __init__(self, probas):
        self.probas = probas
        self.is_trained = True
    
    def predict_proba(self, X):
        return self.probas
    
    def fit(self, X, y):
        pass

def verify_stacking_fix():
    print("Verifying Stacking Fix...")
    
    n_samples = 1
    n_classes = 3
    
    # Create probabilities for a single sample: [0.8, 0.1, 0.1]
    probas = np.array([[0.8, 0.1, 0.1]])
    
    # Initialize HybridEnsemble
    ensemble = HybridEnsemble({'use_stacking': True})
    ensemble.is_trained = True
    # Manually set n_classes to 3
    ensemble.n_classes = 3
    
    # Replace base models with mock
    mock_model = MockModel(probas)
    ensemble.base_models = {'mock1': mock_model, 'mock2': mock_model}
    
    # We want to check stacking logic in predict_proba
    # X is dummy
    X = pd.DataFrame(np.zeros((1, 5)), columns=[f'f{i}' for i in range(5)])
    
    # We need to monkeypath meta_learner to catch input
    original_meta_predict = ensemble.meta_learner.predict_proba
    
    captured_features = []
    def mock_meta_predict(features):
        captured_features.append(features)
        return np.array([[0.5, 0.5]]) # Dummy return
        
    ensemble.meta_learner.predict_proba = mock_meta_predict
    
    # Trigger prediction
    ensemble.predict_proba(X)
    
    stacked_features = captured_features[0]
    print(f"Stacked features shape: {stacked_features.shape}")
    print(f"Stacked features values: {stacked_features}")
    
    # Expected: (n_samples, n_models * n_classes) = (1, 2 * 3) = (1, 6)
    expected_shape = (1, 6)
    
    if stacked_features.shape == expected_shape:
        print("[PASS] Stacking dimension is correct (Includes all classes).")
    else:
        print(f"[FAIL] Stacking dimension incorrect. Expected {expected_shape}, got {stacked_features.shape}")

    # Check that values are not just max
    if np.allclose(stacked_features[0], [0.8, 0.1, 0.1, 0.8, 0.1, 0.1]):
        print("[PASS] Probabilities preserved correctly.")
    else:
        print("[FAIL] Probabilities validation failed.")

if __name__ == "__main__":
    verify_stacking_fix()
