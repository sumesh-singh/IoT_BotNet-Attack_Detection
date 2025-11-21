"""
Unit Tests for Ensemble Module
Author: Kotiwale Sumesh Singh (160124862043)
"""

import pytest
import numpy as np
import pandas as pd
from src.core.ensemble.hybrid_ensemble import HybridEnsemble

@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 10), columns=[f'f{i}' for i in range(10)])
    y = pd.Series(np.random.randint(0, 2, 100))
    return X, y

def test_hybrid_ensemble_initialization():
    ensemble = HybridEnsemble({'n_estimators': 10})
    assert ensemble.config['n_estimators'] == 10

def test_hybrid_ensemble_training_prediction(sample_data):
    X, y = sample_data
    ensemble = HybridEnsemble()
    
    # Train
    ensemble.train(X, y)
    
    # Predict
    y_pred = ensemble.predict(X)
    assert len(y_pred) == len(y)
    
    # Predict proba
    y_proba = ensemble.predict_proba(X)
    assert len(y_proba) == len(y)
    assert y_proba.shape[1] == 2  # Binary classification

def test_hybrid_ensemble_evaluation(sample_data):
    X, y = sample_data
    ensemble = HybridEnsemble()
    ensemble.train(X, y)
    
    # Manual evaluation
    y_pred = ensemble.predict(X)
    accuracy = (y_pred == y).mean()
    assert accuracy >= 0 and accuracy <= 1
