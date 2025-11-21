"""
Unit Tests for Adversarial Attack Module
Author: Kotiwale Sumesh Singh (160124862043)
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from src.core.adversarial.fgsm_attack import FGSMAttack
from src.core.adversarial.pgd_attack import PGDAttack
from src.core.adversarial.cw_attack import CWAttack

# Mock model for testing
class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.rand(100, 10).astype(np.float32)
    y = np.random.randint(0, 2, 100)
    return X, y

@pytest.fixture
def sklearn_model(sample_data):
    X, y = sample_data
    model = LogisticRegression()
    model.fit(X, y)
    return model

def test_fgsm_attack_initialization():
    attack = FGSMAttack({'epsilon': 0.2, 'norm': 'inf'})
    assert attack.epsilon == 0.2
    assert attack.norm == 'inf'

def test_fgsm_attack_generation(sklearn_model, sample_data):
    X, y = sample_data
    attack = FGSMAttack({'epsilon': 0.1})
    X_adv = attack.generate_attack(sklearn_model, X, y)
    
    assert X_adv.shape == X.shape
    assert np.all(X_adv >= 0) and np.all(X_adv <= 1)
    # Check if perturbation is within epsilon bound for L-inf
    diff = np.abs(X_adv - X)
    assert np.all(diff <= 0.1 + 1e-6)

def test_pgd_attack_initialization():
    attack = PGDAttack({'epsilon': 0.2, 'alpha': 0.02, 'num_iter': 20})
    assert attack.epsilon == 0.2
    assert attack.alpha == 0.02
    assert attack.num_iter == 20

def test_pgd_attack_generation(sklearn_model, sample_data):
    X, y = sample_data
    attack = PGDAttack({'epsilon': 0.1, 'num_iter': 5})
    X_adv = attack.generate_attack(sklearn_model, X, y)
    
    assert X_adv.shape == X.shape
    assert np.all(X_adv >= 0) and np.all(X_adv <= 1)

def test_cw_attack_initialization():
    attack = CWAttack({'c': 1.0, 'max_iter': 50})
    assert attack.c == 1.0
    assert attack.max_iter == 50

def test_cw_attack_generation(sklearn_model, sample_data):
    X, y = sample_data
    # Use a small subset for C&W as it's slow
    X_small = X[:5]
    y_small = y[:5]
    
    attack = CWAttack({'max_iter': 10, 'binary_search_steps': 2})
    X_adv = attack.generate_attack(sklearn_model, X_small, y_small)
    
    assert X_adv.shape == X_small.shape
    assert np.all(X_adv >= 0) and np.all(X_adv <= 1)
