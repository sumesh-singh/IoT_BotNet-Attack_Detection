"""
Unit Tests for Drift Detection Module
Author: Kotiwale Sumesh Singh (160124862043)
"""

import pytest
import numpy as np
import pandas as pd
from src.core.drift_detection.kolmogorov_smirnov import KolmogorovSmirnovDriftDetector
from src.core.drift_detection.page_hinkley import PageHinkleyDriftDetector

@pytest.fixture
def reference_data():
    np.random.seed(42)
    return pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 100),
        'feature_2': np.random.normal(5, 2, 100)
    })

@pytest.fixture
def drift_data():
    np.random.seed(43)
    return pd.DataFrame({
        'feature_1': np.random.normal(2, 1, 100),  # Mean shift
        'feature_2': np.random.normal(5, 2, 100)
    })

def test_ks_detector_initialization():
    detector = KolmogorovSmirnovDriftDetector({'alpha': 0.01})
    assert detector.alpha == 0.01

def test_ks_detector_drift_detection(reference_data, drift_data):
    detector = KolmogorovSmirnovDriftDetector({'alpha': 0.05, 'min_samples': 50})
    detector.set_reference_data(reference_data)
    
    # Test with same distribution (should not detect drift)
    drift_result = detector.detect_drift(reference_data)
    assert not drift_result['drift_detected']
    
    # Test with drifted data (should detect drift)
    drift_result = detector.detect_drift(drift_data)
    assert drift_result['drift_detected']
    # Check if any feature drifted (drifted_features might not be in result, check feature_drift)
    assert drift_result['feature_drift']['drift_count'] > 0

def test_ph_detector_initialization():
    detector = PageHinkleyDriftDetector({'threshold': 30, 'min_samples': 10})
    assert detector.threshold == 30
    assert detector.min_samples == 10

def test_ph_detector_drift_detection():
    # Set min_samples to 1 for this test since we initialize with a single row
    detector = PageHinkleyDriftDetector({'threshold': 10, 'min_samples': 1})
    
    # Simulate stream with drift
    np.random.seed(42)
    stream_data = np.concatenate([
        np.random.normal(0, 1, 50),
        np.random.normal(5, 1, 50)  # Drift
    ])
    
    drifts_detected = 0
    for i, value in enumerate(stream_data):
        # Create a dummy dataframe row
        row = pd.DataFrame({'feature_1': [value]})
        if i == 0:
            detector.set_reference_data(row)
        
        result = detector.detect_drift_streaming(row)
        if result['drift_detected']:
            drifts_detected += 1
            
    assert drifts_detected > 0
