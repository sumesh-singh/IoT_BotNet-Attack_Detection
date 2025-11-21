"""
Unit Tests for Preprocessing Module
Author: Kotiwale Sumesh Singh (160124862043)
"""

import pytest
import numpy as np
import pandas as pd
from src.core.preprocessing.data_cleaner import DataCleaner
from src.core.preprocessing.feature_engineer import FeatureEngineer

@pytest.fixture
def dirty_data():
    np.random.seed(42)
    df = pd.DataFrame({
        'A': np.random.normal(0, 1, 100),
        'B': np.random.normal(5, 2, 100),
        'C': np.random.choice(['x', 'y', 'z'], 100)
    })
    # Add missing values
    df.loc[0:5, 'A'] = np.nan
    # Add duplicates
    df = pd.concat([df, df.iloc[0:5]])
    return df

@pytest.fixture
def clean_data():
    np.random.seed(42)
    return pd.DataFrame({
        'A': np.random.normal(0, 1, 100),
        'B': np.random.normal(5, 2, 100),
        'C': np.random.choice(['x', 'y', 'z'], 100)
    })

def test_data_cleaner_initialization():
    cleaner = DataCleaner({'outlier_threshold': 2.5})
    assert cleaner.outlier_threshold == 2.5

def test_data_cleaner_cleaning(dirty_data):
    cleaner = DataCleaner()
    cleaned_df = cleaner.clean_dataset(dirty_data)
    
    assert cleaned_df.isnull().sum().sum() == 0
    assert cleaned_df.duplicated().sum() == 0
    assert len(cleaned_df) <= len(dirty_data)

def test_feature_engineer_initialization():
    engineer = FeatureEngineer({'create_polynomial_features': True})
    assert engineer.create_polynomial_features

def test_feature_engineer_engineering(clean_data):
    engineer = FeatureEngineer({
        'create_statistical_features': True,
        'create_polynomial_features': False,
        'create_interaction_features': True
    })
    
    # Create dummy target
    y = pd.Series(np.random.randint(0, 2, len(clean_data)))
    
    engineered_df = engineer.engineer_features(clean_data, y)
    
    assert len(engineered_df.columns) > len(clean_data.columns)
    assert 'row_mean' in engineered_df.columns
