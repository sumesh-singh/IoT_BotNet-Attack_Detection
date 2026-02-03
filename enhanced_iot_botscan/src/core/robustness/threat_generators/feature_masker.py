"""
Feature Masker for ARM - Simulates sensor failures and missing data.
Author: Enhanced IoT BotScan Team
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class FeatureMasker:
    """Simulate sensor failures by masking/zeroing features."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        logger.info("FeatureMasker initialized")
    
    def mask_random_features(self, X: np.ndarray, mask_rate: float = 0.1) -> np.ndarray:
        """Randomly set features to zero (simulates sensor dropout).
        
        Args:
            X: Input features (n_samples, n_features)
            mask_rate: Fraction of features to mask per sample
            
        Returns:
            Masked features
        """
        X_masked = X.copy()
        n_features = X.shape[1]
        n_mask = max(1, int(n_features * mask_rate))
        
        for i in range(X.shape[0]):
            mask_idx = np.random.choice(n_features, n_mask, replace=False)
            X_masked[i, mask_idx] = 0
        
        return X_masked
    
    # Alias for compatibility with arm_robustness_monitor.py
    def random_feature_masking(self, X: np.ndarray, mask_rate: float = 0.1) -> np.ndarray:
        """Alias for mask_random_features (compatibility with ARM)."""
        return self.mask_random_features(X, mask_rate)
    
    def mask_specific_features(self, X: np.ndarray, 
                               feature_indices: List[int]) -> np.ndarray:
        """Mask specific features (simulates known sensor failure).
        
        Args:
            X: Input features
            feature_indices: Indices of features to mask
            
        Returns:
            Masked features
        """
        X_masked = X.copy()
        for idx in feature_indices:
            if 0 <= idx < X.shape[1]:
                X_masked[:, idx] = 0
        return X_masked
    
    def mask_with_mean(self, X: np.ndarray, mask_rate: float = 0.1) -> np.ndarray:
        """Replace masked values with feature mean (mean imputation).
        
        Args:
            X: Input features
            mask_rate: Fraction of features to mask per sample
            
        Returns:
            Masked features with mean imputation
        """
        X_masked = X.copy()
        feature_means = np.mean(X, axis=0)
        n_features = X.shape[1]
        n_mask = max(1, int(n_features * mask_rate))
        
        for i in range(X.shape[0]):
            mask_idx = np.random.choice(n_features, n_mask, replace=False)
            X_masked[i, mask_idx] = feature_means[mask_idx]
        
        return X_masked
    
    def cascade_failure(self, X: np.ndarray, 
                        failure_rate: float = 0.05) -> np.ndarray:
        """Simulate cascading sensor failure (correlated failures).
        
        Sensors in IoT networks often fail in groups due to network partitions
        or power failures.
        
        Args:
            X: Input features
            failure_rate: Probability of triggering cascade
            
        Returns:
            Features with cascaded failures
        """
        X_masked = X.copy()
        n_features = X.shape[1]
        
        for i in range(X.shape[0]):
            if np.random.rand() < failure_rate:
                # Cascade: fail a contiguous group of features
                start = np.random.randint(0, n_features - 1)
                length = np.random.randint(1, max(2, n_features // 4))
                end = min(start + length, n_features)
                X_masked[i, start:end] = 0
        
        return X_masked
