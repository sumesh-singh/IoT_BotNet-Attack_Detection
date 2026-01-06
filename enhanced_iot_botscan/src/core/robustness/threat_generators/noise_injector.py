"""
Noise Injector for ARM - Simulates sensor noise and environmental interference.
Author: Enhanced IoT BotScan Team
"""

import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class NoiseInjector:
    """Inject various types of noise to test model robustness."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.noise_types = ['gaussian', 'uniform', 'salt_pepper']
        logger.info("NoiseInjector initialized")
    
    def inject_gaussian_noise(self, X: np.ndarray, scale: float = 0.1) -> np.ndarray:
        """Add Gaussian noise scaled by feature standard deviation.
        
        Args:
            X: Input features (n_samples, n_features)
            scale: Noise level as fraction of feature std
            
        Returns:
            Noisy features
        """
        std = np.std(X, axis=0)
        std = np.where(std == 0, 1, std)  # Avoid zero std
        noise = np.random.randn(*X.shape) * std * scale
        return X + noise
    
    def inject_uniform_noise(self, X: np.ndarray, scale: float = 0.1) -> np.ndarray:
        """Add uniform noise scaled by feature range.
        
        Args:
            X: Input features
            scale: Noise level as fraction of feature range
            
        Returns:
            Noisy features
        """
        feature_range = np.ptp(X, axis=0)
        feature_range = np.where(feature_range == 0, 1, feature_range)
        noise = (np.random.rand(*X.shape) - 0.5) * 2 * feature_range * scale
        return X + noise
    
    def inject_salt_pepper_noise(self, X: np.ndarray, rate: float = 0.05) -> np.ndarray:
        """Randomly set some values to min/max (simulates sensor spikes).
        
        Args:
            X: Input features
            rate: Fraction of values to corrupt
            
        Returns:
            Noisy features
        """
        X_noisy = X.copy()
        n_corrupt = int(X.size * rate)
        
        # Random positions
        flat_indices = np.random.choice(X.size, n_corrupt, replace=False)
        row_idx = flat_indices // X.shape[1]
        col_idx = flat_indices % X.shape[1]
        
        # Randomly set to min or max
        for i in range(len(row_idx)):
            r, c = row_idx[i], col_idx[i]
            if np.random.rand() > 0.5:
                X_noisy[r, c] = X[:, c].max()
            else:
                X_noisy[r, c] = X[:, c].min()
        
        return X_noisy
    
    def inject_noise(self, X: np.ndarray, noise_type: str = 'gaussian', 
                     scale: float = 0.1) -> np.ndarray:
        """General noise injection interface.
        
        Args:
            X: Input features
            noise_type: Type of noise ('gaussian', 'uniform', 'salt_pepper')
            scale: Noise intensity
            
        Returns:
            Noisy features
        """
        if noise_type == 'gaussian':
            return self.inject_gaussian_noise(X, scale)
        elif noise_type == 'uniform':
            return self.inject_uniform_noise(X, scale)
        elif noise_type == 'salt_pepper':
            return self.inject_salt_pepper_noise(X, scale)
        else:
            logger.warning(f"Unknown noise type: {noise_type}, using gaussian")
            return self.inject_gaussian_noise(X, scale)
