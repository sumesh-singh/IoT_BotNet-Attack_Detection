"""
Scaler Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Handles feature scaling using various methods including StandardScaler, MinMaxScaler, and RobustScaler.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.preprocessing import PowerTransformer, Normalizer
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class Scaler:
    """Comprehensive feature scaling for IoT botnet detection."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize scaler with configuration."""

        self.config = config or {}
        self.scaler = None
        self.is_fitted = False
        self.scaling_stats = {}

        # Scaling configuration
        self.method = self.config.get('method', 'standard')
        self.scale_features = self.config.get('scale_features', True)
        self.scale_target = self.config.get('scale_target', False)

        # Method-specific parameters
        self.standard_params = self.config.get('standard_params', {})
        self.minmax_params = self.config.get('minmax_params', {})
        self.robust_params = self.config.get('robust_params', {})
        self.quantile_params = self.config.get('quantile_params', {})
        self.power_params = self.config.get('power_params', {})

        logger.info(f"Scaler initialized with method: {self.method}")

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Fit scaler and transform data.

        Args:
            X: Input features
            y: Target labels (optional)

        Returns:
            Tuple of (scaled_features, scaled_target)
        """

        logger.info(
            f"Starting feature scaling on {len(X)} samples with {len(X.columns)} features")

        # Scale features
        X_scaled = self._scale_features(X, fit=True)

        # Scale target if provided
        y_scaled = None
        if y is not None and self.scale_target:
            y_scaled = self._scale_target(y, fit=True)

        # Record scaling statistics
        self.scaling_stats = {
            'method': self.method,
            'n_features': len(X.columns),
            'n_samples': len(X),
            'scale_features': self.scale_features,
            'scale_target': self.scale_target
        }

        logger.info(f"Feature scaling completed using {self.method} method")

        return X_scaled, y_scaled

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Transform new data using fitted scaler."""

        if not self.is_fitted:
            raise ValueError(
                "Scaler must be fitted before transforming new data")

        # Scale features
        X_scaled = self._scale_features(X, fit=False)

        # Scale target if provided
        y_scaled = None
        if y is not None and self.scale_target:
            y_scaled = self._scale_target(y, fit=False)

        return X_scaled, y_scaled

    def inverse_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Inverse transform scaled data back to original scale."""

        if not self.is_fitted:
            raise ValueError(
                "Scaler must be fitted before inverse transforming")

        # Inverse transform features
        X_original = self._inverse_scale_features(X)

        # Inverse transform target if provided
        y_original = None
        if y is not None and self.scale_target:
            y_original = self._inverse_scale_target(y)

        return X_original, y_original

    def _scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale features using specified method."""

        if not self.scale_features:
            return X

        numerical_cols = X.select_dtypes(include=[np.number]).columns

        if len(numerical_cols) == 0:
            return X

        X_numerical = X[numerical_cols]

        # Initialize scaler based on method
        if fit:
            if self.method == 'standard':
                self.scaler = StandardScaler(**self.standard_params)
            elif self.method == 'minmax':
                self.scaler = MinMaxScaler(**self.minmax_params)
            elif self.method == 'robust':
                self.scaler = RobustScaler(**self.robust_params)
            elif self.method == 'quantile':
                self.scaler = QuantileTransformer(**self.quantile_params)
            elif self.method == 'power':
                self.scaler = PowerTransformer(**self.power_params)
            elif self.method == 'normalizer':
                self.scaler = Normalizer()
            else:
                logger.warning(f"Unknown scaling method: {self.method}")
                return X

            X_scaled = self.scaler.fit_transform(X_numerical)
            self.is_fitted = True

        else:
            X_scaled = self.scaler.transform(X_numerical)

        # Create scaled DataFrame
        X_scaled_df = pd.DataFrame(
            X_scaled, columns=numerical_cols, index=X.index)

        # Combine with non-numerical columns
        non_numerical_cols = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numerical_cols) > 0:
            X_scaled_df = pd.concat(
                [X_scaled_df, X[non_numerical_cols]], axis=1)

        return X_scaled_df

    def _scale_target(self, y: pd.Series, fit: bool = True) -> pd.Series:
        """Scale target variable."""

        if not self.scale_target:
            return y

        # Use MinMaxScaler for target scaling
        if fit:
            self.target_scaler = MinMaxScaler()
            y_scaled = self.target_scaler.fit_transform(
                y.values.reshape(-1, 1)).flatten()
        else:
            y_scaled = self.target_scaler.transform(
                y.values.reshape(-1, 1)).flatten()

        return pd.Series(y_scaled, index=y.index, name=y.name)

    def _inverse_scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform scaled features."""

        if not self.scale_features or self.scaler is None:
            return X

        numerical_cols = X.select_dtypes(include=[np.number]).columns

        if len(numerical_cols) == 0:
            return X

        X_numerical = X[numerical_cols]

        # Inverse transform
        X_original = self.scaler.inverse_transform(X_numerical)

        # Create original DataFrame
        X_original_df = pd.DataFrame(
            X_original, columns=numerical_cols, index=X.index)

        # Combine with non-numerical columns
        non_numerical_cols = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numerical_cols) > 0:
            X_original_df = pd.concat(
                [X_original_df, X[non_numerical_cols]], axis=1)

        return X_original_df

    def _inverse_scale_target(self, y: pd.Series) -> pd.Series:
        """Inverse transform scaled target."""

        if not self.scale_target or not hasattr(self, 'target_scaler'):
            return y

        y_original = self.target_scaler.inverse_transform(
            y.values.reshape(-1, 1)).flatten()

        return pd.Series(y_original, index=y.index, name=y.name)

    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling statistics."""

        stats = self.scaling_stats.copy()

        if self.scaler is not None and hasattr(self.scaler, 'mean_'):
            stats['feature_means'] = self.scaler.mean_.tolist()
            stats['feature_stds'] = self.scaler.scale_.tolist()

        if hasattr(self, 'target_scaler') and self.target_scaler is not None:
            stats['target_min'] = self.target_scaler.data_min_[0]
            stats['target_max'] = self.target_scaler.data_max_[0]

        return stats

    def get_scaling_report(self) -> Dict[str, Any]:
        """Get comprehensive scaling report."""

        return {
            'scaling_stats': self.scaling_stats,
            'method': self.method,
            'is_fitted': self.is_fitted,
            'scale_features': self.scale_features,
            'scale_target': self.scale_target
        }

    def save_scaler(self, filepath: str) -> None:
        """Save fitted scaler to disk."""

        if not self.is_fitted:
            raise ValueError("Cannot save unfitted scaler")

        scaler_data = {
            'scaler': self.scaler,
            'target_scaler': getattr(self, 'target_scaler', None),
            'method': self.method,
            'scaling_stats': self.scaling_stats,
            'is_fitted': self.is_fitted,
            'scale_features': self.scale_features,
            'scale_target': self.scale_target,
            'config': self.config
        }

        joblib.dump(scaler_data, filepath)
        logger.info(f"Scaler saved to {filepath}")

    def load_scaler(self, filepath: str) -> None:
        """Load fitted scaler from disk."""

        if not Path(filepath).exists():
            raise FileNotFoundError(f"Scaler file not found: {filepath}")

        scaler_data = joblib.load(filepath)

        self.scaler = scaler_data['scaler']
        self.target_scaler = scaler_data.get('target_scaler')
        self.method = scaler_data['method']
        self.scaling_stats = scaler_data['scaling_stats']
        self.is_fitted = scaler_data['is_fitted']
        self.scale_features = scaler_data['scale_features']
        self.scale_target = scaler_data['scale_target']
        self.config = scaler_data['config']

        logger.info(f"Scaler loaded from {filepath}")

    def compare_scaling_methods(self, X: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Compare different scaling methods on the same data."""

        numerical_cols = X.select_dtypes(include=[np.number]).columns

        if len(numerical_cols) == 0:
            return {}

        X_numerical = X[numerical_cols]

        methods = ['standard', 'minmax', 'robust', 'quantile']
        results = {}

        for method in methods:
            try:
                # Create temporary scaler
                if method == 'standard':
                    temp_scaler = StandardScaler()
                elif method == 'minmax':
                    temp_scaler = MinMaxScaler()
                elif method == 'robust':
                    temp_scaler = RobustScaler()
                elif method == 'quantile':
                    temp_scaler = QuantileTransformer()

                # Fit and transform
                X_scaled = temp_scaler.fit_transform(X_numerical)

                # Calculate statistics
                results[method] = {
                    'mean': np.mean(X_scaled),
                    'std': np.std(X_scaled),
                    'min': np.min(X_scaled),
                    'max': np.max(X_scaled),
                    'skewness': pd.DataFrame(X_scaled).skew().mean(),
                    'kurtosis': pd.DataFrame(X_scaled).kurtosis().mean()
                }

            except Exception as e:
                logger.warning(f"Failed to apply {method} scaling: {e}")
                results[method] = {'error': str(e)}

        return results


# Example usage and testing
if __name__ == '__main__':
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Create target with different scale
    y = pd.Series(np.random.randn(n_samples) * 100 + 1000)

    print("Original data shape:", X.shape)
    print("Original feature means:", X.mean().round(2).tolist())
    print("Original feature stds:", X.std().round(2).tolist())

    # Test StandardScaler
    scaler = Scaler({
        'method': 'standard',
        'scale_features': True,
        'scale_target': True
    })

    X_scaled, y_scaled = scaler.fit_transform(X, y)
    print("\nStandardScaler results:")
    print("Scaled feature means:", X_scaled.mean().round(2).tolist())
    print("Scaled feature stds:", X_scaled.std().round(2).tolist())
    print("Scaled target range:",
          f"{y_scaled.min():.2f} to {y_scaled.max():.2f}")

    # Test inverse transform
    X_original, y_original = scaler.inverse_transform(X_scaled, y_scaled)
    print("\nInverse transform verification:")
    print("Original feature means:", X_original.mean().round(2).tolist())
    print("Original target range:",
          f"{y_original.min():.2f} to {y_original.max():.2f}")

    # Test MinMaxScaler
    scaler_mm = Scaler({
        'method': 'minmax',
        'scale_features': True,
        'scale_target': False
    })

    X_mm, _ = scaler_mm.fit_transform(X)
    print("\nMinMaxScaler results:")
    print("Scaled feature means:", X_mm.mean().round(2).tolist())
    print("Scaled feature stds:", X_mm.std().round(2).tolist())
    print("Scaled feature range:",
          f"{X_mm.min().min():.2f} to {X_mm.max().max():.2f}")

    # Compare scaling methods
    comparison = scaler.compare_scaling_methods(X)
    print("\nScaling Method Comparison:")
    for method, stats in comparison.items():
        if 'error' not in stats:
            print(f"{method}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
                  f"range=[{stats['min']:.2f}, {stats['max']:.2f}]")

    # Get scaling report
    report = scaler.get_scaling_report()
    print("\nScaling Report:")
    print(report)
