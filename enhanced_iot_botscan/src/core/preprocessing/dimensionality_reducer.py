"""
Dimensionality Reducer Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Handles dimensionality reduction using PCA and other techniques for IoT botnet detection.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.manifold import TSNE, Isomap
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class DimensionalityReducer:
    """Comprehensive dimensionality reduction for IoT botnet detection."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize dimensionality reducer with configuration."""

        self.config = config or {}
        self.reducer = None
        self.scaler = None
        self.is_fitted = False
        self.reduction_stats = {}

        # Dimensionality reduction configuration
        self.method = self.config.get('method', 'pca')
        self.n_components = self.config.get('n_components', None)
        self.variance_threshold = self.config.get('variance_threshold', 0.95)
        self.min_variance = self.config.get('min_variance', 0.01)

        # Method-specific parameters
        self.pca_params = self.config.get('pca_params', {})
        self.ica_params = self.config.get('ica_params', {})
        self.tsne_params = self.config.get('tsne_params', {})

        logger.info(
            f"DimensionalityReducer initialized with method: {self.method}")

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit dimensionality reducer and transform data.

        Args:
            X: Input features
            y: Target labels (optional, for supervised methods)

        Returns:
            Transformed features DataFrame
        """

        logger.info(
            f"Starting dimensionality reduction on {len(X)} samples with {len(X.columns)} features")

        # Step 1: Remove low variance features
        X_reduced = self._remove_low_variance_features(X)

        # Step 2: Scale features
        X_scaled = self._scale_features(X_reduced)

        # Step 3: Apply dimensionality reduction
        X_transformed = self._apply_reduction(X_scaled, y)

        # Record reduction statistics
        self.reduction_stats = {
            'original_features': len(X.columns),
            'after_variance_filter': len(X_reduced.columns),
            'final_features': len(X_transformed.columns),
            'reduction_ratio': len(X_transformed.columns) / len(X.columns),
            'method': self.method
        }

        logger.info(
            f"Dimensionality reduction completed. Shape: {X.shape} -> {X_transformed.shape}")

        return X_transformed

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted reducer."""

        if not self.is_fitted:
            raise ValueError(
                "Reducer must be fitted before transforming new data")

        # Apply same preprocessing steps
        X_reduced = self._remove_low_variance_features(X)
        X_scaled = self._scale_features(X_reduced)
        X_transformed = self._apply_reduction(X_scaled, fit=False)

        return X_transformed

    def _remove_low_variance_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove features with low variance."""

        numerical_cols = X.select_dtypes(include=[np.number]).columns

        if len(numerical_cols) == 0:
            return X

        # Remove features with variance below threshold
        variance_selector = VarianceThreshold(threshold=self.min_variance)
        X_numerical = X[numerical_cols]

        try:
            X_numerical_selected = variance_selector.fit_transform(X_numerical)
            selected_features = numerical_cols[variance_selector.get_support()]

            # Combine with non-numerical columns
            non_numerical_cols = X.select_dtypes(exclude=[np.number]).columns
            X_reduced = pd.DataFrame(
                X_numerical_selected, columns=selected_features, index=X.index)
            X_reduced = pd.concat([X_reduced, X[non_numerical_cols]], axis=1)

            logger.info(
                f"Removed {len(numerical_cols) - len(selected_features)} low variance features")

        except Exception as e:
            logger.warning(f"Variance thresholding failed: {e}")
            X_reduced = X

        return X_reduced

    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale features for dimensionality reduction."""

        numerical_cols = X.select_dtypes(include=[np.number]).columns

        if len(numerical_cols) == 0:
            return X

        # Initialize scaler if not already done
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X[numerical_cols])
        else:
            X_scaled = self.scaler.transform(X[numerical_cols])

        # Create scaled DataFrame
        X_scaled_df = pd.DataFrame(
            X_scaled, columns=numerical_cols, index=X.index)

        # Combine with non-numerical columns
        non_numerical_cols = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numerical_cols) > 0:
            X_scaled_df = pd.concat(
                [X_scaled_df, X[non_numerical_cols]], axis=1)

        return X_scaled_df

    def _apply_reduction(self, X: pd.DataFrame, y: Optional[pd.Series] = None, fit: bool = True) -> pd.DataFrame:
        """Apply dimensionality reduction method."""

        numerical_cols = X.select_dtypes(include=[np.number]).columns

        if len(numerical_cols) == 0:
            return X

        X_numerical = X[numerical_cols]

        if self.method == 'pca':
            X_transformed = self._apply_pca(X_numerical, fit)
        elif self.method == 'ica':
            X_transformed = self._apply_ica(X_numerical, fit)
        elif self.method == 'svd':
            X_transformed = self._apply_svd(X_numerical, fit)
        elif self.method == 'tsne':
            X_transformed = self._apply_tsne(X_numerical, fit)
        elif self.method == 'isomap':
            X_transformed = self._apply_isomap(X_numerical, fit)
        else:
            logger.warning(f"Unknown reduction method: {self.method}")
            return X

        # Create transformed DataFrame
        transformed_cols = [
            f'{self.method}_component_{i}' for i in range(X_transformed.shape[1])]
        X_transformed_df = pd.DataFrame(
            X_transformed, columns=transformed_cols, index=X.index)

        return X_transformed_df

    def _apply_pca(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Apply Principal Component Analysis."""

        # Determine number of components
        n_components = self.n_components
        if n_components is None:
            # Use variance threshold
            n_components = min(X.shape[1], X.shape[0] - 1)

        # Initialize PCA
        pca_params = {
            'n_components': n_components,
            'random_state': 42,
            **self.pca_params
        }

        if fit:
            self.reducer = PCA(**pca_params)
            X_transformed = self.reducer.fit_transform(X)

            # Adjust components based on variance threshold
            if self.variance_threshold < 1.0:
                cumsum_variance = np.cumsum(
                    self.reducer.explained_variance_ratio_)
                n_components_threshold = np.argmax(
                    cumsum_variance >= self.variance_threshold) + 1

                if n_components_threshold < n_components:
                    logger.info(f"Reducing components from {n_components} to {n_components_threshold} "
                                f"to achieve {self.variance_threshold*100}% variance")

                    # Refit with adjusted components
                    pca_params['n_components'] = n_components_threshold
                    self.reducer = PCA(**pca_params)
                    X_transformed = self.reducer.fit_transform(X)
        else:
            X_transformed = self.reducer.transform(X)

        logger.info(
            f"PCA applied. Explained variance ratio: {self.reducer.explained_variance_ratio_.sum():.4f}")

        return X_transformed

    def _apply_ica(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Apply Independent Component Analysis."""

        n_components = self.n_components or min(X.shape[1], X.shape[0] - 1)

        ica_params = {
            'n_components': n_components,
            'random_state': 42,
            **self.ica_params
        }

        if fit:
            self.reducer = FastICA(**ica_params)
            X_transformed = self.reducer.fit_transform(X)
        else:
            X_transformed = self.reducer.transform(X)

        logger.info(f"ICA applied with {n_components} components")

        return X_transformed

    def _apply_svd(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Apply Truncated SVD."""

        n_components = self.n_components or min(X.shape[1], X.shape[0] - 1)

        svd_params = {
            'n_components': n_components,
            'random_state': 42
        }

        if fit:
            self.reducer = TruncatedSVD(**svd_params)
            X_transformed = self.reducer.fit_transform(X)
        else:
            X_transformed = self.reducer.transform(X)

        logger.info(f"SVD applied with {n_components} components")

        return X_transformed

    def _apply_tsne(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Apply t-SNE (for visualization, not recommended for large datasets)."""

        n_components = self.n_components or 2

        tsne_params = {
            'n_components': n_components,
            'random_state': 42,
            'perplexity': min(30, len(X) - 1),
            **self.tsne_params
        }

        if fit:
            self.reducer = TSNE(**tsne_params)
            X_transformed = self.reducer.fit_transform(X)
        else:
            # t-SNE doesn't support transform, so we need to refit
            logger.warning(
                "t-SNE doesn't support transform. Refitting on new data.")
            self.reducer = TSNE(**tsne_params)
            X_transformed = self.reducer.fit_transform(X)

        logger.info(f"t-SNE applied with {n_components} components")

        return X_transformed

    def _apply_isomap(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Apply Isomap."""

        n_components = self.n_components or min(X.shape[1], X.shape[0] - 1)
        n_neighbors = min(10, len(X) - 1)

        isomap_params = {
            'n_components': n_components,
            'n_neighbors': n_neighbors
        }

        if fit:
            self.reducer = Isomap(**isomap_params)
            X_transformed = self.reducer.fit_transform(X)
        else:
            X_transformed = self.reducer.transform(X)

        logger.info(f"Isomap applied with {n_components} components")

        return X_transformed

    def get_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """Get explained variance ratio (for PCA)."""

        if self.method == 'pca' and self.reducer is not None:
            return self.reducer.explained_variance_ratio_

        return None

    def get_components(self) -> Optional[np.ndarray]:
        """Get component vectors."""

        if self.reducer is not None and hasattr(self.reducer, 'components_'):
            return self.reducer.components_

        return None

    def get_reduction_report(self) -> Dict[str, Any]:
        """Get comprehensive reduction report."""

        report = {
            'reduction_stats': self.reduction_stats,
            'method': self.method,
            'is_fitted': self.is_fitted
        }

        if self.method == 'pca' and self.reducer is not None:
            report['explained_variance_ratio'] = self.reducer.explained_variance_ratio_.tolist()
            report['cumulative_variance'] = np.cumsum(
                self.reducer.explained_variance_ratio_).tolist()

        return report

    def save_reducer(self, filepath: str) -> None:
        """Save fitted reducer to disk."""

        if not self.is_fitted:
            raise ValueError("Cannot save unfitted reducer")

        reducer_data = {
            'reducer': self.reducer,
            'scaler': self.scaler,
            'method': self.method,
            'reduction_stats': self.reduction_stats,
            'is_fitted': self.is_fitted,
            'config': self.config
        }

        joblib.dump(reducer_data, filepath)
        logger.info(f"Dimensionality reducer saved to {filepath}")

    def load_reducer(self, filepath: str) -> None:
        """Load fitted reducer from disk."""

        if not Path(filepath).exists():
            raise FileNotFoundError(f"Reducer file not found: {filepath}")

        reducer_data = joblib.load(filepath)

        self.reducer = reducer_data['reducer']
        self.scaler = reducer_data['scaler']
        self.method = reducer_data['method']
        self.reduction_stats = reducer_data['reduction_stats']
        self.is_fitted = reducer_data['is_fitted']
        self.config = reducer_data['config']

        logger.info(f"Dimensionality reducer loaded from {filepath}")


# Example usage and testing
if __name__ == '__main__':
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    print("Original data shape:", X.shape)

    # Test PCA
    reducer = DimensionalityReducer({
        'method': 'pca',
        'variance_threshold': 0.95,
        'min_variance': 0.01
    })

    X_reduced = reducer.fit_transform(X)
    print("PCA reduced shape:", X_reduced.shape)

    # Get explained variance
    explained_var = reducer.get_explained_variance_ratio()
    if explained_var is not None:
        print(f"Explained variance: {explained_var.sum():.4f}")
        print(f"First 5 components variance: {explained_var[:5]}")

    # Test transformation on new data
    X_new = pd.DataFrame(np.random.randn(100, n_features), columns=[
                         f'feature_{i}' for i in range(n_features)])
    X_new_reduced = reducer.transform(X_new)
    print(f"New data transformation: {X_new.shape} -> {X_new_reduced.shape}")

    # Get report
    report = reducer.get_reduction_report()
    print("\nReduction Report:")
    print(report)

    # Test ICA
    reducer_ica = DimensionalityReducer({
        'method': 'ica',
        'n_components': 10
    })

    X_ica = reducer_ica.fit_transform(X)
    print(f"\nICA reduced shape: {X_ica.shape}")
