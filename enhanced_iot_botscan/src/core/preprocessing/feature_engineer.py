"""
Feature Engineer Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Handles feature engineering operations including feature creation, selection, and transformation.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
import warnings

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Comprehensive feature engineering for IoT botnet detection."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize feature engineer with configuration."""

        self.config = config or {}
        self.feature_stats = {}
        self.selected_features = None
        self.feature_importance_ = None

        # Feature engineering configuration
        self.create_polynomial_features = self.config.get(
            'create_polynomial_features', False)
        self.polynomial_degree = self.config.get('polynomial_degree', 2)
        self.create_interaction_features = self.config.get(
            'create_interaction_features', False)
        self.create_statistical_features = self.config.get(
            'create_statistical_features', True)

        # Feature selection configuration
        self.feature_selection_method = self.config.get(
            'feature_selection_method', 'mutual_info')
        self.n_features_select = self.config.get('n_features_select', 50)
        self.feature_selection_threshold = self.config.get(
            'feature_selection_threshold', 0.01)

        logger.info("FeatureEngineer initialized")

    def engineer_features(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Perform comprehensive feature engineering.

        Args:
            X: Input features
            y: Target labels (optional, for supervised feature engineering)

        Returns:
            Engineered features DataFrame
        """

        logger.info(
            f"Starting feature engineering on {len(X)} samples with {len(X.columns)} features")

        X_engineered = X.copy()

        # Step 1: Create statistical features
        if self.create_statistical_features:
            X_engineered = self._create_statistical_features(X_engineered)

        # Step 2: Create polynomial features
        if self.create_polynomial_features:
            X_engineered = self._create_polynomial_features(X_engineered)

        # Step 3: Create interaction features
        if self.create_interaction_features:
            X_engineered = self._create_interaction_features(X_engineered)

        # Step 4: Create domain-specific features for IoT botnet detection
        X_engineered = self._create_domain_features(X_engineered)

        # Step 5: Feature selection (if target is provided)
        if y is not None:
            X_engineered = self._select_features(X_engineered, y)

        logger.info(
            f"Feature engineering completed. Shape: {X.shape} -> {X_engineered.shape}")

        return X_engineered

    def _create_statistical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features from existing features."""

        logger.info("Creating statistical features...")

        numerical_cols = X.select_dtypes(include=[np.number]).columns

        if len(numerical_cols) == 0:
            return X

        # Create statistical aggregations
        stats_features = {}

        # Row-wise statistics
        stats_features['row_mean'] = X[numerical_cols].mean(axis=1)
        stats_features['row_std'] = X[numerical_cols].std(axis=1)
        stats_features['row_min'] = X[numerical_cols].min(axis=1)
        stats_features['row_max'] = X[numerical_cols].max(axis=1)
        stats_features['row_median'] = X[numerical_cols].median(axis=1)
        stats_features['row_range'] = stats_features['row_max'] - \
            stats_features['row_min']
        stats_features['row_q75'] = X[numerical_cols].quantile(0.75, axis=1)
        stats_features['row_q25'] = X[numerical_cols].quantile(0.25, axis=1)
        stats_features['row_iqr'] = stats_features['row_q75'] - \
            stats_features['row_q25']

        # Add statistical features to DataFrame
        for name, values in stats_features.items():
            X[name] = values

        logger.info(f"Created {len(stats_features)} statistical features")

        return X

    def _create_polynomial_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features."""

        logger.info("Creating polynomial features...")

        numerical_cols = X.select_dtypes(include=[np.number]).columns

        if len(numerical_cols) == 0:
            return X

        # Limit to top features to avoid explosion
        if len(numerical_cols) > 20:
            # Select top features by variance
            variances = X[numerical_cols].var()
            top_features = variances.nlargest(20).index
            numerical_cols = top_features

        # Create polynomial features
        poly = PolynomialFeatures(
            degree=self.polynomial_degree,
            include_bias=False,
            interaction_only=False
        )

        poly_features = poly.fit_transform(X[numerical_cols])
        poly_feature_names = poly.get_feature_names_out(numerical_cols)

        # Create DataFrame with polynomial features
        poly_df = pd.DataFrame(
            poly_features, columns=poly_feature_names, index=X.index)

        # Remove original features to avoid duplication
        X_poly = X.drop(columns=numerical_cols)
        X_poly = pd.concat([X_poly, poly_df], axis=1)

        logger.info(f"Created {len(poly_feature_names)} polynomial features")

        return X_poly

    def _create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important features."""

        logger.info("Creating interaction features...")

        numerical_cols = X.select_dtypes(include=[np.number]).columns

        if len(numerical_cols) < 2:
            return X

        # Select top features by variance for interactions
        variances = X[numerical_cols].var()
        top_features = variances.nlargest(min(10, len(numerical_cols))).index

        interaction_features = {}

        # Create pairwise interactions
        for i, feat1 in enumerate(top_features):
            for j, feat2 in enumerate(top_features[i+1:], i+1):
                interaction_name = f"{feat1}_x_{feat2}"
                interaction_features[interaction_name] = X[feat1] * X[feat2]

        # Add interaction features
        for name, values in interaction_features.items():
            X[name] = values

        logger.info(
            f"Created {len(interaction_features)} interaction features")

        return X

    def _create_domain_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific features for IoT botnet detection."""

        logger.info("Creating domain-specific features...")

        numerical_cols = X.select_dtypes(include=[np.number]).columns

        if len(numerical_cols) == 0:
            return X

        domain_features = {}

        # Network flow features (assuming some columns represent network metrics)
        flow_cols = [col for col in numerical_cols if any(keyword in col.lower()
                                                          for keyword in ['flow', 'packet', 'byte', 'duration', 'rate'])]

        if len(flow_cols) >= 2:
            # Flow rate features
            if 'packet' in ' '.join(flow_cols).lower() and 'duration' in ' '.join(flow_cols).lower():
                packet_cols = [
                    col for col in flow_cols if 'packet' in col.lower()]
                duration_cols = [
                    col for col in flow_cols if 'duration' in col.lower()]

                if packet_cols and duration_cols:
                    for p_col in packet_cols[:2]:  # Limit to avoid explosion
                        for d_col in duration_cols[:2]:
                            domain_features[f"{p_col}_rate"] = X[p_col] / \
                                (X[d_col] + 1e-8)

            # Byte-to-packet ratio
            byte_cols = [col for col in flow_cols if 'byte' in col.lower()]
            packet_cols = [col for col in flow_cols if 'packet' in col.lower()]

            if byte_cols and packet_cols:
                for b_col in byte_cols[:2]:
                    for p_col in packet_cols[:2]:
                        domain_features[f"{b_col}_per_{p_col}"] = X[b_col] / \
                            (X[p_col] + 1e-8)

        # Statistical features for network behavior
        if len(numerical_cols) >= 5:
            # Coefficient of variation (variability measure)
            domain_features['cv'] = X[numerical_cols].std(
                axis=1) / (X[numerical_cols].mean(axis=1) + 1e-8)

            # Skewness and kurtosis
            domain_features['skewness'] = X[numerical_cols].skew(axis=1)
            domain_features['kurtosis'] = X[numerical_cols].kurtosis(axis=1)

        # Add domain features
        for name, values in domain_features.items():
            X[name] = values

        logger.info(f"Created {len(domain_features)} domain-specific features")

        return X

    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select most important features."""

        logger.info(
            f"Selecting features using {self.feature_selection_method}...")

        numerical_cols = X.select_dtypes(include=[np.number]).columns

        if len(numerical_cols) == 0:
            return X

        X_numerical = X[numerical_cols]

        if self.feature_selection_method == 'mutual_info':
            # Mutual information feature selection
            mi_scores = mutual_info_classif(X_numerical, y, random_state=42)
            feature_scores = pd.Series(mi_scores, index=numerical_cols)

            # Select top features
            top_features = feature_scores.nlargest(
                self.n_features_select).index
            self.feature_importance_ = feature_scores

        elif self.feature_selection_method == 'f_score':
            # F-score feature selection
            f_scores, _ = f_classif(X_numerical, y)
            feature_scores = pd.Series(f_scores, index=numerical_cols)

            # Select top features
            top_features = feature_scores.nlargest(
                self.n_features_select).index
            self.feature_importance_ = feature_scores

        elif self.feature_selection_method == 'random_forest':
            # Random Forest feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_numerical, y)

            feature_scores = pd.Series(
                rf.feature_importances_, index=numerical_cols)

            # Select top features
            top_features = feature_scores.nlargest(
                self.n_features_select).index
            self.feature_importance_ = feature_scores

        elif self.feature_selection_method == 'rfe':
            # Recursive Feature Elimination
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rfe = RFE(rf, n_features_to_select=self.n_features_select)
            rfe.fit(X_numerical, y)

            top_features = numerical_cols[rfe.support_]
            self.feature_importance_ = pd.Series(
                rfe.ranking_, index=numerical_cols)

        else:
            logger.warning(
                f"Unknown feature selection method: {self.feature_selection_method}")
            return X

        # Keep non-numerical columns and selected numerical columns
        non_numerical_cols = X.select_dtypes(exclude=[np.number]).columns
        selected_cols = list(non_numerical_cols) + list(top_features)

        X_selected = X[selected_cols]
        self.selected_features = selected_cols

        logger.info(
            f"Selected {len(top_features)} numerical features from {len(numerical_cols)}")

        return X_selected

    def get_feature_importance(self, top_n: int = 20) -> Optional[pd.Series]:
        """Get feature importance scores."""

        if self.feature_importance_ is None:
            return None

        return self.feature_importance_.nlargest(top_n)

    def get_selected_features(self) -> Optional[List[str]]:
        """Get list of selected features."""

        return self.selected_features

    def get_feature_engineering_report(self) -> Dict[str, Any]:
        """Get comprehensive feature engineering report."""

        return {
            'feature_stats': self.feature_stats,
            'selected_features_count': len(self.selected_features) if self.selected_features else 0,
            'feature_selection_method': self.feature_selection_method,
            'n_features_select': self.n_features_select,
        }

    def get_state(self) -> Dict[str, Any]:
        """Get the internal state of the feature engineer."""
        return {
            'selected_features': self.selected_features,
            'feature_stats': self.feature_stats,
            'feature_importance_': self.feature_importance_,
            'config': self.config
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore internal state."""
        self.selected_features = state.get('selected_features')
        self.feature_stats = state.get('feature_stats', {})
        self.feature_importance_ = state.get('feature_importance_')
        if 'config' in state:
            self.config.update(state['config'])
        
        logger.info(f"FeatureEngineer state restored. Selected features: {len(self.selected_features) if self.selected_features else 0}")

    def restore_state(self, state: Dict[str, Any]):
        """
        Restore internal state (alias for set_state for backward compatibility).
        
        Args:
            state: State dictionary containing selected_features, feature_stats, etc.
        """
        self.set_state(state)
        logger.info("FeatureEngineer state restored via restore_state()")

    def transform_new_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using the same feature engineering pipeline."""
        
        if self.selected_features is None:
            logger.warning("No selected features available. Run feature engineering first.")
            return X
        
        # Apply the same transformations
        X_transformed = X.copy()
        
        # Create statistical features
        if self.create_statistical_features:
            X_transformed = self._create_statistical_features(X_transformed)
        
        # Create polynomial features
        if self.create_polynomial_features:
            X_transformed = self._create_polynomial_features(X_transformed)
        
        # Create interaction features
        if self.create_interaction_features:
            X_transformed = self._create_interaction_features(X_transformed)
        
        # Create domain features
        X_transformed = self._create_domain_features(X_transformed)
        
        # CRITICAL FIX: Handle missing features
        available_features = [col for col in self.selected_features if col in X_transformed.columns]
        missing_features = [col for col in self.selected_features if col not in X_transformed.columns]
        
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features in transformed data. Adding zeros.")
            # Add missing features with zeros
            for feat in missing_features:
                X_transformed[feat] = 0.0
        
        # Select features in the correct order
        X_transformed = X_transformed[self.selected_features]
        
        logger.info(f"Transformed new data. Shape: {X.shape} -> {X_transformed.shape}")
        
        return X_transformed


# Example usage and testing
if __name__ == '__main__':
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Create target with some structure
    y = pd.Series(
        (X.iloc[:, 0] + X.iloc[:, 1] +
         np.random.randn(n_samples) * 0.1 > 0).astype(int)
    )

    print("Original data shape:", X.shape)

    # Initialize feature engineer
    engineer = FeatureEngineer({
        'create_statistical_features': True,
        'create_polynomial_features': False,  # Disable to avoid explosion
        'create_interaction_features': True,
        'feature_selection_method': 'mutual_info',
        'n_features_select': 15
    })

    # Engineer features
    X_engineered = engineer.engineer_features(X, y)

    print("Engineered data shape:", X_engineered.shape)

    # Get feature importance
    importance = engineer.get_feature_importance(10)
    if importance is not None:
        print("\nTop 10 Feature Importance:")
        print(importance)

    # Get selected features
    selected = engineer.get_selected_features()
    print(f"\nSelected features: {len(selected)}")

    # Get report
    report = engineer.get_feature_engineering_report()
    print("\nFeature Engineering Report:")
    print(report)

    # Test transformation on new data
    X_new = pd.DataFrame(np.random.randn(100, n_features), columns=[
                         f'feature_{i}' for i in range(n_features)])
    X_new_transformed = engineer.transform_new_data(X_new)
    print(
        f"\nNew data transformation: {X_new.shape} -> {X_new_transformed.shape}")
