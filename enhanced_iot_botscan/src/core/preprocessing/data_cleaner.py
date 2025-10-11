"""
Data Cleaner Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Handles data cleaning operations including missing values, duplicates, outliers, and data validation.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings

logger = logging.getLogger(__name__)


class DataCleaner:
    """Comprehensive data cleaning for IoT botnet detection datasets."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize data cleaner with configuration."""

        self.config = config or {}
        self.cleaning_stats = {}
        self.outlier_threshold = self.config.get(
            'outlier_threshold', 3.0)  # Z-score threshold
        self.missing_threshold = self.config.get(
            'missing_threshold', 0.5)  # Max missing ratio
        self.duplicate_threshold = self.config.get(
            'duplicate_threshold', 0.95)  # Similarity threshold

        logger.info("DataCleaner initialized")

    def clean_dataset(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Perform comprehensive data cleaning.

        Args:
            df: Input DataFrame
            target_column: Name of target column (if any)

        Returns:
            Cleaned DataFrame
        """

        logger.info(
            f"Starting data cleaning on {len(df)} samples with {len(df.columns)} features")

        original_shape = df.shape
        df_cleaned = df.copy()

        # Step 1: Handle missing values
        df_cleaned = self._handle_missing_values(df_cleaned)

        # Step 2: Remove duplicates
        df_cleaned = self._remove_duplicates(df_cleaned)

        # Step 3: Handle outliers
        df_cleaned = self._handle_outliers(df_cleaned, target_column)

        # Step 4: Data type optimization
        df_cleaned = self._optimize_data_types(df_cleaned)

        # Step 5: Validate data integrity
        df_cleaned = self._validate_data_integrity(df_cleaned)

        # Record cleaning statistics
        self.cleaning_stats = {
            'original_shape': original_shape,
            'cleaned_shape': df_cleaned.shape,
            'rows_removed': original_shape[0] - df_cleaned.shape[0],
            'columns_removed': original_shape[1] - df_cleaned.shape[1],
            'missing_values_handled': True,
            'duplicates_removed': True,
            'outliers_handled': True
        }

        logger.info(
            f"Data cleaning completed. Shape: {original_shape} -> {df_cleaned.shape}")

        return df_cleaned

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""

        logger.info("Handling missing values...")

        missing_stats = df.isnull().sum()
        missing_ratio = missing_stats / len(df)

        # Remove columns with too many missing values
        columns_to_drop = missing_ratio[missing_ratio >
                                        self.missing_threshold].index
        if len(columns_to_drop) > 0:
            logger.info(
                f"Dropping {len(columns_to_drop)} columns with >{self.missing_threshold*100}% missing values")
            df = df.drop(columns=columns_to_drop)

        # Handle remaining missing values
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                if df[column].dtype in ['object', 'category']:
                    # For categorical columns, use mode
                    mode_value = df[column].mode()
                    if len(mode_value) > 0:
                        df[column] = df[column].fillna(mode_value[0])
                    else:
                        df[column] = df[column].fillna('Unknown')
                else:
                    # For numerical columns, use median
                    df[column] = df[column].fillna(df[column].median())

        logger.info("Missing values handled")
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows from the dataset."""

        logger.info("Removing duplicates...")

        initial_rows = len(df)

        # Remove exact duplicates
        df = df.drop_duplicates()

        # Remove near-duplicates using similarity threshold
        if self.duplicate_threshold < 1.0:
            df = self._remove_near_duplicates(df)

        duplicates_removed = initial_rows - len(df)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")

        return df

    def _remove_near_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove near-duplicate rows based on similarity threshold."""

        # For large datasets, sample for efficiency
        if len(df) > 10000:
            sample_size = 5000
            df_sample = df.sample(n=sample_size, random_state=42)
            logger.info(
                f"Sampling {sample_size} rows for near-duplicate detection")
        else:
            df_sample = df

        # Calculate pairwise distances for numerical columns only
        numerical_cols = df_sample.select_dtypes(include=[np.number]).columns

        if len(numerical_cols) == 0:
            return df

        # Normalize numerical data
        df_numerical = df_sample[numerical_cols]
        df_normalized = (df_numerical - df_numerical.mean()) / \
            df_numerical.std()

        # Find near-duplicates using correlation
        from sklearn.metrics.pairwise import cosine_similarity

        similarity_matrix = cosine_similarity(df_normalized)

        # Find pairs above threshold
        near_duplicate_pairs = []
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i, j] > self.duplicate_threshold:
                    near_duplicate_pairs.append((i, j))

        # Remove duplicates (keep first occurrence)
        indices_to_remove = set()
        for i, j in near_duplicate_pairs:
            indices_to_remove.add(j)

        if indices_to_remove:
            logger.info(
                f"Found {len(near_duplicate_pairs)} near-duplicate pairs")
            df_sample = df_sample.drop(
                df_sample.index[list(indices_to_remove)])

        return df_sample

    def _handle_outliers(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """Handle outliers in numerical columns."""

        logger.info("Handling outliers...")

        numerical_cols = df.select_dtypes(include=[np.number]).columns

        if target_column and target_column in numerical_cols:
            numerical_cols = numerical_cols.drop(target_column)

        outliers_removed = 0

        for column in numerical_cols:
            # Use Z-score method for outlier detection
            z_scores = np.abs(stats.zscore(df[column]))
            outlier_mask = z_scores > self.outlier_threshold

            if outlier_mask.sum() > 0:
                # Cap outliers instead of removing them
                q1 = df[column].quantile(0.25)
                q3 = df[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                df.loc[outlier_mask, column] = np.clip(
                    df.loc[outlier_mask, column], lower_bound, upper_bound
                )

                outliers_removed += outlier_mask.sum()

        if outliers_removed > 0:
            logger.info(f"Capped {outliers_removed} outliers")

        return df

    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types to reduce memory usage."""

        logger.info("Optimizing data types...")

        for column in df.columns:
            if df[column].dtype == 'object':
                # Try to convert to numeric
                try:
                    df[column] = pd.to_numeric(df[column], errors='ignore')
                except:
                    pass

            elif df[column].dtype in ['int64', 'int32']:
                # Downcast integers
                if df[column].min() >= 0:
                    if df[column].max() < 255:
                        df[column] = df[column].astype('uint8')
                    elif df[column].max() < 65535:
                        df[column] = df[column].astype('uint16')
                    elif df[column].max() < 4294967295:
                        df[column] = df[column].astype('uint32')
                else:
                    if df[column].min() > -128 and df[column].max() < 127:
                        df[column] = df[column].astype('int8')
                    elif df[column].min() > -32768 and df[column].max() < 32767:
                        df[column] = df[column].astype('int16')
                    elif df[column].min() > -2147483648 and df[column].max() < 2147483647:
                        df[column] = df[column].astype('int32')

            elif df[column].dtype == 'float64':
                # Downcast floats
                df[column] = pd.to_numeric(df[column], downcast='float')

        logger.info("Data types optimized")
        return df

    def _validate_data_integrity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data integrity and fix common issues."""

        logger.info("Validating data integrity...")

        # Check for infinite values
        inf_mask = np.isinf(df.select_dtypes(include=[np.number]))
        if inf_mask.any().any():
            logger.warning("Found infinite values, replacing with NaN")
            df = df.replace([np.inf, -np.inf], np.nan)

            # Fill NaN values created by replacing inf
            for column in df.columns:
                if df[column].isnull().sum() > 0:
                    if df[column].dtype in ['object', 'category']:
                        df[column] = df[column].fillna('Unknown')
                    else:
                        df[column] = df[column].fillna(df[column].median())

        # Check for constant columns
        constant_columns = []
        for column in df.columns:
            if df[column].nunique() <= 1:
                constant_columns.append(column)

        if constant_columns:
            logger.warning(
                f"Found {len(constant_columns)} constant columns: {constant_columns}")
            # Optionally remove constant columns
            if self.config.get('remove_constant_columns', False):
                df = df.drop(columns=constant_columns)

        logger.info("Data integrity validated")
        return df

    def get_cleaning_report(self) -> Dict[str, Any]:
        """Get comprehensive cleaning report."""

        return {
            'cleaning_stats': self.cleaning_stats,
            'outlier_threshold': self.outlier_threshold,
            'missing_threshold': self.missing_threshold,
            'duplicate_threshold': self.duplicate_threshold
        }

    def validate_cleaned_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate cleaned data quality."""

        validation_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }

        # Check for outliers
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        outlier_counts = {}

        for column in numerical_cols:
            z_scores = np.abs(stats.zscore(df[column]))
            outlier_counts[column] = (z_scores > self.outlier_threshold).sum()

        validation_report['outlier_counts'] = outlier_counts

        return validation_report


# Example usage and testing
if __name__ == '__main__':
    # Create sample data with various issues
    np.random.seed(42)
    n_samples = 1000

    # Create data with missing values, duplicates, and outliers
    data = {
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(5, 2, n_samples),
        'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature_4': np.random.exponential(1, n_samples)
    }

    df = pd.DataFrame(data)

    # Introduce issues
    df.loc[0:50, 'feature_1'] = np.nan  # Missing values
    df.loc[100:150, 'feature_2'] = 100  # Outliers
    df = pd.concat([df, df.iloc[200:250]])  # Duplicates

    print("Original data shape:", df.shape)
    print("Missing values:", df.isnull().sum().sum())
    print("Duplicates:", df.duplicated().sum())

    # Clean data
    cleaner = DataCleaner()
    df_cleaned = cleaner.clean_dataset(df)

    print("\nCleaned data shape:", df_cleaned.shape)
    print("Missing values:", df_cleaned.isnull().sum().sum())
    print("Duplicates:", df_cleaned.duplicated().sum())

    # Get cleaning report
    report = cleaner.get_cleaning_report()
    print("\nCleaning Report:")
    print(report)

    # Validate cleaned data
    validation = cleaner.validate_cleaned_data(df_cleaned)
    print("\nValidation Report:")
    print(validation)
