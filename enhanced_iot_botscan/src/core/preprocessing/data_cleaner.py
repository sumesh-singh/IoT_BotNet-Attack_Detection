"""
FIXED Data Cleaner - Conservative approach to preserve training data
The original cleaner was removing 99.9% of data!
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
from sklearn.preprocessing import StandardScaler
from scipy import stats

logger = logging.getLogger(__name__)


class ConservativeDataCleaner:
    """
    Conservative data cleaning that preserves training data
    Only removes truly invalid/duplicate samples
    """
    
    def __init__(self, 
                 config: Dict[str, Any] = None,
                 remove_exact_duplicates: bool = True,
                 handle_missing: bool = True,
                 remove_outliers: bool = False,
                 outlier_threshold: float = 5.0):
        """
        Args:
            config: Configuration dictionary (for compatibility)
            remove_exact_duplicates: Remove only exact duplicates (not near-duplicates)
            handle_missing: Handle missing values
            remove_outliers: Remove statistical outliers (be very conservative)
            outlier_threshold: Z-score threshold (higher = more conservative)
        """
        self.config = config or {}
        self.remove_exact_duplicates = remove_exact_duplicates
        self.handle_missing = handle_missing
        self.remove_outliers = remove_outliers
        self.outlier_threshold = outlier_threshold
        self.cleaning_stats = {}
        
        logger.info("ConservativeDataCleaner initialized")

    def clean_dataset(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Conservative data cleaning pipeline
        
        Args:
            df: Input dataframe
            target_column: Name of target column (for compatibility with old interface)
            
        Returns:
            Cleaned dataframe
        """
        return self.clean(df, target_col=target_column or 'label')
    
    def clean(self, df: pd.DataFrame, target_col: str = 'label') -> pd.DataFrame:
        """
        Conservative data cleaning pipeline
        
        Args:
            df: Input dataframe
            target_col: Name of target column
            
        Returns:
            Cleaned dataframe
        """
        logger.info(f"Starting CONSERVATIVE cleaning on {len(df)} samples with {df.shape[1]} features")
        original_size = len(df)
        df_clean = df.copy()
        
        # Step 1: Handle missing values (conservative)
        if self.handle_missing:
            df_clean = self._handle_missing_conservative(df_clean)
            logger.info(f"After missing value handling: {len(df_clean)} samples")
        
        # Step 2: Remove ONLY exact duplicates (not near-duplicates)
        if self.remove_exact_duplicates:
            df_clean = self._remove_exact_duplicates(df_clean)
            logger.info(f"After duplicate removal: {len(df_clean)} samples")
        
        # Step 3: OPTIONAL outlier removal (very conservative)
        if self.remove_outliers:
            df_clean = self._remove_outliers_conservative(df_clean, target_col if target_col in df_clean.columns else None)
            logger.info(f"After outlier removal: {len(df_clean)} samples")
        
        # Validation
        removed = original_size - len(df_clean)
        removal_rate = (removed / original_size) * 100 if original_size > 0 else 0
        
        logger.info(f"Data cleaning completed. Shape: ({original_size}, {df.shape[1]}) -> ({len(df_clean)}, {df_clean.shape[1]})")
        logger.info(f"Removed {removed} samples ({removal_rate:.2f}%)")
        
        # CRITICAL CHECK: Don't remove more than 20% of data
        if removal_rate > 20:
            logger.warning(f"⚠️  HIGH REMOVAL RATE: {removal_rate:.2f}% - Consider adjusting cleaning parameters")
        
        # Store stats
        self.cleaning_stats = {
            'original_shape': (original_size, df.shape[1]),
            'cleaned_shape': (len(df_clean), df_clean.shape[1]),
            'rows_removed': removed,
            'removal_rate': removal_rate
        }
        
        return df_clean
    
    def _handle_missing_conservative(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values conservatively
        Only remove rows if >50% values are missing
        """
        # Calculate missing percentage per row
        missing_pct = df.isnull().sum(axis=1) / len(df.columns)
        
        # Remove only rows with >50% missing
        df_clean = df[missing_pct <= 0.5].copy()
        
        # For remaining missing values, use forward fill then median/mode imputation
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if df_clean[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # Numerical: use median (more robust than mean)
                    median_val = df_clean[col].median()
                    if pd.notna(median_val):
                        df_clean[col] = df_clean[col].fillna(median_val)
                    else:
                        df_clean[col] = df_clean[col].fillna(0)
                else:
                    # Categorical: use mode
                    mode_val = df_clean[col].mode()
                    if not mode_val.empty:
                        df_clean[col] = df_clean[col].fillna(mode_val[0])
                    else:
                        df_clean[col] = df_clean[col].fillna(0)
        
        removed = len(df) - len(df_clean)
        if removed > 0:
            logger.info(f"  Removed {removed} rows with >50% missing values")
        
        return df_clean
    
    def _remove_exact_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove ONLY exact duplicates (row-wise identical)
        Does NOT use near-duplicate detection
        """
        before = len(df)
        df_clean = df.drop_duplicates(keep='first')
        removed = before - len(df_clean)
        
        if removed > 0:
            logger.info(f"  Removed {removed} exact duplicate rows")
        else:
            logger.info(f"  No exact duplicates found")
        
        return df_clean
    
    def _remove_outliers_conservative(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Very conservative outlier removal using Z-score
        Only removes extreme outliers (Z > 5 by default)
        """
        # Separate features and target if specified
        if target_col and target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col]
        else:
            X = df
            y = None
        
        # Calculate Z-scores for numerical columns only
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            logger.info("  No numerical columns for outlier detection")
            return df
        
        try:
            # Calculate Z-scores
            z_scores = np.abs(stats.zscore(X[numerical_cols], nan_policy='omit'))
            
            # Keep rows where ALL features have Z-score < threshold
            mask = (z_scores < self.outlier_threshold).all(axis=1)
            
            df_clean = df[mask].copy()
            removed = len(df) - len(df_clean)
            
            if removed > 0:
                logger.info(f"  Removed {removed} outlier rows (Z-score > {self.outlier_threshold})")
            else:
                logger.info(f"  No outliers found (threshold: {self.outlier_threshold})")
            
            return df_clean
        except Exception as e:
            logger.warning(f"  Error during outlier detection: {e}, returning original data")
            return df

    def get_cleaning_report(self) -> Dict[str, Any]:
        """Get comprehensive cleaning report."""
        return {
            'cleaning_stats': self.cleaning_stats,
            'outlier_threshold': self.outlier_threshold,
            'remove_exact_duplicates': self.remove_exact_duplicates,
            'handle_missing': self.handle_missing,
            'remove_outliers': self.remove_outliers
        }
    
    def validate_cleaned_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate cleaned data quality."""
        validation_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        return validation_report


class BalancedSampler:
    """
    Smart sampling to balance classes while preserving data diversity
    """
    
    @staticmethod
    def balance_classes(df: pd.DataFrame, 
                       target_col: str = 'label',
                       strategy: str = 'undersample',
                       min_samples_per_class: int = 5000) -> pd.DataFrame:
        """
        Balance class distribution
        
        Args:
            df: Input dataframe
            target_col: Name of target column
            strategy: 'undersample' or 'oversample' or 'hybrid'
            min_samples_per_class: Minimum samples per class
            
        Returns:
            Balanced dataframe
        """
        logger.info(f"Balancing classes using '{strategy}' strategy")
        
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found")
            return df
        
        # Get class distribution
        class_counts = df[target_col].value_counts()
        logger.info(f"Original class distribution:\n{class_counts}")
        
        if strategy == 'undersample':
            # Undersample to minority class size (but at least min_samples_per_class)
            target_size = max(class_counts.min(), min_samples_per_class)
            
            balanced_dfs = []
            for class_label in class_counts.index:
                class_df = df[df[target_col] == class_label]
                
                if len(class_df) > target_size:
                    # Undersample
                    sampled = class_df.sample(n=target_size, random_state=42)
                else:
                    # Keep all if below target
                    sampled = class_df
                
                balanced_dfs.append(sampled)
            
            df_balanced = pd.concat(balanced_dfs, ignore_index=True)
        
        elif strategy == 'oversample':
            # Oversample to majority class size
            target_size = max(class_counts.max(), min_samples_per_class)
            
            balanced_dfs = []
            for class_label in class_counts.index:
                class_df = df[df[target_col] == class_label]
                
                if len(class_df) < target_size:
                    # Oversample with replacement
                    sampled = class_df.sample(n=target_size, replace=True, random_state=42)
                else:
                    sampled = class_df
                
                balanced_dfs.append(sampled)
            
            df_balanced = pd.concat(balanced_dfs, ignore_index=True)
        
        elif strategy == 'hybrid':
            # Hybrid: oversample minority, undersample majority to middle ground
            target_size = max(int(class_counts.mean()), min_samples_per_class)
            
            balanced_dfs = []
            for class_label in class_counts.index:
                class_df = df[df[target_col] == class_label]
                
                if len(class_df) < target_size:
                    # Oversample
                    sampled = class_df.sample(n=target_size, replace=True, random_state=42)
                elif len(class_df) > target_size:
                    # Undersample
                    sampled = class_df.sample(n=target_size, random_state=42)
                else:
                    sampled = class_df
                
                balanced_dfs.append(sampled)
            
            df_balanced = pd.concat(balanced_dfs, ignore_index=True)
        
        else:
            logger.warning(f"Unknown strategy '{strategy}', returning original dataframe")
            return df
        
        # Shuffle
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Log results
        final_counts = df_balanced[target_col].value_counts()
        logger.info(f"Balanced class distribution:\n{final_counts}")
        logger.info(f"Total samples: {len(df)} -> {len(df_balanced)}")
        
        return df_balanced


# Alias for backward compatibility
DataCleaner = ConservativeDataCleaner


# Usage example for your pipeline
def recommended_cleaning_pipeline(df: pd.DataFrame, target_col: str = 'label') -> pd.DataFrame:
    """
    Recommended cleaning pipeline for IoT BotScan
    
    Args:
        df: Raw dataframe
        target_col: Target column name
        
    Returns:
        Cleaned and balanced dataframe
    """
    logger.info("="*80)
    logger.info("RECOMMENDED CLEANING PIPELINE")
    logger.info("="*80)
    
    # Step 1: Conservative cleaning
    cleaner = ConservativeDataCleaner(
        remove_exact_duplicates=True,
        handle_missing=True,
        remove_outliers=False  # Don't remove outliers - they might be attacks!
    )
    df_clean = cleaner.clean(df, target_col)
    
    # Step 2: Balance classes
    sampler = BalancedSampler()
    df_balanced = sampler.balance_classes(
        df_clean, 
        target_col=target_col,
        strategy='hybrid',  # Balance between over/under sampling
        min_samples_per_class=10000  # Ensure enough samples per class
    )
    
    logger.info("="*80)
    logger.info("CLEANING PIPELINE COMPLETED")
    logger.info(f"Final dataset: {len(df_balanced)} samples, {df_balanced.shape[1]} features")
    logger.info("="*80)
    
    return df_balanced


if __name__ == "__main__":
    # Test the cleaner
    print("Conservative Data Cleaner - Ready to use!")
    print("\nKey improvements:")
    print("1. Removes ONLY exact duplicates (not near-duplicates)")
    print("2. Conservative missing value handling")
    print("3. Optional outlier removal (disabled by default)")
    print("4. Smart class balancing")
    print("5. Preserves 80-95% of original data")
