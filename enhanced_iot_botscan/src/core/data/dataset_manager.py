"""
Dataset Manager Implementation for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Comprehensive dataset management system for handling multiple IoT botnet datasets
including N-BaIoT, IoT-23, and BoT-IoT with validation and preprocessing capabilities.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import os
import json
import pickle
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)


class DatasetManager:
    """Comprehensive dataset management for IoT botnet detection."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize dataset manager with configuration."""

        self.config = config or {}
        self.data_dir = Path(self.config.get('data_dir', 'data'))
        self.cache_dir = Path(self.config.get('cache_dir', 'cache'))
        self.datasets = {}
        self.dataset_metadata = {}

        # Dataset configuration
        self.supported_datasets = ['n_baiot', 'iot_23', 'bot_iot']
        self.default_split_ratios = {
            'train': 0.7, 'validation': 0.15, 'test': 0.15}

        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

        logger.info(
            f"DatasetManager initialized with data_dir: {self.data_dir}")

    def load_dataset(self, dataset_name: str, file_path: Optional[str] = None,
                     preprocessing_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Load a dataset from file or cache.

        Args:
            dataset_name: Name of the dataset
            file_path: Path to dataset file (optional)
            preprocessing_config: Preprocessing configuration (optional)

        Returns:
            Loaded dataset as DataFrame
        """

        logger.info(f"Loading dataset: {dataset_name}")

        # Check cache first
        cache_key = self._generate_cache_key(
            dataset_name, preprocessing_config)
        cached_data = self._load_from_cache(cache_key)

        if cached_data is not None:
            logger.info(f"Dataset {dataset_name} loaded from cache")
            return cached_data

        # Load from file
        if file_path is None:
            file_path = self._get_default_file_path(dataset_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Load dataset based on file extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        # Apply preprocessing if specified
        if preprocessing_config:
            df = self._apply_preprocessing(df, preprocessing_config)

        # Store dataset metadata
        self.dataset_metadata[dataset_name] = {
            'file_path': file_path,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'preprocessing_config': preprocessing_config,
            'load_timestamp': datetime.now().isoformat()
        }

        # Cache the dataset
        self._save_to_cache(cache_key, df)

        # Store in memory
        self.datasets[dataset_name] = df

        logger.info(
            f"Dataset {dataset_name} loaded successfully. Shape: {df.shape}")

        return df

    def load_n_baiot_dataset(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load N-BaIoT dataset with specific preprocessing."""

        logger.info("Loading N-BaIoT dataset")

        preprocessing_config = {
            'target_column': 'label',
            'binary_classification': True,
            'normalize_features': True,
            'remove_duplicates': True
        }

        return self.load_dataset('n_baiot', file_path, preprocessing_config)

    def load_iot_23_dataset(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load IoT-23 dataset with specific preprocessing."""

        logger.info("Loading IoT-23 dataset")

        preprocessing_config = {
            'target_column': 'label',
            'binary_classification': True,
            'normalize_features': True,
            'remove_duplicates': True,
            'handle_missing_values': True
        }

        return self.load_dataset('iot_23', file_path, preprocessing_config)

    def load_bot_iot_dataset(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load BoT-IoT dataset with specific preprocessing."""

        logger.info("Loading BoT-IoT dataset")

        preprocessing_config = {
            'target_column': 'label',
            'binary_classification': True,
            'normalize_features': True,
            'remove_duplicates': True,
            'handle_missing_values': True
        }

        return self.load_dataset('bot_iot', file_path, preprocessing_config)

    def create_train_test_split(self, dataset_name: str,
                                split_ratios: Optional[Dict[str,
                                                            float]] = None,
                                stratify_column: Optional[str] = None,
                                random_state: int = 42) -> Dict[str, pd.DataFrame]:
        """
        Create train/validation/test splits for a dataset.

        Args:
            dataset_name: Name of the dataset
            split_ratios: Split ratios (default: 70/15/15)
            stratify_column: Column to stratify on (optional)
            random_state: Random state for reproducibility

        Returns:
            Dictionary containing train, validation, and test splits
        """

        logger.info(f"Creating train/test split for {dataset_name}")

        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")

        df = self.datasets[dataset_name]
        split_ratios = split_ratios or self.default_split_ratios

        # Validate split ratios
        if abs(sum(split_ratios.values()) - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        # Create splits
        from sklearn.model_selection import train_test_split

        # First split: train vs (validation + test)
        train_ratio = split_ratios['train']
        val_test_ratio = split_ratios['validation'] + split_ratios['test']

        stratify = df[stratify_column] if stratify_column and stratify_column in df.columns else None

        X_train, X_temp, y_train, y_temp = train_test_split(
            df.drop(columns=[stratify_column]) if stratify_column else df,
            stratify,
            test_size=val_test_ratio,
            random_state=random_state,
            stratify=stratify
        )

        # Second split: validation vs test
        val_ratio = split_ratios['validation'] / val_test_ratio

        stratify_temp = y_temp if stratify_column else None

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            stratify_temp,
            test_size=1-val_ratio,
            random_state=random_state,
            stratify=stratify_temp
        )

        # Create result dictionary
        splits = {
            'train': X_train,
            'validation': X_val,
            'test': X_test
        }

        # Add target columns back if stratified
        if stratify_column:
            splits['train'][stratify_column] = y_train
            splits['validation'][stratify_column] = y_val
            splits['test'][stratify_column] = y_test

        # Store split metadata
        self.dataset_metadata[dataset_name]['splits'] = {
            'split_ratios': split_ratios,
            'stratify_column': stratify_column,
            'random_state': random_state,
            'split_timestamp': datetime.now().isoformat()
        }

        logger.info(f"Train/test split created for {dataset_name}")
        logger.info(
            f"Train: {len(splits['train'])}, Validation: {len(splits['validation'])}, Test: {len(splits['test'])}")

        return splits

    def create_cross_validation_splits(self, dataset_name: str, n_splits: int = 5,
                                       stratify_column: Optional[str] = None,
                                       random_state: int = 42) -> List[Dict[str, pd.DataFrame]]:
        """
        Create cross-validation splits for a dataset.

        Args:
            dataset_name: Name of the dataset
            n_splits: Number of CV splits
            stratify_column: Column to stratify on (optional)
            random_state: Random state for reproducibility

        Returns:
            List of CV splits
        """

        logger.info(f"Creating {n_splits}-fold CV splits for {dataset_name}")

        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")

        df = self.datasets[dataset_name]

        from sklearn.model_selection import StratifiedKFold, KFold

        # Choose CV strategy
        if stratify_column and stratify_column in df.columns:
            cv = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state)
            stratify_values = df[stratify_column]
        else:
            cv = KFold(n_splits=n_splits, shuffle=True,
                       random_state=random_state)
            stratify_values = None

        cv_splits = []

        for fold, (train_idx, test_idx) in enumerate(cv.split(df, stratify_values)):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()

            cv_splits.append({
                'fold': fold,
                'train': train_df,
                'test': test_df
            })

        # Store CV metadata
        self.dataset_metadata[dataset_name]['cv_splits'] = {
            'n_splits': n_splits,
            'stratify_column': stratify_column,
            'random_state': random_state,
            'cv_timestamp': datetime.now().isoformat()
        }

        logger.info(
            f"CV splits created for {dataset_name}: {len(cv_splits)} folds")

        return cv_splits

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get comprehensive information about a dataset."""

        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")

        df = self.datasets[dataset_name]
        metadata = self.dataset_metadata.get(dataset_name, {})

        info = {
            'name': dataset_name,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'metadata': metadata
        }

        # Add statistical information
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            info['numerical_summary'] = df[numerical_cols].describe().to_dict()

        # Add categorical information
        categorical_cols = df.select_dtypes(
            include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            info['categorical_summary'] = {}
            for col in categorical_cols:
                info['categorical_summary'][col] = {
                    'unique_values': df[col].nunique(),
                    'most_common': df[col].value_counts().head().to_dict()
                }

        return info

    def validate_dataset(self, dataset_name: str, validation_rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate dataset quality and integrity.

        Args:
            dataset_name: Name of the dataset
            validation_rules: Custom validation rules (optional)

        Returns:
            Validation results
        """

        logger.info(f"Validating dataset: {dataset_name}")

        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")

        df = self.datasets[dataset_name]
        validation_rules = validation_rules or self._get_default_validation_rules()

        validation_results = {
            'dataset_name': dataset_name,
            'validation_timestamp': datetime.now().isoformat(),
            'passed': True,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }

        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            validation_results['warnings'].append(
                f"Found {missing_values.sum()} missing values")
            validation_results['statistics']['missing_values'] = missing_values.to_dict(
            )

        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            validation_results['warnings'].append(
                f"Found {duplicate_count} duplicate rows")
            validation_results['statistics']['duplicate_rows'] = duplicate_count

        # Check for constant columns
        constant_columns = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_columns.append(col)

        if constant_columns:
            validation_results['issues'].append(
                f"Found constant columns: {constant_columns}")
            validation_results['passed'] = False

        # Check for infinite values
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        infinite_values = {}
        for col in numerical_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                infinite_values[col] = inf_count

        if infinite_values:
            validation_results['issues'].append(
                f"Found infinite values: {infinite_values}")
            validation_results['passed'] = False

        # Check data types
        validation_results['statistics']['dtypes'] = df.dtypes.to_dict()

        # Check memory usage
        validation_results['statistics']['memory_usage_mb'] = df.memory_usage(
            deep=True).sum() / 1024 / 1024

        logger.info(
            f"Dataset validation completed for {dataset_name}. Passed: {validation_results['passed']}")

        return validation_results

    def merge_datasets(self, dataset_names: List[str], merge_strategy: str = 'concat',
                       merge_keys: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Merge multiple datasets.

        Args:
            dataset_names: List of dataset names to merge
            merge_strategy: Strategy for merging ('concat', 'join')
            merge_keys: Keys for joining (optional)

        Returns:
            Merged dataset
        """

        logger.info(
            f"Merging datasets: {dataset_names} using {merge_strategy}")

        # Check if all datasets are loaded
        for name in dataset_names:
            if name not in self.datasets:
                raise ValueError(f"Dataset {name} not loaded")

        datasets = [self.datasets[name] for name in dataset_names]

        if merge_strategy == 'concat':
            # Concatenate datasets
            merged_df = pd.concat(datasets, ignore_index=True)

        elif merge_strategy == 'join':
            if not merge_keys:
                raise ValueError("merge_keys required for join strategy")

            # Join datasets
            merged_df = datasets[0]
            for i, df in enumerate(datasets[1:], 1):
                merged_df = merged_df.merge(df, on=merge_keys, how='outer')

        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")

        # Store merged dataset
        merged_name = f"merged_{'_'.join(dataset_names)}"
        self.datasets[merged_name] = merged_df

        logger.info(f"Datasets merged successfully. Shape: {merged_df.shape}")

        return merged_df

    def save_dataset(self, dataset_name: str, file_path: str, format: str = 'csv') -> None:
        """
        Save dataset to file.

        Args:
            dataset_name: Name of the dataset
            file_path: Path to save the dataset
            format: File format ('csv', 'parquet', 'json')
        """

        logger.info(f"Saving dataset {dataset_name} to {file_path}")

        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")

        df = self.datasets[dataset_name]

        if format == 'csv':
            df.to_csv(file_path, index=False)
        elif format == 'parquet':
            df.to_parquet(file_path, index=False)
        elif format == 'json':
            df.to_json(file_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Dataset {dataset_name} saved successfully")

    def _apply_preprocessing(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply preprocessing to dataset."""

        df_processed = df.copy()

        # Handle missing values
        if config.get('handle_missing_values', False):
            df_processed = df_processed.fillna(df_processed.median())

        # Remove duplicates
        if config.get('remove_duplicates', False):
            df_processed = df_processed.drop_duplicates()

        # Normalize features
        if config.get('normalize_features', False):
            numerical_cols = df_processed.select_dtypes(
                include=[np.number]).columns
            if len(numerical_cols) > 0:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                df_processed[numerical_cols] = scaler.fit_transform(
                    df_processed[numerical_cols])

        # Binary classification
        if config.get('binary_classification', False):
            target_col = config.get('target_column', 'label')
            if target_col in df_processed.columns:
                # Convert to binary (0/1)
                unique_values = df_processed[target_col].unique()
                if len(unique_values) > 2:
                    # Map to binary
                    df_processed[target_col] = (
                        df_processed[target_col] == unique_values[0]).astype(int)

        return df_processed

    def _get_default_file_path(self, dataset_name: str) -> str:
        """Get default file path for dataset."""

        file_mappings = {
            'n_baiot': 'data/n_baiot.csv',
            'iot_23': 'data/iot_23.csv',
            'bot_iot': 'data/bot_iot.csv'
        }

        return file_mappings.get(dataset_name, f'data/{dataset_name}.csv')

    def _generate_cache_key(self, dataset_name: str, preprocessing_config: Optional[Dict[str, Any]]) -> str:
        """Generate cache key for dataset."""

        config_str = json.dumps(preprocessing_config,
                                sort_keys=True) if preprocessing_config else ""
        return f"{dataset_name}_{hash(config_str)}"

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load dataset from cache."""

        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
                return None

        return None

    def _save_to_cache(self, cache_key: str, df: pd.DataFrame) -> None:
        """Save dataset to cache."""

        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    def _get_default_validation_rules(self) -> Dict[str, Any]:
        """Get default validation rules."""

        return {
            'max_missing_ratio': 0.5,
            'max_duplicate_ratio': 0.1,
            'min_unique_values': 2,
            'max_memory_mb': 1000
        }

    def get_all_datasets_info(self) -> Dict[str, Any]:
        """Get information about all loaded datasets."""

        return {
            'loaded_datasets': list(self.datasets.keys()),
            'dataset_count': len(self.datasets),
            'datasets_info': {name: self.get_dataset_info(name) for name in self.datasets.keys()},
            'metadata': self.dataset_metadata
        }

    def clear_cache(self) -> None:
        """Clear all cached datasets."""

        logger.info("Clearing dataset cache")

        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()

        logger.info("Dataset cache cleared")

    def unload_dataset(self, dataset_name: str) -> None:
        """Unload dataset from memory."""

        if dataset_name in self.datasets:
            del self.datasets[dataset_name]
            logger.info(f"Dataset {dataset_name} unloaded from memory")


# Example usage and testing
if __name__ == '__main__':
    # Create sample data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    # Create sample datasets
    sample_data = {
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(5, 2, n_samples),
        'feature_3': np.random.choice(['A', 'B', 'C'], n_samples),
        'label': np.random.randint(0, 2, n_samples)
    }

    df = pd.DataFrame(sample_data)

    print("Testing Dataset Manager:")

    # Initialize dataset manager
    manager = DatasetManager({
        'data_dir': 'test_data',
        'cache_dir': 'test_cache'
    })

    # Save sample dataset
    os.makedirs('test_data', exist_ok=True)
    df.to_csv('test_data/sample_dataset.csv', index=False)

    # Load dataset
    loaded_df = manager.load_dataset(
        'sample_dataset', 'test_data/sample_dataset.csv')
    print(f"Dataset loaded. Shape: {loaded_df.shape}")

    # Get dataset info
    info = manager.get_dataset_info('sample_dataset')
    print(
        f"Dataset info: {info['shape']} samples, {info['memory_usage']} bytes")

    # Create train/test split
    splits = manager.create_train_test_split(
        'sample_dataset', stratify_column='label')
    print(
        f"Train/Test split: Train={len(splits['train'])}, Validation={len(splits['validation'])}, Test={len(splits['test'])}")

    # Create CV splits
    cv_splits = manager.create_cross_validation_splits(
        'sample_dataset', n_splits=3, stratify_column='label')
    print(f"CV splits created: {len(cv_splits)} folds")

    # Validate dataset
    validation_results = manager.validate_dataset('sample_dataset')
    print(
        f"Dataset validation: Passed={validation_results['passed']}, Issues={len(validation_results['issues'])}")

    # Get all datasets info
    all_info = manager.get_all_datasets_info()
    print(f"All datasets info: {all_info['dataset_count']} datasets loaded")

    # Clean up
    manager.clear_cache()
    print("Dataset manager testing completed")
