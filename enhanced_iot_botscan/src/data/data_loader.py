"""
Data Loader for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Handles loading and preprocessing of IoT botnet datasets (N-BaIoT, IoT-23, BoT-IoT).
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from datetime import datetime
import glob
from pathlib import Path


logger = logging.getLogger(__name__)


class DataLoader:
    """Comprehensive data loader for IoT botnet detection datasets."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_paths = config.get('data_paths', {})
        self.supported_datasets = ['n_baiot', 'iot_23', 'bot_iot']

        # Cache for loaded datasets
        self.loaded_datasets = {}
        self.dataset_info = {}

        # Processing parameters
        self.chunk_size = config.get('chunk_size', 10000)
        self.memory_limit_gb = config.get('memory_limit_gb', 8)

        print(
            f"DataLoader initialized. Supported datasets: {self.supported_datasets}")

    def load_n_baiot_dataset(self, device_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Load N-BaIoT dataset from flat directory structure (raw/n_baiot/*.csv).

        Args:
            device_types: List of device prefixes (e.g., '1', '2') to load. If None, loads all.

        Returns:
            Dictionary containing features, labels, and metadata
        """

        n_baiot_path = self.data_paths.get('n_baiot', './data/raw/n_baiot/')

        # If configured, use the optimized loader implemented in src/data/optimized_data_loader.py
        if self.config.get('use_optimized_loader', False):
            try:
                from .optimized_data_loader import OptimizedDataLoader
            except Exception:
                # fallback import path
                from src.data.optimized_data_loader import OptimizedDataLoader

            loader = OptimizedDataLoader(
                n_baiot_path=n_baiot_path,
                max_samples_per_device=self.config.get(
                    'max_samples_per_device', 50000),
                chunk_size=self.chunk_size
            )

            X, y = loader.load_n_baiot_optimized()

            # Check if data was actually loaded
            if len(X) == 0 or len(y) == 0:
                logger.warning("OptimizedDataLoader returned empty dataset")
                raise ValueError("No data loaded from N-BaIoT dataset")

            dataset = {
                'features': X.values.astype(np.float32) if hasattr(X, 'values') else X,
                'labels': y.values if hasattr(y, 'values') else y,
                'feature_names': list(X.columns) if hasattr(X, 'columns') else [f'feat_{i}' for i in range(X.shape[1])],
                'dataset_name': 'N-BaIoT',
                'device_metadata': {},
                'label_mapping': {int(i): str(i) for i in np.unique(y if not hasattr(y, 'values') else y.values)},
                'total_samples': len(X),
                'n_features': X.shape[1] if getattr(X, 'shape', None) else 0,
                'n_classes': len(np.unique(y if not hasattr(y, 'values') else y.values))
            }

            self.loaded_datasets['n_baiot'] = dataset
            print(
                f"N-BaIoT dataset loaded (optimized): {dataset['total_samples']} samples, {dataset['n_features']} features")
            return dataset

        if not os.path.exists(n_baiot_path):
            raise FileNotFoundError(
                f"N-BaIoT dataset path not found: {n_baiot_path}")

        print(f"Loading N-BaIoT dataset from {n_baiot_path}...")

        all_data = []
        all_labels = []
        device_metadata = {}

        # Find all CSV files
        csv_files = glob.glob(os.path.join(n_baiot_path, "*.csv"))

        # Group by device prefix (e.g., "1.benign.csv" -> "1")
        device_files = {}
        for f in csv_files:
            basename = os.path.basename(f)
            prefix = basename.split('.')[0]
            if prefix.isdigit():
                if prefix not in device_files:
                    device_files[prefix] = []
                device_files[prefix].append(f)

        # Filter if specific devices requested
        if device_types:
            device_files = {k: v for k,
                            v in device_files.items() if k in device_types}

        for device_id, files in device_files.items():
            print(f"Processing device {device_id} with {len(files)} files...")

            device_data = []
            device_labels = []

            for file_path in files:
                try:
                    # Using chunking to manage memory
                    chunks = []
                    for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                        chunks.append(chunk)

                    if not chunks:
                        continue

                    df = pd.concat(chunks, ignore_index=True)

                    # Store data
                    device_data.append(df)

                    # Determine label
                    filename = os.path.basename(file_path).lower()
                    if 'benign' in filename:
                        label = 0
                    elif 'mirai' in filename:
                        label = 1
                    elif 'gafgyt' in filename:
                        label = 2
                    elif 'bashlite' in filename:
                        label = 3
                    else:
                        label = 4

                    device_labels.extend([label] * len(df))

                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

            if device_data:
                device_df = pd.concat(device_data, ignore_index=True)
                all_data.append(device_df)
                all_labels.extend(device_labels)

                device_metadata[device_id] = {
                    'samples': len(device_df),
                    'features': len(device_df.columns),
                    'benign_samples': device_labels.count(0),
                    'malware_samples': len(device_labels) - device_labels.count(0)
                }
                print(f"Loaded Device {device_id}: {len(device_df)} samples")

        if not all_data:
            # Fallback for empty directory or no matches
            # Try loading ANY csv if "1.", "2." patterns missed
            other_files = [f for f in csv_files if not os.path.basename(f).split('.')[
                0].isdigit()]
            if other_files:
                print(
                    f"Warning: Standard N-BaIoT naming not found. Loading {len(other_files)} generic CSVs.")
                for file_path in other_files:
                    try:
                        # Be careful with memory here
                        df = pd.read_csv(file_path)
                        all_data.append(df)
                        # Assume filename indicates class roughly
                        label = 0 if 'benign' in os.path.basename(
                            file_path).lower() else 1
                        all_labels.extend([label] * len(df))
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

        if not all_data:
            raise ValueError("No data loaded from N-BaIoT dataset")

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Handle missing values and infinite values
        combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
        combined_df = combined_df.fillna(0)

        # Force numeric column names if they are mixed strings/floats (common issue)
        # But commonly N-BaIoT has 'MI_dir_L5_weight' etc. Keep string names.

        dataset = {
            'features': combined_df.values.astype(np.float32),
            'labels': np.array(all_labels),
            'feature_names': list(combined_df.columns),
            'dataset_name': 'N-BaIoT',
            'device_metadata': device_metadata,
            'label_mapping': {0: 'Benign', 1: 'Mirai', 2: 'Gafgyt', 3: 'Bashlite', 4: 'Other'},
            'total_samples': len(combined_df),
            'n_features': len(combined_df.columns),
            'n_classes': len(np.unique(all_labels))
        }

        self.loaded_datasets['n_baiot'] = dataset
        print(
            f"N-BaIoT dataset loaded: {dataset['total_samples']} samples, {dataset['n_features']} features")

        return dataset

    def load_iot_23_dataset(self) -> Dict[str, Any]:
        """Load IoT-23 dataset."""

        iot_23_path = self.data_paths.get('iot_23', './data/raw/iot_23/')

        if not os.path.exists(iot_23_path):
            raise FileNotFoundError(
                f"IoT-23 dataset path not found: {iot_23_path}")

        print("Loading IoT-23 dataset...")

        # Find CSV files
        csv_files = glob.glob(os.path.join(iot_23_path, "*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {iot_23_path}")

        all_data = []

        for file_path in csv_files:
            try:
                # Load in chunks to handle large files
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                    chunks.append(chunk)

                df = pd.concat(chunks, ignore_index=True)
                all_data.append(df)
                print(
                    f"Loaded {os.path.basename(file_path)}: {len(df)} samples")

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        if not all_data:
            raise ValueError("No data loaded from IoT-23 dataset")

        # Combine data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Assume last column is label (adjust as needed for actual IoT-23 format)
        feature_columns = combined_df.columns[:-1]
        label_column = combined_df.columns[-1]

        features = combined_df[feature_columns].values.astype(np.float32)
        labels = combined_df[label_column].values

        # Convert labels to numeric if needed
        if labels.dtype == 'object':
            unique_labels = np.unique(labels)
            label_map = {label: i for i, label in enumerate(unique_labels)}
            labels = np.array([label_map[label] for label in labels])

        dataset = {
            'features': features,
            'labels': labels,
            'feature_names': list(feature_columns),
            'dataset_name': 'IoT-23',
            'label_mapping': {i: f'Class_{i}' for i in np.unique(labels)},
            'total_samples': len(features),
            'n_features': len(feature_columns),
            'n_classes': len(np.unique(labels))
        }

        self.loaded_datasets['iot_23'] = dataset
        print(
            f"IoT-23 dataset loaded: {dataset['total_samples']} samples, {dataset['n_features']} features")

        return dataset

    def load_bot_iot_dataset(self) -> Dict[str, Any]:
        """Load BoT-IoT dataset."""

        bot_iot_path = self.data_paths.get('bot_iot', './data/raw/bot_iot/')

        if not os.path.exists(bot_iot_path):
            raise FileNotFoundError(
                f"BoT-IoT dataset path not found: {bot_iot_path}")

        print("Loading BoT-IoT dataset...")

        sample_files = glob.glob(os.path.join(bot_iot_path, "*sample*.csv"))
        csv_files = sample_files if sample_files else glob.glob(
            os.path.join(bot_iot_path, "*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {bot_iot_path}")

        all_data = []

        for file_path in csv_files:
            try:
                # Load in chunks
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                    chunks.append(chunk)

                df = pd.concat(chunks, ignore_index=True)
                all_data.append(df)
                print(
                    f"Loaded {os.path.basename(file_path)}: {len(df)} samples")

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        if not all_data:
            raise ValueError("No data loaded from BoT-IoT dataset")

        # Combine data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Handle BoT-IoT specific format (adjust based on actual format)
        if 'label' in combined_df.columns:
            label_column = 'label'
        elif 'Label' in combined_df.columns:
            label_column = 'Label'
        else:
            label_column = combined_df.columns[-1]  # Assume last column

        feature_columns = [
            col for col in combined_df.columns if col != label_column]

        # Select only numeric features
        numeric_features = []
        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(combined_df[col]):
                numeric_features.append(col)

        features = combined_df[numeric_features].values.astype(np.float32)
        labels = combined_df[label_column].values

        # Convert labels to numeric
        if labels.dtype == 'object':
            unique_labels = np.unique(labels)
            label_map = {label: i for i, label in enumerate(unique_labels)}
            labels = np.array([label_map[label] for label in labels])
            label_mapping = {i: label for label, i in label_map.items()}
        else:
            label_mapping = {i: f'Class_{i}' for i in np.unique(labels)}

        dataset = {
            'features': features,
            'labels': labels,
            'feature_names': numeric_features,
            'dataset_name': 'BoT-IoT',
            'label_mapping': label_mapping,
            'total_samples': len(features),
            'n_features': len(numeric_features),
            'n_classes': len(np.unique(labels))
        }

        self.loaded_datasets['bot_iot'] = dataset
        print(
            f"BoT-IoT dataset loaded: {dataset['total_samples']} samples, {dataset['n_features']} features")

        return dataset

    def load_dataset(self, dataset_name: str, **kwargs) -> Dict[str, Any]:
        """Load specified dataset."""

        if dataset_name not in self.supported_datasets:
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. Supported: {self.supported_datasets}")

        if dataset_name == 'n_baiot':
            return self.load_n_baiot_dataset(**kwargs)
        elif dataset_name == 'iot_23':
            return self.load_iot_23_dataset(**kwargs)
        elif dataset_name == 'bot_iot':
            return self.load_bot_iot_dataset(**kwargs)

    def load_unified_dataset(self, max_samples: int = 1000000) -> Dict[str, Any]:
        """
        Load and unify all supported datasets into a single training set.

        Args:
            max_samples: Maximum number of samples (subsampled to prevent OOM)

        Returns:
            Unified dataset dictionary
        """
        print("Starting unified dataset loading...")

        # Load individual datasets (with internal subsampling to be safe)
        datasets = []
        common_features = set()

        # 1. N-BaIoT
        try:
            # Load partial or full based on memory constraints.
            # Ideally we'd pass a limit to individual loaders, but for now we load and downsample.
            nb_data = self.load_n_baiot_dataset()
            df_nb = pd.DataFrame(
                nb_data['features'], columns=nb_data['feature_names'])
            # Use simple labels for now, mapping needed
            df_nb['label'] = nb_data['labels']
            # Normalize labels to binary: 0=Benign, 1=Malicious
            df_nb['binary_label'] = df_nb['label'].apply(
                lambda x: 0 if x == 0 else 1)
            datasets.append(df_nb)
        except Exception as e:
            print(f"Skipping N-BaIoT: {e}")

        # 2. IoT-23
        try:
            iot23_data = self.load_iot_23_dataset()
            df_i23 = pd.DataFrame(
                iot23_data['features'], columns=iot23_data['feature_names'])
            # Logic to normalize IoT-23 labels often complex (strings). Assumed encoded in loader?
            # Existing loader maps strings to ints. We need to know which int is benign.
            # Usually 'Benign' is mapped. Let's assume class 0 is Benign if not specified.
            # TODO: robust label mapping
            df_i23['binary_label'] = (iot23_data['labels'] != 0).astype(int)
            datasets.append(df_i23)
        except Exception as e:
            print(f"Skipping IoT-23: {e}")

        # 3. BoT-IoT
        try:
            bot_data = self.load_bot_iot_dataset()
            df_bot = pd.DataFrame(
                bot_data['features'], columns=bot_data['feature_names'])
            df_bot['binary_label'] = (bot_data['labels'] != 0).astype(int)
            datasets.append(df_bot)
        except Exception as e:
            print(f"Skipping BoT-IoT: {e}")

        # Combine all data
        # STRATEGY: Union of features with zero-fill for missing values.
        # This allows the model to utilize all available information.

        all_features = set()
        for df in datasets:
            # Exclude our label columns
            feats = [c for c in df.columns if c not in [
                'label', 'binary_label']]
            all_features.update(feats)

        common_features = list(all_features)
        print(
            f"Merging {len(datasets)} datasets. Total unique features: {len(common_features)}")

        final_dfs = []
        for df in datasets:
            # 1. Align columns
            current_feats = set(df.columns)
            missing_feats = list(all_features - current_feats)

            # Efficiently add missing columns as 0
            if missing_feats:
                # Create a DataFrame of 0s and concat specifically to avoid fragmentation
                zeros = pd.DataFrame(0, index=df.index, columns=missing_feats)
                df = pd.concat([df, zeros], axis=1)

            # 2. Reorder to match common_features + label
            # Ensure we use 'binary_label' as the target 'label'
            cols_to_keep = common_features + ['binary_label']
            df_aligned = df[cols_to_keep].copy()
            df_aligned.rename(columns={'binary_label': 'label'}, inplace=True)

            final_dfs.append(df_aligned)

        unified_df = pd.concat(final_dfs, ignore_index=True)

        # Subsample if too large
        if len(unified_df) > max_samples:
            print(
                f"Subsampling unified dataset from {len(unified_df)} to {max_samples}")
            unified_df = unified_df.sample(n=max_samples, random_state=42)

        # Return structured like single dataset
        X = unified_df.drop(columns=['label']).values.astype(np.float32)
        y = unified_df['label'].values

        print(f"Unified dataset created: {X.shape}")

        return {
            'features': X,
            'labels': y,
            'feature_names': common_features,
            'dataset_name': 'Unified_Multi_Dataset',
            'label_mapping': {0: 'Benign', 1: 'Malicious'},
            'total_samples': len(unified_df),
            'n_features': len(common_features),
            'n_classes': 2
        }

    def get_dataset_statistics(self, dataset_name: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a dataset."""

        if dataset_name not in self.loaded_datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")

        dataset = self.loaded_datasets[dataset_name]
        features = dataset['features']
        labels = dataset['labels']

        # Basic statistics
        stats = {
            'dataset_name': dataset['dataset_name'],
            'total_samples': dataset['total_samples'],
            'n_features': dataset['n_features'],
            'n_classes': dataset['n_classes'],
            'label_distribution': {},
            'feature_statistics': {},
            'memory_usage_mb': features.nbytes / (1024 * 1024)
        }

        # Label distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            label_name = dataset['label_mapping'].get(label, f'Class_{label}')
            stats['label_distribution'][label_name] = {
                'count': int(count),
                'percentage': float(count / len(labels) * 100)
            }

        # Feature statistics
        stats['feature_statistics'] = {
            'mean': np.mean(features, axis=0).tolist(),
            'std': np.std(features, axis=0).tolist(),
            'min': np.min(features, axis=0).tolist(),
            'max': np.max(features, axis=0).tolist(),
            'missing_values': np.sum(np.isnan(features), axis=0).tolist()
        }

        return stats

    def save_processed_dataset(self, dataset_name: str, output_path: str) -> None:
        """Save processed dataset to disk."""

        if dataset_name not in self.loaded_datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")

        dataset = self.loaded_datasets[dataset_name]

        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save as compressed numpy arrays
        np.savez_compressed(
            output_path,
            features=dataset['features'],
            labels=dataset['labels'],
            feature_names=dataset['feature_names'],
            label_mapping=dataset['label_mapping'],
            metadata=dataset
        )

        print(f"Dataset {dataset_name} saved to {output_path}")

    def load_processed_dataset(self, filepath: str) -> Dict[str, Any]:
        """Load previously processed dataset."""

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Processed dataset not found: {filepath}")

        data = np.load(filepath, allow_pickle=True)

        dataset = {
            'features': data['features'],
            'labels': data['labels'],
            'feature_names': data['feature_names'].tolist(),
            'label_mapping': data['label_mapping'].item(),
            'dataset_name': data['metadata'].item().get('dataset_name', 'Unknown'),
            'total_samples': len(data['features']),
            'n_features': data['features'].shape[1],
            'n_classes': len(np.unique(data['labels']))
        }

        print(f"Processed dataset loaded from {filepath}")
        return dataset
