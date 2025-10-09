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

        print(f"DataLoader initialized. Supported datasets: {self.supported_datasets}")

    def load_n_baiot_dataset(self, device_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Load N-BaIoT dataset with specified device types.

        Args:
            device_types: List of device types to load

        Returns:
            Dictionary containing features, labels, and metadata
        """

        n_baiot_path = self.data_paths.get('n_baiot', './data/raw/n_baiot/')

        if not os.path.exists(n_baiot_path):
            raise FileNotFoundError(f"N-BaIoT dataset path not found: {n_baiot_path}")

        # Default device types if not specified
        if device_types is None:
            device_types = [
                'Danmini_Doorbell', 'Ecobee_Thermostat', 'Ennio_Doorbell',
                'Philips_B120N10_Baby_Monitor', 'Provision_PT_737E_Security_Camera',
                'Provision_PT_838_Security_Camera', 'Samsung_SNH_1011_N_Webcam',
                'SimpleHome_XCS7_1002_WHT_Security_Camera', 'SimpleHome_XCS7_1003_WHT_Security_Camera'
            ]

        print(f"Loading N-BaIoT dataset for device types: {device_types}")

        all_data = []
        all_labels = []
        device_metadata = {}

        for device_type in device_types:
            device_path = os.path.join(n_baiot_path, device_type)

            if not os.path.exists(device_path):
                print(f"Warning: Device path not found: {device_path}")
                continue

            # Load benign data
            benign_files = glob.glob(os.path.join(device_path, "*benign*.csv"))
            malware_files = glob.glob(os.path.join(device_path, "*[!benign]*.csv"))

            device_data = []
            device_labels = []

            # Load benign samples
            for file_path in benign_files:
                try:
                    df = pd.read_csv(file_path)
                    device_data.append(df)
                    device_labels.extend([0] * len(df))  # Benign = 0
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

            # Load malware samples
            for file_path in malware_files:
                try:
                    df = pd.read_csv(file_path)
                    device_data.append(df)

                    # Determine malware type from filename
                    filename = os.path.basename(file_path).lower()
                    if 'mirai' in filename:
                        label = 1
                    elif 'gafgyt' in filename:
                        label = 2
                    elif 'bashlite' in filename:
                        label = 3
                    else:
                        label = 4  # Other malware

                    device_labels.extend([label] * len(df))

                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

            if device_data:
                device_df = pd.concat(device_data, ignore_index=True)
                all_data.append(device_df)
                all_labels.extend(device_labels)

                device_metadata[device_type] = {
                    'samples': len(device_df),
                    'features': len(device_df.columns),
                    'benign_samples': device_labels.count(0),
                    'malware_samples': len(device_labels) - device_labels.count(0)
                }

                print(f"Loaded {device_type}: {len(device_df)} samples")

        if not all_data:
            raise ValueError("No data loaded from N-BaIoT dataset")

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Handle missing values and infinite values
        combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
        combined_df = combined_df.fillna(0)

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
        print(f"N-BaIoT dataset loaded: {dataset['total_samples']} samples, {dataset['n_features']} features")

        return dataset

    def load_iot_23_dataset(self) -> Dict[str, Any]:
        """Load IoT-23 dataset."""

        iot_23_path = self.data_paths.get('iot_23', './data/raw/iot_23/')

        if not os.path.exists(iot_23_path):
            raise FileNotFoundError(f"IoT-23 dataset path not found: {iot_23_path}")

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
                print(f"Loaded {os.path.basename(file_path)}: {len(df)} samples")

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
        print(f"IoT-23 dataset loaded: {dataset['total_samples']} samples, {dataset['n_features']} features")

        return dataset

    def load_bot_iot_dataset(self) -> Dict[str, Any]:
        """Load BoT-IoT dataset."""

        bot_iot_path = self.data_paths.get('bot_iot', './data/raw/bot_iot/')

        if not os.path.exists(bot_iot_path):
            raise FileNotFoundError(f"BoT-IoT dataset path not found: {bot_iot_path}")

        print("Loading BoT-IoT dataset...")

        # Find CSV files
        csv_files = glob.glob(os.path.join(bot_iot_path, "*.csv"))

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
                print(f"Loaded {os.path.basename(file_path)}: {len(df)} samples")

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

        feature_columns = [col for col in combined_df.columns if col != label_column]

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
        print(f"BoT-IoT dataset loaded: {dataset['total_samples']} samples, {dataset['n_features']} features")

        return dataset

    def load_dataset(self, dataset_name: str, **kwargs) -> Dict[str, Any]:
        """Load specified dataset."""

        if dataset_name not in self.supported_datasets:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: {self.supported_datasets}")

        if dataset_name == 'n_baiot':
            return self.load_n_baiot_dataset(**kwargs)
        elif dataset_name == 'iot_23':
            return self.load_iot_23_dataset(**kwargs)
        elif dataset_name == 'bot_iot':
            return self.load_bot_iot_dataset(**kwargs)

    def load_all_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Load all supported datasets."""

        datasets = {}

        for dataset_name in self.supported_datasets:
            try:
                datasets[dataset_name] = self.load_dataset(dataset_name)
                print(f"✅ {dataset_name} loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load {dataset_name}: {e}")

        return datasets

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
