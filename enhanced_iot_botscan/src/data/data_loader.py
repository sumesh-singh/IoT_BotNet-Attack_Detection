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


# CRITICAL: Detailed 11-class N-BaIoT label mapping per seminar document (Table 2)
# 1 Benign class + 10 Attack types (Mirai variants, Gafgyt variants)
NBAIOT_ATTACK_LABELS = {
    'benign': 0,
    # Mirai variants
    'mirai.ack': 1,
    'mirai.scan': 2,
    'mirai.syn': 3,
    'mirai.udp': 4,
    'mirai.udpplain': 5,
    # Gafgyt/Bashlite variants
    'gafgyt.combo': 6,
    'gafgyt.junk': 7,
    'gafgyt.scan': 8,
    'gafgyt.tcp': 9,
    'gafgyt.udp': 10,
    # Bashlite aliases (same as gafgyt)
    'bashlite.combo': 6,
    'bashlite.junk': 7,
    'bashlite.scan': 8,
    'bashlite.tcp': 9,
    'bashlite.udp': 10
}

# Simplified label mapping for coarse-grained classification
NBAIOT_COARSE_LABELS = {
    'benign': 0,
    'mirai': 1,
    'gafgyt': 2,
    'bashlite': 2  # Bashlite is same as Gafgyt
}

def parse_nbaiot_attack_label(filename: str, fine_grained: bool = False) -> int:
    """
    Parse attack type from N-BaIoT filename.
    
    Args:
        filename: The filename to parse (e.g., '1.benign.csv', '2.mirai.ack.csv')
        fine_grained: If True, use 11-class detailed labels. If False, use coarse 4-class labels.
    
    Returns:
        Integer label for the attack type
    """
    filename_lower = filename.lower()
    
    if fine_grained:
        # Try to match detailed attack patterns
        for attack_name, label in NBAIOT_ATTACK_LABELS.items():
            if attack_name in filename_lower:
                return label
        # Default: unknown attack
        return 10 if 'gafgyt' in filename_lower or 'bashlite' in filename_lower else (
            4 if 'mirai' in filename_lower else 0)
    else:
        # Coarse-grained classification
        for attack_name, label in NBAIOT_COARSE_LABELS.items():
            if attack_name in filename_lower:
                return label
        return 0  # Default to benign


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

        # Memory-safe limits (per individual dataset, before unification)
        self.max_samples_per_dataset = config.get('max_samples_per_dataset', 50000)

        print(
            f"DataLoader initialized. Supported datasets: {self.supported_datasets}, "
            f"max_samples_per_dataset: {self.max_samples_per_dataset}")

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

                    # FIXED: Use detailed label parsing function for granular attack classification
                    filename = os.path.basename(file_path).lower()
                    use_fine_grained = self.config.get('fine_grained_labels', False)
                    label = parse_nbaiot_attack_label(filename, fine_grained=use_fine_grained)

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
                        # FIXED: Use proper label parsing for fallback files
                        label = parse_nbaiot_attack_label(
                            os.path.basename(file_path), 
                            fine_grained=self.config.get('fine_grained_labels', False)
                        )
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
            # FIXED: Dynamic label mapping based on actual labels found
            'label_mapping': self._get_nbaiot_label_mapping(all_labels),
            'total_samples': len(combined_df),
            'n_features': len(combined_df.columns),
            'n_classes': len(np.unique(all_labels))
        }

        self.loaded_datasets['n_baiot'] = dataset
        print(
            f"N-BaIoT dataset loaded: {dataset['total_samples']} samples, {dataset['n_features']} features")

        return dataset
    
    def _get_nbaiot_label_mapping(self, labels: List[int]) -> Dict[int, str]:
        """Generate label mapping based on actual labels found in data."""
        unique_labels = set(labels)
        
        # Full 11-class mapping
        full_mapping = {
            0: 'Benign',
            1: 'Mirai_ACK',
            2: 'Mirai_Scan',
            3: 'Mirai_SYN',
            4: 'Mirai_UDP',
            5: 'Mirai_UDPPlain',
            6: 'Gafgyt_Combo',
            7: 'Gafgyt_Junk',
            8: 'Gafgyt_Scan',
            9: 'Gafgyt_TCP',
            10: 'Gafgyt_UDP'
        }
        
        # Coarse mapping fallback
        coarse_mapping = {
            0: 'Benign',
            1: 'Mirai',
            2: 'Gafgyt',
            3: 'Bashlite',
            4: 'Other'
        }
        
        # Return only the labels that are present in the data
        result = {}
        for label in sorted(unique_labels):
            if label in full_mapping:
                result[label] = full_mapping[label]
            elif label in coarse_mapping:
                result[label] = coarse_mapping[label]
            else:
                result[label] = f'Class_{label}'
        
        return result

    def load_iot_23_dataset(self) -> Dict[str, Any]:
        """
        Load IoT-23 dataset.
        
        Supports two data formats:
        1. Raw Zeek/Bro conn.log.labeled files (nested under opt/Malware-Project/
           BigDataset/IoTScenarios/*/bro/conn.log.labeled)
        2. Pre-processed CSV files (e.g., iot_23_sample.csv) as fallback
        """

        iot_23_path = self.data_paths.get('iot_23', './data/raw/iot_23/')

        if not os.path.exists(iot_23_path):
            raise FileNotFoundError(
                f"IoT-23 dataset path not found: {iot_23_path}")

        print("Loading IoT-23 dataset...")

        # -----------------------------------------------------------------
        # Strategy 1: Load raw Zeek/Bro conn.log.labeled files
        # -----------------------------------------------------------------
        zeek_files = []
        for root, dirs, files in os.walk(iot_23_path):
            for fname in files:
                if fname == 'conn.log.labeled':
                    zeek_files.append(os.path.join(root, fname))

        if zeek_files:
            print(f"Found {len(zeek_files)} Zeek conn.log.labeled files")
            all_data = []
            max_samples = self.max_samples_per_dataset
            total_rows = 0

            for file_path in zeek_files:
                if total_rows >= max_samples:
                    print(f"  Reached {max_samples} sample cap, skipping remaining scenarios")
                    break
                try:
                    df = self._parse_zeek_conn_log(file_path)
                    if df is not None and len(df) > 0:
                        # Cap rows from this scenario
                        remaining = max_samples - total_rows
                        if len(df) > remaining:
                            df = df.sample(n=remaining, random_state=42)
                        all_data.append(df)
                        total_rows += len(df)
                        scenario = Path(file_path).parts
                        scenario_name = 'unknown'
                        for part in scenario:
                            if part.startswith('CTU-'):
                                scenario_name = part
                                break
                        print(f"  Loaded {scenario_name}: {len(df)} samples (total: {total_rows})")
                except Exception as e:
                    print(f"  Error parsing {file_path}: {e}")

            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)

                # Separate features and labels
                label_column = 'label'
                feature_columns = [c for c in combined_df.columns if c != label_column]

                # Handle missing/infinite values
                combined_df[feature_columns] = combined_df[feature_columns].replace(
                    [np.inf, -np.inf], np.nan)
                combined_df[feature_columns] = combined_df[feature_columns].fillna(0)

                features = combined_df[feature_columns].values.astype(np.float32)
                labels = combined_df[label_column].values

                # Build label mapping from actual label values
                label_mapping = {}
                for lbl in np.unique(labels):
                    label_mapping[int(lbl)] = 'Benign' if lbl == 0 else 'Malicious'

                dataset = {
                    'features': features,
                    'labels': labels,
                    'feature_names': feature_columns,
                    'dataset_name': 'IoT-23',
                    'label_mapping': label_mapping,
                    'total_samples': len(features),
                    'n_features': len(feature_columns),
                    'n_classes': len(np.unique(labels))
                }

                self.loaded_datasets['iot_23'] = dataset
                print(f"IoT-23 dataset loaded (Zeek): {dataset['total_samples']} samples, "
                      f"{dataset['n_features']} features, {dataset['n_classes']} classes")
                return dataset

        # -----------------------------------------------------------------
        # Strategy 2: Fallback to CSV files in top-level directory
        # -----------------------------------------------------------------
        print("No Zeek logs found, falling back to CSV files...")
        csv_files = glob.glob(os.path.join(iot_23_path, "*.csv"))

        if not csv_files:
            raise FileNotFoundError(
                f"No data files found in {iot_23_path} "
                "(neither conn.log.labeled nor *.csv)")

        all_data = []

        for file_path in csv_files:
            try:
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                    chunks.append(chunk)

                df = pd.concat(chunks, ignore_index=True)
                all_data.append(df)
                print(f"  Loaded {os.path.basename(file_path)}: {len(df)} samples")

            except Exception as e:
                print(f"  Error loading {file_path}: {e}")

        if not all_data:
            raise ValueError("No data loaded from IoT-23 dataset")

        combined_df = pd.concat(all_data, ignore_index=True)

        # Assume last column is label
        feature_columns = list(combined_df.columns[:-1])
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
            'feature_names': feature_columns,
            'dataset_name': 'IoT-23',
            'label_mapping': {int(i): f'Class_{i}' for i in np.unique(labels)},
            'total_samples': len(features),
            'n_features': len(feature_columns),
            'n_classes': len(np.unique(labels))
        }

        self.loaded_datasets['iot_23'] = dataset
        print(f"IoT-23 dataset loaded (CSV): {dataset['total_samples']} samples, "
              f"{dataset['n_features']} features")

        return dataset

    def _parse_zeek_conn_log(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Parse a Zeek/Bro conn.log.labeled file into a DataFrame.
        
        These files are tab-separated with header lines starting with '#'.
        Fields of interest (numeric): duration, orig_bytes, resp_bytes,
        missed_bytes, orig_pkts, orig_ip_bytes, resp_pkts, resp_ip_bytes.
        Label fields: 'label' (Malicious/benign), 'detailed-label'.
        
        Args:
            file_path: Path to conn.log.labeled file
            
        Returns:
            DataFrame with numeric features and binary label column
        """
        # Read the file and parse header
        field_names = None
        data_lines = []

        # IoT-23 conn.log.labeled files use a MIX of tab and multi-space
        # delimiters.  The last few columns (tunnel_parents, label,
        # detailed-label) are separated by 3 spaces instead of tabs.
        # We use a regex that matches either a tab or 2+ consecutive spaces.
        import re
        _split_re = re.compile(r'\t|  +')  # tab OR 2+ spaces

        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('#fields'):
                    # Parse field names: split by tab-or-multispace, skip '#fields'
                    parts = _split_re.split(line)
                    field_names = [fn.strip() for fn in parts[1:] if fn.strip()]
                    continue
                if line.startswith('#'):
                    continue  # Skip other header lines
                data_lines.append(line)

        if field_names is None:
            print(f"  Warning: No #fields header found in {file_path}")
            return None

        if not data_lines:
            print(f"  Warning: No data rows in {file_path}")
            return None

        # Parse data rows using the same mixed-delimiter regex
        rows = []
        for line in data_lines:
            parts = _split_re.split(line)
            if len(parts) >= len(field_names):
                rows.append(parts[:len(field_names)])

        if not rows:
            return None

        df = pd.DataFrame(rows, columns=field_names)

        # Define numeric features to extract from Zeek conn logs
        numeric_fields = [
            'duration', 'orig_bytes', 'resp_bytes', 'missed_bytes',
            'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes'
        ]

        # Keep only numeric fields that exist in this file
        available_numeric = [f for f in numeric_fields if f in df.columns]

        if not available_numeric:
            print(f"  Warning: No numeric fields found in {file_path}")
            return None

        # Convert numeric fields, replacing '-' (Zeek unset) with NaN
        for col in available_numeric:
            df[col] = pd.to_numeric(df[col].replace('-', np.nan), errors='coerce')

        # Extract label column
        if 'label' not in df.columns:
            print(f"  Warning: No 'label' column in {file_path}")
            return None

        # Map labels to binary: benign -> 0, anything else -> 1
        benign_keywords = ['benign', 'normal', 'legitimate']
        df['label'] = df['label'].apply(
            lambda x: 0 if any(kw in str(x).lower() for kw in benign_keywords) else 1
        ).astype(int)

        # Build result with numeric features + label
        result_df = df[available_numeric + ['label']].copy()
        result_df[available_numeric] = result_df[available_numeric].fillna(0)

        return result_df

    def load_bot_iot_dataset(self) -> Dict[str, Any]:
        """
        Load BoT-IoT dataset.
        
        Memory-safe: Prefers sample files (~5MB) over full data files
        (~15GB). Falls back to loading only a few full files with row cap.
        """

        bot_iot_path = self.data_paths.get('bot_iot', './data/raw/bot_iot/')

        if not os.path.exists(bot_iot_path):
            raise FileNotFoundError(
                f"BoT-IoT dataset path not found: {bot_iot_path}")

        print("Loading BoT-IoT dataset...")
        max_samples = self.max_samples_per_dataset

        # MEMORY SAFE: Strongly prefer sample files (small, curated)
        sample_files = glob.glob(os.path.join(bot_iot_path, "*sample*.csv"))

        if sample_files:
            csv_files = sample_files
            print(f"  Using {len(sample_files)} sample file(s) (memory-safe)")
        else:
            # Fallback: load only first 2 data files with nrows cap
            all_csvs = sorted(glob.glob(os.path.join(bot_iot_path, "*.csv")))
            csv_files = all_csvs[:2]  # Only first 2 files
            print(f"  No sample files found. Loading first {len(csv_files)} data files (capped)")

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {bot_iot_path}")

        all_data = []
        total_rows = 0

        for file_path in csv_files:
            if total_rows >= max_samples:
                print(f"  Reached {max_samples} sample cap, skipping remaining files")
                break
            try:
                remaining = max_samples - total_rows
                df = pd.read_csv(file_path, nrows=remaining)
                all_data.append(df)
                total_rows += len(df)
                print(f"  Loaded {os.path.basename(file_path)}: {len(df)} samples")

            except Exception as e:
                print(f"  Error loading {file_path}: {e}")

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
        
        FIXES APPLIED:
        1. Dynamic 'Benign' label detection instead of assuming label=0
        2. Guard against empty dataset concatenation crashes
        3. Better error messages for debugging

        Args:
            max_samples: Maximum number of samples (subsampled to prevent OOM)

        Returns:
            Unified dataset dictionary
        """
        print("Starting unified dataset loading...")

        # Load individual datasets (with internal subsampling to be safe)
        datasets = []
        common_features = set()

        # ========================================================================
        # 1. N-BaIoT
        # ========================================================================
        try:
            nb_data = self.load_n_baiot_dataset()
            df_nb = pd.DataFrame(
                nb_data['features'], columns=nb_data['feature_names'])
            df_nb['label'] = nb_data['labels']
            
            # Normalize labels to binary: 0=Benign, 1=Malicious
            # N-BaIoT: label 0 is always Benign (from NBAIOT_ATTACK_LABELS)
            df_nb['binary_label'] = df_nb['label'].apply(
                lambda x: 0 if x == 0 else 1)
            
            datasets.append(df_nb)
            print(f"N-BaIoT dataset loaded: {len(df_nb)} samples, {len(df_nb.columns)-2} features")
            
        except Exception as e:
            print(f"Skipping N-BaIoT: {e}")

        # ========================================================================
        # 2. IoT-23 - FIXED LABEL MAPPING
        # ========================================================================
        try:
            iot23_data = self.load_iot_23_dataset()
            df_i23 = pd.DataFrame(
                iot23_data['features'], columns=iot23_data['feature_names'])
            
            # FIXED: Dynamically find the 'Benign' label ID from metadata
            labels_map = iot23_data.get('label_mapping', {})
            benign_id = None
            
            # Search for 'Benign' or 'Normal' in the label mapping (case-insensitive)
            # Keywords to identify benign class
            benign_keywords = ['benign', 'normal', 'clean', 'legit']
            
            if labels_map:
                for lbl_id, lbl_name in labels_map.items():
                    if any(keyword in str(lbl_name).lower() for keyword in benign_keywords):
                        benign_id = lbl_id
                        print(f"IoT-23: Found Benign-equivalent label at ID={benign_id} ('{lbl_name}')")
                        break
            
            # If 'Benign' not found, fallback to 0 but warn
            if benign_id is None:
                print(f"Warning: IoT-23 'Benign' label not found in mapping: {labels_map}. Assuming ID=0 is Benign.")
                benign_id = 0
            
            # Classify as Malicious (1) if NOT Benign (0)
            df_i23['binary_label'] = (iot23_data['labels'] != benign_id).astype(int)
            
            datasets.append(df_i23)
            print(f"IoT-23 dataset loaded: {len(df_i23)} samples, {len(df_i23.columns)-1} features")
            
        except Exception as e:
            print(f"Skipping IoT-23: {e}")

        # ========================================================================
        # 3. BoT-IoT - FIXED LABEL MAPPING
        # ========================================================================
        try:
            bot_data = self.load_bot_iot_dataset()
            df_bot = pd.DataFrame(
                bot_data['features'], columns=bot_data['feature_names'])
            
            # FIXED: Dynamically find the 'Benign' label ID from metadata
            labels_map = bot_data.get('label_mapping', {})
            benign_id = None
            
            # Search for 'Benign' or 'Normal' in the label mapping
            benign_keywords = ['benign', 'normal', 'clean', 'legit']
            
            if labels_map:
                for lbl_id, lbl_name in labels_map.items():
                    if any(keyword in str(lbl_name).lower() for keyword in benign_keywords):
                        benign_id = lbl_id
                        print(f"BoT-IoT: Found Benign-equivalent label at ID={benign_id} ('{lbl_name}')")
                        break
            
            # If 'Benign' not found, fallback to 0 but warn
            if benign_id is None:
                print(f"Warning: BoT-IoT 'Benign' label not found in mapping: {labels_map}. Assuming ID=0 is Benign.")
                benign_id = 0
            
            # Classify as Malicious (1) if NOT Benign (0)
            df_bot['binary_label'] = (bot_data['labels'] != benign_id).astype(int)
            
            datasets.append(df_bot)
            print(f"BoT-IoT dataset loaded: {len(df_bot)} samples, {len(df_bot.columns)-1} features")
            
        except Exception as e:
            print(f"Skipping BoT-IoT: {e}")

        # ========================================================================
        # GUARD: Check if any datasets were loaded
        # ========================================================================
        if not datasets:
            raise ValueError(
                "No datasets could be loaded for unification. "
                "Check data paths and availability:\n"
                f"  - N-BaIoT: {self.data_paths.get('n_baiot', 'NOT SET')}\n"
                f"  - IoT-23: {self.data_paths.get('iot_23', 'NOT SET')}\n"
                f"  - BoT-IoT: {self.data_paths.get('bot_iot', 'NOT SET')}"
            )

        # ========================================================================
        # Combine all datasets with feature alignment
        # ========================================================================
        # STRATEGY: Union of features with zero-fill for missing values.
        # This allows the model to utilize all available information.

        all_features = set()
        for df in datasets:
            # Exclude our label columns
            feats = [c for c in df.columns if c not in ['label', 'binary_label']]
            all_features.update(feats)

        common_features = list(all_features)
        print(f"Merging {len(datasets)} datasets. Total unique features: {len(common_features)}")

        # Align all datasets to have the same feature columns
        final_dfs = []
        for i, df in enumerate(datasets):
            # 1. Identify missing features
            current_feats = set(df.columns)
            missing_feats = list(all_features - current_feats)

            # 2. Add missing columns as zeros
            if missing_feats:
                zeros = pd.DataFrame(0, index=df.index, columns=missing_feats, dtype=np.float32)
                df_aligned = pd.concat([df, zeros], axis=1)
            else:
                df_aligned = df.copy()

            # 3. Reorder columns to match common_features + binary_label
            cols_to_keep = common_features + ['binary_label']
            df_final = df_aligned[cols_to_keep].copy()
            df_final.rename(columns={'binary_label': 'label'}, inplace=True)

            final_dfs.append(df_final)
            print(f"  Dataset {i+1} aligned: {len(df_final)} samples, {len(common_features)} features")

        # ========================================================================
        # GUARD: Check if alignment succeeded
        # ========================================================================
        if not final_dfs:
            raise ValueError(
                "Dataset alignment failed. All DataFrames are empty after processing. "
                "This indicates a critical bug in feature alignment logic."
            )

        # Concatenate all aligned datasets
        unified_df = pd.concat(final_dfs, ignore_index=True)
        print(f"Unified dataset created: {len(unified_df)} samples, {len(common_features)} features")

        # ========================================================================
        # Subsample if too large
        # ========================================================================
        if len(unified_df) > max_samples:
            print(f"Subsampling unified dataset from {len(unified_df)} to {max_samples}")
            unified_df = unified_df.sample(n=max_samples, random_state=42)

        # ========================================================================
        # Return structured dataset
        # ========================================================================
        X = unified_df.drop(columns=['label']).values.astype(np.float32)
        y = unified_df['label'].values

        print(f"Final unified dataset: {X.shape}")
        print(f"Label distribution: {np.bincount(y)}")

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
