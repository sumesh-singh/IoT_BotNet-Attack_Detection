"""
Enhanced Data Loader with Memory Optimization and Drift Detection
Applied from provided attachment and integrated for project use.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import gc

logger = logging.getLogger(__name__)


class OptimizedDataLoader:
    """Memory-efficient data loader with chunked processing"""

    def __init__(self,
                 n_baiot_path: str = "./data/raw/n_baiot/",
                 max_samples_per_device: int = 50000,
                 chunk_size: int = 10000):
        """
        Args:
            n_baiot_path: Path to N-BaIoT dataset
            max_samples_per_device: Maximum samples to load per device
            chunk_size: Size of chunks for processing
        """
        self.n_baiot_path = Path(n_baiot_path)
        self.max_samples_per_device = max_samples_per_device
        self.chunk_size = chunk_size
        self.label_encoder = LabelEncoder()

    def load_n_baiot_optimized(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load N-BaIoT dataset with memory optimization
        Handles flat CSV structure (1.benign.csv, 1.mirai.csv, etc.)

        Returns:
            X: Features (float32)
            y: Labels (encoded)
        """
        logger.info("Loading N-BaIoT dataset with memory optimization...")

        all_data = []
        all_labels = []

        if not self.n_baiot_path.exists():
            logger.warning(f"N-BaIoT path does not exist: {self.n_baiot_path}")
            return pd.DataFrame(), pd.Series()

        # Check if structure is flat (CSV files) or nested (device directories)
        csv_files = list(self.n_baiot_path.glob("*.csv"))
        device_dirs = [d for d in self.n_baiot_path.iterdir() if d.is_dir()]

        # If we have flat CSV files, use them directly
        if csv_files:
            logger.info(f"Detected flat CSV structure with {len(csv_files)} files")
            # Group by device prefix (e.g., "1.benign.csv" -> device "1")
            device_groups = {}
            for csv_file in csv_files:
                device_prefix = csv_file.stem.split('.')[0]  # "1" from "1.benign.csv"
                if device_prefix.isdigit():
                    if device_prefix not in device_groups:
                        device_groups[device_prefix] = []
                    device_groups[device_prefix].append(csv_file)

            samples_per_device = self.max_samples_per_device

            for device_id, files_in_device in sorted(device_groups.items()):
                logger.info(f"Processing device {device_id} with {len(files_in_device)} files")

                device_samples = []
                device_labels = []
                samples_loaded = 0

                for csv_file in files_in_device:
                    if samples_loaded >= samples_per_device:
                        break

                    try:
                        # Extract attack type from filename (benign/mirai/gafgyt/bashlite)
                        filename_parts = csv_file.stem.split('.')
                        attack_type = filename_parts[1] if len(
                            filename_parts) > 1 else 'unknown'

                        # Load in chunks with float32
                        chunk_iter = pd.read_csv(
                            csv_file,
                            chunksize=self.chunk_size,
                            dtype=np.float32
                        )

                        file_samples = []
                        for chunk in chunk_iter:
                            if samples_loaded >= samples_per_device:
                                break

                            # Sample if we're near the limit
                            if samples_loaded + len(chunk) > samples_per_device:
                                n_take = samples_per_device - samples_loaded
                                chunk = chunk.sample(
                                    n=n_take, random_state=42)

                            file_samples.append(chunk)
                            samples_loaded += len(chunk)

                        if file_samples:
                            file_data = pd.concat(
                                file_samples, ignore_index=True)
                            device_samples.append(file_data)
                            device_labels.extend([attack_type] * len(file_data))

                            logger.info(
                                f"  Loaded {len(file_data)} samples from {attack_type}")

                        del file_samples
                        gc.collect()

                    except Exception as e:
                        logger.error(f"Error loading {csv_file}: {e}")
                        continue

                if device_samples:
                    device_data = pd.concat(
                        device_samples, ignore_index=True)
                    all_data.append(device_data)
                    all_labels.extend(device_labels)

        # Otherwise, use nested directory structure
        elif device_dirs:
            logger.info(f"Detected nested directory structure with {len(device_dirs)} devices")
            for device_dir in sorted(device_dirs):
                device_name = device_dir.name
                logger.info(f"Processing device: {device_name}")

                csv_files = list(device_dir.glob("*.csv"))
                samples_per_file = self.max_samples_per_device // len(
                    csv_files) if csv_files else 0

                for csv_file in csv_files:
                    try:
                        # Extract attack type from filename
                        attack_type = csv_file.stem.split('_')[-1]

                        # Load in chunks with float32
                        chunk_iter = pd.read_csv(
                            csv_file,
                            chunksize=self.chunk_size,
                            dtype=np.float32
                        )

                        device_samples = []
                        total_loaded = 0

                        for chunk in chunk_iter:
                            if total_loaded >= samples_per_file:
                                break

                            # Sample from chunk if needed
                            n_samples = min(
                                len(chunk), samples_per_file - total_loaded)
                            if n_samples < len(chunk):
                                chunk = chunk.sample(n=n_samples, random_state=42)

                            device_samples.append(chunk)
                            total_loaded += len(chunk)
                        if device_samples:
                            device_data = pd.concat(
                                device_samples, ignore_index=True)
                            all_data.append(device_data)
                            all_labels.extend([attack_type] * len(device_data))

                            logger.info(
                                f"  Loaded {len(device_data)} samples from {attack_type}")

                        # Clear memory
                        del device_samples
                        gc.collect()

                    except Exception as e:
                        logger.error(f"Error loading {csv_file}: {e}")
                        continue
        else:
            logger.warning(f"No CSV files or device directories found in {self.n_baiot_path}")

        # Combine all data
        if not all_data:
            # Fallback: try CSV files directly under folder
            fallback_files = sorted(
                [f for f in self.n_baiot_path.glob('*.csv')])
            for csv_file in fallback_files:
                try:
                    df = pd.read_csv(csv_file, dtype=np.float32)
                    all_data.append(df)
                    attack_type = csv_file.stem.split('_')[-1]
                    all_labels.extend([attack_type] * len(df))
                except Exception:
                    continue

        X = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
        y = pd.Series(all_labels) if all_labels else pd.Series(dtype=object)

        # Encode labels
        y = pd.Series(self.label_encoder.fit_transform(y))

        logger.info(f"N-BaIoT loaded: {len(X)} samples, {X.shape[1]} features")
        logger.info(f"Class distribution:\n{y.value_counts()}")

        # Clear memory
        del all_data, all_labels
        gc.collect()

        return X, y

    def create_balanced_dataset(self, X: pd.DataFrame, y: pd.Series,
                                min_samples: int = 5000) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create balanced dataset using stratified sampling
        """
        logger.info("Creating balanced dataset...")

        # Get class counts
        class_counts = y.value_counts()

        # Undersample majority classes or oversample minority
        balanced_indices = []

        for class_label in class_counts.index:
            class_indices = y[y == class_label].index

            if len(class_indices) > min_samples:
                # Undersample
                sampled_indices = np.random.choice(
                    class_indices,
                    size=min_samples,
                    replace=False
                )
            else:
                # Oversample
                sampled_indices = np.random.choice(
                    class_indices,
                    size=min_samples,
                    replace=True
                )

            balanced_indices.extend(sampled_indices)

        # Shuffle
        np.random.shuffle(balanced_indices)

        X_balanced = X.iloc[balanced_indices].reset_index(drop=True)
        y_balanced = y.iloc[balanced_indices].reset_index(drop=True)

        logger.info(f"Balanced dataset: {len(X_balanced)} samples")
        logger.info(f"Class distribution:\n{y_balanced.value_counts()}")

        return X_balanced, y_balanced


class DriftSimulator:
    """Generate synthetic drift for testing drift detection"""

    @staticmethod
    def generate_covariate_shift(X: pd.DataFrame,
                                 drift_magnitude: float = 0.3) -> pd.DataFrame:
        X_drift = X.copy()
        n_drift_features = int(len(X.columns) * 0.3)
        drift_features = np.random.choice(
            X.columns, n_drift_features, replace=False)

        for feature in drift_features:
            shift = X[feature].std() * drift_magnitude
            X_drift[feature] = X[feature] + shift
            noise = np.random.normal(0, X[feature].std() * 0.1, len(X))
            X_drift[feature] += noise

        logger.info(
            f"Generated covariate shift in {n_drift_features} features")
        return X_drift

    @staticmethod
    def generate_label_shift(y: pd.Series,
                             shift_ratio: float = 0.3) -> pd.Series:
        y_shift = y.copy()
        classes = y.unique()
        n_shift = int(len(y) * shift_ratio)
        shift_indices = np.random.choice(len(y), n_shift, replace=False)

        for idx in shift_indices:
            current_class = y_shift.iloc[idx]
            new_class = np.random.choice(
                [c for c in classes if c != current_class])
            y_shift.iloc[idx] = new_class

        logger.info(f"Generated label shift: {n_shift} samples modified")
        return y_shift


class AdversarialSimulator:
    """Simulate adversarial attacks without PyTorch"""

    @staticmethod
    def fgsm_simulation(X: pd.DataFrame,
                        epsilon: float = 0.1) -> pd.DataFrame:
        X_adv = X.copy()
        for col in X.columns:
            sign = np.random.choice([-1, 1], size=len(X))
            perturbation = epsilon * X[col].std() * sign
            X_adv[col] = X[col] + perturbation
        X_adv = X_adv.clip(X.min().min(), X.max().max())
        logger.info(
            f"Generated FGSM adversarial samples with epsilon={epsilon}")
        return X_adv

    @staticmethod
    def random_noise_attack(X: pd.DataFrame,
                            noise_level: float = 0.2) -> pd.DataFrame:
        X_noisy = X.copy()
        for col in X.columns:
            noise = np.random.normal(0, X[col].std() * noise_level, len(X))
            X_noisy[col] = X[col] + noise
        logger.info(f"Generated random noise attack with level={noise_level}")
        return X_noisy


if __name__ == "__main__":
    loader = OptimizedDataLoader(
        n_baiot_path="./data/raw/n_baiot/",
        max_samples_per_device=50000,
        chunk_size=10000
    )
    X, y = loader.load_n_baiot_optimized()
    X_balanced, y_balanced = loader.create_balanced_dataset(
        X, y, min_samples=5000)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    drift_sim = DriftSimulator()
    X_drift = drift_sim.generate_covariate_shift(X_test, drift_magnitude=0.3)
    adv_sim = AdversarialSimulator()
    X_adv_fgsm = adv_sim.fgsm_simulation(X_test, epsilon=0.1)
    X_adv_noise = adv_sim.random_noise_attack(X_test, noise_level=0.2)
    print("✓ Data loading and simulation complete!")
