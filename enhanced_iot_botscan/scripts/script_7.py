# Continue with more essential components

# 10. Main Script for Dataset Downloading
download_datasets_content = '''#!/usr/bin/env python3
"""
Dataset Download Script for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Downloads and prepares N-BaIoT, IoT-23, and BoT-IoT datasets for training.
"""

import os
import sys
import urllib.request
import zipfile
import tarfile
import gzip
import shutil
from pathlib import Path
import argparse
import requests
from tqdm import tqdm

class DatasetDownloader:
    """Downloads and prepares IoT botnet datasets."""
    
    def __init__(self, data_dir: str = './data/raw'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset URLs and information
        self.datasets = {
            'n_baiot': {
                'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00442/',
                'description': 'N-BaIoT Dataset - Network-based Detection of IoT Botnet Attacks',
                'files': [
                    'Danmini_Doorbell.zip',
                    'Ecobee_Thermostat.zip', 
                    'Ennio_Doorbell.zip',
                    'Philips_B120N10_Baby_Monitor.zip',
                    'Provision_PT_737E_Security_Camera.zip',
                    'Provision_PT_838_Security_Camera.zip',
                    'Samsung_SNH_1011_N_Webcam.zip',
                    'SimpleHome_XCS7_1002_WHT_Security_Camera.zip',
                    'SimpleHome_XCS7_1003_WHT_Security_Camera.zip'
                ]
            },
            'iot_23': {
                'url': 'https://www.stratosphereips.org/datasets-iot23',
                'description': 'IoT-23 Dataset - Labeled dataset with IoT network traffic',
                'files': ['iot_23_dataset.zip']
            },
            'bot_iot': {
                'url': 'https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/',
                'description': 'BoT-IoT Dataset - Bot-IoT dataset for IoT botnet detection',
                'files': ['bot_iot.zip']
            }
        }
    
    def download_file(self, url: str, filename: str, chunk_size: int = 8192) -> bool:
        """Download file with progress bar."""
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(filename)) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"✅ Downloaded: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
            return False
    
    def extract_archive(self, archive_path: str, extract_to: str) -> bool:
        """Extract various archive formats."""
        
        try:
            archive_path = Path(archive_path)
            extract_to = Path(extract_to)
            extract_to.mkdir(parents=True, exist_ok=True)
            
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
                    
            elif archive_path.suffix in ['.tar', '.tar.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
                    
            elif archive_path.suffix == '.gz':
                with gzip.open(archive_path, 'rb') as f_in:
                    with open(extract_to / archive_path.stem, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            
            print(f"✅ Extracted: {archive_path} -> {extract_to}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to extract {archive_path}: {e}")
            return False
    
    def download_n_baiot(self) -> bool:
        """Download N-BaIoT dataset."""
        
        print("\\n📥 Downloading N-BaIoT Dataset...")
        print("=" * 50)
        
        dataset_dir = self.data_dir / 'n_baiot'
        dataset_dir.mkdir(exist_ok=True)
        
        base_url = self.datasets['n_baiot']['url']
        
        success_count = 0
        
        for device_file in self.datasets['n_baiot']['files']:
            device_name = device_file.replace('.zip', '')
            device_dir = dataset_dir / device_name
            
            # Skip if already downloaded and extracted
            if device_dir.exists() and any(device_dir.iterdir()):
                print(f"⏭️ Skipping {device_name} (already exists)")
                success_count += 1
                continue
            
            # Download device data
            file_url = f"{base_url}{device_file}"
            download_path = dataset_dir / device_file
            
            if self.download_file(file_url, str(download_path)):
                # Extract the downloaded file
                if self.extract_archive(str(download_path), str(device_dir)):
                    # Clean up zip file
                    download_path.unlink()
                    success_count += 1
                    
                    # Verify extraction
                    csv_files = list(device_dir.glob('*.csv'))
                    print(f"   Found {len(csv_files)} CSV files in {device_name}")
        
        print(f"\\n✅ N-BaIoT download completed: {success_count}/{len(self.datasets['n_baiot']['files'])} devices")
        return success_count == len(self.datasets['n_baiot']['files'])
    
    def download_iot_23(self) -> bool:
        """Download IoT-23 dataset (placeholder - adjust URL as needed)."""
        
        print("\\n📥 Downloading IoT-23 Dataset...")
        print("=" * 50)
        print("⚠️ Note: IoT-23 requires manual download from Stratosphere IPS")
        print("Please visit: https://www.stratosphereips.org/datasets-iot23")
        print("Download the dataset manually and place in:", self.data_dir / 'iot_23')
        
        # Create directory structure
        dataset_dir = self.data_dir / 'iot_23'
        dataset_dir.mkdir(exist_ok=True)
        
        # Check if files already exist
        csv_files = list(dataset_dir.glob('*.csv'))
        if csv_files:
            print(f"✅ Found {len(csv_files)} CSV files in IoT-23 directory")
            return True
        else:
            print("❌ No CSV files found. Please download manually.")
            return False
    
    def download_bot_iot(self) -> bool:
        """Download BoT-IoT dataset (placeholder - adjust URL as needed)."""
        
        print("\\n📥 Downloading BoT-IoT Dataset...")
        print("=" * 50)
        print("⚠️ Note: BoT-IoT requires manual download from UNSW")
        print("Please visit: https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/")
        print("Download the BoT-IoT dataset manually and place in:", self.data_dir / 'bot_iot')
        
        # Create directory structure
        dataset_dir = self.data_dir / 'bot_iot'
        dataset_dir.mkdir(exist_ok=True)
        
        # Check if files already exist
        csv_files = list(dataset_dir.glob('*.csv'))
        if csv_files:
            print(f"✅ Found {len(csv_files)} CSV files in BoT-IoT directory")
            return True
        else:
            print("❌ No CSV files found. Please download manually.")
            return False
    
    def create_sample_data(self) -> None:
        """Create sample data for testing when real datasets are not available."""
        
        print("\\n🔬 Creating sample data for testing...")
        print("=" * 50)
        
        import pandas as pd
        import numpy as np
        
        # Create sample N-BaIoT data
        n_baiot_dir = self.data_dir / 'n_baiot' / 'Sample_Device'
        n_baiot_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate sample features (115 features as in real N-BaIoT)
        n_samples = 10000
        n_features = 115
        
        # Benign samples
        benign_data = np.random.normal(0, 1, (n_samples // 2, n_features))
        benign_df = pd.DataFrame(benign_data, columns=[f'feature_{i}' for i in range(n_features)])
        benign_df.to_csv(n_baiot_dir / 'benign_traffic.csv', index=False)
        
        # Malware samples (Mirai)
        malware_data = np.random.normal(2, 1.5, (n_samples // 2, n_features))
        malware_df = pd.DataFrame(malware_data, columns=[f'feature_{i}' for i in range(n_features)])
        malware_df.to_csv(n_baiot_dir / 'mirai_attacks.csv', index=False)
        
        print(f"✅ Created sample N-BaIoT data: {n_samples} samples, {n_features} features")
        
        # Create sample IoT-23 data
        iot_23_dir = self.data_dir / 'iot_23'
        iot_23_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate sample IoT-23 data with different feature set
        iot_23_data = np.random.random((5000, 50))
        iot_23_labels = np.random.randint(0, 3, 5000)  # 3 classes
        
        iot_23_df = pd.DataFrame(iot_23_data, columns=[f'feat_{i}' for i in range(50)])
        iot_23_df['label'] = iot_23_labels
        iot_23_df.to_csv(iot_23_dir / 'iot_23_sample.csv', index=False)
        
        print(f"✅ Created sample IoT-23 data: 5000 samples, 50 features")
        
        # Create sample BoT-IoT data
        bot_iot_dir = self.data_dir / 'bot_iot'
        bot_iot_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate sample BoT-IoT data
        bot_iot_data = np.random.exponential(1, (8000, 30))
        bot_iot_labels = np.random.randint(0, 4, 8000)  # 4 classes
        
        bot_iot_df = pd.DataFrame(bot_iot_data, columns=[f'f_{i}' for i in range(30)])
        bot_iot_df['Label'] = ['Normal' if l == 0 else f'Attack_{l}' for l in bot_iot_labels]
        bot_iot_df.to_csv(bot_iot_dir / 'bot_iot_sample.csv', index=False)
        
        print(f"✅ Created sample BoT-IoT data: 8000 samples, 30 features")
    
    def download_all(self, create_samples: bool = True) -> Dict[str, bool]:
        """Download all datasets."""
        
        print("🚀 Enhanced IoT BotScan - Dataset Downloader")
        print("=" * 60)
        
        results = {}
        
        # Download each dataset
        results['n_baiot'] = self.download_n_baiot()
        results['iot_23'] = self.download_iot_23()
        results['bot_iot'] = self.download_bot_iot()
        
        # Create sample data if requested
        if create_samples:
            self.create_sample_data()
        
        # Summary
        print("\\n📊 Download Summary:")
        print("=" * 30)
        for dataset, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"{dataset.upper()}: {status}")
        
        return results

def main():
    """Main function for command-line usage."""
    
    parser = argparse.ArgumentParser(
        description="Download IoT botnet datasets for Enhanced IoT BotScan"
    )
    parser.add_argument(
        '--data-dir', 
        default='./data/raw',
        help='Directory to store downloaded datasets'
    )
    parser.add_argument(
        '--dataset',
        choices=['n_baiot', 'iot_23', 'bot_iot', 'all'],
        default='all',
        help='Specific dataset to download'
    )
    parser.add_argument(
        '--create-samples',
        action='store_true',
        help='Create sample data for testing'
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = DatasetDownloader(args.data_dir)
    
    # Download specified dataset(s)
    if args.dataset == 'all':
        results = downloader.download_all(create_samples=args.create_samples)
    elif args.dataset == 'n_baiot':
        results = {'n_baiot': downloader.download_n_baiot()}
    elif args.dataset == 'iot_23':
        results = {'iot_23': downloader.download_iot_23()}
    elif args.dataset == 'bot_iot':
        results = {'bot_iot': downloader.download_bot_iot()}
    
    # Create samples if requested
    if args.create_samples and args.dataset != 'all':
        downloader.create_sample_data()
    
    # Exit with appropriate code
    if all(results.values()):
        print("\\n🎉 All downloads completed successfully!")
        sys.exit(0)
    else:
        print("\\n⚠️ Some downloads failed. Check the logs above.")
        sys.exit(1)

if __name__ == '__main__':
    main()
'''

with open('./enhanced_iot_botscan/scripts/download_datasets.py', 'w') as f:
    f.write(download_datasets_content)

print("✅ Created download_datasets.py")

# 11. Main Training Script
train_models_content = '''#!/usr/bin/env python3
"""
Model Training Script for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Trains hybrid ensemble models with adversarial robustness and concept drift adaptation.
"""

import sys
import os
import argparse
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.ensemble.hybrid_ensemble import HybridEnsemble
from core.adversarial.adversarial_trainer import AdversarialTrainer
from core.drift_detection.drift_detector import DriftDetector
from data.data_loader import DataLoader
from evaluation.performance_evaluator import PerformanceEvaluator
from utils.config_manager import ConfigManager
from utils.logger import setup_logging

class ModelTrainer:
    """Main training orchestrator for Enhanced IoT BotScan."""
    
    def __init__(self, config_path: str = None):
        """Initialize trainer with configuration."""
        
        # Setup configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # Setup logging
        setup_logging(self.config.get('logging', {}))
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_loader = DataLoader(self.config.get('data', {}))
        self.ensemble = HybridEnsemble(config_path)
        self.adversarial_trainer = AdversarialTrainer(self.config.get('adversarial_training', {}))
        self.drift_detector = DriftDetector(self.config.get('concept_drift', {}))
        self.evaluator = PerformanceEvaluator(self.config.get('evaluation', {}))
        
        # Training state
        self.datasets = {}
        self.training_results = {}
        
        self.logger.info("ModelTrainer initialized successfully")
    
    def load_datasets(self, dataset_names: list = None) -> None:
        """Load specified datasets."""
        
        if dataset_names is None:
            dataset_names = ['n_baiot', 'iot_23', 'bot_iot']
        
        self.logger.info(f"Loading datasets: {dataset_names}")
        
        for dataset_name in dataset_names:
            try:
                dataset = self.data_loader.load_dataset(dataset_name)
                self.datasets[dataset_name] = dataset
                
                # Log dataset info
                self.logger.info(f"Loaded {dataset_name}: {dataset['total_samples']} samples, "
                               f"{dataset['n_features']} features, {dataset['n_classes']} classes")
                
            except Exception as e:
                self.logger.error(f"Failed to load {dataset_name}: {e}")
    
    def prepare_training_data(self, dataset_name: str = 'n_baiot') -> tuple:
        """Prepare training and validation data."""
        
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        dataset = self.datasets[dataset_name]
        X = dataset['features']
        y = dataset['labels']
        
        # Train-test split
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Further split training into train/validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        self.logger.info(f"Data split - Train: {len(X_train_split)}, "
                        f"Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train_split, X_val, X_test, y_train_split, y_val, y_test
    
    def train_baseline_models(self, dataset_name: str = 'n_baiot') -> dict:
        """Train baseline ensemble models without adversarial training."""
        
        self.logger.info(f"Training baseline models on {dataset_name}")
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_training_data(dataset_name)
        
        # Add reference data for drift detection
        self.drift_detector.add_reference_data(X_train)
        
        # Train ensemble
        training_results = self.ensemble.train(
            X=pd.DataFrame(X_train),
            y=pd.Series(y_train),
            validation_data=(pd.DataFrame(X_val), pd.Series(y_val))
        )
        
        # Evaluate on test set
        test_results = self.evaluator.comprehensive_evaluation(
            self.ensemble, 
            pd.DataFrame(X_test), 
            pd.Series(y_test)
        )
        
        baseline_results = {
            'dataset': dataset_name,
            'training_results': training_results,
            'test_results': test_results,
            'model_info': self.ensemble.get_model_info()
        }
        
        self.training_results[f'{dataset_name}_baseline'] = baseline_results
        
        self.logger.info(f"Baseline training completed. Test accuracy: {test_results['accuracy']:.4f}")
        
        return baseline_results
    
    def train_adversarial_robust_models(self, dataset_name: str = 'n_baiot') -> dict:
        """Train models with adversarial robustness."""
        
        self.logger.info(f"Training adversarially robust models on {dataset_name}")
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_training_data(dataset_name)
        
        # Create a fresh ensemble for adversarial training
        robust_ensemble = HybridEnsemble(self.config_manager.config_path)
        
        # Train with adversarial examples
        adversarial_results = self.adversarial_trainer.train_robust_model(
            robust_ensemble,
            X_train, y_train,
            X_val, y_val
        )
        
        # Evaluate robustness
        robustness_results = self.evaluator.evaluate_adversarial_robustness(
            robust_ensemble,
            pd.DataFrame(X_test),
            pd.Series(y_test)
        )
        
        # Regular evaluation
        test_results = self.evaluator.comprehensive_evaluation(
            robust_ensemble,
            pd.DataFrame(X_test),
            pd.Series(y_test)
        )
        
        robust_model_results = {
            'dataset': dataset_name,
            'adversarial_training_results': adversarial_results,
            'robustness_evaluation': robustness_results,
            'test_results': test_results,
            'model_info': robust_ensemble.get_model_info()
        }
        
        self.training_results[f'{dataset_name}_robust'] = robust_model_results
        
        self.logger.info(f"Adversarial training completed. "
                        f"Clean accuracy: {test_results['accuracy']:.4f}, "
                        f"Robust accuracy: {robustness_results.get('overall_robustness', 0):.4f}")
        
        return robust_model_results
    
    def validate_cross_dataset(self) -> dict:
        """Perform cross-dataset validation."""
        
        self.logger.info("Starting cross-dataset validation")
        
        cross_validation_results = {}
        dataset_names = list(self.datasets.keys())
        
        for train_dataset in dataset_names:
            for test_dataset in dataset_names:
                if train_dataset == test_dataset:
                    continue
                
                self.logger.info(f"Training on {train_dataset}, testing on {test_dataset}")
                
                # Prepare training data
                train_data = self.datasets[train_dataset]
                X_train, y_train = train_data['features'], train_data['labels']
                
                # Prepare test data
                test_data = self.datasets[test_dataset]
                X_test, y_test = test_data['features'], test_data['labels']
                
                # Train model
                cross_ensemble = HybridEnsemble(self.config_manager.config_path)
                cross_ensemble.train(pd.DataFrame(X_train), pd.Series(y_train))
                
                # Evaluate
                results = self.evaluator.comprehensive_evaluation(
                    cross_ensemble,
                    pd.DataFrame(X_test),
                    pd.Series(y_test)
                )
                
                cross_validation_results[f'{train_dataset}_to_{test_dataset}'] = results
                
                self.logger.info(f"Cross-validation {train_dataset}->{test_dataset}: "
                               f"Accuracy = {results['accuracy']:.4f}")
        
        self.training_results['cross_dataset_validation'] = cross_validation_results
        return cross_validation_results
    
    def test_concept_drift_detection(self, dataset_name: str = 'n_baiot') -> dict:
        """Test concept drift detection capabilities."""
        
        self.logger.info(f"Testing concept drift detection on {dataset_name}")
        
        # Get dataset
        dataset = self.datasets[dataset_name]
        X, y = dataset['features'], dataset['labels']
        
        # Split data into batches to simulate streaming
        n_batches = 10
        batch_size = len(X) // n_batches
        
        drift_results = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X))
            
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            
            # Simulate concept drift by modifying later batches
            if i > 6:  # Introduce drift after batch 6
                # Add noise to simulate drift
                noise = np.random.normal(0, 0.5, X_batch.shape)
                X_batch = X_batch + noise
            
            # Detect drift
            drift_result = self.drift_detector.detect_drift(X_new=X_batch)
            drift_results.append({
                'batch': i,
                'drift_detected': drift_result['drift_detected'],
                'drift_methods': drift_result.get('individual_results', {})
            })
            
            if drift_result['drift_detected']:
                self.logger.info(f"Drift detected in batch {i}")
        
        # Get comprehensive drift statistics
        drift_stats = self.drift_detector.get_comprehensive_statistics()
        
        concept_drift_results = {
            'dataset': dataset_name,
            'batch_results': drift_results,
            'drift_statistics': drift_stats,
            'total_batches': n_batches,
            'drift_detections': sum(1 for r in drift_results if r['drift_detected'])
        }
        
        self.training_results['concept_drift_test'] = concept_drift_results
        
        self.logger.info(f"Concept drift testing completed. "
                        f"Drift detected in {concept_drift_results['drift_detections']} batches")
        
        return concept_drift_results
    
    def save_results(self, output_dir: str = './data/results') -> None:
        """Save all training results."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results as YAML
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = output_path / f'training_results_{timestamp}.yaml'
        
        with open(results_file, 'w') as f:
            yaml.dump(self.training_results, f, default_flow_style=False)
        
        # Save models
        models_dir = output_path / 'models'
        models_dir.mkdir(exist_ok=True)
        
        if hasattr(self.ensemble, 'is_trained') and self.ensemble.is_trained:
            model_file = models_dir / f'ensemble_model_{timestamp}.pkl'
            self.ensemble.save_model(str(model_file))
        
        self.logger.info(f"Results saved to {results_file}")
    
    def run_full_training_pipeline(self, datasets: list = None) -> dict:
        """Run the complete training pipeline."""
        
        self.logger.info("Starting full training pipeline")
        
        # Load datasets
        self.load_datasets(datasets)
        
        # Train baseline models on primary dataset
        primary_dataset = 'n_baiot'
        if primary_dataset in self.datasets:
            self.train_baseline_models(primary_dataset)
            self.train_adversarial_robust_models(primary_dataset)
        
        # Cross-dataset validation
        if len(self.datasets) > 1:
            self.validate_cross_dataset()
        
        # Test concept drift detection
        if primary_dataset in self.datasets:
            self.test_concept_drift_detection(primary_dataset)
        
        # Save results
        self.save_results()
        
        self.logger.info("Full training pipeline completed")
        
        return self.training_results

def main():
    """Main function for command-line usage."""
    
    parser = argparse.ArgumentParser(
        description="Train Enhanced IoT BotScan models"
    )
    parser.add_argument(
        '--config',
        default='./config/config.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['n_baiot'],
        choices=['n_baiot', 'iot_23', 'bot_iot'],
        help='Datasets to use for training'
    )
    parser.add_argument(
        '--mode',
        choices=['baseline', 'adversarial', 'full'],
        default='full',
        help='Training mode'
    )
    parser.add_argument(
        '--output-dir',
        default='./data/results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(args.config)
        
        # Load datasets
        trainer.load_datasets(args.datasets)
        
        # Run training based on mode
        if args.mode == 'baseline':
            trainer.train_baseline_models(args.datasets[0])
        elif args.mode == 'adversarial':
            trainer.train_adversarial_robust_models(args.datasets[0])
        elif args.mode == 'full':
            trainer.run_full_training_pipeline(args.datasets)
        
        # Save results
        trainer.save_results(args.output_dir)
        
        print("\\n🎉 Training completed successfully!")
        
    except Exception as e:
        print(f"\\n❌ Training failed: {e}")
        logging.exception("Training failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
'''

with open('./enhanced_iot_botscan/scripts/train_models.py', 'w') as f:
    f.write(train_models_content)

print("✅ Created train_models.py")

print("\n🎯 Essential training and data components completed! Let's check our progress...")