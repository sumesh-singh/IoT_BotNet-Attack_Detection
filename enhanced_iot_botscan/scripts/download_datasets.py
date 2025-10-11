#!/usr/bin/env python3
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
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

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

        print("\n📥 Downloading N-BaIoT Dataset...")
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

        print(f"\n✅ N-BaIoT download completed: {success_count}/{len(self.datasets['n_baiot']['files'])} devices")
        return success_count == len(self.datasets['n_baiot']['files'])

    def download_iot_23(self) -> bool:
        print("\n📥 IoT-23 requires manual download from Stratosphere. Skipping automated download.")
        dataset_dir = self.data_dir / 'iot_23'
        dataset_dir.mkdir(exist_ok=True)
        csv_files = list(dataset_dir.glob('*.csv'))
        if csv_files:
            print(f"✅ Found {len(csv_files)} CSV files in IoT-23 directory")
            return True
        # Not found -> return False but keep directory
        return False

    def download_bot_iot(self) -> bool:
        print("\n📥 BoT-IoT requires manual download from Stratosphere. Skipping automated download.")
        dataset_dir = self.data_dir / 'bot_iot'
        dataset_dir.mkdir(exist_ok=True)
        csv_files = list(dataset_dir.glob('*.csv'))
        if csv_files:
            print(f"✅ Found {len(csv_files)} CSV files in BoT-IoT directory")
            return True
        # Not found -> return False but keep directory
        return False

    def create_sample_data(self) -> None:
        """Create sample data for testing when real datasets are not available."""

        print("\n🔬 Creating sample data for testing...")
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
        print("\n📊 Download Summary:")
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
        print("\n🎉 All downloads completed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️ Some downloads failed. Check the logs above.")
        sys.exit(1)

if __name__ == '__main__':
    main()
