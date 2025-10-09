# Continue with more core components

# 8. Drift Detector (Main orchestrator)
drift_detector_content = '''"""
Main Drift Detector for Enhanced IoT BotScan
Author: Kotiwale Sumesh Singh (160124862043)

Orchestrates multiple drift detection methods and provides unified drift detection interface.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from datetime import datetime
from collections import defaultdict

from .kolmogorov_smirnov import KolmogorovSmirnovDriftDetector
from .page_hinkley import PageHinkleyDriftDetector

class DriftDetector:
    """Main drift detector that combines multiple detection methods."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.methods = config.get('methods', ['kolmogorov_smirnov', 'page_hinkley'])
        
        # Initialize detectors
        self.detectors = {}
        
        if 'kolmogorov_smirnov' in self.methods:
            self.detectors['ks'] = KolmogorovSmirnovDriftDetector(
                config.get('kolmogorov_smirnov', {})
            )
        
        if 'page_hinkley' in self.methods:
            self.detectors['ph'] = PageHinkleyDriftDetector(
                config.get('page_hinkley', {})
            )
        
        # Consensus parameters
        self.consensus_threshold = config.get('consensus_threshold', 0.5)
        self.voting_strategy = config.get('voting_strategy', 'majority')  # majority, unanimous, any
        
        # State tracking
        self.drift_detected = False
        self.detection_history = []
        self.performance_history = []
        self.consensus_history = []
        
        print(f"Drift detector initialized with methods: {list(self.detectors.keys())}")
    
    def add_reference_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Add reference data to all applicable detectors."""
        
        if 'ks' in self.detectors:
            self.detectors['ks'].add_reference_data(X)
        
        print(f"Reference data added to {len(self.detectors)} detectors")
    
    def detect_drift(self, 
                    X_new: Optional[np.ndarray] = None,
                    y_true: Optional[np.ndarray] = None,
                    y_pred: Optional[np.ndarray] = None,
                    performance_metric: Optional[float] = None) -> Dict[str, Any]:
        """
        Comprehensive drift detection using multiple methods.
        
        Args:
            X_new: New feature data
            y_true: True labels (for performance monitoring)
            y_pred: Predicted labels (for performance monitoring) 
            performance_metric: Direct performance metric value
            
        Returns:
            Comprehensive drift detection results
        """
        
        detection_results = {}
        individual_detections = {}
        
        # K-S test on feature distribution
        if 'ks' in self.detectors and X_new is not None:
            ks_result = self.detectors['ks'].detect_drift(X_new)
            individual_detections['ks'] = ks_result['drift_detected']
            detection_results['ks_test'] = ks_result
        
        # Page-Hinkley test on performance
        if 'ph' in self.detectors:
            if performance_metric is not None:
                ph_result = self.detectors['ph'].add_element(performance_metric)
            elif y_true is not None and y_pred is not None:
                ph_result = self.detectors['ph'].detect_performance_drift(y_true, y_pred)
            else:
                ph_result = {'drift_detected': False, 'reason': 'No performance data provided'}
            
            individual_detections['ph'] = ph_result['drift_detected']
            detection_results['ph_test'] = ph_result
        
        # Consensus decision
        consensus_result = self._make_consensus_decision(individual_detections)
        
        # Update state
        self.drift_detected = consensus_result['drift_detected']
        
        # Store results
        detection_record = {
            'timestamp': datetime.now().isoformat(),
            'individual_detections': individual_detections,
            'consensus_result': consensus_result,
            'drift_detected': self.drift_detected
        }
        
        self.detection_history.append(detection_record)
        self.consensus_history.append(consensus_result)
        
        # Combine all results
        final_result = {
            'drift_detected': self.drift_detected,
            'consensus': consensus_result,
            'individual_results': detection_results,
            'methods_used': list(self.detectors.keys()),
            'timestamp': detection_record['timestamp']
        }
        
        if self.drift_detected:
            print(f"🚨 CONSENSUS DRIFT DETECTED! Methods: {consensus_result['agreeing_methods']}")
        
        return final_result
    
    def _make_consensus_decision(self, individual_detections: Dict[str, bool]) -> Dict[str, Any]:
        """Make consensus decision based on individual detector results."""
        
        if not individual_detections:
            return {
                'drift_detected': False,
                'agreeing_methods': [],
                'disagreeing_methods': [],
                'consensus_score': 0.0,
                'voting_strategy': self.voting_strategy
            }
        
        # Count agreements
        positive_detections = [method for method, detected in individual_detections.items() if detected]
        negative_detections = [method for method, detected in individual_detections.items() if not detected]
        
        n_positive = len(positive_detections)
        n_total = len(individual_detections)
        consensus_score = n_positive / n_total if n_total > 0 else 0.0
        
        # Apply voting strategy
        if self.voting_strategy == 'majority':
            drift_detected = consensus_score > 0.5
        elif self.voting_strategy == 'unanimous':
            drift_detected = consensus_score == 1.0
        elif self.voting_strategy == 'any':
            drift_detected = consensus_score > 0.0
        elif self.voting_strategy == 'threshold':
            drift_detected = consensus_score >= self.consensus_threshold
        else:
            drift_detected = consensus_score > 0.5  # Default to majority
        
        return {
            'drift_detected': drift_detected,
            'agreeing_methods': positive_detections,
            'disagreeing_methods': negative_detections,
            'consensus_score': consensus_score,
            'voting_strategy': self.voting_strategy,
            'n_methods_positive': n_positive,
            'n_methods_total': n_total
        }
    
    def monitor_model_performance(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Monitor model performance and detect performance drift."""
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate performance metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        performance_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Store performance history
        performance_record = {
            'timestamp': datetime.now().isoformat(),
            'metrics': performance_metrics,
            'n_samples': len(X_test)
        }
        self.performance_history.append(performance_record)
        
        # Detect drift using performance metrics
        drift_result = self.detect_drift(
            X_new=X_test,
            y_true=y_test,
            y_pred=y_pred
        )
        
        return {
            'performance_metrics': performance_metrics,
            'drift_detection': drift_result,
            'performance_trend': self._analyze_performance_trend()
        }
    
    def _analyze_performance_trend(self, window_size: int = 10) -> Dict[str, Any]:
        """Analyze recent performance trend."""
        
        if len(self.performance_history) < 2:
            return {'trend': 'insufficient_data'}
        
        recent_performance = self.performance_history[-window_size:]
        
        # Extract accuracy values
        accuracies = [record['metrics']['accuracy'] for record in recent_performance]
        
        if len(accuracies) < 2:
            return {'trend': 'insufficient_data'}
        
        # Calculate trend
        x = np.arange(len(accuracies))
        slope, _ = np.polyfit(x, accuracies, 1)
        
        # Determine trend direction
        if slope > 0.01:
            trend = 'improving'
        elif slope < -0.01:
            trend = 'degrading'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'slope': slope,
            'recent_mean_accuracy': np.mean(accuracies),
            'recent_std_accuracy': np.std(accuracies),
            'samples_analyzed': len(recent_performance)
        }
    
    def reset_detectors(self) -> None:
        """Reset all detectors after handling drift."""
        
        for detector_name, detector in self.detectors.items():
            detector.reset()
        
        self.drift_detected = False
        print("All drift detectors reset")
    
    def update_reference_data(self, X_new: np.ndarray) -> None:
        """Update reference data for applicable detectors."""
        
        if 'ks' in self.detectors:
            self.detectors['ks'].update_reference_window(X_new)
        
        print("Reference data updated for applicable detectors")
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all detectors."""
        
        detector_stats = {}
        
        for name, detector in self.detectors.items():
            detector_stats[name] = detector.get_drift_statistics()
        
        # Overall statistics
        total_detections = len([h for h in self.detection_history if h['drift_detected']])
        detection_rate = total_detections / max(len(self.detection_history), 1)
        
        return {
            'overall_statistics': {
                'total_detection_events': len(self.detection_history),
                'total_drift_detections': total_detections,
                'detection_rate': detection_rate,
                'current_drift_status': self.drift_detected,
                'methods_enabled': list(self.detectors.keys()),
                'consensus_threshold': self.consensus_threshold,
                'voting_strategy': self.voting_strategy
            },
            'detector_statistics': detector_stats,
            'recent_consensus': self.consensus_history[-10:] if self.consensus_history else []
        }
    
    def export_detection_history(self, filepath: str) -> None:
        """Export complete detection history to CSV."""
        
        if not self.detection_history:
            print("No detection history to export")
            return
        
        # Flatten detection history for CSV export
        flattened_history = []
        
        for record in self.detection_history:
            flat_record = {
                'timestamp': record['timestamp'],
                'drift_detected': record['drift_detected'],
                'consensus_score': record['consensus_result']['consensus_score'],
                'voting_strategy': record['consensus_result']['voting_strategy']
            }
            
            # Add individual detector results
            for method, result in record['individual_detections'].items():
                flat_record[f'{method}_detection'] = result
            
            flattened_history.append(flat_record)
        
        df = pd.DataFrame(flattened_history)
        df.to_csv(filepath, index=False)
        
        print(f"Detection history exported to {filepath}")
    
    def get_drift_summary_report(self) -> str:
        """Generate a comprehensive drift detection summary report."""
        
        stats = self.get_comprehensive_statistics()
        
        report = f"""
ENHANCED IOT BOTSCAN - DRIFT DETECTION SUMMARY REPORT
====================================================

Overall Statistics:
- Total Detection Events: {stats['overall_statistics']['total_detection_events']}
- Total Drift Detections: {stats['overall_statistics']['total_drift_detections']}
- Detection Rate: {stats['overall_statistics']['detection_rate']:.2%}
- Current Drift Status: {'ACTIVE' if stats['overall_statistics']['current_drift_status'] else 'STABLE'}
- Methods Enabled: {', '.join(stats['overall_statistics']['methods_enabled'])}
- Voting Strategy: {stats['overall_statistics']['voting_strategy']}

Individual Detector Statistics:
"""
        
        for detector_name, detector_stats in stats['detector_statistics'].items():
            report += f"""
{detector_name.upper()} Detector:
- Samples Processed: {detector_stats.get('total_samples_processed', 'N/A')}
- Drift Detections: {detector_stats.get('drift_detections', 'N/A')}
- Current Status: {'DRIFT DETECTED' if detector_stats.get('drift_detected', False) else 'STABLE'}
"""
        
        return report
'''

with open('./enhanced_iot_botscan/src/core/drift_detection/drift_detector.py', 'w') as f:
    f.write(drift_detector_content)

print("✅ Created drift_detector.py")

# 9. Data Loader Implementation
data_loader_content = '''"""
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
'''

with open('./enhanced_iot_botscan/src/data/data_loader.py', 'w') as f:
    f.write(data_loader_content)

print("✅ Created data_loader.py")

print("\n🎯 Core components are taking shape! More implementations coming...")