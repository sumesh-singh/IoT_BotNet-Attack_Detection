# -*- coding: utf-8 -*-
"""
Comprehensive Test Script for Enhanced IoT BotScan
===================================================
Author: Kotiwale Sumesh Singh (160124862043)

This script validates all core functionality:
1. Data Loading (N-BaIoT, IoT-23, BoT-IoT)
2. Feature Engineering
3. Model Training (RF, XGBoost, LightGBM, HybridEnsemble)
4. Predictions
5. Drift Detection (KS, Page-Hinkley)
6. ARM Robustness Evaluation
7. Monitor and Adapt (Automated Drift Adaptation)

Run from project root: python tests/test_all_components.py
"""

import sys
import os
import io
import numpy as np
import pandas as pd
import warnings
import logging
import shutil
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestScript")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def record(self, name: str, passed: bool, error: str = None):
        if passed:
            self.passed += 1
            print(f"  ✓ [PASS] {name}")
        else:
            self.failed += 1
            self.errors.append(f"{name}: {error}")
            print(f"  ✗ [FAIL] {name}: {error}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY: {self.passed}/{total} passed ({100*self.passed/total:.1f}%)")
        if self.errors:
            print("\nFailed Tests:")
            for e in self.errors:
                print(f"  - {e}")
        print('='*60)
        return self.failed == 0


def create_sample_data(n_samples=1000, n_features=50, n_classes=2):
    """Create synthetic data for testing."""
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randint(0, n_classes, n_samples), name='label')
    return X, y


def test_data_loader(results: TestResults):
    """Test DataLoader functionality."""
    print("\n[TEST] Data Loader...")
    
    try:
        from src.data.data_loader import DataLoader
        
        # Test 1: DataLoader initialization
        try:
            loader = DataLoader({})
            results.record("DataLoader initialization", True)
        except Exception as e:
            results.record("DataLoader initialization", False, str(e))
            return
        
        # Test 2: Check dataset methods exist
        methods = ['load_n_baiot_dataset', 'load_iot_23_dataset', 
                   'load_bot_iot_dataset', 'load_unified_dataset']
        for method in methods:
            has_method = hasattr(loader, method)
            results.record(f"DataLoader.{method} exists", has_method)
        
        # Test 3: Check supported datasets
        try:
            has_supported = hasattr(loader, 'supported_datasets')
            results.record("DataLoader.supported_datasets exists", has_supported)
        except Exception as e:
            results.record("DataLoader.supported_datasets", False, str(e))
            
    except ImportError as e:
        results.record("DataLoader import", False, str(e))


def test_feature_engineering(results: TestResults):
    """Test FeatureEngineer functionality."""
    print("\n[TEST] Feature Engineering...")
    
    try:
        from src.core.preprocessing.feature_engineer import FeatureEngineer
        
        # Create sample data
        X, y = create_sample_data(500, 30)
        
        # Test 1: Initialization
        try:
            engineer = FeatureEngineer({})
            results.record("FeatureEngineer initialization", True)
        except Exception as e:
            results.record("FeatureEngineer initialization", False, str(e))
            return
        
        # Test 2: engineer_features
        try:
            X_eng = engineer.engineer_features(X, y)
            results.record("engineer_features()", X_eng is not None and len(X_eng) == len(X))
        except Exception as e:
            results.record("engineer_features()", False, str(e))
            return
        
        # Test 3: get_state
        try:
            state = engineer.get_state()
            has_features = 'selected_features' in state and state['selected_features'] is not None
            results.record("get_state() with selected_features", has_features)
        except Exception as e:
            results.record("get_state()", False, str(e))
        
        # Test 4: transform_new_data
        try:
            X_new, _ = create_sample_data(100, 30)
            X_transformed = engineer.transform_new_data(X_new)
            results.record("transform_new_data()", X_transformed is not None)
        except Exception as e:
            results.record("transform_new_data()", False, str(e))
        
        # Test 5: set_state (restore state)
        try:
            new_engineer = FeatureEngineer({})
            new_engineer.set_state(state)
            results.record("set_state() restores state", 
                          new_engineer.selected_features == engineer.selected_features)
        except Exception as e:
            results.record("set_state()", False, str(e))
            
    except ImportError as e:
        results.record("FeatureEngineer import", False, str(e))


def test_base_models(results: TestResults):
    """Test individual base models."""
    print("\n[TEST] Base Models...")
    
    X, y = create_sample_data(500, 30, n_classes=2)
    
    # Test Random Forest
    try:
        from src.core.ensemble.random_forest_model import RandomForestModel
        model = RandomForestModel({})
        model.train(X, y)
        preds = model.predict(X[:10])
        proba = model.predict_proba(X[:10])
        results.record("RandomForest train + predict", preds is not None and len(preds) == 10)
        results.record("RandomForest predict_proba", proba is not None and proba.shape[0] == 10)
    except Exception as e:
        results.record("RandomForest", False, str(e))
    
    # Test XGBoost
    try:
        from src.core.ensemble.xgboost_model import XGBoostModel
        model = XGBoostModel({})
        model.train(X, y)
        preds = model.predict(X[:10])
        proba = model.predict_proba(X[:10])
        results.record("XGBoost train + predict", preds is not None and len(preds) == 10)
        results.record("XGBoost predict_proba", proba is not None and proba.shape[0] == 10)
    except Exception as e:
        results.record("XGBoost", False, str(e))
    
    # Test LightGBM
    try:
        from src.core.ensemble.lightgbm_model import LightGBMModel
        model = LightGBMModel({})
        model.train(X, y)
        preds = model.predict(X[:10])
        proba = model.predict_proba(X[:10])
        results.record("LightGBM train + predict", preds is not None and len(preds) == 10)
        results.record("LightGBM predict_proba", proba is not None and proba.shape[0] == 10)
    except Exception as e:
        results.record("LightGBM", False, str(e))


def test_hybrid_ensemble(results: TestResults):
    """Test HybridEnsemble with stacking."""
    print("\n[TEST] Hybrid Ensemble...")
    
    try:
        from src.core.ensemble.hybrid_ensemble import HybridEnsemble
        from src.core.preprocessing.feature_engineer import FeatureEngineer
        
        X, y = create_sample_data(800, 30, n_classes=2)
        
        # Test 1: Initialization with stacking
        try:
            config = {'use_stacking': True}
            ensemble = HybridEnsemble(config)
            results.record("HybridEnsemble initialization", True)
        except Exception as e:
            results.record("HybridEnsemble initialization", False, str(e))
            return
        
        # Test 2: Training with feature engineer
        try:
            engineer = FeatureEngineer({})
            X_eng = engineer.engineer_features(X, y)
            
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X_eng, y, test_size=0.2)
            
            ensemble.train(X_train, y_train, validation_data=(X_val, y_val), 
                          feature_engineer=engineer)
            results.record("HybridEnsemble train with feature_engineer", ensemble.is_trained)
        except Exception as e:
            results.record("HybridEnsemble train", False, str(e))
            return
        
        # Test 3: Predictions
        try:
            preds = ensemble.predict(X_val)
            probs = ensemble.predict_proba(X_val)
            results.record("HybridEnsemble predict", len(preds) == len(X_val))
            results.record("HybridEnsemble predict_proba", probs.shape[0] == len(X_val))
        except Exception as e:
            results.record("HybridEnsemble predict", False, str(e))
        
        # Test 4: Feature engineer state stored
        try:
            has_fe = hasattr(ensemble, 'feature_engineer') and ensemble.feature_engineer is not None
            results.record("HybridEnsemble stores feature_engineer", has_fe)
        except Exception as e:
            results.record("HybridEnsemble feature_engineer", False, str(e))
        
        # Test 5: n_classes stored
        try:
            has_n_classes = hasattr(ensemble, 'n_classes') and ensemble.n_classes is not None
            results.record("HybridEnsemble stores n_classes", has_n_classes)
        except Exception as e:
            results.record("HybridEnsemble n_classes", False, str(e))
        
        # Test 6: Save and load model
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
                tmp_path = tmp.name
            
            fe_state = engineer.get_state()
            ensemble.save_model(tmp_path, feature_engineer_state=fe_state)
            
            # Load in new instance
            new_ensemble = HybridEnsemble(config)
            new_ensemble.load_model(tmp_path)
            
            results.record("HybridEnsemble save + load", new_ensemble.is_trained)
            
            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
        except Exception as e:
            results.record("HybridEnsemble save/load", False, str(e))
            
    except ImportError as e:
        results.record("HybridEnsemble import", False, str(e))


def test_multi_class(results: TestResults):
    """Test multi-class classification support."""
    print("\n[TEST] Multi-class Support...")
    
    try:
        from src.core.ensemble.hybrid_ensemble import HybridEnsemble
        
        # Create 5-class data
        X, y = create_sample_data(600, 30, n_classes=5)
        
        config = {'use_stacking': True}
        ensemble = HybridEnsemble(config)
        
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        
        ensemble.train(X_train, y_train, validation_data=(X_val, y_val))
        
        probs = ensemble.predict_proba(X_val)
        
        # Check probability shape matches n_classes
        correct_shape = probs.shape[1] == 5
        results.record("Multi-class (5 classes) probability shape", correct_shape)
        
        # Check n_classes stored
        results.record("Multi-class n_classes == 5", 
                      hasattr(ensemble, 'n_classes') and ensemble.n_classes == 5)
        
        # Check predictions are valid
        preds = ensemble.predict(X_val)
        valid_preds = all(0 <= p < 5 for p in preds)
        results.record("Multi-class predictions in valid range", valid_preds)
        
    except Exception as e:
        results.record("Multi-class support", False, str(e))


def test_drift_detection(results: TestResults):
    """Test Drift Detection components."""
    print("\n[TEST] Drift Detection...")
    
    try:
        from src.core.drift_detection.drift_detector import DriftDetector
        
        # Test 1: Initialization
        try:
            detector = DriftDetector({})
            results.record("DriftDetector initialization", True)
        except Exception as e:
            results.record("DriftDetector initialization", False, str(e))
            return
        
        # Test 2: Set reference data
        try:
            X_ref = np.random.randn(500, 20)
            y_ref = np.random.randint(0, 2, 500)
            detector.set_reference_data(X_ref, y_ref)
            results.record("set_reference_data()", True)
        except Exception as e:
            results.record("set_reference_data()", False, str(e))
            return
        
        # Test 3: Detect drift (no drift - same distribution)
        try:
            X_new = np.random.randn(200, 20)
            drift_result = detector.detect_drift(X_new)
            results.record("detect_drift() returns result", 'drift_detected' in drift_result)
        except Exception as e:
            results.record("detect_drift()", False, str(e))
        
        # Test 4: Detect drift with shifted data
        try:
            X_shifted = np.random.randn(200, 20) + 5  # Shifted mean
            drift_result = detector.detect_drift(X_shifted)
            # Should detect drift
            results.record("detect_drift() detects shifted data", 
                          drift_result.get('drift_detected', False))
        except Exception as e:
            results.record("detect_drift() with shift", False, str(e))
        
        # Test 5: Get drift statistics
        try:
            stats = detector.get_drift_statistics()
            results.record("get_drift_statistics()", stats is not None)
        except Exception as e:
            results.record("get_drift_statistics()", False, str(e))
            
    except ImportError as e:
        results.record("DriftDetector import", False, str(e))


def test_arm_robustness(results: TestResults):
    """Test ARM Robustness Monitor."""
    print("\n[TEST] ARM Robustness Monitor...")
    
    try:
        # FIXED: Use correct import path
        from src.core.robustness.robustness_monitor import AdaptiveRobustnessMonitor
        from src.core.ensemble.random_forest_model import RandomForestModel
        
        # Create and train a simple model
        X, y = create_sample_data(500, 20, n_classes=2)
        model = RandomForestModel({})
        model.train(X, y)
        
        # Use consistent sample size for baseline and evaluation
        X_test = X.values[:200]
        y_test = y.values[:200]
        
        # Test 1: ARM initialization
        try:
            arm = AdaptiveRobustnessMonitor({})
            results.record("ARM initialization", True)
        except Exception as e:
            results.record("ARM initialization", False, str(e))
            return
        
        # Test 2: Establish baseline with SAME data size as evaluation
        try:
            baseline = arm.establish_baseline(model, X_test, y_test)
            results.record("establish_baseline()", 'accuracy' in baseline)
        except Exception as e:
            results.record("establish_baseline()", False, str(e))
            return
        
        # Test 3: Comprehensive robustness evaluation with SAME data
        try:
            robustness = arm.evaluate_comprehensive_robustness(model, X_test, y_test)
            has_scores = 'aggregate_scores' in robustness
            results.record("evaluate_comprehensive_robustness()", has_scores)
            
            if has_scores:
                overall_rob = robustness['aggregate_scores'].get('overall_robustness', 0)
                results.record(f"Overall robustness score: {overall_rob:.2%}", overall_rob > 0.5)
        except Exception as e:
            results.record("evaluate_comprehensive_robustness()", False, str(e))
        
        # Test 4: Get report
        try:
            report = arm.get_robustness_report()
            has_summary = 'summary' in report
            results.record("get_robustness_report()", has_summary)
        except Exception as e:
            results.record("get_robustness_report()", False, str(e))
            
    except ImportError as e:
        results.record("ARM import", False, str(e))


def test_threat_generators(results: TestResults):
    """Test threat generators with correct method names."""
    print("\n[TEST] Threat Generators...")
    
    X = np.random.randn(100, 20)
    
    # Test NoiseInjector
    try:
        from src.core.robustness.threat_generators.noise_injector import NoiseInjector
        injector = NoiseInjector({})
        
        # Test actual methods
        X_gaussian = injector.inject_gaussian_noise(X, scale=0.1)
        results.record("NoiseInjector.inject_gaussian_noise()", X_gaussian is not None)
        
        X_uniform = injector.inject_uniform_noise(X, scale=0.1)
        results.record("NoiseInjector.inject_uniform_noise()", X_uniform is not None)
        
        X_salt = injector.inject_salt_pepper_noise(X, rate=0.05)
        results.record("NoiseInjector.inject_salt_pepper_noise()", X_salt is not None)
        
    except Exception as e:
        results.record("NoiseInjector", False, str(e))
    
    # Test FeatureMasker
    try:
        from src.core.robustness.threat_generators.feature_masker import FeatureMasker
        masker = FeatureMasker({})
        
        # Test actual methods
        X_masked = masker.mask_random_features(X, mask_rate=0.1)
        results.record("FeatureMasker.mask_random_features()", X_masked is not None)
        
        X_specific = masker.mask_specific_features(X, [0, 5, 10])
        results.record("FeatureMasker.mask_specific_features()", X_specific is not None)
        
        X_cascade = masker.cascade_failure(X, failure_rate=0.05)
        results.record("FeatureMasker.cascade_failure()", X_cascade is not None)
        
    except Exception as e:
        results.record("FeatureMasker", False, str(e))
    
    # Test BurstGenerator
    try:
        from src.core.robustness.threat_generators.burst_generator import BurstGenerator
        burst = BurstGenerator({})
        
        X_burst = burst.simulate_burst_traffic(X, intensity=1.5)
        results.record("BurstGenerator.simulate_burst_traffic()", X_burst is not None)
        
        # Test zero-clipping fix
        X_zeros = np.zeros((50, 20))
        X_burst_zeros = burst.simulate_burst_traffic(X_zeros, intensity=2.0)
        results.record("BurstGenerator handles zero values", X_burst_zeros is not None)
        
        # Test other burst patterns
        X_ddos = burst.simulate_ddos_pattern(X, attack_rate=0.3, amplification=5.0)
        results.record("BurstGenerator.simulate_ddos_pattern()", X_ddos is not None)
        
    except Exception as e:
        results.record("BurstGenerator", False, str(e))


def test_backend_interface(results: TestResults):
    """Test BackendInterface functionality."""
    print("\n[TEST] Backend Interface...")
    
    try:
        from src.streamlit_app.backend_interface import BackendInterface
        
        # Test 1: Initialization (singleton)
        try:
            backend = BackendInterface()
            results.record("BackendInterface initialization", True)
        except Exception as e:
            results.record("BackendInterface initialization", False, str(e))
            return
        
        # Test 2: Singleton pattern
        try:
            backend2 = BackendInterface()
            is_singleton = backend is backend2
            results.record("BackendInterface singleton pattern", is_singleton)
        except Exception as e:
            results.record("BackendInterface singleton", False, str(e))
        
        # Test 3: System status
        try:
            status = backend.get_system_status()
            results.record("get_system_status()", 'model_loaded' in status)
        except Exception as e:
            results.record("get_system_status()", False, str(e))
        
        # Test 4: monitor_and_adapt exists
        try:
            has_method = hasattr(backend, 'monitor_and_adapt')
            results.record("monitor_and_adapt() method exists", has_method)
        except Exception as e:
            results.record("monitor_and_adapt() exists", False, str(e))
        
        # Test 5: Key methods exist
        methods = ['train_model', 'predict', 'check_drift', 'retrain_model', 
                  'evaluate_robustness', 'train_robust_model']
        for method in methods:
            results.record(f"BackendInterface.{method} exists", hasattr(backend, method))
            
    except ImportError as e:
        results.record("BackendInterface import", False, str(e))


def test_meta_learner(results: TestResults):
    """Test MetaLearner with multi-class support."""
    print("\n[TEST] Meta Learner...")
    
    try:
        from src.core.ensemble.meta_learner import MetaLearner, StackingEnsemble
        
        # Test 1: MetaLearner initialization
        try:
            meta = MetaLearner({})
            results.record("MetaLearner initialization", True)
        except Exception as e:
            results.record("MetaLearner initialization", False, str(e))
        
        # FIXED: Test static method existence instead of initialization
        try:
            has_method = hasattr(StackingEnsemble, 'generate_stacking_data')
            results.record("StackingEnsemble.generate_stacking_data exists", has_method)
        except Exception as e:
            results.record("StackingEnsemble method check", False, str(e))
            
    except ImportError as e:
        results.record("MetaLearner import", False, str(e))


def test_monitor_and_adapt(results: TestResults):
    """Test automated drift adaptation."""
    print("\n[TEST] Monitor and Adapt (Automated Drift Adaptation)...")
    
    # Create sample data files
    X1, y1 = create_sample_data(300, 30, n_classes=2)
    X2, y2 = create_sample_data(300, 30, n_classes=2)
    
    # Save to temporary files
    temp_dir = "temp_test_data"
    os.makedirs(temp_dir, exist_ok=True)
    
    train_path = os.path.join(temp_dir, "train.csv")
    drift_path = os.path.join(temp_dir, "drift.csv")
    
    try:
        # Save training data
        train_df = X1.copy()
        train_df['label'] = y1
        train_df.to_csv(train_path, index=False)
        
        # Save drift data (shifted to induce drift)
        X2_shifted = X2 + 3  # Introduce significant drift
        drift_df = X2_shifted.copy()
        drift_df['label'] = y2
        drift_df.to_csv(drift_path, index=False)
        
        from src.streamlit_app.backend_interface import BackendInterface
        
        backend = BackendInterface()
        
        # Train initial model
        config = {'target_column': 'label', 'use_stacking': False}
        train_result = backend.train_model(train_path, config)
        
        if train_result['status'] != 'success':
            results.record("monitor_and_adapt (training failed)", False, "Initial training failed")
            return
        
        # Test 1: monitor_and_adapt with auto_retrain=False
        try:
            adapt_result = backend.monitor_and_adapt(
                drift_path, config, auto_retrain=False
            )
            
            detected = adapt_result.get('drift_detected', False)
            results.record("monitor_and_adapt() detects drift", detected)
            results.record("monitor_and_adapt() respects auto_retrain=False", 
                          not adapt_result.get('retraining_triggered', True))
        except Exception as e:
            results.record("monitor_and_adapt() with auto_retrain=False", False, str(e))
        
        # Test 2: Check status messages
        try:
            status = adapt_result.get('status')
            valid_status = status in ['drift_detected', 'stable', 'adapted', 'error']
            results.record("monitor_and_adapt() returns valid status", valid_status)
        except Exception as e:
            results.record("monitor_and_adapt() status check", False, str(e))
        
    except Exception as e:
        results.record("monitor_and_adapt", False, str(e))
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_model_performance(results: TestResults):
    """Test that models meet minimum performance thresholds."""
    print("\n[TEST] Model Performance Thresholds...")
    
    try:
        from src.core.ensemble.hybrid_ensemble import HybridEnsemble
        
        # Create well-separated data (easy classification task)
        np.random.seed(42)
        X_class0 = np.random.randn(400, 20) - 2  # Shifted left
        X_class1 = np.random.randn(400, 20) + 2  # Shifted right
        
        X = pd.DataFrame(np.vstack([X_class0, X_class1]))
        y = pd.Series([0]*400 + [1]*400)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y
        )
        
        config = {'use_stacking': True}
        ensemble = HybridEnsemble(config)
        ensemble.train(X_train, y_train, validation_data=(X_test, y_test))
        
        # Test predictions
        y_pred = ensemble.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        
        # Model should achieve >95% on this easy task
        results.record("Model achieves >95% accuracy on easy task", accuracy > 0.95)
        
        # Log actual accuracy
        print(f"    Achieved accuracy: {accuracy:.2%}")
        
    except Exception as e:
        results.record("Model performance test", False, str(e))


def test_data_cleaner(results: TestResults):
    """Test DataCleaner functionality."""
    print("\n[TEST] Data Cleaner...")
    
    try:
        from src.core.preprocessing.data_cleaner import DataCleaner
        
        # Test 1: Initialization
        try:
            cleaner = DataCleaner({})
            results.record("DataCleaner initialization", True)
        except Exception as e:
            results.record("DataCleaner initialization", False, str(e))
            return
        
        # Test 2: Clean dataset with duplicates
        try:
            X, _ = create_sample_data(200, 20)
            # Add duplicates
            X_with_dupes = pd.concat([X, X.head(50)], ignore_index=True)
            
            X_clean = cleaner.clean_dataset(X_with_dupes)
            removed_dupes = len(X_with_dupes) - len(X_clean)
            results.record("DataCleaner removes duplicates", removed_dupes >= 50)
        except Exception as e:
            results.record("DataCleaner clean_dataset", False, str(e))
        
        # Test 3: Handle NaN values
        try:
            X_with_nan = X.copy()
            X_with_nan.iloc[0:10, 0] = np.nan
            
            X_clean = cleaner.clean_dataset(X_with_nan)
            has_no_nan = not X_clean.isnull().any().any()
            results.record("DataCleaner handles NaN values", has_no_nan)
        except Exception as e:
            results.record("DataCleaner NaN handling", False, str(e))
            
    except ImportError as e:
        results.record("DataCleaner import", False, str(e))


def run_all_tests():
    """Run all tests and report results."""
    print("="*60)
    print("ENHANCED IoT BotScan - COMPREHENSIVE TEST SUITE")
    print("Author: Kotiwale Sumesh Singh (160124862043)")
    print("="*60)
    
    results = TestResults()
    
    # Run all test categories
    test_data_loader(results)
    test_data_cleaner(results)
    test_feature_engineering(results)
    test_base_models(results)
    test_hybrid_ensemble(results)
    test_multi_class(results)
    test_meta_learner(results)
    test_drift_detection(results)
    test_arm_robustness(results)
    test_threat_generators(results)
    test_backend_interface(results)
    test_monitor_and_adapt(results)
    test_model_performance(results)
    
    # Print summary
    all_passed = results.summary()
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
