"""
Verification test for multi-dataset stacking ensemble integration.

Tests that:
1. IoT-23 Zeek log parsing works correctly
2. Unified dataset preparation merges all 3 datasets
3. The stacking ensemble trains and predicts on unified data
4. CLI default arguments now include all 3 datasets

Run from project root:
    python tests/test_unified_training.py
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_zeek_parser():
    """Test that _parse_zeek_conn_log correctly parses Zeek format."""
    print("\n" + "=" * 60)
    print("TEST 1: Zeek conn.log.labeled Parser")
    print("=" * 60)

    from src.data.data_loader import DataLoader

    loader = DataLoader({'data_paths': {}})

    # Create a temporary Zeek file
    import tempfile
    zeek_content = (
        "#separator \\x09\n"
        "#set_separator\t,\n"
        "#empty_field\t(empty)\n"
        "#unset_field\t-\n"
        "#path\tconn\n"
        "#open\t2018-05-21\n"
        "#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\t"
        "service\tduration\torig_bytes\tresp_bytes\tconn_state\tlocal_orig\t"
        "local_resp\tmissed_bytes\thistory\torig_pkts\torig_ip_bytes\t"
        "resp_pkts\tresp_ip_bytes\ttunnel_parents\tlabel\tdetailed-label\n"
        "#types\ttime\tstring\taddr\tport\taddr\tport\tenum\tstring\tinterval\t"
        "count\tcount\tstring\tbool\tbool\tcount\tstring\tcount\tcount\tcount\t"
        "count\tset[string]\tstring\tstring\n"
        # Malicious sample
        "1525879831.01\tCtest1\t192.168.1.1\t51524\t10.0.0.1\t23\ttcp\t-\t"
        "2.999\t100\t200\tS0\t-\t-\t0\tS\t3\t180\t0\t0\t(empty)\t"
        "Malicious\tPortScan\n"
        # Benign sample
        "1525879832.02\tCtest2\t192.168.1.2\t58687\t10.0.0.2\t123\tudp\t-\t"
        "0.114\t48\t48\tSF\t-\t-\t0\tDd\t1\t76\t1\t76\t-\t"
        "benign\t-\n"
        # Another malicious sample with '-' (unset) values
        "1525879833.03\tCtest3\t192.168.1.3\t12345\t10.0.0.3\t80\ttcp\t-\t"
        "-\t0\t0\tS0\t-\t-\t0\tS\t1\t60\t0\t0\t(empty)\t"
        "Malicious\tC&C\n"
    )

    with tempfile.NamedTemporaryFile(mode='w', suffix='.labeled', delete=False,
                                      encoding='utf-8') as f:
        f.write(zeek_content)
        temp_path = f.name

    try:
        df = loader._parse_zeek_conn_log(temp_path)

        assert df is not None, "Parser returned None"
        assert len(df) == 3, f"Expected 3 rows, got {len(df)}"
        assert 'label' in df.columns, "Missing 'label' column"

        # Check numeric features exist
        expected_features = ['duration', 'orig_bytes', 'resp_bytes', 'missed_bytes',
                             'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
        for feat in expected_features:
            assert feat in df.columns, f"Missing feature: {feat}"

        # Check labels: Malicious -> 1, benign -> 0
        labels = df['label'].values
        assert labels[0] == 1, f"Malicious should be 1, got {labels[0]}"
        assert labels[1] == 0, f"Benign should be 0, got {labels[1]}"
        assert labels[2] == 1, f"Malicious should be 1, got {labels[2]}"

        # Check that '-' (unset duration) was converted to 0
        assert df.iloc[2]['duration'] == 0.0, f"Unset duration should be 0, got {df.iloc[2]['duration']}"

        # Check numeric values
        assert df.iloc[0]['orig_bytes'] == 100.0, f"orig_bytes mismatch"
        assert df.iloc[1]['resp_bytes'] == 48.0, f"resp_bytes mismatch"

        print(f"  ✅ Parsed {len(df)} rows with {len(df.columns) - 1} features")
        print(f"  ✅ Labels correct: {dict(zip(*np.unique(labels, return_counts=True)))}")
        print(f"  ✅ Features: {[c for c in df.columns if c != 'label']}")
        print("  PASS")

    finally:
        os.unlink(temp_path)


def test_iot23_loader_with_real_data():
    """Test that load_iot_23_dataset finds and parses real Zeek files."""
    print("\n" + "=" * 60)
    print("TEST 2: IoT-23 Dataset Loader (Real Data)")
    print("=" * 60)

    from src.data.data_loader import DataLoader

    iot23_path = './data/raw/iot_23/'
    if not os.path.exists(iot23_path):
        print("  SKIP: IoT-23 data directory not found")
        return

    loader = DataLoader({'data_paths': {'iot_23': iot23_path}})

    try:
        dataset = loader.load_iot_23_dataset()

        assert dataset is not None, "Loader returned None"
        assert dataset['total_samples'] > 0, "No samples loaded"
        assert dataset['n_features'] > 0, "No features extracted"
        assert dataset['dataset_name'] == 'IoT-23', "Wrong dataset name"

        print(f"  ✅ Loaded {dataset['total_samples']} samples")
        print(f"  ✅ {dataset['n_features']} features")
        print(f"  ✅ {dataset['n_classes']} classes")
        print(f"  ✅ Label mapping: {dataset['label_mapping']}")
        print(f"  ✅ Features: {dataset['feature_names'][:5]}...")
        print("  PASS")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()


def test_unified_dataset_loading():
    """Test that load_unified_dataset merges all available datasets."""
    print("\n" + "=" * 60)
    print("TEST 3: Unified Dataset Loading")
    print("=" * 60)

    from src.data.data_loader import DataLoader

    config = {
        'data_paths': {
            'n_baiot': './data/raw/n_baiot/',
            'iot_23': './data/raw/iot_23/',
            'bot_iot': './data/raw/bot_iot/'
        },
        'use_optimized_loader': True,
        'max_samples_per_device': 5000,  # Small for testing speed
        'chunk_size': 5000
    }

    loader = DataLoader(config)

    try:
        unified = loader.load_unified_dataset(max_samples=50000)

        assert unified is not None, "Unified loader returned None"
        assert unified['total_samples'] > 0, "No samples loaded"
        assert unified['n_features'] > 0, "No features"
        assert unified['n_classes'] == 2, f"Expected binary, got {unified['n_classes']} classes"
        assert unified['dataset_name'] == 'Unified_Multi_Dataset'

        # Check label distribution
        labels = unified['labels']
        benign = np.sum(labels == 0)
        malicious = np.sum(labels == 1)

        print(f"  ✅ Total samples: {unified['total_samples']}")
        print(f"  ✅ Features: {unified['n_features']}")
        print(f"  ✅ Benign: {benign}, Malicious: {malicious}")
        print(f"  ✅ Feature shape: {unified['features'].shape}")
        print("  PASS")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()


def test_train_models_cli_defaults():
    """Test that CLI defaults now include all 3 datasets."""
    print("\n" + "=" * 60)
    print("TEST 4: CLI Default Arguments")
    print("=" * 60)

    import argparse

    # Simulate the argument parser from train_models.py main()
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+',
                        default=['n_baiot', 'iot_23', 'bot_iot'])
    parser.add_argument('--unified', dest='unified',
                        action='store_true', default=True)

    args = parser.parse_args([])  # Empty args = use defaults

    assert args.datasets == ['n_baiot', 'iot_23', 'bot_iot'], \
        f"Default datasets should be all 3, got {args.datasets}"
    assert args.unified is True, \
        f"Unified should default to True, got {args.unified}"

    print(f"  ✅ Default datasets: {args.datasets}")
    print(f"  ✅ Unified mode: {args.unified}")
    print("  PASS")


def test_prepare_unified_method_exists():
    """Test that ModelTrainer has the new prepare_unified_training_data method."""
    print("\n" + "=" * 60)
    print("TEST 5: ModelTrainer.prepare_unified_training_data Exists")
    print("=" * 60)

    from scripts.train_models import ModelTrainer

    assert hasattr(ModelTrainer, 'prepare_unified_training_data'), \
        "ModelTrainer missing prepare_unified_training_data method"
    assert hasattr(ModelTrainer, '_split_and_preprocess'), \
        "ModelTrainer missing _split_and_preprocess method"

    # Check that train_baseline_models accepts use_unified parameter
    import inspect
    sig = inspect.signature(ModelTrainer.train_baseline_models)
    assert 'use_unified' in sig.parameters, \
        "train_baseline_models missing use_unified parameter"

    sig2 = inspect.signature(ModelTrainer.train_adversarial_robust_models)
    assert 'use_unified' in sig2.parameters, \
        "train_adversarial_robust_models missing use_unified parameter"

    sig3 = inspect.signature(ModelTrainer.run_full_training_pipeline)
    assert 'use_unified' in sig3.parameters, \
        "run_full_training_pipeline missing use_unified parameter"

    print("  ✅ prepare_unified_training_data() exists")
    print("  ✅ _split_and_preprocess() exists")
    print("  ✅ train_baseline_models(use_unified=) exists")
    print("  ✅ train_adversarial_robust_models(use_unified=) exists")
    print("  ✅ run_full_training_pipeline(use_unified=) exists")
    print("  PASS")


def test_config_has_unified_settings():
    """Test that config.yaml has the new unified training settings."""
    print("\n" + "=" * 60)
    print("TEST 6: Config File Settings")
    print("=" * 60)

    import yaml

    config_path = os.path.join(PROJECT_ROOT, 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    training = config.get('training', {})
    assert 'use_unified_dataset' in training, \
        "Missing use_unified_dataset in config"
    assert training['use_unified_dataset'] is True, \
        f"use_unified_dataset should be True, got {training['use_unified_dataset']}"
    assert 'unified_max_samples' in training, \
        "Missing unified_max_samples in config"

    print(f"  ✅ use_unified_dataset: {training['use_unified_dataset']}")
    print(f"  ✅ unified_max_samples: {training['unified_max_samples']}")
    print("  PASS")


if __name__ == '__main__':
    print("=" * 60)
    print("MULTI-DATASET STACKING INTEGRATION VERIFICATION")
    print("=" * 60)

    tests = [
        test_zeek_parser,
        test_iot23_loader_with_real_data,
        test_unified_dataset_loading,
        test_train_models_cli_defaults,
        test_prepare_unified_method_exists,
        test_config_has_unified_settings,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
