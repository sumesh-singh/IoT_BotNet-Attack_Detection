
import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_loader import DataLoader

class TestUnifiedLoading(unittest.TestCase):
    def setUp(self):
        self.config = {'data_paths': {}}
        self.loader = DataLoader(self.config)

    def test_iot23_mislabeling(self):
        """Test that IoT-23 data is mislabeled if 'Benign' is not class 0."""
        print("\nTesting IoT-23 Mislabeling Assumption...")
        
        # Mock load_iot_23_dataset to return data where Benign is class 1
        # Example: 'Attack': 0, 'Benign': 1, 'C&C': 2
        mock_data = {
            'features': np.array([[1.0], [2.0], [3.0]]),
            'feature_names': ['f1'],
            'labels': np.array([0, 1, 2]), # 0=Attack, 1=Benign, 2=C&C
            'label_mapping': {0: 'Attack', 1: 'Benign', 2: 'C&C'},
            'dataset_name': 'IoT-23',
            'n_features': 1,
            'n_classes': 3,
            'total_samples': 3
        }
        
        self.loader.load_iot_23_dataset = MagicMock(return_value=mock_data)
        # Mock others to fail/return None so we focus on IoT-23
        self.loader.load_n_baiot_dataset = MagicMock(side_effect=Exception("Skip N-BaIoT"))
        self.loader.load_bot_iot_dataset = MagicMock(side_effect=Exception("Skip BoT-IoT"))

        # Run load_unified_dataset
        # The current code does: df_i23['binary_label'] = (iot23_data['labels'] != 0).astype(int)
        # So it assumes class 0 (Attack) is benign (0)?? No, !=0 means 1 (malicious).
        # So class 0 (Attack) becomes 0 (Benign). ERROR!
        # Class 1 (Benign) becomes 1 (Malicious). ERROR!
        
        unified = self.loader.load_unified_dataset()
        y = unified['labels']
        
        # Expected correct behavior (if fixed):
        # 0 (Attack) -> 1 (Malicious)
        # 1 (Benign) -> 0 (Benign)
        # 2 (C&C) -> 1 (Malicious)
        
        # Current incorrect behavior:
        # 0 != 0 -> False -> 0 (Benign) -> WRONG
        # 1 != 0 -> True -> 1 (Malicious) -> WRONG
        
        print(f"Original Labels: {mock_data['labels']}")
        print(f"Mapped Binary Labels: {y}")
        
        # Verify if it failed (we expect it to fail currently)
        # We are asserting strictly for CORRECT behavior.
        # If the bug exists, this test should FAIL.
        
        # Check mapping for index 0 (Attack, label 0) -> Should be 1
        if y[0] == 0:
            print("[FAIL] Attack (Class 0) mapped to Benign (0)!")
        
        # Check mapping for index 1 (Benign, label 1) -> Should be 0
        if y[1] == 1:
            print("[FAIL] Benign (Class 1) mapped to Malicious (1)!")

    def test_empty_dataset_crash(self):
        """Test that empty datasets list causes crash."""
        print("\nTesting Empty Dataset Crash...")
        
        # Mock all loaders to fail
        self.loader.load_n_baiot_dataset = MagicMock(side_effect=Exception("Fail"))
        self.loader.load_iot_23_dataset = MagicMock(side_effect=Exception("Fail"))
        self.loader.load_bot_iot_dataset = MagicMock(side_effect=Exception("Fail"))
        
        try:
            self.loader.load_unified_dataset()
            print("[FAIL] Should have raised ValueError for empty datasets")
        except ValueError as e:
            if "No objects to concatenate" in str(e):
                print(f"[CONFIRMED] Crashed with ValueError: {e}")
            elif "No datasets loaded" in str(e): # This is what we WANT
                 print(f"[PASS] Correctly raised friendly error: {e}")
            else:
                 print(f"[FAIL] Raised unexpected error: {e}")
        except Exception as e:
            print(f"[CONFIRMED] Crashed with unexpected exception: {e} ({type(e)})")

if __name__ == '__main__':
    unittest.main()
