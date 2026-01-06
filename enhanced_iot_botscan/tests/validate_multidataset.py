"""
Multi-Dataset Validation Script
Validates REQ-013 (N-BaIoT, IoT-23, BoT-IoT support)
"""

import sys
import os
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.config_manager import ConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_dataset_path(name: str, path_str: str) -> bool:
    """Validate that a dataset path exists and contains data."""
    path = Path(path_str)
    if not path.exists():
        logger.error(f"❌ {name}: Path does not exist -> {path}")
        return False
    
    if path.is_file():
        logger.info(f"✅ {name}: Found file -> {path}")
        return True
        
    if path.is_dir():
        # Check for CSV files
        files = list(path.glob("*.csv"))
        if files:
            logger.info(f"✅ {name}: Found directory with {len(files)} CSV files -> {path}")
            return True
        else:
            logger.warning(f"⚠️ {name}: Directory exists but contains no CSV files -> {path}")
            return False
            
    return False

def validate_datasets():
    """Validate all configured datasets."""
    print("="*50)
    print("MULTI-DATASET COMPLIANCE VALIDATION")
    print("="*50)
    
    config = ConfigManager()
    data_paths = config.get('data.data_paths', {})
    
    if not data_paths:
        logger.error("❌ No data paths configured in config.yaml")
        return
    
    results = {}
    
    # Required datasets per REQ-013
    required_datasets = ['n_baiot', 'iot_23', 'bot_iot']
    
    for dataset in required_datasets:
        path = data_paths.get(dataset)
        if path:
            is_valid = validate_dataset_path(dataset, path)
            results[dataset] = "PASS" if is_valid else "FAIL - Invalid Path"
        else:
            logger.error(f"❌ {dataset}: Not configured in config.yaml")
            results[dataset] = "FAIL - Not Configured"
            
    # Summary
    print("\nValidation Summary:")
    all_passed = True
    for name, status in results.items():
        print(f"{name.upper().ljust(10)}: {status}")
        if "FAIL" in status:
            all_passed = False
            
    if all_passed:
        print("\n✅ REQ-013 (Multi-Dataset Support): PASS (All paths configured and valid)")
    else:
        print("\n⚠️ REQ-013 (Multi-Dataset Support): PARTIAL/FAIL (See details above)")

if __name__ == "__main__":
    validate_datasets()
