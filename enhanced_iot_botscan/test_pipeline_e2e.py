
import pandas as pd
import numpy as np
import logging
import shutil
import os
from src.core.preprocessing.data_cleaner import DataCleaner
from src.core.preprocessing.feature_engineer import FeatureEngineer
from src.core.ensemble.hybrid_ensemble import HybridEnsemble
from src.streamlit_app.backend_interface import BackendInterface

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def test_pipeline():
    logger.info("STARTING PIPELINE TEST")
    
    # 1. Generate Dummy Training Data (115 features -> to simulate N-BaIoT)
    logger.info("Generating dummy training data...")
    n_samples = 200
    n_features = 115
    X = pd.DataFrame(np.random.randn(n_samples, n_features), columns=[f'feat_{i}' for i in range(n_features)])
    # Make some features predictive
    y = (X['feat_0'] + X['feat_1'] > 0).astype(int)
    
    # 2. Preprocessing & Engineering
    logger.info("Running Feature Engineering...")
    config = {'n_features_select': 10} # reduced for speed test
    
    cleaner = DataCleaner(config)
    X_clean = cleaner.clean_dataset(X)
    
    engineer = FeatureEngineer(config)
    X_eng = engineer.engineer_features(X_clean, y)
    logger.info(f"Engineered features: {X_eng.shape[1]}")
    
    assert X_eng.shape[1] == 10, f"Expected 10 features, got {X_eng.shape[1]}"
    
    # 3. Model Training
    logger.info("Training HybridEnsemble...")
    model = HybridEnsemble(config)
    
    # Mock label encoder (backend does this)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y), index=y.index)
    model.label_encoder = le
    
    model.train(X_eng, y_enc)
    
    # 4. Save Model with State (The Critical Step)
    logger.info("Saving Model with State...")
    os.makedirs("models_test", exist_ok=True)
    model_path = "models_test/test_model.joblib"
    
    fe_state = engineer.get_state()
    model.save_model(model_path, feature_engineer_state=fe_state)
    
    # 5. Check In-Memory State Update
    logger.info("Checking in-memory state update...")
    if hasattr(model, 'feature_engineer_state') and model.feature_engineer_state:
         logger.info("✓ In-memory state updated successfully")
    else:
         logger.error("❌ In-memory state NOT updated!")
         return
         
    # 6. Mock Robustness Evaluation (Simulating Backend Logic)
    logger.info("Simulating Robustness Evaluation on New Data...")
    
    # New data with original 115 features
    X_test_raw = pd.DataFrame(np.random.randn(50, n_features), columns=[f'feat_{i}' for i in range(n_features)])
    y_test = pd.Series(np.random.randint(0, 2, 50))
    
    # Instantiate new engineer
    logger.info("Restoring feature engineer from model...")
    new_engineer = FeatureEngineer(config)
    
    # Get state from model (simulating backend logic)
    # Check both keys
    restored_state = getattr(model, 'feature_engineer_state', None)
    if not restored_state:
        restored_state = getattr(model, '_feature_engineer_state', None)
        
    if restored_state:
        new_engineer.restore_state(restored_state)
        logger.info(f"✓ Restored state. Selected: {len(new_engineer.selected_features)}")
    else:
        logger.error("❌ Failed to retrieve state from model!")
        return

    # Transform
    X_test_eng = new_engineer.transform_new_data(X_test_raw)
    
    # Verify dimensions
    logger.info(f"Test Transformed Shape: {X_test_eng.shape}")
    assert X_test_eng.shape[1] == 10, f"Expected 10 features, got {X_test_eng.shape[1]}"
    
    # Verify features match
    assert list(X_test_eng.columns) == list(X_eng.columns), "Feature columns do not match!"
    logger.info("✓ Feature columns match exactly")
    
    # Clean up
    shutil.rmtree("models_test")
    logger.info("\nPIPELINE TEST SUCCESSFUL! 🎉")

if __name__ == "__main__":
    test_pipeline()
