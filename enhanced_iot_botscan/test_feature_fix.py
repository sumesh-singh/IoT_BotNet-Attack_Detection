from src.streamlit_app.backend_interface import BackendInterface
import pandas as pd
import numpy as np
import logging

# Configure logging to see the output
logging.basicConfig(level=logging.INFO)

print("\n--- Testing Feature Engineer State Persistence ---")

backend = BackendInterface()

# Check if model is loaded
if backend.model and backend.model.is_trained:
    print("✓ Model loaded successfully")
    
    # Check if feature engineer state exists in the model
    # (Checking both public and private attributes as per our fix)
    fe_state = getattr(backend.model, 'feature_engineer_state', None)
    if fe_state is None:
        fe_state = getattr(backend.model, '_feature_engineer_state', None)
        
    if fe_state:
        # Check if selected features are present
        selected_features = fe_state.get('selected_features')
        if selected_features:
            print(f"✓ Feature engineer state found in model")
            print(f"✓ Number of selected features: {len(selected_features)}")
            print("✓ SUCCESS: State is persisted correctly!")
        else:
            print("❌ Feature engineer state found but 'selected_features' is empty!")
            print("⚠️ ACTION REQUIRED: You must re-train the model to save the state.")
    else:
        print("❌ No feature engineer state found in the loaded model.")
        print("⚠️ ACTION REQUIRED: You must re-train the model to save the state.")
        
else:
    print("❌ Model not loaded or not trained - Train model first!")

print("\n--------------------------------------------------")
