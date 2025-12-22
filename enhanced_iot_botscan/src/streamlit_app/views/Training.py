import streamlit as st
import os
import time
from utils import render_header
from src.streamlit_app.backend_interface import BackendInterface

def get_available_datasets():
    """Scan data directories recursively for CSV files."""
    datasets = []
    for root_dir in ['data/raw/n_baiot', 'data/processed', 'data/raw/iot_23', 'data/raw/bot_iot']:
        if os.path.exists(root_dir):
            for dirpath, _, filenames in os.walk(root_dir):
                for file in filenames:
                    if file.endswith('.csv'):
                        # Store relative path for display, or absolute if needed
                        # Using relative path from project root
                        full_path = os.path.join(dirpath, file)
                        datasets.append(full_path)
    return datasets

def app():
    render_header("Model Training", "Train and evaluate new models")
    
    backend = BackendInterface()
    
    # Help Section
    with st.expander("ℹ️ How to Train a Model"):
        st.write("""
        1. **Select Dataset**: Choose a CSV file from the `data/` directory.
        2. **Configure Model**: Select the architecture (Hybrid Ensemble is recommended).
        3. **Set Hyperparameters**: Adjust training settings if needed.
        4. **Start Training**: Click the button to begin the pipeline (Cleaning -> Engineering -> Training).
        """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Configuration")
        
        data_source = st.radio("Data Source", ["Existing Dataset", "Upload New Dataset"], horizontal=True)
        
        dataset_path = None
        
        if data_source == "Existing Dataset":
            datasets = get_available_datasets()
            if not datasets:
                st.warning("No CSV datasets found in `data/raw` or `data/processed`.")
            else:
                dataset_path = st.selectbox(
                    "Select Dataset",
                    datasets,
                    help="Path to the CSV file containing training data."
                )
        else:
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded_file is not None:
                # Save the file
                upload_dir = os.path.join("data", "uploads")
                os.makedirs(upload_dir, exist_ok=True)
                file_path = os.path.join(upload_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                dataset_path = file_path
                st.success(f"File saved: {uploaded_file.name}")
        
        target_col = st.text_input(
            "Target Column Name", 
            value="label",
            help="Name of the column containing the class labels (e.g., 'label', 'class')."
        )
        
        model_type = st.selectbox(
            "Model Architecture",
            ["Hybrid Ensemble (Recommended)", "Random Forest", "XGBoost", "LightGBM"],
            help="The machine learning algorithm to use. Hybrid Ensemble combines multiple models for better performance."
        )
        
        st.markdown("#### Hyperparameters")
        
        optimize = st.checkbox(
            "Optimize Hyperparameters", 
            value=False,
            help="If checked, the system will search for optimal parameters (takes longer)."
        )
        
        use_stacking = st.checkbox(
            "Use Stacking", 
            value=True,
            help="Enable stacking meta-learner for Hybrid Ensemble."
        )
        
        train_btn = st.button("Start Training", type="primary", disabled=not dataset_path)

    with col2:
        st.subheader("Training Progress")
        
        if train_btn and dataset_path:
            config = {
                'target_column': target_col,
                'optimize_base_models': optimize,
                'use_stacking': use_stacking,
                # Add other config params as needed
            }
            
            with st.status("Running Training Pipeline...", expanded=True) as status:
                st.write("1. Loading and Cleaning Data...")
                # We can't easily stream progress from the backend yet without callbacks, 
                # so we'll show indeterminate progress for each step.
                
                start_time = time.time()
                result = backend.train_model(dataset_path, config)
                
                if result['status'] == 'success':
                    st.write("✅ Data Cleaned & Features Engineered")
                    st.write("✅ Model Trained Successfully")
                    status.update(label="Training Complete!", state="complete", expanded=False)
                    
                    # Show Results
                    metrics = result['results']
                    st.balloons()
                    
                    st.success(f"Training completed in {time.time() - start_time:.2f} seconds")
                    
                    st.markdown("### Evaluation Metrics")
                    res_col1, res_col2 = st.columns(2)
                    
                    acc = metrics.get('ensemble_validation_accuracy', 0.0)
                    res_col1.metric("Validation Accuracy", f"{acc:.2%}")
                    
                    # If we had other metrics in the result, we'd show them here
                    
                    st.json(metrics) # Show full details
                    
                else:
                    status.update(label="Training Failed", state="error")
                    st.error(f"Error: {result['message']}")
                    
        elif not train_btn:
            st.info("Configure parameters and click 'Start Training' to begin.")
            st.image("https://img.icons8.com/clouds/200/000000/development.png", width=200)
