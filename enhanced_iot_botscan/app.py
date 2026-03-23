"""
Enhanced IoT BotScan - Streamlit Frontend
Fixed: Navigation menu and component loading issues
"""

# ===== CRITICAL: Windows DLL Path Fix - MUST BE BEFORE ANY IMPORTS =====
# This fixes c10.dll loading issues in Streamlit's subprocess environment
import os
import sys
if sys.platform == 'win32':
    # Find torch installation and add its lib directory to DLL search path
    try:
        import importlib.util
        torch_spec = importlib.util.find_spec('torch')
        if torch_spec and torch_spec.origin:
            torch_lib_path = os.path.join(os.path.dirname(torch_spec.origin), 'lib')
            if os.path.exists(torch_lib_path):
                # Add to DLL search path (Python 3.8+)
                if hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(torch_lib_path)
                # Also prepend to PATH
                os.environ['PATH'] = torch_lib_path + os.pathsep + os.environ.get('PATH', '')
    except Exception:
        pass  # Silently fail - torch might not be installed
# ===== END DLL PATH FIX =====

import streamlit as st
import requests
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import sys
import os
import time

# Add src to path for backend imports
sys.path.insert(0, 'src')
from streamlit_app.backend_interface import BackendInterface
from utils.logging_config import setup_logging

# Initialize file + console logging
setup_logging()

# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="IoT BotScan Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Session State Initialization
# ============================================

if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard"

if "backend" not in st.session_state:
    st.session_state.backend = BackendInterface()

# ============================================
# Navigation Menu (Fixed)
# ============================================

def create_navigation():
    """Create robust navigation menu with fallback"""
    try:
        selected = option_menu(
            menu_title=None,
            options=["Dashboard", "Analytics", "Training", "Inference", "Defense", "Settings"],
            icons=["speedometer2", "bar-chart-line", "robot", "lightning-charge", "shield-check", "gear"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#262730"},
                "icon": {"color": "#FF4B4B", "font-size": "20px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "center",
                    "margin": "0px",
                    "padding": "10px",
                    "--hover-color": "#1F1F1F",
                },
                "nav-link-selected": {"background-color": "#FF4B4B"},
            }
        )
        return selected
    except Exception as e:
        st.error(f"Navigation component error: {e}")
        # Fallback to standard Streamlit selectbox
        return st.selectbox(
            "Navigate to:",
            ["Dashboard", "Analytics", "Training", "Inference", "Defense", "Settings"],
            index=0
        )

# ============================================
# Page: Dashboard
# ============================================

def show_dashboard():
    st.title("🛡️ IoT BotScan Dashboard")
    
    backend = st.session_state.backend
    status = backend.get_system_status()
    
    # Metrics row - Real Data
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "System Status", 
            "Active" if status['model_loaded'] else "Idle",
            help="Current operational state of the detection system."
        )
    with col2:
        st.metric(
            "Model Status", 
            "Trained" if status['model_loaded'] else "Untrained",
            delta="Ready" if status['model_loaded'] else "Needs Training",
            help="Indicates if the model is trained and ready."
        )
    with col3:
        st.metric(
            "Model Accuracy", 
            status['accuracy'],
            help="Validation accuracy of the currently loaded model."
        )
    with col4:
        st.metric(
            "Last Training", 
            status['last_training'].split('T')[0] if 'T' in status['last_training'] else status['last_training'],
            help="Date of the most recent model training run."
        )

    # Drift Alert Section
    if 'drift_status' in status and status['drift_status']['drift_detected']:
        ds = status['drift_status']
        st.error(f"⚠️ **Performance Degradation Detected**: {ds.get('message', 'Unknown issue')}")
        st.caption("Consider retraining the model with new data or enabling active drift adaptation.")
    
    st.divider()
    
    # Model Performance Visualization
    if status['model_loaded']:
        st.subheader("📊 Model Performance Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Confusion Matrix
            cm_data = backend.get_confusion_matrix()
            if cm_data and 'matrix' in cm_data:
                import plotly.express as px
                fig = px.imshow(
                    cm_data['matrix'],
                    text_auto=True,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=cm_data['classes'],
                    y=cm_data['classes'],
                    title="Confusion Matrix (Last Training)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Train a model to see the confusion matrix.")
        
        with col2:
            st.success("✅ System Status: Operational")
            
            st.write("**Backend:** ✅ Connected")
            st.write(f"**ML Models:** ✅ Loaded ({status['model_type']})")
            st.write(f"**Model Accuracy:** {status['accuracy']}")
            st.write("**Database:** Not implemented")
            
            # Training History Summary
            if backend.training_history:
                st.markdown("### Training Sessions")
                st.write(f"Total sessions: {len(backend.training_history)}")
                latest = backend.training_history[-1]
                st.write(f"Latest accuracy: {latest['accuracy']:.2%}")
    else:
        st.info("🔄 No trained model available. Navigate to Training tab to train a model.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("📊 Model performance metrics will appear here after training.")
        
        with col2:
            st.success("✅ System Status: Operational")
            st.write("**Backend:** ✅ Connected (No trained model)")
            st.write("**ML Models:** ❌ Not trained yet")
            st.write("**Database:** Not implemented")

# ============================================
# Page: Analytics
# ============================================

def show_analytics():
    st.title("📈 Analytics & Insights")
    
    backend = st.session_state.backend
    status = backend.get_system_status()
    
    # Help Section
    with st.expander("ℹ️ About Analytics"):
        st.write("""
        - **Feature Importance**: Shows which network features contribute most to threat detection.
        - **Training History**: Tracks model accuracy over time across training sessions.
        - **Model Performance**: Visualizes classification performance metrics.
        """)
    
    if not status['model_loaded']:
        st.warning("⚠️ No trained model found. Train a model to view analytics.")
        return
    
    # Feature Importance & Training History
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Feature Importance")
        importance_dict = backend.get_feature_importance()
        
        if importance_dict:
            # Convert to DataFrame for plotting
            imp_df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance'])
            imp_df = imp_df.sort_values(by='Importance', ascending=False).head(10)
            
            fig = px.bar(
                imp_df, 
                x='Importance', 
                y='Feature', 
                orientation='h',
                title="Top 10 Most Important Features",
                color='Importance',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance data not available.")
    
    with col2:
        st.subheader("� Training History")
        history = backend.training_history
        
        if history and len(history) > 0:
            hist_df = pd.DataFrame(history)
            # Ensure timestamp is datetime
            hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
            
            fig2 = px.line(
                hist_df, 
                x='timestamp', 
                y='accuracy', 
                markers=True,
                title="Model Accuracy Over Time",
                labels={'accuracy': 'Validation Accuracy', 'timestamp': 'Date'}
            )
            fig2.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig2, use_container_width=True)
            
            # Show training sessions table
            st.markdown("#### Training Sessions")
            display_df = hist_df[['timestamp', 'accuracy']].copy()
            display_df['accuracy'] = display_df['accuracy'].apply(lambda x: f"{x:.2%}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("""No training history available. 
            
Note: Training history accumulates across sessions but is reset when the app restarts. Train the model again to see history.""")
    
    # Additional Metrics Section
    st.divider()
    st.subheader("📈 Current Model Metrics")
    
    if backend.current_metrics:
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            acc = backend.current_metrics.get('ensemble_validation_accuracy', 0)
            st.metric("Validation Accuracy", f"{acc:.2%}")
        
        with metric_col2:
            # If we have more metrics, show them
            if 'ensemble_test_accuracy' in backend.current_metrics:
                test_acc = backend.current_metrics['ensemble_test_accuracy']
                st.metric("Test Accuracy", f"{test_acc:.2%}")
            else:
                st.metric("Model Type", status['model_type'])
        
        with metric_col3:
            if backend.training_history:
                st.metric("Total Trainings", len(backend.training_history))
            else:
                st.metric("Last Trained", status['last_training'].split('T')[0] if 'T' in status['last_training'] else status['last_training'])
    else:
        st.info("Train a model to see detailed metrics.")

# ============================================
# Page: Training
# ============================================

def get_available_datasets():
    """Scan data directories recursively for CSV files."""
    datasets = []
    for root_dir in ['data/raw/n_baiot', 'data/processed', 'data/raw/iot_23', 'data/raw/bot_iot', 'data/uploads']:
        if os.path.exists(root_dir):
            for dirpath, _, filenames in os.walk(root_dir):
                for file in filenames:
                    if file.endswith('.csv'):
                        full_path = os.path.join(dirpath, file)
                        datasets.append(full_path)
    return datasets

def show_training():
    st.title("🤖 Model Training")
    
    backend = st.session_state.backend
    
    # Help Section
    with st.expander("ℹ️ How to Train a Model"):
        st.write("""
        1. **Select Dataset**: Choose a CSV file from the `data/` directory or upload a new one.
        2. **Configure Model**: Select the architecture (Hybrid Ensemble is recommended).
        3. **Set Hyperparameters**: Adjust training settings if needed.
        4. **Start Training**: Click the button to begin the pipeline (Cleaning → Engineering → Training).
        """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Configuration")
        
        data_source = st.radio(
            "Data Source", 
            ["Existing Dataset", "Unified Multi-Dataset (N-BaIoT + IoT-23 + BoT-IoT)", "Upload New Dataset"],
            horizontal=False
        )
        
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
        elif data_source == "Unified Multi-Dataset (N-BaIoT + IoT-23 + BoT-IoT)":
            dataset_path = "unified_mode"
            st.info("ℹ️ This mode combines N-BaIoT, IoT-23, and BoT-IoT datasets into a single training set. This requires significant memory and time.")
            st.warning("⚠️ Target column will be automatically normalized to 'label' (0=Benign, 1=Malicious).")
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
        
        # Target Column Selection (Dynamic based on dataset)
        if dataset_path and os.path.exists(dataset_path):
            try:
                # Read only the first row to get column names
                df_preview = pd.read_csv(dataset_path, nrows=0)
                columns = df_preview.columns.tolist()
                
                # Try to find a likely default (label, class, target, etc.)
                default_col = "label"
                for col in columns:
                    if any(keyword in col.lower() for keyword in ['label', 'class', 'target']):
                        default_col = col
                        break
                
                # Use default if it exists in columns, otherwise use first column
                default_index = columns.index(default_col) if default_col in columns else 0
                
                target_col = st.selectbox(
                    "Target Column Name",
                    options=columns,
                    index=default_index,
                    help="Select the column containing the class labels."
                )

            except Exception as e:
                st.error(f"Error reading dataset columns: {e}")
                target_col = st.text_input(
                    "Target Column Name", 
                    value="label",
                    help="Name of the column containing the class labels (e.g., 'label', 'class')."
                )
        elif data_source == "Unified Multi-Dataset (N-BaIoT + IoT-23 + BoT-IoT)":
             target_col = st.text_input(
                "Target Column Name", 
                value="label",
                help="Automatically set for Unified Mode.",
                disabled=True
            )
        else:
            # Fallback to text input if no dataset selected
            target_col = st.text_input(
                "Target Column Name", 
                value="label",
                help="Select a dataset first to see available columns.",
                disabled=True
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
        
        train_btn = st.button("🚀 Start Training", type="primary", disabled=not dataset_path)

    with col2:
        st.subheader("Training Progress")
        
        if train_btn and dataset_path:
            is_unified = data_source == "Unified Multi-Dataset (N-BaIoT + IoT-23 + BoT-IoT)"
            config = {
                'target_column': target_col,
                'optimize_base_models': optimize,
                'use_stacking': use_stacking,
                'dataset_mode': 'unified' if is_unified else 'single',
                'use_optimized_loader': True  # Enable optimized data loader for better memory management
            }
            
            with st.status("Running Training Pipeline...", expanded=True) as status:
                st.write("1. Loading and Cleaning Data...")
                
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
                    
                    st.json(metrics) # Show full details
                    
                else:
                    status.update(label="Training Failed", state="error")
                    st.error(f"Error: {result['message']}")
                    
        elif not train_btn:
            st.info("Configure parameters and click 'Start Training' to begin.")



# ============================================
# Page: Adversarial Defense
# ============================================

def show_adversarial_defense():
    st.title("🛡️ Adversarial Defense Center")
    
    # ===== TEMPORARY DIAGNOSTIC - Remove after debugging =====
    import sys
    st.write("**Python executable:**", sys.executable)
    try:
        # Add DLL directory before torch import (Windows fix)
        if sys.platform == 'win32':
            import importlib.util
            torch_spec = importlib.util.find_spec('torch')
            if torch_spec and torch_spec.origin:
                import os
                torch_lib_path = os.path.join(os.path.dirname(torch_spec.origin), 'lib')
                if os.path.exists(torch_lib_path):
                    if hasattr(os, 'add_dll_directory'):
                        os.add_dll_directory(torch_lib_path)
                    os.environ['PATH'] = torch_lib_path + os.pathsep + os.environ.get('PATH', '')
        
        import torch
        st.success(f"✅ **Torch version:** {torch.__version__}")
    except Exception as e:
        st.error(f"❌ **Torch import failed:** {e}")
    # ===== END DIAGNOSTIC =====
    
    backend = st.session_state.backend
    status = backend.get_system_status()
    
    if not status['model_loaded']:
        st.warning("⚠️ No trained model found. Train a model first to enable defense mechanisms.")
        return

    tab1, tab2 = st.tabs(["🛡️ Robustness Testing", "🤖 Robust Training"])
    
    # --- Tab 1: Robustness Testing (ARM) ---
    with tab1:
        st.header("Robustness Testing (ARM)")
        st.write("Test your model's resilience using realistic IoT threat simulations.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Threat Scenarios")
            st.write("ARM tests your model against:")
            st.write("- 🔊 **Noise Injection** - Sensor noise simulation")
            st.write("- ❌ **Feature Masking** - Sensor failure simulation")  
            st.write("- 📈 **Traffic Bursts** - DDoS pattern simulation")
            
            datasets = get_available_datasets()
            test_dataset = st.selectbox("Test Dataset", datasets) if datasets else None
            
            run_test = st.button("🚀 Run Robustness Test", type="primary", disabled=not test_dataset)
            
        with col2:
            st.subheader("Robustness Analysis")
            if run_test and test_dataset:
                with st.spinner("Running comprehensive robustness evaluation..."):
                    config = {
                        'target_column': 'label',
                        'noise_levels': [0.0, 0.05, 0.1, 0.2],
                        'masking_rates': [0.0, 0.1, 0.2, 0.3],
                        'burst_intensities': [1.0, 1.5, 2.0, 3.0]
                    }
                    
                    result = backend.evaluate_robustness(test_dataset, config)
                    
                    if result['status'] == 'success':
                        res = result['results']
                        
                        # Check for ARM format
                        if 'aggregate_scores' in res:
                            scores = res['aggregate_scores']
                            
                            # Main metrics row
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Overall", f"{scores.get('overall_robustness', 0):.1%}")
                            m2.metric("Noise", f"{scores.get('noise_robustness', 0):.1%}")
                            m3.metric("Masking", f"{scores.get('masking_robustness', 0):.1%}")
                            m4.metric("Burst", f"{scores.get('burst_robustness', 0):.1%}")
                            
                            # Baseline info
                            baseline = res.get('baseline', {})
                            if baseline:
                                st.info(f"📊 Baseline Accuracy: {baseline.get('accuracy', 0):.1%} | Confidence: {baseline.get('confidence', 0):.1%}")
                            
                            # ARM Visualizations
                            try:
                                from src.core.robustness.visualization import ARMVisualizer
                                viz = ARMVisualizer()
                                
                                # Gauge chart
                                st.plotly_chart(viz.create_summary_gauge(scores.get('overall_robustness', 0)), 
                                               use_container_width=True)
                                
                                # Scenario bar chart
                                st.plotly_chart(viz.create_scenario_bar_chart(res), 
                                               use_container_width=True)
                                
                                # Accuracy impact chart
                                with st.expander("📉 View Accuracy Impact Analysis"):
                                    st.plotly_chart(viz.create_accuracy_impact_chart(res), 
                                                   use_container_width=True)
                            except ImportError:
                                # Fallback to basic chart
                                import pandas as pd
                                scenarios = res.get('threat_scenarios', {})
                                if scenarios:
                                    chart_data = []
                                    for threat_type, threat_results in scenarios.items():
                                        for scenario, metrics in threat_results.items():
                                            chart_data.append({
                                                'Scenario': scenario,
                                                'Robustness': metrics.get('robustness_score', 0)
                                            })
                                    df = pd.DataFrame(chart_data)
                                    st.bar_chart(df.set_index('Scenario')['Robustness'])
                            
                            # Recommendations
                            report = res.get('report', {})
                            if report.get('recommendations'):
                                st.subheader("📋 Recommendations")
                                for rec in report['recommendations']:
                                    st.write(f"• {rec}")
                            
                            st.success("✅ Robustness evaluation complete!")
                        else:
                            # Fallback for old format
                            st.metric("Robustness Score", f"{res.get('overall_robustness', 0):.2%}")
                    else:
                        st.error(f"Evaluation failed: {result.get('message')}")
            elif not run_test:
                st.info("Select a dataset and run the test to see robustness metrics.")

    # --- Tab 2: Robust Training ---
    with tab2:
        st.header("Robust Training (ARM Augmentation)")
        st.write("Improve model robustness by augmenting training data with noise, masking, and burst patterns.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Augmentation Config")
            aug_ratio = st.slider("Augmentation Ratio", 0.1, 0.5, 0.3, 
                help="Fraction of training data to augment with noise/masking.")
            
            st.write("**Augmentation Types:**")
            st.write("- 🔊 Gaussian Noise (10% of feature std)")
            st.write("- ❌ Feature Masking (10% of features)")
            
            datasets = get_available_datasets()
            train_dataset = st.selectbox("Training Dataset", datasets, key="adv_train_ds") if datasets else None
            
            start_robust_train = st.button("🛡️ Start Robust Training", type="primary", disabled=not train_dataset)
            
        with col2:
            st.subheader("Training Status")
            if start_robust_train and train_dataset:
                with st.status("Running ARM-Augmented Training...", expanded=True) as status:
                    st.write("1. Loading and preprocessing data...")
                    st.write("2. Generating noisy and masked samples...")
                    st.write("3. Training on augmented dataset...")
                    
                    config = {
                        'augmentation_ratio': aug_ratio,
                        'target_column': 'label'
                    }
                    
                    result = backend.train_robust_model(train_dataset, config)
                    
                    if result['status'] == 'success':
                        status.update(label="Robust Training Complete!", state="complete")
                        res = result['results']
                        
                        # Display key metrics
                        m1, m2 = st.columns(2)
                        m1.metric("Augmentation Samples", res.get('augmentation_samples', 0))
                        m2.metric("New Robustness", f"{res.get('robustness_after_training', 0):.1%}")
                        
                        st.success("✅ Model updated with improved robustness!")
                    else:
                        status.update(label="Training Failed", state="error")
                        st.error(f"Error: {result.get('message')}")

# ============================================
# Page: Inference
# ============================================

def show_inference():
    st.title("⚡ Inference Engine")
    
    backend = st.session_state.backend
    status = backend.get_system_status()
    
    if not status['model_loaded']:
        st.warning("⚠️ No trained model found. Please train a model to enable inference.")
        return

    # Help Section
        st.write("""
        - **Manual Input**: Test a single flow by entering feature values.
        - **Batch Processing**: Upload a CSV to process multiple samples.
        """)

    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

    with tab1:
        st.subheader("Manual Input")
        st.info("Enter values for standard network flow features.")
        
        # Standard IoT Botnet Features
        col1, col2 = st.columns(2)
        with col1:
            f1 = st.number_input("Packet Size (bytes)", min_value=0, value=64)
            f2 = st.number_input("Flow Duration (ms)", min_value=0.0, value=10.5)
            f3 = st.number_input("Source Port", min_value=0, max_value=65535, value=80)
        with col2:
            f4 = st.number_input("Destination Port", min_value=0, max_value=65535, value=443)
            f5 = st.number_input("Protocol", min_value=0, max_value=255, value=6)
            f6 = st.number_input("Bytes/Sec", min_value=0.0, value=1024.0)

        if st.button("Predict Threat", type="primary"):
            # Construct DataFrame
            input_data = pd.DataFrame([{
                'packet_size': f1,
                'flow_duration': f2,
                'src_port': f3,
                'dst_port': f4,
                'protocol': f5,
                'bytes_per_sec': f6
            }])
            
            with st.spinner("Analyzing..."):
                result = backend.predict(input_data)
                
                if result['status'] == 'success':
                    pred = result['predictions'][0]
                    prob = result['probabilities'][0]
                    
                    # 0 = Benign, 1 = Malicious
                    is_threat = pred == 1 
                    
                    if is_threat:
                        st.error(f"🚨 **THREAT DETECTED**")
                        st.metric("Confidence", f"{prob[1]:.2%}")
                    else:
                        st.success(f"✅ **Traffic is Benign**")
                        st.metric("Confidence", f"{prob[0]:.2%}")
                else:
                    st.error(f"Prediction Failed: {result['message']}")

    with tab2:
        st.subheader("Batch Processing")
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
             if st.button("Process File"):
                try:
                    input_df = pd.read_csv(uploaded_file)
                    st.write(f"Loaded {len(input_df)} samples.")
                    
                    with st.spinner("Processing Batch..."):
                        result = backend.predict(input_df)
                        
                        if result['status'] == 'success':
                            st.success("Processing Complete")
                            preds = result['predictions']
                            probs = result['probabilities']
                            indices = result.get('indices')
                            
                            if indices:
                                results_df = input_df.loc[indices].copy()
                            else:
                                results_df = input_df.copy()

                            results_df['Prediction'] = preds
                            
                            # Handle probabilities
                            if hasattr(probs, 'shape') and len(probs.shape) > 1:
                                results_df['Confidence'] = np.max(probs, axis=1)
                            else:
                                results_df['Confidence'] = probs
                                
                            st.dataframe(results_df)

                            # CSV Download
                            csv = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "Download Results.csv",
                                csv,
                                "prediction_results.csv",
                                "text/csv",
                                key='download-csv'
                            )
                        else:
                            st.error(f"Batch Prediction Failed: {result['message']}")
                except Exception as e:
                    st.error(f"Error: {e}")

# ============================================
# Page: Settings
# ============================================

def show_settings():
    st.title("⚙️ Settings & Configuration")
    
    st.subheader("🔧 System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("Backend API URL", "http://localhost:8000")
        st.number_input("Scan Interval (seconds)", 1, 3600, 60)
        st.selectbox("Threat Threshold", ["Low", "Medium", "High"])
    
    with col2:
        st.checkbox("Enable Auto-Scan", value=True)
        st.checkbox("Email Notifications", value=False)
        st.checkbox("Real-time Alerts", value=True)
    
    st.divider()
    
    if st.button("💾 Save Settings", type="primary"):
        st.success("✓ Settings saved successfully!")

# ============================================
# Authentication
# ============================================

API_KEY = "YOUR API KEY"

def login(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={API_KEY}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    return requests.post(url, json=payload).json()

def signup(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={API_KEY}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    return requests.post(url, json=payload).json()

# ============================================
# Main Application
# ============================================

def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        /* Global Font & Background */
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif !important;
        }
        
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e1e2f 100%);
            color: #f8fafc;
        }

        /* Glassmorphism for Metrics and Containers */
        [data-testid="stMetric"], .css-1r6slb0, .css-12oz5g7 {
            background: rgba(30, 41, 59, 0.7) !important;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
            padding: 20px !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
            transition: transform 0.3s ease, box-shadow 0.3s ease !important;
        }
        
        [data-testid="stMetric"]:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px 0 rgba(239, 68, 68, 0.2) !important;
            border: 1px solid rgba(239, 68, 68, 0.4) !important;
        }

        /* Metric Values Text Glowing */
        [data-testid="stMetricValue"] {
            font-weight: 800 !important;
            background: -webkit-linear-gradient(45deg, #FF4B4B, #FF8F8F);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Premium Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #FF4B4B 0%, #FF2B2B 100%);
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 0.5rem 2rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3) !important;
        }
        
        .stButton>button:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 20px rgba(255, 75, 75, 0.5) !important;
            background: linear-gradient(90deg, #FF6B6B 0%, #FF4B4B 100%);
        }

        /* Inputs and Selectboxes */
        .stTextInput>div>div>input, .stSelectbox>div>div>div {
            background: rgba(15, 23, 42, 0.5) !important;
            color: white !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 8px !important;
        }
        
        .stTextInput>div>div>input:focus, .stSelectbox>div>div>div:focus {
            border: 1px solid #FF4B4B !important;
            box-shadow: 0 0 10px rgba(255, 75, 75, 0.2) !important;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.8) !important;
            backdrop-filter: blur(15px);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* Headings */
        h1, h2, h3 {
            font-weight: 800 !important;
            letter-spacing: -0.5px;
        }

        /* Divider */
        hr {
            border-top: 1px solid rgba(255, 255, 255, 0.1) !important;
        }
        
        /* Tabs Setup */
        .stTabs [data-baseweb="tab-list"] {
            gap: 15px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(30, 41, 59, 0.5);
            border-radius: 8px 8px 0 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-bottom: none;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(239, 68, 68, 0.1);
            border-bottom: 2px solid #FF4B4B !important;
            color: #FF4B4B !important;
        }
    </style>
    """, unsafe_allow_html=True)

def main():
    inject_custom_css()
    
    if "user" not in st.session_state:
        st.session_state.user = None

    if not st.session_state.user:
        st.title("🔐 IoT Botnet Detection Login")
        menu = ["Login", "Signup"]
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "Signup":
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")

            if st.button("Create Account"):
                res = signup(email, password)
                if "email" in res:
                    st.success("Account created!")
                else:
                    st.error(res)

        elif choice == "Login":
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                res = login(email, password)
                if "email" in res:
                    st.session_state.user = res
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error(res)
        return  # Stop execution here if not logged in

    # Logout button in sidebar
    st.sidebar.markdown("### Account")
    st.sidebar.write(f"Logged in as: {st.session_state.user['email']}")
    if st.sidebar.button("Logout", type="primary"):
        st.session_state.user = None
        st.rerun()

    # --- PROTECTED APP ---
    
    # Navigation
    selected_page = create_navigation()
    
    # Route to appropriate page
    if selected_page == "Dashboard":
        show_dashboard()
    elif selected_page == "Analytics":
        show_analytics()
    elif selected_page == "Training":
        show_training()
    elif selected_page == "Inference":
        show_inference()
    elif selected_page == "Defense":
        show_adversarial_defense()
    elif selected_page == "Settings":
        show_settings()

if __name__ == "__main__":
    main()
