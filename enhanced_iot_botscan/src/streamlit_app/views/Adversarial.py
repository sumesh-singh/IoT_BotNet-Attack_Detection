import streamlit as st
import pandas as pd
import plotly.express as px
import os
from utils import render_header
from src.streamlit_app.backend_interface import BackendInterface

def get_available_datasets():
    """Scan data directories recursively for CSV files."""
    datasets = []
    for root_dir in ['data/raw', 'data/processed', 'data/uploads']:
        if os.path.exists(root_dir):
            for dirpath, _, filenames in os.walk(root_dir):
                for file in filenames:
                    if file.endswith('.csv'):
                        full_path = os.path.join(dirpath, file)
                        datasets.append(full_path)
    return datasets

def app():
    render_header("Adversarial & Drift", "Test robustness and monitor concept drift")
    
    backend = BackendInterface()
    
    # Get datasets once for all tabs
    datasets = get_available_datasets()
    if not datasets:
        st.warning("No CSV datasets found. Please upload a dataset in the 'Training' page.")
    
    tab1, tab2, tab3 = st.tabs(["🛡️ Adversarial Attacks", "🤖 Adversarial Training", "📉 Drift Analysis"])
    
    with tab1:
        st.subheader("Adversarial Attack Generation")
        
        if not backend.model or not backend.model.is_trained:
            st.error("Model must be trained first. Please go to the 'Training' page.")
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                dataset_path = st.selectbox("Select Test Dataset", datasets, key="adv_dataset") if datasets else None
                
                attack_type = st.selectbox("Attack Method", ["FGSM", "PGD", "C&W"])
                
                # Attack params
                epsilon = st.slider("Epsilon (Perturbation Magnitude)", 0.01, 0.5, 0.1)
                
                if st.button("Generate Attack"):
                    with st.spinner("Generating adversarial examples..."):
                        # We'll run evaluation which includes generation
                        config = {
                            'target_column': 'label', # assume standard
                            'attack_config': {
                                'fgsm': {'epsilon': epsilon},
                                'pgd': {'epsilon': epsilon},
                                # Add others
                            },
                             # Only enable selected attack for this run
                            'enabled_attacks': [attack_type.lower()] 
                        }
                        # We utilize evaluate_robustness to generate and test
                        result = backend.evaluate_robustness(dataset_path, config)
                        
                        if result['status'] == 'success':
                            st.session_state['adv_results'] = result['results']
                            st.success("Attack generation complete!")
                        else:
                            st.error(f"Attack failed: {result['message']}")

            with col2:
                if 'adv_results' in st.session_state:
                    res = st.session_state['adv_results']
                    st.metric("Robustness Score", f"{res['overall_robustness']:.2%}")
                    
                    # Show per-attack details
                    metrics = res.get('attack_metrics', {})
                    if metrics:
                         # Convert to DF for chart
                         data = []
                         for atk, m in metrics.items():
                             data.append({'Attack': atk, 'Success Rate': m['success_rate'], 'Accuracy Drop': m['accuracy_drop']})
                         
                         df_res = pd.DataFrame(data)
                         st.dataframe(df_res)
                         
                         fig = px.bar(df_res, x='Attack', y='Success Rate', title="Attack Success Rate", color='Attack')
                         st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Adversarial Training")
        st.write("Train a robust model by augmenting training data with adversarial examples.")
        
        train_ds = st.selectbox("Select Training Dataset", datasets, key="adv_train_ds")
        ratio = st.slider("Adversarial Ratio", 0.1, 0.5, 0.3, help="Proportion of adversarial examples in training batch")
        
        if st.button("Start Adversarial Training", type="primary"):
            if not train_ds:
                st.warning("Please select a dataset.")
            else:
                with st.status("Running Adversarial Training...", expanded=True) as status:
                    st.write("Initializing training pipeline...")
                    config = {
                        'target_column': 'label', 
                        'adversarial_ratio': ratio,
                        'attack_types': ['fgsm', 'pgd'] # Default mix
                    }
                    
                    res = backend.train_robust_model(train_ds, config)
                    
                    if res['status'] == 'success':
                        status.update(label="Robust Training Complete!", state="complete")
                        st.success(f"Best Robustness Achieved: {res['results']['best_robustness']:.2%}")
                    else:
                        status.update(label="Training Failed", state="error")
                        st.error(str(res.get('message', 'Unknown error')))

    with tab3:
        st.subheader("Concept Drift Analysis")
        st.write("Compare new incoming data against the training baseline to detect distribution shifts.")
        
        drift_ds = st.selectbox("Select New Data (to check for drift)", datasets, key="drift_ds")
        
        if st.button("Analyze for Drift"):
             if not drift_ds:
                st.warning("Please select a dataset.")
             else:
                with st.spinner("Calculating drift statistics..."):
                    res = backend.check_drift(drift_ds, {'target_column': 'label'})
                    
                    if res['status'] == 'success':
                         drift_res = res['results']
                         
                         st.divider()
                         if drift_res['drift_detected']:
                             st.error("🚨 DRIFT DETECTED!")
                             st.write("The statistical properties of this dataset differ significantly from the training data.")
                         else:
                             st.success("✅ No Significant Drift Detected")
                             
                         # Show details
                         st.json(drift_res)
                    else:
                        st.error(f"Analysis failed: {res['message']}")
