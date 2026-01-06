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
    
    # TEMPORARY DIAGNOSTIC - Remove after debugging
    import sys
    st.write("**Python:**", sys.executable)
    try:
        # Add DLL directory before torch import (Windows fix)
        if sys.platform == 'win32':
            import importlib.util
            torch_spec = importlib.util.find_spec('torch')
            if torch_spec and torch_spec.origin:
                torch_lib_path = os.path.join(os.path.dirname(torch_spec.origin), 'lib')
                if os.path.exists(torch_lib_path):
                    if hasattr(os, 'add_dll_directory'):
                        os.add_dll_directory(torch_lib_path)
                    os.environ['PATH'] = torch_lib_path + os.pathsep + os.environ.get('PATH', '')
        
        import torch
        st.success(f"✅ Torch version: {torch.__version__}")
    except Exception as e:
        st.error(f"❌ Torch import failed: {e}")
    # END DIAGNOSTIC
    
    backend = BackendInterface()
    
    # Get datasets once for all tabs
    datasets = get_available_datasets()
    if not datasets:
        st.warning("No CSV datasets found. Please upload a dataset in the 'Training' page.")
    
    tab1, tab2, tab3 = st.tabs(["🛡️ Adversarial Attacks", "🤖 Adversarial Training", "📉 Drift Analysis"])
    
    with tab1:
        st.subheader("Robustness Testing (ARM)")
        st.write("Test model robustness using realistic IoT threat simulations.")
        
        if not backend.model or not backend.model.is_trained:
            st.error("Model must be trained first. Please go to the 'Training' page.")
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                dataset_path = st.selectbox("Select Test Dataset", datasets, key="adv_dataset") if datasets else None
                
                st.write("**Threat Scenarios:**")
                st.write("- 🔊 Noise Injection (sensor noise)")
                st.write("- ❌ Feature Masking (sensor failures)")
                st.write("- 📈 Traffic Bursts (DDoS patterns)")
                
                if st.button("🚀 Run Robustness Test", type="primary"):
                    with st.spinner("Running comprehensive robustness evaluation..."):
                        config = {
                            'target_column': 'label',
                            'noise_levels': [0.0, 0.05, 0.1, 0.2],
                            'masking_rates': [0.0, 0.1, 0.2, 0.3],
                            'burst_intensities': [1.0, 1.5, 2.0, 3.0]
                        }
                        result = backend.evaluate_robustness(dataset_path, config)
                        
                        if result['status'] == 'success':
                            st.session_state['adv_results'] = result['results']
                            st.success("✅ Robustness evaluation complete!")
                        else:
                            st.error(f"Evaluation failed: {result['message']}")

            with col2:
                if 'adv_results' in st.session_state:
                    res = st.session_state['adv_results']
                    
                    # Handle ARM results format
                    if 'aggregate_scores' in res:
                        scores = res['aggregate_scores']
                        
                        # Main metrics
                        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                        mcol1.metric("Overall Robustness", f"{scores.get('overall_robustness', 0):.1%}")
                        mcol2.metric("Noise Robustness", f"{scores.get('noise_robustness', 0):.1%}")
                        mcol3.metric("Masking Robustness", f"{scores.get('masking_robustness', 0):.1%}")
                        mcol4.metric("Burst Robustness", f"{scores.get('burst_robustness', 0):.1%}")
                        
                        st.divider()
                        
                        # Detailed results per threat
                        scenarios = res.get('threat_scenarios', {})
                        
                        if scenarios:
                            import numpy as np
                            
                            # Create data for visualization
                            chart_data = []
                            for threat_type, threat_results in scenarios.items():
                                for scenario, metrics in threat_results.items():
                                    chart_data.append({
                                        'Threat': threat_type.title(),
                                        'Scenario': scenario,
                                        'Accuracy': metrics.get('accuracy', 0),
                                        'Robustness': metrics.get('robustness_score', 0)
                                    })
                            
                            df_res = pd.DataFrame(chart_data)
                            
                            # Show chart
                            fig = px.bar(df_res, x='Scenario', y='Robustness', color='Threat',
                                        title="Robustness Score by Threat Scenario",
                                        barmode='group')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show detailed table
                            with st.expander("View Detailed Results"):
                                st.dataframe(df_res, use_container_width=True)
                        
                        # Show report if available
                        report = res.get('report', {})
                        if report.get('recommendations'):
                            st.subheader("📋 Recommendations")
                            for rec in report['recommendations']:
                                st.write(f"• {rec}")
                    
                    # Fallback for old adversarial attack format
                    elif 'overall_robustness' in res:
                        st.metric("Robustness Score", f"{res['overall_robustness']:.2%}")
                        metrics = res.get('attack_metrics', {})
                        if metrics:
                            data = [{'Attack': atk, 'Success Rate': m.get('success_rate', 0)} 
                                   for atk, m in metrics.items()]
                            st.dataframe(pd.DataFrame(data))


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
                        stats = res.get('statistics', {})
                        
                        # Summary Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Drifts Detected", stats.get('n_detections', 0))
                        col2.metric("Drift Rate", f"{stats.get('overall_drift_rate', 0):.2%}")
                        col3.metric("Last Detection", str(stats.get('last_detection', '-'))[:10])
                        
                        # Severity assessment
                        if drift_res.get('drift_detected', False):
                            severity = "Severe" if stats.get('overall_drift_rate', 0) >= 0.5 else "Moderate"
                        else:
                            severity = "None"
                        col4.metric("Severity", severity)
                        
                        st.divider()
                        if drift_res.get('drift_detected', False):
                            st.error("🚨 DRIFT DETECTED!")
                            st.write("The statistical properties of this dataset differ significantly from the training data.")
                            
                            # Feature-level visualization
                            method_results = drift_res.get('method_results', {})
                            if 'ks' in method_results:
                                ks_result = method_results['ks']
                                feature_drift = ks_result.get('feature_drift', {})
                                feature_results = feature_drift.get('feature_results', [])
                                
                                flagged = [f['feature_index'] for f in feature_results if f.get('drift_detected', False)]
                                if flagged:
                                    st.write(f"**Features flagged by KS test:** {flagged[:10]}...")
                                    
                                    # Load incoming data for plotting
                                    try:
                                        import numpy as np
                                        df_new = pd.read_csv(drift_ds)
                                        target_col = 'label'
                                        if target_col not in df_new.columns:
                                            possible = [c for c in df_new.columns if 'label' in c.lower()]
                                            target_col = possible[0] if possible else df_new.columns[-1]
                                        X_new = df_new.drop(columns=[target_col], errors='ignore')
                                        
                                        # Plot up to 3 flagged features
                                        for i in flagged[:3]:
                                            if i < len(X_new.columns):
                                                values_new = X_new.iloc[:, i]
                                                
                                                # Get reference stats
                                                ref_mean = None
                                                ref_std = None
                                                for f in feature_results:
                                                    if f.get('feature_index') == i:
                                                        ref_mean = f.get('reference_mean', values_new.mean())
                                                        ref_std = f.get('reference_std', values_new.std())
                                                
                                                # Generate synthetic baseline
                                                if ref_mean is not None and ref_std is not None and ref_std > 0:
                                                    baseline_vals = np.random.normal(ref_mean, ref_std, size=min(len(values_new), 1000))
                                                else:
                                                    baseline_vals = values_new.sample(min(len(values_new), 1000)).values
                                                
                                                df_plot = pd.DataFrame({
                                                    'Value': np.concatenate([baseline_vals, values_new.head(1000).values]),
                                                    'Dataset': ['Baseline']*len(baseline_vals) + ['Incoming']*min(len(values_new), 1000)
                                                })
                                                fig = px.histogram(
                                                    df_plot, x='Value', color='Dataset', barmode='overlay',
                                                    histnorm='probability density', nbins=30,
                                                    title=f"Feature {i} ({X_new.columns[i]}) Distribution"
                                                )
                                                st.plotly_chart(fig, use_container_width=True)
                                    except Exception as e:
                                        st.warning(f"Could not visualize features: {e}")
                            
                            # Auto-retraining option
                            st.divider()
                            st.write("**Recommendation:** Retrain the model on this new data to adapt to the drift.")
                            if st.button("Start Automatic Retraining", type="primary"):
                                with st.spinner("Retraining model on new data..."):
                                    retrain_res = backend.retrain_model(drift_ds, {'target_column': 'label'})
                                    if retrain_res['status'] == 'success':
                                        st.success("✅ Model retrained successfully!")
                                        new_acc = retrain_res['results'].get('ensemble_validation_accuracy', 0.0)
                                        st.metric("New Model Accuracy", f"{new_acc:.2%}")
                                    else:
                                        st.error(f"Retraining failed: {retrain_res.get('message', 'Unknown error')}")
                        else:
                            st.success("✅ No Significant Drift Detected")
                            st.write("The incoming data distribution matches the training baseline.")
                        
                        # Show detailed results
                        with st.expander("View Detailed Results"):
                            st.json(drift_res)
                    else:
                        st.error(f"Analysis failed: {res.get('message', '')}")
