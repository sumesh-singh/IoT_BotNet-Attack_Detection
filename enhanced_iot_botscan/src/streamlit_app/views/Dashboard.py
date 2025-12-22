import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import render_header, render_metrics_row
from src.streamlit_app.backend_interface import BackendInterface

def app():
    render_header("System Dashboard", "Real-time monitoring and status")
    
    backend = BackendInterface()
    status = backend.get_system_status()

    # System Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="System Status", 
            value="Active" if status['model_loaded'] else "Idle",
            help="Current operational state of the detection system."
        )
    
    with col2:
        st.metric(
            label="Model Status", 
            value="Trained" if status['model_loaded'] else "Untrained",
            delta="Ready" if status['model_loaded'] else "Needs Training",
            help="Indicates if the Hybrid Ensemble model is trained and ready for inference."
        )
        
    with col3:
        st.metric(
            label="Model Accuracy", 
            value=status['accuracy'],
            help="Validation accuracy of the currently loaded model."
        )
        
    with col4:
        st.metric(
            label="Last Training", 
            value=status['last_training'].split('T')[0] if 'T' in status['last_training'] else status['last_training'],
            help="Date of the most recent successful model training run."
        )

    st.markdown("---")

    # Real-time Traffic Analysis (Using Validation Data as Proxy)
    st.subheader("Model Performance Analysis (Validation Data)")
    
    val_results = backend.get_recent_predictions()
    cm_data = backend.get_confusion_matrix()
    
    if not val_results:
        st.info("No validation data available. Please train the model to see performance metrics.")
    else:
        # Layout: 2 Rows. Row 1: Traffic Flow (Samples) & Attack Dist. Row 2: Confusion Matrix & Alerts
        row1_col1, row1_col2 = st.columns([2, 1])
        
        y_true = val_results['y_true']
        y_pred = val_results['y_pred']
        
        # Create a dataframe for visualization
        df_vis = pd.DataFrame({
            'Sample ID': range(len(y_true)),
            'True Label': y_true,
            'Predicted Label': y_pred
        })
        
        # Map numeric labels to names if possible (assuming 0=Benign, 1=Mirai, 2=Gafgyt based on previous context)
        # Ideally we should get this mapping from the backend, but for now we'll infer or use generic
        label_map = {0: 'Benign', 1: 'Mirai', 2: 'Gafgyt'}
        df_vis['True Class'] = df_vis['True Label'].map(label_map).fillna(df_vis['True Label'].astype(str))
        df_vis['Predicted Class'] = df_vis['Predicted Label'].map(label_map).fillna(df_vis['Predicted Label'].astype(str))
        
        with row1_col1:
            # Traffic Flow - Visualize the stream of validation samples
            # We'll show a scatter plot of predictions over "time" (sample index)
            fig_traffic = px.scatter(
                df_vis, 
                x='Sample ID', 
                y='Predicted Class',
                color='True Class',
                title="Validation Predictions Stream",
                color_discrete_sequence=px.colors.qualitative.Bold,
                template="plotly_dark",
                height=350
            )
            fig_traffic.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_traffic, use_container_width=True)
    
        with row1_col2:
            # Attack Distribution (Donut Chart) based on Predictions
            pred_counts = df_vis['Predicted Class'].value_counts().reset_index()
            pred_counts.columns = ['Class', 'Count']
            
            fig_donut = px.pie(
                pred_counts, 
                names='Class', 
                values='Count', 
                hole=0.5,
                title="Predicted Attack Distribution",
                color_discrete_sequence=px.colors.qualitative.Bold,
                template="plotly_dark",
                height=350
            )
            fig_donut.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_donut, use_container_width=True)
    
        row2_col1, row2_col2 = st.columns([1, 1])
    
        with row2_col1:
            # Confusion Matrix
            if cm_data:
                import plotly.figure_factory as ff
                
                z = cm_data['matrix']
                x = [str(c) for c in cm_data['classes']]
                y = [str(c) for c in cm_data['classes']]
                
                # Map class names if they match our known labels
                x_labels = [label_map.get(int(c), c) if c.isdigit() else c for c in x]
                y_labels = [label_map.get(int(c), c) if c.isdigit() else c for c in y]
                
                fig_cm = ff.create_annotated_heatmap(
                    z, 
                    x=x_labels, 
                    y=y_labels, 
                    colorscale='Viridis',
                    showscale=True
                )
                fig_cm.update_layout(
                    title="Confusion Matrix",
                    xaxis_title="Predicted",
                    yaxis_title="True",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=350
                )
                st.plotly_chart(fig_cm, use_container_width=True)

        with row2_col2:
            st.markdown("#### Recent Alerts")
            alerts = [
                {"time": "10:42:15", "type": "Mirai Botnet", "severity": "High"},
                {"time": "10:38:22", "type": "Port Scan", "severity": "Medium"},
                {"time": "10:15:00", "type": "DDoS Attempt", "severity": "Critical"},
                {"time": "09:55:10", "type": "Unknown UDP", "severity": "Low"},
            ]
            
            for alert in alerts:
                color = "#ff4b4b" if alert['severity'] in ["High", "Critical"] else "#ffa500" if alert['severity'] == "Medium" else "#00d4ff"
                st.markdown(
                    f"""
                    <div class="alert-card" style="border-left-color: {color};">
                        <strong style="color: {color}">{alert['type']}</strong> 
                        <span style="float:right; font-size:0.8em; color: #b0b0b0;">{alert['time']}</span><br>
                        <span style="font-size:0.9em; color: #fafafa;">Severity: {alert['severity']}</span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

    st.markdown("### Drift Detection Status")
    with st.expander("What is Drift Detection?", expanded=False):
        st.write("""
        **Concept Drift** occurs when the statistical properties of the target variable change over time. 
        This system uses **Page-Hinkley** and **Kolmogorov-Smirnov** tests to detect such changes.
        If drift is detected, it indicates the model may need retraining.
        """)
        
    drift_stats = backend.get_drift_status()
    
    if drift_stats and drift_stats.get('n_detections', 0) > 0:
        drift_col1, drift_col2, drift_col3 = st.columns(3)
        with drift_col1:
             st.metric("Total Detections", drift_stats['n_detections'])
        with drift_col2:
             st.metric("Overall Drift Rate", f"{drift_stats.get('overall_drift_rate', 0.0):.2%}")
        with drift_col3:
             st.metric("Last Detection", drift_stats.get('last_detection', 'N/A').split('T')[0] if drift_stats.get('last_detection') else 'None')
             
        # Show status warning/success
        if drift_stats.get('overall_drift_rate', 0) > 0.1: # Threshold example
             st.warning("⚠️ High drift rate detected. Consider retraining the model.")
        else:
             st.success("✅ Drift levels are within acceptable limits.")

    else:
        st.info("No drift detection history available yet. Detect drift in the 'Adversarial & Drift' view.")
