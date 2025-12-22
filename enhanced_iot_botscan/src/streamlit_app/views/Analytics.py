import streamlit as st
import plotly.express as px
import pandas as pd
from utils import render_header
from src.streamlit_app.backend_interface import BackendInterface

def app():
    render_header("System Analytics", "Performance metrics and insights")
    
    backend = BackendInterface()
    status = backend.get_system_status()
    
    # Help Section
    with st.expander("ℹ️ About Analytics"):
        st.write("""
        - **Feature Importance**: Shows which network features contribute most to the detection of threats.
        - **Training History**: Tracks model accuracy over time.
        - **Confusion Matrix**: Visualizes the performance of the classification model (True Positives vs False Positives).
        """)

    if not status['model_loaded']:
        st.warning("⚠️ No trained model found. Train a model to view analytics.")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feature Importance")
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
            st.info("Feature importance not available.")

    with col2:
        st.subheader("Training History")
        history = backend.training_history
        
        if history:
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
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No training history available.")

    st.markdown("### Confusion Matrix (Last Training)")
    # In a real scenario, we'd store the confusion matrix in the training results
    # For now, we'll check if we have results in current_metrics
    if backend.current_metrics and 'confusion_matrix' in backend.current_metrics:
        cm = backend.current_metrics['confusion_matrix']
        # Visualize CM
        st.write(cm) # Placeholder
    else:
        st.info("Confusion matrix not available for the current model.")
        
        # Placeholder for visual appeal if no real data
        if not status['model_loaded']:
            data = [[800, 50], [30, 120]]
            fig_cm = px.imshow(
                data, 
                text_auto=True, 
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Benign', 'Malicious'],
                y=['Benign', 'Malicious'],
                title="Example Confusion Matrix"
            )
            st.plotly_chart(fig_cm, use_container_width=True)
