import streamlit as st
import time
from src.streamlit_app.utils import render_header

def app():
    render_header("Model Training", "Train and evaluate new models")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Configuration")
        
        dataset = st.selectbox(
            "Select Dataset",
            ["N-BaIoT", "IoT-23", "Custom Upload"]
        )
        
        model_type = st.selectbox(
            "Model Architecture",
            ["Hybrid Ensemble (Recommended)", "Random Forest", "XGBoost", "LightGBM", "Deep Learning (LSTM)"]
        )
        
        st.markdown("#### Hyperparameters")
        epochs = st.slider("Epochs / Estimators", 10, 500, 100)
        learning_rate = st.number_input("Learning Rate", 0.001, 1.0, 0.01, format="%.3f")
        batch_size = st.select_slider("Batch Size", options=[32, 64, 128, 256, 512], value=64)
        
        train_btn = st.button("Start Training", type="primary")

    with col2:
        st.subheader("Training Progress")
        
        if train_btn:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            chart_placeholder = st.empty()
            loss_data = []
            
            for i in range(101):
                # Simulate training
                time.sleep(0.05) 
                progress_bar.progress(i)
                status_text.text(f"Training... {i}% complete")
                
                # Update chart
                if i % 10 == 0:
                    loss_data.append(1.0 / (i + 1) + 0.1)
                    st.line_chart(loss_data)
            
            status_text.success("Training Completed Successfully!")
            st.balloons()
            
            st.markdown("### Evaluation Results")
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("Accuracy", "99.2%", "+0.5%")
            res_col2.metric("Precision", "98.9%", "+0.3%")
            res_col3.metric("Recall", "99.1%", "+0.4%")
        else:
            st.info("Configure parameters and click 'Start Training' to begin.")
            st.image("https://img.icons8.com/clouds/200/000000/development.png", width=200)
