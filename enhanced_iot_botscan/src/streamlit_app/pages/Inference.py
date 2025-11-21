import streamlit as st
import pandas as pd
from src.streamlit_app.utils import render_header

def app():
    render_header("Inference", "Test model with new data")

    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

    with tab1:
        st.subheader("Manual Input")
        
        col1, col2 = st.columns(2)
        with col1:
            f1 = st.number_input("Packet Size (bytes)", min_value=0, value=64)
            f2 = st.number_input("Flow Duration (ms)", min_value=0.0, value=10.5)
            f3 = st.number_input("Source Port", min_value=0, max_value=65535, value=80)
        with col2:
            f4 = st.number_input("Destination Port", min_value=0, max_value=65535, value=443)
            f5 = st.number_input("Protocol", min_value=0, max_value=255, value=6)
            f6 = st.number_input("Bytes/Sec", min_value=0.0, value=1024.0)

        if st.button("Predict Threat"):
            with st.spinner("Analyzing..."):
                # Simulate prediction
                import time
                time.sleep(1)
                prediction = "Benign" if f1 < 1000 else "Mirai Botnet"
                confidence = 0.98
                
                if prediction == "Benign":
                    st.success(f"Prediction: **{prediction}** (Confidence: {confidence})")
                else:
                    st.error(f"Prediction: **{prediction}** (Confidence: {confidence})")

    with tab2:
        st.subheader("Batch Processing")
        uploaded_file = st.file_uploader("Upload CSV or PCAP file", type=['csv', 'pcap'])
        
        if uploaded_file is not None:
            st.info(f"File '{uploaded_file.name}' uploaded successfully.")
            if st.button("Process File"):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                st.success("Processing Complete")
                
                # Dummy results
                results = pd.DataFrame({
                    'Flow ID': range(1, 6),
                    'Prediction': ['Benign', 'Mirai', 'Benign', 'Gafgyt', 'Benign'],
                    'Confidence': [0.99, 0.95, 0.98, 0.89, 0.97]
                })
                st.dataframe(results)
