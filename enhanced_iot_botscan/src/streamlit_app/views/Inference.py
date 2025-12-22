import streamlit as st
import pandas as pd
from utils import render_header
from src.streamlit_app.backend_interface import BackendInterface

def app():
    render_header("Inference", "Test model with new data")
    
    backend = BackendInterface()
    status = backend.get_system_status()
    
    if not status['model_loaded']:
        st.error("⚠️ No trained model found. Please go to the **Training** page to train a model first.")
        return

    # Help Section
    with st.expander("ℹ️ How to Run Inference"):
        st.write("""
        - **Manual Input**: Enter feature values manually to test a single sample.
        - **Batch Processing**: Upload a CSV file containing multiple samples.
        - The system will automatically clean and engineer features before prediction.
        """)

    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

    with tab1:
        st.subheader("Manual Input")
        st.info("Enter values for standard network flow features.")
        
        # Standard IoT Botnet Features (Simplified set for demo)
        col1, col2 = st.columns(2)
        with col1:
            # These keys should match what the model expects or what the cleaner/engineer can handle
            # Ideally, we'd inspect the model's expected features, but for now we'll use a standard set
            # and let the backend handle missing columns (cleaner fills them)
            f1 = st.number_input("Packet Size (bytes)", min_value=0, value=64, help="Average size of packets in the flow.")
            f2 = st.number_input("Flow Duration (ms)", min_value=0.0, value=10.5, help="Duration of the flow in milliseconds.")
            f3 = st.number_input("Source Port", min_value=0, max_value=65535, value=80, help="Source port number.")
        with col2:
            f4 = st.number_input("Destination Port", min_value=0, max_value=65535, value=443, help="Destination port number.")
            f5 = st.number_input("Protocol", min_value=0, max_value=255, value=6, help="IP Protocol number (e.g., 6 for TCP, 17 for UDP).")
            f6 = st.number_input("Bytes/Sec", min_value=0.0, value=1024.0, help="Rate of bytes transferred per second.")

        if st.button("Predict Threat"):
            # Construct DataFrame
            input_data = pd.DataFrame([{
                'packet_size': f1,
                'flow_duration': f2,
                'src_port': f3,
                'dst_port': f4,
                'protocol': f5,
                'bytes_per_sec': f6
                # Add more default columns if needed by the model to avoid "missing column" errors
                # The DataCleaner should handle missing columns by filling them, 
                # but it's best to match training data schema if possible.
            }])
            
            with st.spinner("Analyzing..."):
                result = backend.predict(input_data)
                
                if result['status'] == 'success':
                    pred = result['predictions'][0]
                    prob = result['probabilities'][0]
                    
                    # Assuming binary classification: 0 = Benign, 1 = Malicious
                    # Or multiclass. Adjust display based on result type.
                    
                    is_threat = pred == 1 # Simplified check
                    
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
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'], help="Upload a CSV file with network flow data.")
        
        if uploaded_file is not None:
            st.info(f"File '{uploaded_file.name}' uploaded successfully.")
            
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
                            
                            # Create results DataFrame
                            results_df = input_df.copy()
                            results_df['Prediction'] = preds
                            # Handle probability display depending on shape
                            if len(probs.shape) > 1:
                                results_df['Confidence'] = np.max(probs, axis=1)
                            else:
                                results_df['Confidence'] = probs
                            
                            st.dataframe(results_df)
                            
                            # Download button
                            csv = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "Download Results",
                                csv,
                                "prediction_results.csv",
                                "text/csv",
                                key='download-csv'
                            )
                        else:
                            st.error(f"Batch Prediction Failed: {result['message']}")
                            
                except Exception as e:
                    st.error(f"Error reading file: {e}")
