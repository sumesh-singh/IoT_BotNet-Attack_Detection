import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.streamlit_app.utils import render_header

def app():
    render_header("Analytics", "Historical data analysis and insights")

    # Date Range Picker
    col1, col2 = st.columns(2)
    with col1:
        st.date_input("Start Date", pd.Timestamp.now() - pd.Timedelta(days=7))
    with col2:
        st.date_input("End Date", pd.Timestamp.now())

    st.markdown("### Attack Distribution")
    
    # Dummy Data for Attack Types
    attack_data = pd.DataFrame({
        'Attack Type': ['Mirai', 'Bashlite', 'Gafgyt', 'Benign', 'Other'],
        'Count': [4500, 3200, 1500, 50000, 800]
    })
    
    fig_pie = px.pie(
        attack_data, 
        values='Count', 
        names='Attack Type', 
        title='Distribution of Detected Threats',
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("### Feature Correlation Analysis")
    st.write("Correlation heatmap of top features involved in recent attacks.")
    
    # Dummy Correlation Matrix
    corr_matrix = [
        [1.0, 0.8, 0.3, 0.1],
        [0.8, 1.0, 0.4, 0.2],
        [0.3, 0.4, 1.0, 0.7],
        [0.1, 0.2, 0.7, 1.0]
    ]
    features = ['Packet Size', 'Flow Duration', 'Byte Rate', 'Packet Rate']
    
    fig_corr = px.imshow(
        corr_matrix,
        x=features,
        y=features,
        color_continuous_scale='RdBu_r',
        aspect="auto"
    )
    st.plotly_chart(fig_corr, use_container_width=True)
