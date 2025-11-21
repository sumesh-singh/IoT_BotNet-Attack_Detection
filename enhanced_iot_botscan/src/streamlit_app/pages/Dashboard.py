import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from src.streamlit_app.utils import render_header, render_metrics_row

def app():
    render_header("System Dashboard", "Real-time monitoring and status")

    # Simulated Metrics
    metrics = [
        {"label": "System Status", "value": "Online", "delta": None},
        {"label": "Active Threats", "value": "12", "delta": "-2"},
        {"label": "Traffic (req/s)", "value": "1,245", "delta": "+15%"},
        {"label": "Model Accuracy", "value": "98.5%", "delta": "+0.2%"}
    ]
    render_metrics_row(metrics)

    st.markdown("### Real-time Traffic Analysis")
    
    # Simulated real-time data
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Live traffic chart
        chart_placeholder = st.empty()
        # Generate some dummy data
        now = pd.Timestamp.now()
        times = pd.date_range(now - pd.Timedelta(minutes=60), now, freq='1min')
        data = pd.DataFrame({
            'Time': times,
            'Normal Traffic': np.random.randint(800, 1200, len(times)),
            'Malicious Traffic': np.random.randint(0, 100, len(times))
        })
        
        fig = px.area(
            data, 
            x='Time', 
            y=['Normal Traffic', 'Malicious Traffic'],
            title="Network Traffic Flow",
            color_discrete_map={'Normal Traffic': '#2ecc71', 'Malicious Traffic': '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Recent Alerts")
        alerts = [
            {"time": "10:42:15", "type": "Mirai Botnet", "severity": "High"},
            {"time": "10:38:22", "type": "Port Scan", "severity": "Medium"},
            {"time": "10:15:00", "type": "DDoS Attempt", "severity": "Critical"},
            {"time": "09:55:10", "type": "Unknown UDP", "severity": "Low"},
        ]
        
        for alert in alerts:
            color = "red" if alert['severity'] in ["High", "Critical"] else "orange" if alert['severity'] == "Medium" else "blue"
            st.markdown(
                f"""
                <div style="padding: 10px; border-left: 5px solid {color}; background-color: #f8f9fa; margin-bottom: 10px;">
                    <strong>{alert['type']}</strong> <span style="float:right; font-size:0.8em; color:gray">{alert['time']}</span><br>
                    <span style="font-size:0.9em;">Severity: {alert['severity']}</span>
                </div>
                """, 
                unsafe_allow_html=True
            )

    st.markdown("### Drift Detection Status")
    st.info("No significant concept drift detected in the last 24 hours. Model performance remains stable.")
