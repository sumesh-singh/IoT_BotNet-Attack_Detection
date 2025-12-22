import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List

def load_css():
    """Deprecated: CSS is now loaded globally in app.py"""
    pass

def render_header(title: str, subtitle: str = ""):
    """Render page header."""
    st.title(title)
    if subtitle:
        st.markdown(f"*{subtitle}*")
    st.markdown("---")

def render_metrics_row(metrics: List[Dict[str, Any]]):
    """Render a row of metrics."""
    cols = st.columns(len(metrics))
    for i, metric in enumerate(metrics):
        with cols[i]:
            st.metric(
                label=metric['label'],
                value=metric['value'],
                delta=metric.get('delta')
            )

def plot_confusion_matrix(cm: np.ndarray, labels: List[str]):
    """Plot confusion matrix using Plotly."""
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        color_continuous_scale='Blues',
        aspect="auto"
    )
    fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)

def plot_feature_importance(importance_dict: Dict[str, float]):
    """Plot feature importance."""
    df = pd.DataFrame({
        'Feature': list(importance_dict.keys()),
        'Importance': list(importance_dict.values())
    }).sort_values('Importance', ascending=True)

    fig = px.bar(
        df,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Feature Importance"
    )
    st.plotly_chart(fig, use_container_width=True)
