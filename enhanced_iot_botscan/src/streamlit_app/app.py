import streamlit as st
from streamlit_option_menu import option_menu
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

st.set_page_config(
    page_title="Enhanced IoT BotScan",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    h1 {
        color: #1f77b4;
    }
    .stMetric {
        background-color: #ffffff;
        border: 1px solid #e6e9ef;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

def main():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/security-shield-green.png", width=50)
        st.title("IoT BotScan")
        
        selected = option_menu(
            "Navigation",
            ["Dashboard", "Analytics", "Training", "Inference"],
            icons=['speedometer2', 'graph-up', 'cpu', 'search'],
            menu_icon="cast",
            default_index=0,
        )
        
        st.info("Enhanced IoT Botnet Detection System\nv1.0.0")

    # Routing
    if selected == "Dashboard":
        import pages.Dashboard as dashboard
        dashboard.app()
    elif selected == "Analytics":
        import pages.Analytics as analytics
        analytics.app()
    elif selected == "Training":
        import pages.Training as training
        training.app()
    elif selected == "Inference":
        import pages.Inference as inference
        inference.app()

if __name__ == "__main__":
    main()
