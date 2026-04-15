import streamlit as st
from streamlit_option_menu import option_menu
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

st.set_page_config(
    page_title="Enhanced IoT BotNet Detection Through Hybrid Ensemble Learning",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

css_path = os.path.join(os.path.dirname(__file__), 'assets', 'custom.css')
load_css(css_path)

def main():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/security-shield-green.png", width=50)
        st.title("IoT BotScan")
        
        selected = option_menu(
            "Navigation",
            ["Dashboard", "Analytics", "Training", "Adversarial", "Inference"],
            icons=['speedometer2', 'graph-up', 'cpu', 'shield-lock', 'search'],
            menu_icon="cast",
            default_index=0,
        )
        
        st.info("Enhanced IoT Botnet Detection System\nv1.0.0")

    # Routing
    if selected == "Dashboard":
        import views.Dashboard as dashboard
        dashboard.app()
    elif selected == "Analytics":
        import views.Analytics as analytics
        analytics.app()
    elif selected == "Training":
        import views.Training as training
        training.app()
    elif selected == "Adversarial":
        import views.Adversarial as adversarial
        adversarial.app()
    elif selected == "Inference":
        import views.Inference as inference
        inference.app()

if __name__ == "__main__":
    main()
