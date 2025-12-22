@echo off
set PYTHONPATH=%PYTHONPATH%;%CD%
streamlit run src/streamlit_app/app.py
