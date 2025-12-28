@echo off
title Data Science Dashboard - Fixed Version
color 0A

echo ================================================
echo    DATA SCIENCE PIPELINE DASHBOARD - FIXED
echo ================================================
echo.

echo Step 1: Checking Python version...
python --version
echo.

echo Step 2: Upgrading pip to latest version...
python -m pip install --upgrade pip --user
echo.

echo Step 3: Installing core packages...
pip install streamlit==1.24.0
pip install pandas==1.5.3
pip install numpy==1.24.3
pip install matplotlib==3.5.3
pip install seaborn==0.12.2
pip install scikit-learn==1.2.2
echo.

echo Step 4: Installing optional packages (for file conversion)...
pip install openpyxl==3.0.10
echo [INFO] PDF and OCR support disabled for compatibility
echo.

echo Step 5: Starting Dashboard...
echo.
echo If browser doesn't open automatically, go to: http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
streamlit run data_analysis_dashboard.py --server.port 8501

pause