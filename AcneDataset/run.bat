@echo off
title DermaScan AI - Clinical Acne Diagnostics
echo ===================================================
echo   Starting DermaScan AI Clinical Platform...
echo ===================================================
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    py -m streamlit run app.py
) else (
    python -m streamlit run app.py
)
pause
