@echo off
title CSE 402: N-Body Defense Platform
echo =====================================================================
echo   🪐 CSE 402: N-Body Simulation - Supervisor Oral Defense Platform
echo   Group C_G8 - Live Demonstration & Benchmark Evaluation Hub
echo =====================================================================
echo.
echo Launching Streamlit defense app on local browser...
echo.

if exist ".venv\Scripts\streamlit.exe" (
    ".venv\Scripts\streamlit.exe" run "bench\defense_app.py"
) else (
    uv run streamlit run "bench\defense_app.py"
)

pause
