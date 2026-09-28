#!/usr/bin/env bash
# CSE 402: N-Body Defense Platform Launcher for Git Bash / Zsh
echo "====================================================================="
echo "  🪐 CSE 402: N-Body Simulation - Supervisor Oral Defense Platform"
echo "  Group C_G8 - Live Demonstration & Benchmark Evaluation Hub"
echo "====================================================================="

if [ -f ".venv/Scripts/streamlit.exe" ]; then
    .venv/Scripts/streamlit.exe run bench/defense_app.py
else
    uv run streamlit run bench/defense_app.py
fi
