#!/bin/bash
# Run the Doppler Speed Analyser Streamlit app inside the local venv.
# Usage:  ./run_doppler.sh
cd "$(dirname "$0")"
.venv/bin/streamlit run doppler_app.py "$@"
