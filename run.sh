#!/bin/bash
# Run the TDOA analyser Streamlit app inside the local venv.
# Usage:  ./run.sh
cd "$(dirname "$0")"
.venv/bin/streamlit run app.py "$@"
