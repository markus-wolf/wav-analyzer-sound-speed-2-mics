# Host TDOA analysis

Streamlit app and Python helpers to analyse recordings from the ESP32 dual-mic TDOA project: load stereo WAVs (or two mono files), estimate time-difference-of-arrival, and explore sound speed vs. temperature.

## Setup

```bash
cd host_analysis
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
./run.sh
```

Or: `streamlit run app.py`

Open the URL shown in the terminal (usually `http://localhost:8501`).

## Related firmware

The mic-array firmware and web UI live in the separate `ESP32-S3-2mics` repo (sibling directory in this workspace tree). Record from the device, download WAV, then analyse here.
