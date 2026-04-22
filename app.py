"""
Two-Microphone TDOA Analyser
Streamlit interactive dashboard for analysing recordings from the ESP32
dual-mic sound-speed measurement system.

Run:  streamlit run app.py
"""

import copy
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import soundfile as sf

from analysis import (
    AudioData, TDOAResult,
    load_stereo, load_two_mono,
    trim_window, bandpass, normalise,
    compute_tdoa, compute_speed, theoretical_speed,
    detect_events, waveform_envelope,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TDOA Analyser",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔊 Two-Microphone TDOA Analyser")
st.caption("Load a stereo recording (or two mono files) from the ESP32 mic array "
           "to measure the time difference of arrival and calculate sound speed.")

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Microphone geometry")
    sep_m = st.number_input(
        "Mic separation (m)", min_value=0.01, max_value=20.0,
        value=6.0, step=0.1,
        help="Physical distance between the two microphones."
    )
    temp_c = st.number_input(
        "Air temperature (°C)", min_value=-20.0, max_value=50.0,
        value=20.0, step=0.5,
        help="Used to calculate the theoretical speed of sound."
    )

    st.subheader("Bandpass filter")
    do_filter = st.toggle("Enable bandpass filter", value=True)
    col1, col2 = st.columns(2)
    with col1:
        low_hz = st.number_input("Low cut (Hz)",  min_value=10,  max_value=4000,  value=100)
    with col2:
        high_hz = st.number_input("High cut (Hz)", min_value=500, max_value=24000, value=20000)

    st.subheader("Correlation window")
    use_phat = st.toggle("GCC-PHAT weighting", value=True,
        help="Phase Transform: whitens the cross-spectrum so sidelobes collapse "
             "into a single sharp peak. Recommended when the two channels have "
             "very different amplitudes or when clap/transient sources are used.")
    max_lag_auto = st.toggle("Auto max lag", value=True,
        help="Limit lag to the maximum physically possible given mic separation.")
    if not max_lag_auto:
        max_lag_ms = st.slider("Max lag (ms)", 0.1, 50.0, 20.0, 0.1)
    else:
        max_lag_ms = None

    st.subheader("Event detection")
    show_events = st.toggle("Detect transient events", value=False)
    if show_events:
        evt_thresh = st.slider("Threshold (× median RMS)", 1.0, 20.0, 5.0, 0.5)

    st.divider()
    st.subheader("Theoretical speed of sound")
    theory = theoretical_speed(temp_c)
    st.metric("at given temperature", f"{theory:.1f} m/s")
    st.caption(f"Formula: 331.3 × √(1 + T/273.15)")

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------
st.header("1 · Load recording")

mode = st.radio("Input format", ["Stereo WAV (both mics in one file)",
                                  "Two mono WAV files"],
                horizontal=True)

audio: AudioData | None = None

if mode.startswith("Stereo"):
    uploaded = st.file_uploader("Upload stereo WAV", type=["wav", "flac", "aiff"],
                                 key="stereo")
    if uploaded:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(uploaded.read())
            tmp_path = f.name
        try:
            audio = load_stereo(tmp_path)
            st.success(f"Loaded **{uploaded.name}** — "
                       f"{audio.sample_rate} Hz, "
                       f"{audio.duration_s:.3f} s, "
                       f"{audio.n_channels_original} ch")
        except Exception as e:
            st.error(f"Could not load file: {e}")
else:
    col_a, col_b = st.columns(2)
    with col_a:
        up1 = st.file_uploader("Mic 1 (mono WAV)", type=["wav", "flac"], key="mono1")
    with col_b:
        up2 = st.file_uploader("Mic 2 (mono WAV)", type=["wav", "flac"], key="mono2")

    if up1 and up2:
        with (tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f1,
              tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f2):
            f1.write(up1.read()); f2.write(up2.read())
            try:
                audio = load_two_mono(f1.name, f2.name)
                st.success(f"Loaded **{up1.name}** + **{up2.name}** — "
                           f"{audio.sample_rate} Hz, {audio.duration_s:.3f} s")
            except Exception as e:
                st.error(f"Could not load files: {e}")

# ---------------------------------------------------------------------------
# Demo / sample data  — persisted in session_state so it survives reruns
# ---------------------------------------------------------------------------
if audio is None:
    # Recover previously generated demo
    if "demo_audio" in st.session_state:
        audio = st.session_state["demo_audio"]
    else:
        st.info("No file loaded yet — upload a WAV above, or generate the built-in demo.")

        col_d1, col_d2 = st.columns([2, 1])
        with col_d1:
            delay_ms = st.slider("Demo delay (ms)", 0.1, 10.0, 1.2, 0.1,
                                 help="Time offset between the two synthetic channels.")
        with col_d2:
            freq_hz = st.selectbox("Pulse frequency", [500, 1000, 2000, 4000], index=1)

        if st.button("▶ Generate demo", type="primary"):
            _sr = 48000
            _t  = np.linspace(0, 0.1, int(0.1 * _sr), endpoint=False)
            _pulse = np.zeros_like(_t)
            _ps = int(0.02 * _sr)
            _pl = int(0.005 * _sr)
            _pulse[_ps: _ps + _pl] = np.hanning(_pl) * np.sin(2 * np.pi * freq_hz * _t[:_pl])
            _noise = lambda: np.random.normal(0, 0.01, len(_t))
            _ch1 = _pulse + _noise()
            _ch2 = np.roll(_pulse, int(delay_ms * 1e-3 * _sr)) + _noise()
            _audio = AudioData(
                ch1=_ch1, ch2=_ch2, sample_rate=_sr,
                duration_s=0.1, source_file=f"demo · {freq_hz} Hz · {delay_ms} ms delay",
                n_channels_original=2,
            )
            st.session_state["demo_audio"] = _audio
            st.rerun()

        st.stop()

# Clear demo button (shown whenever demo data is active)
if audio is not None and "demo_audio" in st.session_state:
    if st.button("✕ Clear demo", help="Remove demo data and upload a real file."):
        del st.session_state["demo_audio"]
        st.rerun()

# ---------------------------------------------------------------------------
# Time-window trim
# ---------------------------------------------------------------------------
st.header("2 · Select analysis window")

dur = float(audio.duration_s)
# Slider step: ~2000 steps across the full recording (min 1 ms)
_step = max(0.001, round(dur / 2000, 3))

t_range = st.slider(
    "Analysis window — drag handles to isolate a single event",
    min_value=0.0, max_value=dur,
    value=(0.0, dur), step=_step, format="%.3f s",
    help="Drag the left/right handles to select the portion sent to the "
         "cross-correlator.  Isolating one clap removes the bias from "
         "multiple events and gives a clean TDOA estimate.",
)
t_start, t_end = float(t_range[0]), float(t_range[1])

# Fine-tune number inputs (values initialise from slider; type to override)
col_t1, col_t2, col_t3 = st.columns([2, 2, 1])
with col_t1:
    t_start = st.number_input("Fine-tune start (s)", min_value=0.0, max_value=dur,
                               value=t_start, step=0.001, format="%.3f",
                               key="t_start_fine")
with col_t2:
    t_end = st.number_input("Fine-tune end (s)", min_value=0.0, max_value=dur,
                             value=t_end, step=0.001, format="%.3f",
                             key="t_end_fine")
with col_t3:
    st.metric("Window", f"{t_end - t_start:.3f} s")

if t_end <= t_start:
    st.error("End must be after start."); st.stop()

audio_win = trim_window(audio, t_start, t_end)
sr = audio_win.sample_rate
times = np.linspace(t_start, t_end, len(audio_win.ch1))

# ---------------------------------------------------------------------------
# Event detection overlay
# ---------------------------------------------------------------------------
events_ch1, events_ch2 = [], []
if show_events:
    events_ch1 = detect_events(audio_win.ch1, sr, evt_thresh)
    events_ch2 = detect_events(audio_win.ch2, sr, evt_thresh)

# ---------------------------------------------------------------------------
# Audacity-style waveform display — both channels on one dark canvas
# ---------------------------------------------------------------------------
st.header("3 · Waveform")

# Colour scheme (matches Audacity's default dark track colours)
C1_FILL  = "rgba(74, 158, 255, 0.35)"   # Mic 1 fill  — blue
C1_LINE  = "rgba(74, 158, 255, 0.90)"   # Mic 1 edge
C2_FILL  = "rgba(255, 140, 50,  0.35)"  # Mic 2 fill  — orange
C2_LINE  = "rgba(255, 140, 50,  0.90)"  # Mic 2 edge
BG       = "#1a1a1a"
GRID     = "#2e2e2e"
ZERO     = "#444444"
FONT     = "#cccccc"

def _print_theme(fig: go.Figure) -> go.Figure:
    """Return a deep copy of fig restyled for white-background print output."""
    f = copy.deepcopy(fig)
    f.update_layout(
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color="black"),
    )
    f.update_xaxes(gridcolor="#dddddd", zerolinecolor="#aaaaaa",
                   tickfont=dict(color="black"), title_font=dict(color="black"))
    f.update_yaxes(gridcolor="#dddddd", zerolinecolor="#aaaaaa",
                   tickfont=dict(color="black"), title_font=dict(color="black"))
    return f


def _fig_bytes(fig: go.Figure, fmt: str, width: int, scale: int,
               light: bool) -> bytes | str:
    """Render a Plotly figure to bytes (PNG/SVG) or str (HTML)."""
    f = _print_theme(fig) if light else copy.deepcopy(fig)
    if fmt == "html":
        return f.to_html(include_plotlyjs="cdn", full_html=True)
    return f.to_image(format=fmt, width=width,
                      height=fig.layout.height or 400, scale=scale)


n_bins = st.slider("Display resolution (columns)", 500, 5000, 2000, 100,
                   help="More columns = finer detail, but slower to render.")

t1, mn1, mx1 = waveform_envelope(audio_win.ch1, times, n_bins)
t2, mn2, mx2 = waveform_envelope(audio_win.ch2, times, n_bins)

# Normalise both together so amplitudes are comparable
peak = max(np.abs(mn1).max(), mx1.max(), np.abs(mn2).max(), mx2.max(), 1e-9)
mn1n, mx1n = mn1 / peak, mx1 / peak
mn2n, mx2n = mn2 / peak, mx2 / peak

fig_aud = go.Figure()

# --- zero lines ---
fig_aud.add_hline(y=0,  line=dict(color=ZERO, width=1, dash="dot"))

# --- Mic 2 (drawn first so Mic 1 sits on top) ---
# Upper envelope (invisible border, used as fill reference)
fig_aud.add_trace(go.Scatter(
    x=t2, y=mx2n, mode="lines",
    line=dict(color=C2_LINE, width=0.6),
    name="Mic 2",
    legendgroup="mic2",
    hovertemplate="t=%{x:.4f}s  max=%{y:.3f}<extra>Mic 2</extra>",
))
# Lower envelope filled to upper
fig_aud.add_trace(go.Scatter(
    x=t2, y=mn2n, mode="lines",
    fill="tonexty", fillcolor=C2_FILL,
    line=dict(color=C2_LINE, width=0.6),
    name="Mic 2",
    legendgroup="mic2", showlegend=False,
    hovertemplate="t=%{x:.4f}s  min=%{y:.3f}<extra>Mic 2</extra>",
))

# --- Mic 1 ---
fig_aud.add_trace(go.Scatter(
    x=t1, y=mx1n, mode="lines",
    line=dict(color=C1_LINE, width=0.6),
    name="Mic 1",
    legendgroup="mic1",
    hovertemplate="t=%{x:.4f}s  max=%{y:.3f}<extra>Mic 1</extra>",
))
fig_aud.add_trace(go.Scatter(
    x=t1, y=mn1n, mode="lines",
    fill="tonexty", fillcolor=C1_FILL,
    line=dict(color=C1_LINE, width=0.6),
    name="Mic 1",
    legendgroup="mic1", showlegend=False,
    hovertemplate="t=%{x:.4f}s  min=%{y:.3f}<extra>Mic 1</extra>",
))

# --- Event markers ---
for et in events_ch1:
    fig_aud.add_vline(x=et, line=dict(color="#00ff88", width=1, dash="dot"),
                      annotation_text="●", annotation_font=dict(color="#00ff88", size=10))
for et in events_ch2:
    fig_aud.add_vline(x=et, line=dict(color="#ffdd00", width=1, dash="dot"))

fig_aud.update_layout(
    height=320,
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(color=FONT),
    xaxis=dict(
        title="Time (s)",
        gridcolor=GRID, zerolinecolor=ZERO,
        title_font=dict(color=FONT),
        tickfont=dict(color=FONT),
    ),
    yaxis=dict(
        title="Amplitude (normalised)",
        range=[-1.05, 1.05],
        gridcolor=GRID, zerolinecolor=ZERO,
        title_font=dict(color=FONT),
        tickfont=dict(color=FONT),
        tickvals=[-1, -0.5, 0, 0.5, 1],
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor="#555",
        borderwidth=1,
        font=dict(color=FONT),
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="left", x=0,
    ),
    margin=dict(l=70, r=20, t=40, b=50),
    hovermode="x unified",
)

st.plotly_chart(fig_aud, width="stretch")

st.caption(
    f"Envelope resolution: {n_bins} columns over "
    f"{audio_win.duration_s*1000:.1f} ms "
    f"({len(audio_win.ch1):,} samples · {sr} Hz).  "
    "Zoom with scroll or box-select; double-click to reset."
)

# Separate stacked view (collapsible)
with st.expander("Stacked view (separate tracks)"):
    fig_stack = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=("Mic 1", "Mic 2"),
                               vertical_spacing=0.06)
    for row, (t_b, mn_n, mx_n, cfill, cline, label) in enumerate([
        (t1, mn1n, mx1n, C1_FILL, C1_LINE, "Mic 1"),
        (t2, mn2n, mx2n, C2_FILL, C2_LINE, "Mic 2"),
    ], start=1):
        fig_stack.add_trace(go.Scatter(x=t_b, y=mx_n, mode="lines",
                                        line=dict(color=cline, width=0.6),
                                        name=label, legendgroup=label),
                            row=row, col=1)
        fig_stack.add_trace(go.Scatter(x=t_b, y=mn_n, mode="lines",
                                        fill="tonexty", fillcolor=cfill,
                                        line=dict(color=cline, width=0.6),
                                        showlegend=False),
                            row=row, col=1)
        fig_stack.add_hline(y=0, line=dict(color=ZERO, width=1, dash="dot"),
                            row=row, col=1)

    fig_stack.update_layout(
        height=380, paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=FONT), showlegend=False,
        margin=dict(l=70, r=20, t=40, b=50),
    )
    for ax in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
        fig_stack.update_layout(**{ax: dict(gridcolor=GRID, zerolinecolor=ZERO,
                                            tickfont=dict(color=FONT))})
    fig_stack.update_xaxes(title_text="Time (s)", row=2, col=1)
    st.plotly_chart(fig_stack, width="stretch")

# Playback (browser native audio player)
with st.expander("🎧 Playback"):
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        buf1 = io.BytesIO()
        sf.write(buf1, audio_win.ch1, sr, format="WAV", subtype="FLOAT")
        st.caption("Mic 1")
        st.audio(buf1.getvalue(), format="audio/wav")
    with col_p2:
        buf2 = io.BytesIO()
        sf.write(buf2, audio_win.ch2, sr, format="WAV", subtype="FLOAT")
        st.caption("Mic 2")
        st.audio(buf2.getvalue(), format="audio/wav")

# ---------------------------------------------------------------------------
# Cross-correlation
# ---------------------------------------------------------------------------
st.header("4 · Cross-correlation & TDOA")

# Compute max_lag in samples
if max_lag_auto:
    max_lag_s = sep_m / theoretical_speed(temp_c) + 0.005   # add 5 ms margin
    max_lag_samp = int(max_lag_s * sr)
else:
    max_lag_samp = int(max_lag_ms * 1e-3 * sr)

try:
    result: TDOAResult = compute_tdoa(
        audio_win,
        low_hz=float(low_hz),
        high_hz=float(high_hz),
        do_filter=do_filter,
        use_phat=use_phat,
        max_lag_samples=max_lag_samp,
    )
except Exception as e:
    st.error(f"Correlation failed: {e}"); st.stop()

# Correlation plot
fig_corr = go.Figure()
fig_corr.add_trace(go.Scatter(
    x=result.lags_us, y=result.correlation,
    name="GCC-PHAT" if use_phat else "GCC",
    line=dict(color="#2ca02c", width=1.2),
    hovertemplate="Lag: %{x:.1f} µs<br>Corr: %{y:.4f}<extra></extra>",
))

# Peak marker
fig_corr.add_trace(go.Scatter(
    x=[result.delay_us], y=[result.peak_value],
    mode="markers+text",
    marker=dict(size=12, color="red", symbol="diamond"),
    text=[f"  {result.delay_us:+.2f} µs"],
    textposition="middle right",
    textfont=dict(size=13, color="red"),
    name=f"Peak",
    hovertemplate=f"Delay: {result.delay_us:+.2f} µs<br>Corr: {result.peak_value:.4f}<extra></extra>",
))

fig_corr.add_vline(x=0, line=dict(color="grey", dash="dot", width=1))
fig_corr.update_layout(
    height=320,
    xaxis_title="Lag (µs)",
    yaxis_title="Normalised correlation",
    showlegend=True,
    margin=dict(l=60, r=20, t=20, b=40),
)
st.plotly_chart(fig_corr, width="stretch")

# ---------------------------------------------------------------------------
# Results metrics
# ---------------------------------------------------------------------------
st.header("5 · Results")

speed_result = compute_speed(result, sep_m, temp_c)

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Time delay",
    f"{result.delay_us:+.2f} µs",
    help="Positive → sound reached Mic 1 first (source closer to Mic 1).",
)
col2.metric(
    "Delay (samples)",
    f"{result.delay_samples:+.3f}",
    help="Sub-sample precision via parabolic interpolation.",
)
col3.metric(
    "Correlation peak",
    f"{result.peak_value:.4f}",
    delta=f"SNR {result.snr_db:.1f} dB",
    help="Normalised peak of the cross-correlation (1.0 = perfect match).",
)
col4.metric(
    "Speed of sound",
    f"{speed_result.speed_ms:.1f} m/s" if abs(speed_result.speed_ms) > 1 else "—",
    delta=f"{speed_result.error_pct:+.1f}% vs theory" if abs(speed_result.speed_ms) > 1 else None,
    delta_color="off",
    help="Derived from delay and mic separation. Only valid for on-axis sound.",
)

# Direction indicator
st.subheader("Source direction")
direction = "← closer to Mic 1" if result.delay_us > 0 else ("→ closer to Mic 2" if result.delay_us < 0 else "⊙ centred")
dist_from_centre_m = abs(result.delay_us * 1e-6) * theoretical_speed(temp_c) / 2
bar_pos = 0.5 + (result.delay_us / (sep_m / theoretical_speed(temp_c) * 1e6)) * 0.5
bar_pos = float(np.clip(bar_pos, 0.0, 1.0))

fig_dir = go.Figure()
fig_dir.add_trace(go.Scatter(
    x=[0, sep_m], y=[0, 0],
    mode="lines",
    line=dict(color="lightgrey", width=6),
    showlegend=False,
))
fig_dir.add_trace(go.Scatter(
    x=[0], y=[0], mode="markers+text",
    marker=dict(size=18, color="#1f77b4", symbol="triangle-right"),
    text=["Mic 1"], textposition="bottom center", showlegend=False,
))
fig_dir.add_trace(go.Scatter(
    x=[sep_m], y=[0], mode="markers+text",
    marker=dict(size=18, color="#ff7f0e", symbol="triangle-left"),
    text=["Mic 2"], textposition="bottom center", showlegend=False,
))
src_x = bar_pos * sep_m
fig_dir.add_trace(go.Scatter(
    x=[src_x], y=[0], mode="markers+text",
    marker=dict(size=22, color="red", symbol="star"),
    text=[f"  source\n  ({direction})"],
    textposition="top center",
    showlegend=False,
))
fig_dir.update_layout(
    height=160,
    xaxis=dict(title=f"Position along mic axis (m, sep = {sep_m} m)",
               range=[-0.5, sep_m + 0.5]),
    yaxis=dict(visible=False, range=[-0.5, 0.5]),
    margin=dict(l=20, r=20, t=20, b=50),
)
st.plotly_chart(fig_dir, width="stretch")

# Speed of sound detail
if abs(speed_result.speed_ms) > 1:
    with st.expander("Speed of sound detail"):
        df = pd.DataFrame({
            "Parameter": [
                "Measured delay",
                "Mic separation",
                "Measured speed",
                f"Theoretical ({temp_c}°C)",
                "Error",
            ],
            "Value": [
                f"{speed_result.delay_us:.2f} µs",
                f"{speed_result.separation_m:.2f} m",
                f"{speed_result.speed_ms:.2f} m/s",
                f"{speed_result.theoretical_ms:.2f} m/s",
                f"{speed_result.error_pct:+.2f}%",
            ],
        })
        st.table(df)

# ---------------------------------------------------------------------------
# Multi-event analysis
# ---------------------------------------------------------------------------
fig_hist = None   # may be populated below; referenced by export section
if show_events and (events_ch1 or events_ch2):
    st.header("6 · Event-by-event analysis")
    st.caption(f"Detected {len(events_ch1)} events on Mic 1, "
               f"{len(events_ch2)} on Mic 2.")

    window_s = 0.05   # 50 ms window around each event
    event_results = []

    for i, evt in enumerate(events_ch1[:20]):   # limit to 20
        t0 = max(0, evt - 0.005)
        t1 = min(audio.duration_s, t0 + window_s)
        win = trim_window(audio, t0, t1)
        if len(win.ch1) < 100:
            continue
        try:
            r = compute_tdoa(win, low_hz=float(low_hz), high_hz=float(high_hz),
                             do_filter=do_filter, use_phat=use_phat,
                             max_lag_samples=max_lag_samp)
            event_results.append({
                "Event #": i + 1,
                "Time (s)": f"{evt:.4f}",
                "Delay (µs)": f"{r.delay_us:+.2f}",
                "Peak corr": f"{r.peak_value:.4f}",
                "SNR (dB)": f"{r.snr_db:.1f}",
            })
        except Exception:
            pass

    if event_results:
        st.dataframe(pd.DataFrame(event_results), width="stretch")

        delays_us = [float(e["Delay (µs)"]) for e in event_results]
        fig_hist = go.Figure(go.Histogram(
            x=delays_us, nbinsx=20,
            marker_color="#9467bd",
            hovertemplate="Delay: %{x:.1f} µs<br>Count: %{y}<extra></extra>",
        ))
        fig_hist.add_vline(x=np.mean(delays_us), line=dict(color="red", dash="dash"),
                           annotation_text=f"mean {np.mean(delays_us):.1f} µs")
        fig_hist.update_layout(
            title="Distribution of delays across events",
            xaxis_title="Delay (µs)", yaxis_title="Count",
            height=280, margin=dict(l=60, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_hist, width="stretch")

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
st.header("7 · Export figures")

_catalog: dict[str, tuple[str, go.Figure]] = {
    "waveform":    ("Waveform overlay",    fig_aud),
    "stacked":     ("Stacked view",        fig_stack),
    "correlation": ("Cross-correlation",   fig_corr),
    "direction":   ("Source direction",    fig_dir),
}
if fig_hist is not None:
    _catalog["histogram"] = ("Event histogram", fig_hist)

with st.expander("📥 Select figures and download", expanded=True):

    # ── Component checkboxes ─────────────────────────────────────────────
    st.markdown("**Select components**")
    _ncols = 3
    _ck_cols = st.columns(_ncols)
    _selected: dict[str, bool] = {}
    for i, (key, (label, _)) in enumerate(_catalog.items()):
        with _ck_cols[i % _ncols]:
            _selected[key] = st.checkbox(label, value=True, key=f"exp_ck_{key}")

    st.divider()

    # ── Format & quality ────────────────────────────────────────────────
    _fc, _sc, _lc = st.columns([2, 2, 1])
    with _fc:
        exp_fmt = st.radio(
            "Format",
            ["PNG (high-res)", "SVG (vector)", "HTML (interactive)"],
            horizontal=False,
            help="SVG is infinitely scalable — ideal for print/LaTeX.\n"
                 "PNG at ×3 scale ≈ 300 DPI at A4 width.\n"
                 "HTML is self-contained interactive.",
        )
    with _sc:
        exp_width = st.slider("Export width (px)", 800, 3200, 1400, 100,
                              help="PNG/SVG canvas width before scale factor.")
        if exp_fmt == "PNG (high-res)":
            exp_scale = st.slider("Scale factor (×)", 1, 5, 3,
                                  help="Effective resolution = width × scale. "
                                       "×3 on 1400 px → 4200 px ≈ 300 DPI at A4.")
        else:
            exp_scale = 1
    with _lc:
        exp_light = st.checkbox(
            "White background", value=True,
            help="Replace dark theme with white background and black text — "
                 "required for most print workflows.",
        )

    st.divider()

    # ── Generate downloads ───────────────────────────────────────────────
    _fmt_ext = {"PNG (high-res)": "png", "SVG (vector)": "svg",
                "HTML (interactive)": "html"}
    _fmt_mime = {"png": "image/png", "svg": "image/svg+xml",
                 "html": "text/html"}
    ext  = _fmt_ext[exp_fmt]
    mime = _fmt_mime[ext]

    _active = [(key, label, fig)
               for key, (label, fig) in _catalog.items()
               if _selected.get(key)]

    if not _active:
        st.info("Select at least one component above.")
    else:
        _btn_cols = st.columns(min(len(_active), 3))
        for i, (key, label, fig) in enumerate(_active):
            with _btn_cols[i % 3]:
                try:
                    _data = _fig_bytes(fig, ext, exp_width, exp_scale, exp_light)
                    if isinstance(_data, str):
                        _data = _data.encode()
                    st.download_button(
                        label=f"⬇ {label}",
                        data=_data,
                        file_name=f"tdoa_{key}.{ext}",
                        mime=mime,
                        key=f"dl_{key}",
                        width="stretch",
                    )
                except Exception as ex:
                    st.error(f"{label}: {ex}")
                    if "kaleido" in str(ex).lower() or "orca" in str(ex).lower():
                        st.code("pip install kaleido", language="bash")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    f"Sample rate: {sr} Hz · "
    f"Window: {audio_win.duration_s*1000:.1f} ms · "
    f"Samples: {len(audio_win.ch1):,} · "
    f"Source: {audio.source_file}"
)
