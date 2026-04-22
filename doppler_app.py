"""
Doppler Effect Speed Analyser
===============================
Upload a WAV recording of a sound source moving past a microphone
(e.g. a buzzer on a swinging pendulum).  The app extracts the Doppler
frequency shift to compute either:

  • the maximum source speed  (given the speed of sound), or
  • the speed of sound         (given the maximum source speed)

Physics recap
-------------
When a source moves at speed v_s relative to the medium (speed of sound v_c):

    f_approach  =  f₀ · v_c / (v_c − v_s)     (source moving toward mic)
    f_retreat   =  f₀ · v_c / (v_c + v_s)     (source moving away)

Taking the ratio r = f_approach / f_retreat:

    r = (v_c + v_s) / (v_c − v_s)

Solving for β ≡ v_s / v_c:

    β = (r − 1) / (r + 1)

So:
    v_s = β · v_c     (if v_c is known)
    v_c = v_s / β     (if v_s is known)

Run:  streamlit run doppler_app.py
"""

import io

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d
from scipy.signal import medfilt, savgol_filter, spectrogram

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Doppler Speed Analyser",
    page_icon="〜",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_BG   = "#1a1a1a"
DARK_GRID = "#2e2e2e"
DARK_ZERO = "#444444"
DARK_FONT = "#cccccc"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_wav(uploaded_file) -> tuple[int, np.ndarray]:
    """Load WAV, collapse to mono float64 in [−1, 1]."""
    raw = uploaded_file.read()
    rate, data = wavfile.read(io.BytesIO(raw))
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float64)
    peak = np.max(np.abs(data))
    if peak > 0:
        data /= peak
    return rate, data


def compute_spectrogram(
    data: np.ndarray,
    rate: int,
    nperseg: int = 2048,
    overlap_pct: float = 90.0,
    freq_min: float = 0.0,
    freq_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (f_filtered, t, Sxx_filtered) for the chosen frequency band."""
    noverlap = int(nperseg * overlap_pct / 100)
    f, t, Sxx = spectrogram(data, fs=rate, nperseg=nperseg,
                            noverlap=noverlap, window="hann")
    if freq_max is None:
        freq_max = rate / 2.0
    mask = (f >= freq_min) & (f <= freq_max)
    return f[mask], t, Sxx[mask, :]


def track_frequency(
    f: np.ndarray, Sxx: np.ndarray, smooth_ms: float = 50.0, t_spec: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the dominant frequency at each time frame with sub-bin accuracy.

    Uses parabolic interpolation around the spectral peak so frequency
    resolution is not limited to df = fs/N.  Returns (f_interp, f_smooth).

    f_interp  – sub-bin interpolated track (used for peak/trough measurement)
    f_smooth  – lightly smoothed version for display and transition detection
    """
    n_freq, n_time = Sxx.shape
    df = f[1] - f[0] if len(f) > 1 else 1.0

    f_interp = np.empty(n_time)
    for i in range(n_time):
        col = Sxx[:, i]
        k = int(np.argmax(col))
        # Parabolic interpolation: δ = 0.5*(α−γ)/(α−2β+γ)
        if 0 < k < n_freq - 1:
            alpha, beta_v, gamma = col[k - 1], col[k], col[k + 1]
            denom = alpha - 2.0 * beta_v + gamma
            delta = 0.5 * (alpha - gamma) / denom if denom != 0 else 0.0
            delta = float(np.clip(delta, -1.0, 1.0))
        else:
            delta = 0.0
        f_interp[i] = f[k] + delta * df

    # Light smoothing: median to kill outliers, then optional Savitzky-Golay
    n = n_time
    # Target ~smooth_ms of temporal smoothing; fall back to 3 if t_spec absent
    if t_spec is not None and len(t_spec) > 1:
        dt = float(t_spec[1] - t_spec[0])
        med_k = max(3, int(round(smooth_ms * 1e-3 / dt / 2) * 2 + 1))
    else:
        med_k = max(3, n // 20)
    if med_k % 2 == 0:
        med_k += 1
    f_smooth = medfilt(f_interp, min(med_k, n if n % 2 == 1 else n - 1))

    sg_win = max(5, med_k // 2)
    if sg_win % 2 == 0:
        sg_win += 1
    if sg_win < n:
        try:
            f_smooth = savgol_filter(f_smooth, sg_win, 3)
        except Exception:
            pass

    return f_interp, f_smooth


def auto_detect_windows(
    t_spec: np.ndarray, f_smooth: np.ndarray
) -> tuple[float, float, float, float]:
    """Heuristically find the approaching-peak and retreating-trough windows.

    Strategy (pendulum with mic along the path):
      1. Find the steepest frequency drop — this is the crossing point.
      2. Search *before* the crossing for the maximum frequency (f_peak).
      3. Search *after*  the crossing for the minimum frequency (f_trough).
      4. Return windows centred on those extremes, each ±1/8 of total length.
    """
    n = len(f_smooth)
    df = np.gradient(f_smooth, t_spec)
    df_s = uniform_filter1d(df, size=max(3, n // 10))
    trans = int(np.argmin(df_s))

    half = max(1, n // 8)   # half-width of each search/window

    # Peak before the crossing
    search_a = f_smooth[: trans + 1]
    pk_a = int(np.argmax(search_a))
    a_start = max(0, pk_a - half)
    a_end   = min(trans, pk_a + half)

    # Trough after the crossing
    search_r = f_smooth[trans:]
    tr_r = trans + int(np.argmin(search_r))
    r_start = max(trans, tr_r - half)
    r_end   = min(n - 1, tr_r + half)

    return (
        float(t_spec[a_start]), float(t_spec[a_end]),
        float(t_spec[r_start]), float(t_spec[r_end]),
    )


def doppler_calc(
    f_approach: float, f_retreat: float
) -> tuple[float, float, float]:
    """Return (ratio r, beta β, estimated true source frequency f₀)."""
    r = f_approach / f_retreat
    beta = (r - 1.0) / (r + 1.0)
    f0 = f_approach * (1.0 - beta)   # = f_retreat * (1 + beta)
    return r, beta, f0


def dark_layout(fig: go.Figure, height: int = 320, **extra) -> None:
    """Apply consistent dark-theme layout to a Plotly figure."""
    fig.update_layout(
        height=height,
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=DARK_FONT),
        margin=dict(l=60, r=20, t=30, b=50),
        **extra,
    )
    fig.update_xaxes(gridcolor=DARK_GRID, zerolinecolor=DARK_ZERO,
                     tickfont=dict(color=DARK_FONT))
    fig.update_yaxes(gridcolor=DARK_GRID, zerolinecolor=DARK_ZERO,
                     tickfont=dict(color=DARK_FONT))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Frequency band")
    freq_min_inp = st.number_input(
        "Min frequency (Hz)", min_value=0, max_value=20_000,
        value=200, step=50,
        help="Ignore everything below this frequency (removes low-frequency rumble).",
    )
    freq_max_inp = st.number_input(
        "Max frequency (Hz)", min_value=100, max_value=24_000,
        value=4_000, step=100,
        help="Ignore everything above this frequency.",
    )

    st.subheader("Spectrogram resolution")
    nperseg_inp = st.select_slider(
        "FFT window (samples)",
        options=[256, 512, 1024, 2048, 4096, 8192, 16384],
        value=8192,
        help=(
            "Larger window → finer frequency resolution, coarser time resolution.  "
            "For a 1 kHz source at ~2 m/s the Doppler shift is ~6 Hz — "
            "use ≥ 8192 samples so one bin ≈ 5 Hz at 44.1 kHz."
        ),
    )
    overlap_inp = st.slider(
        "Window overlap (%)", min_value=50, max_value=98,
        value=95, step=1,
        help="Higher overlap = more time frames = smoother track.",
    )
    smooth_ms_inp = st.slider(
        "Track smoothing (ms)", min_value=5, max_value=500,
        value=80, step=5,
        help=(
            "Temporal smoothing of the frequency track.  "
            "Keep short enough that the peak and trough are not flattened."
        ),
    )

    st.divider()
    st.subheader("Calculation mode")
    calc_mode = st.radio(
        "What do you want to find?",
        options=["Source speed  (v_s)", "Speed of sound  (v_c)"],
        index=0,
    )

    if calc_mode.startswith("Source"):
        v_sound_inp = st.number_input(
            "Speed of sound (m/s)", min_value=200.0, max_value=500.0,
            value=343.0, step=1.0,
        )
        v_source_inp = None
    else:
        speed_input_mode = st.radio(
            "Specify source speed as",
            options=["Direct (m/s)", "Release height (m)"],
            horizontal=True,
            help=(
                "Release height uses energy conservation "
                "v = √(2gh) to compute the max speed at the swing bottom "
                "(air resistance ignored)."
            ),
        )
        if speed_input_mode == "Direct (m/s)":
            v_source_inp = st.number_input(
                "Known max source speed (m/s)", min_value=0.001, max_value=500.0,
                value=2.0, step=0.1,
            )
        else:
            release_h = st.number_input(
                "Release height above swing bottom (m)",
                min_value=0.001, max_value=100.0,
                value=0.20, step=0.01,
                help="Height h above the lowest point from which the pendulum is released.",
            )
            v_source_inp = float(np.sqrt(2 * 9.80665 * release_h))
            st.metric(
                "Max speed at bottom",
                f"{v_source_inp:.4f} m/s",
                help="v = √(2 g h)",
            )
        v_sound_inp = None

    st.divider()
    st.subheader("Geometry")
    st.caption(
        "**Mic along the path** (offset in swing direction): "
        "max speed and max radial angle coincide at the swing bottom → "
        "the Doppler extremes are a **sharp peak then trough**, not flat plateaus. "
        "Place the approach window around the frequency peak and the "
        "retreat window around the trough."
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("〜 Doppler Effect Speed Analyser")
st.markdown(
    "Upload a WAV recording of a **moving sound source** (e.g. a buzzer on a "
    "swinging pendulum) to measure source speed or the speed of sound via the "
    "Doppler frequency shift."
)

# ---------------------------------------------------------------------------
# 1 · File upload
# ---------------------------------------------------------------------------
st.header("1 · Load recording")

uploaded = st.file_uploader(
    "Upload a mono or stereo WAV file",
    type=["wav"],
    help="Stereo files are collapsed to mono by averaging the two channels.",
)

if uploaded is None:
    st.info(
        "No file loaded yet.  Upload a WAV recording above to begin.\n\n"
        "**Expected signal:** A continuous tone (buzzer / whistle / tuning fork) "
        "recorded while the source moves past the microphone — you should hear the "
        "pitch drop as the source passes.\n\n"
        "**Tip:** A pendulum swinging through the mic's field gives a clear, "
        "repeatable Doppler shift."
    )
    st.stop()

rate, data = load_wav(uploaded)
duration = len(data) / rate
t_audio = np.linspace(0, duration, len(data))

st.success(
    f"**{uploaded.name}** — {rate:,} Hz sample rate · "
    f"{duration:.3f} s · {len(data):,} samples"
)

# Audio player
st.audio(uploaded.getvalue(), format="audio/wav")

# ---------------------------------------------------------------------------
# 2 · Waveform + analysis window
# ---------------------------------------------------------------------------
st.header("2 · Waveform")

ds = max(1, len(t_audio) // 6_000)   # downsample for display speed
fig_wave = go.Figure()
fig_wave.add_trace(go.Scatter(
    x=t_audio[::ds], y=data[::ds],
    mode="lines",
    line=dict(color="rgba(74,158,255,0.8)", width=0.7),
    name="amplitude",
    hovertemplate="t=%{x:.4f} s<br>amp=%{y:.3f}<extra></extra>",
))
dark_layout(fig_wave, height=220)
fig_wave.update_layout(
    xaxis_title="Time (s)",
    yaxis_title="Amplitude (normalised)",
    yaxis=dict(range=[-1.1, 1.1]),
)
st.plotly_chart(fig_wave, width="stretch")

# ---------------------------------------------------------------------------
# 2b · Analysis window selection
# ---------------------------------------------------------------------------
st.subheader("Analysis window")

_step = max(0.001, round(duration / 2000, 3))
t_range = st.slider(
    "Drag handles to select the region to analyse",
    min_value=0.0, max_value=duration,
    value=(0.0, duration),
    step=_step, format="%.3f s",
    help="Only the audio inside this window is sent to the spectrogram and Doppler analysis.",
)
t_win_start, t_win_end = float(t_range[0]), float(t_range[1])

col_t1, col_t2, col_t3 = st.columns([2, 2, 1])
with col_t1:
    t_win_start = st.number_input(
        "Fine-tune start (s)", min_value=0.0, max_value=duration,
        value=t_win_start, step=0.001, format="%.3f", key="t_win_start",
    )
with col_t2:
    t_win_end = st.number_input(
        "Fine-tune end (s)", min_value=0.0, max_value=duration,
        value=t_win_end, step=0.001, format="%.3f", key="t_win_end",
    )
with col_t3:
    st.metric("Window", f"{t_win_end - t_win_start:.3f} s")

if t_win_end <= t_win_start:
    st.error("End must be after start."); st.stop()

# Slice audio to the selected window
i_start = int(t_win_start * rate)
i_end   = int(t_win_end   * rate)
data_win   = data[i_start:i_end]
t_audio_win = t_audio[i_start:i_end]

# Shade the selected window on the waveform
fig_wave.add_vrect(
    x0=t_win_start, x1=t_win_end,
    fillcolor="rgba(255,200,0,0.08)", line_width=0,
)
fig_wave.add_vline(x=t_win_start, line=dict(color="#ffcc00", width=1, dash="dot"))
fig_wave.add_vline(x=t_win_end,   line=dict(color="#ffcc00", width=1, dash="dot"))

# ---------------------------------------------------------------------------
# 3 · Spectrogram + frequency track
# ---------------------------------------------------------------------------
st.header("3 · Spectrogram & frequency track")

with st.spinner("Computing spectrogram…"):
    f_spec, t_spec, Sxx = compute_spectrogram(
        data_win, rate,
        nperseg=nperseg_inp,
        overlap_pct=float(overlap_inp),
        freq_min=float(freq_min_inp),
        freq_max=float(freq_max_inp),
    )
    # t_spec is relative to window start; shift to absolute recording time
    t_spec = t_spec + t_win_start
    f_raw, f_smooth = track_frequency(f_spec, Sxx,
                                      smooth_ms=float(smooth_ms_inp),
                                      t_spec=t_spec)

bin_hz = rate / nperseg_inp
frame_ms = (nperseg_inp * (1.0 - overlap_inp / 100.0)) / rate * 1000
st.caption(
    f"FFT bin width: **{bin_hz:.2f} Hz/bin** · "
    f"Frame step: **{frame_ms:.1f} ms** · "
    f"Sub-bin interpolation enabled (effective resolution < 1 bin)"
)

Sxx_db = 10 * np.log10(Sxx + 1e-12)
vmin = float(np.percentile(Sxx_db, 20))
vmax = float(np.percentile(Sxx_db, 99.5))

fig_spec = go.Figure()
fig_spec.add_trace(go.Heatmap(
    x=t_spec,
    y=f_spec,
    z=Sxx_db,
    zmin=vmin, zmax=vmax,
    colorscale="Viridis",
    colorbar=dict(title=dict(text="dB", font=dict(color=DARK_FONT)),
                  tickfont=dict(color=DARK_FONT)),
    hovertemplate="t=%{x:.3f} s<br>f=%{y:.0f} Hz<br>%{z:.1f} dB<extra></extra>",
))
fig_spec.add_trace(go.Scatter(
    x=t_spec, y=f_smooth,
    mode="lines",
    line=dict(color="red", width=2),
    name="tracked freq",
    hovertemplate="t=%{x:.3f} s<br>f=%{y:.1f} Hz<extra>tracked</extra>",
))
dark_layout(fig_spec, height=350)
fig_spec.update_layout(
    xaxis_title="Time (s)",
    yaxis_title="Frequency (Hz)",
    legend=dict(
        bgcolor="rgba(0,0,0,0.5)", font=dict(color=DARK_FONT),
        x=0.01, y=0.97,
    ),
)
st.plotly_chart(fig_spec, width="stretch")

# Frequency track standalone
with st.expander("Frequency track detail"):
    fig_ftrack = go.Figure()
    fig_ftrack.add_trace(go.Scatter(
        x=t_spec, y=f_raw,
        mode="lines",
        line=dict(color="rgba(100,160,255,0.5)", width=1),
        name="raw peak",
    ))
    fig_ftrack.add_trace(go.Scatter(
        x=t_spec, y=f_smooth,
        mode="lines",
        line=dict(color="orange", width=2),
        name="smoothed",
    ))
    dark_layout(fig_ftrack, height=280)
    fig_ftrack.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color=DARK_FONT)),
    )
    st.plotly_chart(fig_ftrack, width="stretch")

# ---------------------------------------------------------------------------
# 4 · Doppler analysis — window selection
# ---------------------------------------------------------------------------
st.header("4 · Doppler analysis")

# Auto-detect starting windows
auto_a_start, auto_a_end, auto_r_start, auto_r_end = auto_detect_windows(
    t_spec, f_smooth
)

st.markdown(
    "**Mic along the swing path:** the Doppler extremes are a sharp **peak** "
    "(just before the source passes the mic) and a sharp **trough** (just after). "
    "Place each window to bracket that peak/trough — the app takes the "
    "**maximum** frequency in the approach window and the **minimum** in the retreat window."
)

col_a, col_b = st.columns(2, gap="large")

_win_dur  = t_win_end - t_win_start
_win_step = max(0.001, round(_win_dur / 1000, 3))

with col_a:
    st.markdown("##### 📈 Approach peak window  *(highest frequency)*")
    a_start = st.slider(
        "Start (s)", key="a_start",
        min_value=t_win_start, max_value=t_win_end,
        value=float(np.clip(round(auto_a_start, 3), t_win_start, t_win_end)),
        step=_win_step,
    )
    a_end = st.slider(
        "End (s)", key="a_end",
        min_value=t_win_start, max_value=t_win_end,
        value=float(np.clip(round(auto_a_end, 3), t_win_start, t_win_end)),
        step=_win_step,
    )

with col_b:
    st.markdown("##### 📉 Retreat trough window  *(lowest frequency)*")
    r_start = st.slider(
        "Start (s)", key="r_start",
        min_value=t_win_start, max_value=t_win_end,
        value=float(np.clip(round(auto_r_start, 3), t_win_start, t_win_end)),
        step=_win_step,
    )
    r_end = st.slider(
        "End (s)", key="r_end",
        min_value=t_win_start, max_value=t_win_end,
        value=float(np.clip(round(auto_r_end, 3), t_win_start, t_win_end)),
        step=_win_step,
    )

# Validate windows
a_mask = (t_spec >= a_start) & (t_spec <= a_end)
r_mask = (t_spec >= r_start) & (t_spec <= r_end)

if a_mask.sum() == 0:
    st.error("Approach window contains no spectrogram frames — widen the window.")
    st.stop()
if r_mask.sum() == 0:
    st.error("Retreat window contains no spectrogram frames — widen the window.")
    st.stop()

# Use peak of the interpolated (not smoothed) track for maximum accuracy
f_approach_val = float(np.max(f_raw[a_mask]))
f_retreat_val  = float(np.min(f_raw[r_mask]))

if f_approach_val <= f_retreat_val:
    st.warning(
        f"The approaching frequency ({f_approach_val:.1f} Hz) is not higher than the "
        f"retreating frequency ({f_retreat_val:.1f} Hz).  "
        "Check that the windows are placed in the correct phases, "
        "or that the recording actually contains a Doppler shift."
    )

# Annotated frequency track — show both raw interpolated and smoothed
fig_annot = go.Figure()
fig_annot.add_trace(go.Scatter(
    x=t_spec, y=f_raw,
    mode="lines",
    line=dict(color="rgba(100,160,255,0.4)", width=1),
    name="interpolated",
))
fig_annot.add_trace(go.Scatter(
    x=t_spec, y=f_smooth,
    mode="lines",
    line=dict(color="orange", width=2),
    name="smoothed",
))

# Shade approach window
fig_annot.add_vrect(
    x0=a_start, x1=a_end,
    fillcolor="rgba(0,200,100,0.15)",
    line_width=0,
    annotation_text="approach peak", annotation_position="top left",
    annotation_font_color="#00cc66",
)
# Shade retreat window
fig_annot.add_vrect(
    x0=r_start, x1=r_end,
    fillcolor="rgba(255,80,80,0.15)",
    line_width=0,
    annotation_text="retreat trough", annotation_position="top right",
    annotation_font_color="#ff5050",
)
# Mark the actual peak / trough on the raw track
t_a_peak = float(t_spec[a_mask][np.argmax(f_raw[a_mask])])
t_r_trough = float(t_spec[r_mask][np.argmin(f_raw[r_mask])])
fig_annot.add_trace(go.Scatter(
    x=[t_a_peak], y=[f_approach_val],
    mode="markers", marker=dict(size=10, color="#00cc66", symbol="diamond"),
    name=f"f_peak = {f_approach_val:.2f} Hz", showlegend=True,
))
fig_annot.add_trace(go.Scatter(
    x=[t_r_trough], y=[f_retreat_val],
    mode="markers", marker=dict(size=10, color="#ff5050", symbol="diamond"),
    name=f"f_trough = {f_retreat_val:.2f} Hz", showlegend=True,
))
fig_annot.add_hline(
    y=f_approach_val,
    line=dict(color="#00cc66", width=1.2, dash="dash"),
    annotation_text=f"f_peak = {f_approach_val:.2f} Hz",
    annotation_font_color="#00cc66",
    annotation_position="right",
)
fig_annot.add_hline(
    y=f_retreat_val,
    line=dict(color="#ff5050", width=1.2, dash="dash"),
    annotation_text=f"f_trough = {f_retreat_val:.2f} Hz",
    annotation_font_color="#ff5050",
    annotation_position="right",
)

dark_layout(fig_annot, height=300)
fig_annot.update_layout(
    xaxis_title="Time (s)",
    yaxis_title="Frequency (Hz)",
    legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color=DARK_FONT)),
)
st.plotly_chart(fig_annot, width="stretch")

# ---------------------------------------------------------------------------
# 5 · Results
# ---------------------------------------------------------------------------
st.header("5 · Results")

ratio, beta, f0_est = doppler_calc(f_approach_val, f_retreat_val)

# Guard against degenerate cases
if f_approach_val <= 0 or f_retreat_val <= 0 or ratio <= 1.0:
    st.error(
        "Cannot compute a valid result: the frequency ratio must be > 1 "
        "(approaching frequency must exceed retreating frequency)."
    )
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("f_peak (approach)", f"{f_approach_val:.2f} Hz",
            help="Maximum interpolated frequency in the approach window.")
col2.metric("f_trough (retreat)", f"{f_retreat_val:.2f} Hz",
            help="Minimum interpolated frequency in the retreat window.")
col3.metric("f_peak / f_trough", f"{ratio:.5f}",
            help="Doppler frequency ratio r.")

col4, col5, col6 = st.columns(3)
col4.metric("β = v_s / v_c", f"{beta:.5f}",
            help="Fractional source speed (dimensionless).")
col5.metric("True source freq f₀ (est.)", f"{f0_est:.1f} Hz",
            help="f₀ = f_approach × (1 − β)")

if calc_mode.startswith("Source"):
    v_source_calc = beta * v_sound_inp
    col6.metric(
        "Max source speed",
        f"{v_source_calc:.3f} m/s",
        delta=f"{v_source_calc * 3.6:.2f} km/h",
        delta_color="off",
    )

    st.success(
        f"**Max source speed = {v_source_calc:.3f} m/s  "
        f"({v_source_calc * 3.6:.2f} km/h)**  "
        f"using v_c = {v_sound_inp} m/s"
    )
else:
    v_sound_calc = v_source_inp / beta
    col6.metric(
        "Estimated speed of sound",
        f"{v_sound_calc:.2f} m/s",
    )

    # Compare with classic formula
    theoretical_20c = 343.0
    error_pct = (v_sound_calc - theoretical_20c) / theoretical_20c * 100
    st.success(
        f"**Estimated speed of sound = {v_sound_calc:.2f} m/s**  "
        f"(theory at 20 °C: {theoretical_20c} m/s, "
        f"deviation {error_pct:+.1f}%)"
    )

# ---------------------------------------------------------------------------
# Speed-vs-time (pendulum profile)
# ---------------------------------------------------------------------------
st.subheader("Instantaneous radial speed profile")
st.caption(
    "Derived from the Doppler formula at each spectrogram frame, "
    "using the estimated true frequency f₀.  "
    "Positive = approaching, negative = retreating."
)

if calc_mode.startswith("Source"):
    v_c_for_profile = float(v_sound_inp)
else:
    v_c_for_profile = float(v_source_inp / beta)

# v(t) = v_c · (1 − f₀/f(t))  ... from f_obs = f₀·v_c/(v_c − v_r)
# where v_r is the radial velocity (positive toward mic)
with np.errstate(divide="ignore", invalid="ignore"):
    v_radial = v_c_for_profile * (1.0 - f0_est / np.where(f_smooth > 0, f_smooth, np.nan))

fig_speed = go.Figure()
fig_speed.add_hline(y=0, line=dict(color=DARK_ZERO, width=1, dash="dot"))

# Shade approach / retreat regions
fig_speed.add_vrect(x0=a_start, x1=a_end,
                    fillcolor="rgba(0,200,100,0.10)", line_width=0)
fig_speed.add_vrect(x0=r_start, x1=r_end,
                    fillcolor="rgba(255,80,80,0.10)", line_width=0)

fig_speed.add_trace(go.Scatter(
    x=t_spec, y=v_radial,
    mode="lines",
    line=dict(color="#58a6ff", width=1.8),
    name="v_radial(t)",
    hovertemplate="t=%{x:.3f} s<br>v=%{y:.2f} m/s<extra></extra>",
))

# Mark extremes within the selected windows
v_approach_profile = v_radial[a_mask]
v_retreat_profile  = v_radial[r_mask]
if len(v_approach_profile) and not np.all(np.isnan(v_approach_profile)):
    v_peak_approach = float(np.nanmax(v_approach_profile))
    t_peak_approach = float(t_spec[a_mask][np.nanargmax(v_approach_profile)])
    fig_speed.add_trace(go.Scatter(
        x=[t_peak_approach], y=[v_peak_approach],
        mode="markers+text",
        marker=dict(size=10, color="#00cc66", symbol="diamond"),
        text=[f" {v_peak_approach:.2f} m/s"],
        textposition="top right",
        textfont=dict(color="#00cc66"),
        name="peak approach",
        showlegend=False,
    ))

if len(v_retreat_profile) and not np.all(np.isnan(v_retreat_profile)):
    v_peak_retreat = float(np.nanmin(v_retreat_profile))
    t_peak_retreat = float(t_spec[r_mask][np.nanargmin(v_retreat_profile)])
    fig_speed.add_trace(go.Scatter(
        x=[t_peak_retreat], y=[v_peak_retreat],
        mode="markers+text",
        marker=dict(size=10, color="#ff5050", symbol="diamond"),
        text=[f" {v_peak_retreat:.2f} m/s"],
        textposition="bottom right",
        textfont=dict(color="#ff5050"),
        name="peak retreat",
        showlegend=False,
    ))

dark_layout(fig_speed, height=300)
fig_speed.update_layout(
    xaxis_title="Time (s)",
    yaxis_title="Radial speed (m/s)  [+ toward mic]",
    legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(color=DARK_FONT)),
)
st.plotly_chart(fig_speed, width="stretch")

# ---------------------------------------------------------------------------
# Physics explainer
# ---------------------------------------------------------------------------
with st.expander("📐 Physics & formulas"):
    st.markdown(
        r"""
### Doppler effect (moving source, stationary observer)

$$
f_{obs} = f_0 \cdot \frac{v_c}{v_c \mp v_s}
$$

where the **−** applies when the source is **approaching** and the **+**
when it is **retreating**.

---

### Frequency ratio

$$
r = \frac{f_{approach}}{f_{retreat}}
  = \frac{v_c + v_s}{v_c - v_s}
$$

### Solving for speed ratio β = v_s / v_c

$$
\beta = \frac{r - 1}{r + 1}
$$

### Recovering quantities

| Known | Sought | Formula |
|-------|--------|---------|
| v_c   | v_s    | $v_s = \beta \cdot v_c$ |
| v_s   | v_c    | $v_c = v_s / \beta$ |

### True source frequency

$$
f_0 = f_{approach} \cdot (1 - \beta) = f_{retreat} \cdot (1 + \beta)
$$

---

### Pendulum geometry — mic offset along the swing path

The microphone is placed at the same height as the swing bottom,
slightly ahead of the lowest point **in the direction of motion**.

At the bottom of the swing the pendulum bob has **maximum speed** and its
velocity is **horizontal** — pointing directly toward (then away from) the mic.
Both the speed and the radial angle are maximised simultaneously, so the
observed frequency reaches its **extreme values at that single instant**.

The frequency profile therefore looks like a sharp peak → steep drop → sharp trough,
not two flat plateaus.  The correct quantities to extract are:

- $f_{peak}$ — **maximum** frequency in the approach window
- $f_{trough}$ — **minimum** frequency in the retreat window

These correspond to $v_s = v_{max}$ and full radial alignment:

$$
f_{peak}   = f_0 \cdot \frac{v_c}{v_c - v_{max}}, \qquad
f_{trough} = f_0 \cdot \frac{v_c}{v_c + v_{max}}
$$

$$
r = \frac{f_{peak}}{f_{trough}} = \frac{v_c + v_{max}}{v_c - v_{max}},
\qquad \beta = \frac{r-1}{r+1}
$$

### Sub-bin frequency resolution

Standard FFT bin width is $\Delta f = f_s / N$.  For $f_s = 44100$ Hz, $N = 8192$:
$\Delta f \approx 5.4$ Hz per bin — comparable to the expected Doppler shift
at pendulum speeds.

**Parabolic interpolation** fits a parabola to the three spectral bins around
the peak and analytically estimates the true peak location to sub-bin accuracy:

$$
\delta = \frac{1}{2} \cdot \frac{S[k-1] - S[k+1]}{S[k-1] - 2S[k] + S[k+1]},
\qquad f_{interp} = (k + \delta)\,\frac{f_s}{N}
$$
        """
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    f"Sample rate: {rate:,} Hz · Total duration: {duration:.3f} s · "
    f"Analysis window: {t_win_start:.3f} – {t_win_end:.3f} s "
    f"({t_win_end - t_win_start:.3f} s, {len(data_win):,} samples) · "
    f"File: {uploaded.name}"
)
