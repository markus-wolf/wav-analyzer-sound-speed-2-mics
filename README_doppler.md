# Doppler Effect Speed Analyser

Extract the **maximum speed of a moving sound source** — or inversely estimate
the **speed of sound** — from a single-microphone WAV/M4A recording of a source
moving past the microphone (e.g. a buzzer on a swinging pendulum).

```
streamlit run doppler_app.py
```

---

## Purpose

When a sound source moves relative to a stationary observer, the observed
frequency shifts above the true frequency while the source approaches and below
it while retreating — the Doppler effect.  By measuring the peak and trough of
the observed frequency the app recovers the source speed (or speed of sound)
without any specialised hardware: a phone recording and a known-frequency
buzzer are sufficient.

---

## Physical setup

```
              pendulum swing arc
         ←———————————————————————→
    [extreme]     [bottom]    [extreme]
         \           |           /
          \          |          /
           \         ↓         /
            ——————[buzzer]————
                    ↑
               max speed here
                    ↓
               [microphone]  ← offset along the swing direction
```

**Critical geometry — mic offset along the path:**
The microphone sits at the same height as the swing bottom, displaced slightly
*in the direction of motion* (not to the side).  At the lowest point of the
swing the pendulum bob has both its **maximum speed** and its velocity vector
pointing **directly toward then away from the mic**.  Both conditions are
satisfied simultaneously, so the Doppler shift is maximised at that single
instant and the frequency track shows a sharp **peak → drop → trough** rather
than two flat plateaus.

---

## Architecture

`doppler_app.py` is a self-contained Streamlit application.  All signal
processing is embedded directly in the file alongside the UI.

### Data flow

```
WAV / M4A file
    │
    ▼
load_audio                    mono float64, normalised −1…1
    │                         (scipy for WAV; pydub/ffmpeg for M4A)
    ▼
[analysis window slice]       user-selected time range
    │
    ▼
compute_spectrogram           scipy.signal.spectrogram, Hann window
    │
    ▼
track_frequency               sub-bin parabolic interpolation + smoothing
    │
    ▼
auto_detect_windows           locate frequency peak & trough automatically
    │
    ▼
[user adjusts windows]        sliders bracket the peak and trough
    │
    ▼
doppler_calc                  ratio → β → v_source or v_sound
    │
    ▼
speed profile                 v_radial(t) reconstructed for every frame
```

---

## Algorithms

### Short-Time Fourier Transform (spectrogram)

```python
scipy.signal.spectrogram(data, fs=rate, nperseg=N, noverlap=N*overlap, window='hann')
```

A Hann-windowed STFT converts the 1-D audio signal into a 2-D
time–frequency power map.  Key settings:

| Parameter | Effect |
|---|---|
| `nperseg` (FFT window) | Larger → finer frequency resolution, coarser time resolution |
| `overlap` | Higher (e.g. 95 %) → more time frames → smoother frequency track |

**Recommended:** 8192 samples at 44.1 kHz → bin width ≈ 5.4 Hz/bin.

### Sub-bin parabolic interpolation

The spectral peak bin `k` is refined by fitting a parabola to the three
surrounding bins:

```
δ = 0.5 × (S[k−1] − S[k+1]) / (S[k−1] − 2·S[k] + S[k+1])

f_interp = (k + δ) × fs / N
```

This achieves sub-Hz effective frequency resolution independently of the FFT
size — essential for a 1 kHz source at 2 m/s where the Doppler shift is only
≈ 6 Hz per side.

### Frequency track smoothing

Two-stage:
1. **Median filter** (width ≈ target smoothing time in frames) — removes
   impulsive bin-jumping without distorting the shape.
2. **Savitzky-Golay filter** (polynomial order 3) — further smooths while
   preserving peaks and troughs.

The smoothed track is used for display and transition detection; the raw
interpolated track is used for peak/trough measurement to avoid flattening.

### Automatic window detection

1. Compute the gradient of the (heavily smoothed) frequency track.
2. Find the index of maximum negative gradient → **crossing point** where the
   source passes the mic.
3. Search *before* the crossing for the frequency maximum → centre of the
   **approach peak window**.
4. Search *after* the crossing for the frequency minimum → centre of the
   **retreat trough window**.

### Doppler analysis

Given the peak and trough observed frequencies:

```
r = f_peak / f_trough = (v_c + v_max) / (v_c − v_max)

β = v_max / v_c = (r − 1) / (r + 1)
```

Two operating modes:

| Mode | Known | Computed |
|---|---|---|
| Source speed | v_c (speed of sound) | v_max = β · v_c |
| Speed of sound | v_max (source speed) | v_c = v_max / β |

The source speed can be supplied directly or derived from a **pendulum release
height** via energy conservation (air resistance neglected):

```
v_max = √(2 g h)   →   h = v_max² / (2g)
```

### Instantaneous radial speed profile

Every spectrogram frame is converted to a radial speed via the inverse Doppler
formula:

```
v_radial(t) = v_c · (1 − f₀ / f_obs(t))
```

where `f₀ = f_peak · (1 − β)` is the estimated true source frequency.
Positive values indicate the source is approaching; negative values indicate
retreat.

---

## UI walkthrough

| Section | What to do |
|---|---|
| **Sidebar — Frequency band** | Narrow the band to ±50–100 Hz around the source tone to suppress noise. |
| **Sidebar — FFT window** | Use 8192 or 16384 for a ~1 kHz source; smaller values for faster sources with large shifts. |
| **Sidebar — Overlap** | 90–95 % gives a smooth track. |
| **Sidebar — Track smoothing** | Keep short enough (50–100 ms) that the sharp peak is not flattened. |
| **Sidebar — Mode** | Choose whether to find source speed or speed of sound. |
| **Sidebar — Release height** | Optionally compute pendulum max speed from drop height instead of entering it manually. |
| **1 · Load recording** | Upload WAV or M4A. The file is collapsed to mono if stereo. |
| **2 · Waveform + window** | Drag the range slider to crop to one swing cycle. Fine-tune with the number inputs. |
| **3 · Spectrogram** | The red overlay is the tracked frequency.  Inspect the spectrogram to confirm the tone is visible and the track follows it. |
| **4 · Doppler analysis** | Two shaded windows are placed automatically. Adjust sliders so the green window brackets the frequency peak and the red window brackets the trough. The diamonds mark the measured extremes. |
| **5 · Results** | f_peak, f_trough, ratio, β, computed speed or speed of sound, and the full radial speed profile. |
| **Physics expander** | Full derivation of all formulas and geometry notes. |

---

## Key formulas

```
Doppler (approaching):   f_obs = f₀ · v_c / (v_c − v_s · cosθ)
Doppler (retreating):    f_obs = f₀ · v_c / (v_c + v_s · cosθ)

Frequency ratio:         r = f_peak / f_trough = (v_c + v_max) / (v_c − v_max)

Speed ratio:             β = (r − 1) / (r + 1)

Source speed:            v_max = β · v_c
Speed of sound:          v_c   = v_max / β
True source frequency:   f₀    = f_peak · (1 − β)

Pendulum max speed:      v_max = √(2 g h)       [h = release height, g = 9.807 m/s²]
Release height:          h     = v_max² / (2 g)

Speed of sound (dry air, temperature T in °C):
                         v_c   = 331.3 × √(1 + T / 273.15)
```

---

## Expected accuracy

For a 1 kHz buzzer on a pendulum released from ~2 m:

| Source of error | Typical effect on h |
|---|---|
| Spectrogram peak/trough spread (±2 Hz) | ±0.1–0.2 m |
| Speed-of-sound temperature offset (±5 °C) | ±0.3 % in v → ±0.6 % in h |
| Mic slightly off-axis | underestimates v → underestimates h |
| Air resistance (neglected) | reduces actual v → underestimates h |

Overall: **5–15 % accuracy** on height/speed without calibration is typical.

---

## Tips for best results

- Use a **continuous, single-frequency tone** (buzzer, tuning fork, signal
  generator + speaker) — a pure tone gives the cleanest spectrogram track.
- **Narrow the frequency band** in the sidebar to just around the source
  frequency to remove harmonics and noise from the tracker.
- **Increase the FFT window** (8192 or 16384) for small Doppler shifts
  (slow sources, high frequencies).
- **Reduce smoothing** if the peak/trough look flattened in the frequency
  track detail view.
- Use the **analysis window** to isolate a single swing pass and exclude
  the start/stop transients at the ends of the recording.
- **Multiple recordings** of the same swing, averaged, reduce measurement
  scatter significantly.
- For the speed-of-sound mode, weigh the pendulum bob to minimise air-
  resistance losses and use as large a swing height as safely possible.
