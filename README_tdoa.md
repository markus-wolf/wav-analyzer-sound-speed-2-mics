# Two-Microphone TDOA Analyser

Measure the **speed of sound** from a stereo (or dual-mono) recording by computing
the Time Difference Of Arrival (TDOA) between two microphones placed at a known
separation.

```
streamlit run app.py
```

---

## Purpose

A sound pulse (clap, snap, buzzer tone, etc.) travels from a source to two
microphones.  Because the mics are at different distances from the source, the
same wavefront arrives at a slightly different time at each mic.  By measuring
that delay and knowing the physical distance between the mics, you can recover
the speed of sound:

```
v = d / Δt
```

where `d` is the mic separation (m) and `Δt` is the TDOA (s).

---

## Hardware setup

```
  [source]
     |
     |  (on-axis, equidistant from each mic is not required)
     |
  [Mic 1] ←——— d metres ———→ [Mic 2]
```

- Place both mics along the same axis as the sound source.
- Record to a **stereo WAV** (both mics on L/R channels) or two separate mono
  WAV files.
- Any sample rate works; 44.1 kHz or 48 kHz is recommended.
- Mic separation of 1–10 m gives comfortably measurable delays
  (≈ 3–30 ms at 343 m/s).

---

## Architecture

```
app.py          — Streamlit UI (all display logic)
analysis.py     — Pure signal-processing library (no Streamlit imports)
```

`analysis.py` is fully independent and can be imported by other scripts or
test harnesses without Streamlit.

### Data flow

```
WAV file(s)
    │
    ▼
load_stereo / load_two_mono          AudioData(ch1, ch2, sample_rate, …)
    │
    ▼
trim_window                          slice to user-selected time window
    │
    ▼
bandpass (Butterworth, order 4)      remove out-of-band noise
    │
    ▼
normalise                            zero-mean, unit-RMS per channel
    │
    ▼
compute_tdoa                         GCC or GCC-PHAT → TDOAResult
    │
    ▼
compute_speed                        v = d / Δt → SpeedResult
```

---

## Algorithms

### GCC-PHAT (Generalised Cross-Correlation with Phase Transform)

The preferred method.  Steps:

1. FFT both channels to length `2^⌈log₂(2N−1)⌉` (zero-padded for linear
   correlation).
2. Compute the cross-spectrum: `G(f) = X₁(f) · X₂*(f)`
3. Apply PHAT weighting — divide by the magnitude:
   `G̃(f) = G(f) / |G(f)|`
   This whitens the spectrum so every frequency bin contributes equally,
   collapsing sidelobes into a single sharp peak regardless of the spectral
   shape of the source.
4. IFFT → circular cross-correlation, re-arranged to linear lags.

### Plain GCC

Conventional time-domain cross-correlation via `scipy.signal.correlate`,
normalised by the product of RMS values.  Less robust when channels have very
different amplitudes or strong tonal components.

### Sub-sample interpolation (parabolic)

The integer-bin lag is refined by fitting a parabola to the three samples
around the peak:

```
δ = 0.5 × (y₋₁ − y₊₁) / (y₋₁ − 2y₀ + y₊₁)
Δt_frac = (k + δ) / fs
```

This pushes timing resolution well below one sample period (e.g. < 5 µs at
48 kHz) without any up-sampling.

### SNR metric

```
SNR = 20 log₁₀( peak / median(|correlation|) )   [dB]
```

A high SNR (> 20 dB) indicates a clean, reliable delay estimate.
Low SNR typically means the window contains noise, multiple reflections,
or the source is not on-axis.

### Waveform display (envelope)

Rather than plotting every sample, the signal is divided into `N` equal columns
and the per-column min/max is plotted as a filled band — the same technique
used by Audacity.  This gives a faithful waveform representation at any zoom
level with a fixed number of draw calls.

### Event detection

A 2 ms sliding RMS envelope is compared to a threshold of
`k × median(RMS)`.  Rising edges above threshold are reported as event
timestamps, with a minimum gap to suppress double-triggers.

---

## UI walkthrough

| Section | What to do |
|---|---|
| **1 · Load recording** | Upload a stereo WAV (both mics) or two mono WAVs. A synthetic demo can be generated in-browser. |
| **Sidebar — Mic geometry** | Enter the physical mic separation in metres and the air temperature. |
| **Sidebar — Bandpass filter** | Enable and set cut-off frequencies to isolate the source signal. |
| **Sidebar — Correlation** | Toggle GCC-PHAT (recommended) vs plain GCC; set max lag. |
| **2 · Analysis window** | Drag the range slider to isolate a single event (clap, pulse). Narrowing the window removes bias from multiple reflections. |
| **3 · Waveform** | Inspect both channels; use the stacked view or zoom to see the time offset. |
| **4 · Cross-correlation** | The peak position on the lag axis is Δt. The red diamond marks the interpolated peak. |
| **5 · Results** | TDOA in µs, speed of sound in m/s, % deviation from theoretical. The direction diagram shows which mic was closer. |
| **6 · Event analysis** | (Optional) Detects multiple claps/events and histograms their individual delay estimates. |
| **7 · Export** | Download any plot as PNG, SVG, or self-contained HTML. |

---

## Key formulas

```
Speed of sound (dry air):    v = 331.3 × √(1 + T / 273.15)   [m/s, T in °C]

Measured speed:              v = d / |Δt|

TDOA from correlation peak:  Δt = (k + δ) / fs               [s]
```

---

## Tips for best results

- Use an **impulsive source** (clap, wooden block tap, starter pistol) —
  GCC-PHAT performs best with broadband transients.
- **Isolate one event** using the analysis window slider; multiple overlapping
  events bias the peak.
- Enable the **bandpass filter** and set it to the dominant frequency range
  of your source to reject low-frequency rumble and high-frequency hiss.
- Keep the source **on the microphone axis** — off-axis sources produce a
  projected delay `Δt = d cos θ / v`, underestimating speed.
- Record in a **large, open space** to minimise reflections (which appear as
  secondary correlation peaks and reduce SNR).
- Check the **SNR metric**: values below ~15 dB suggest the estimate may be
  unreliable.
