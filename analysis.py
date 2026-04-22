"""
Signal processing for two-microphone TDOA analysis.

All functions are pure (no Streamlit imports) so they can be tested
independently and reused by other frontends.
"""

import numpy as np
import soundfile as sf
from scipy.signal import correlate, butter, sosfilt, resample_poly
from scipy.interpolate import UnivariateSpline
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class AudioData:
    """Two-channel audio ready for analysis."""
    ch1: np.ndarray        # mic 1 samples (float64, normalised –1…1)
    ch2: np.ndarray        # mic 2 samples
    sample_rate: int
    duration_s: float
    source_file: str
    n_channels_original: int


@dataclass
class TDOAResult:
    """Result of a TDOA computation."""
    delay_samples: float       # fractional sample delay (ch2 – ch1)
    delay_us: float            # µs
    correlation: np.ndarray    # full normalised cross-correlation array
    lags_samples: np.ndarray   # lag axis (samples)
    lags_us: np.ndarray        # lag axis (µs)
    peak_index: int            # index into correlation / lags arrays
    peak_value: float          # normalised peak height (0…1)
    snr_db: float              # peak / median of |correlation| in dB


@dataclass
class SpeedResult:
    """Speed of sound derived from TDOA + known mic separation."""
    speed_ms: float            # m/s
    separation_m: float
    delay_us: float
    theoretical_ms: float      # at given temperature
    error_pct: float


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_stereo(path: str | Path) -> AudioData:
    """
    Load a stereo WAV/FLAC/etc.  Both channels returned as float64 –1…1.
    Supports mono files too (ch1 == ch2).
    """
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    ch1 = data[:, 0]
    ch2 = data[:, 1] if data.shape[1] > 1 else data[:, 0]
    return AudioData(
        ch1=ch1, ch2=ch2, sample_rate=sr,
        duration_s=len(ch1) / sr,
        source_file=str(Path(path).name),
        n_channels_original=data.shape[1],
    )


def load_two_mono(path1: str | Path, path2: str | Path) -> AudioData:
    """
    Load two separate mono files as mic1 and mic2.
    Resamples the second file to match the first if sample rates differ.
    """
    d1, sr1 = sf.read(str(path1), dtype="float64", always_2d=True)
    d2, sr2 = sf.read(str(path2), dtype="float64", always_2d=True)
    ch1 = d1[:, 0]
    ch2 = d2[:, 0]
    if sr1 != sr2:
        # Resample ch2 to sr1
        from math import gcd
        g = gcd(sr1, sr2)
        ch2 = resample_poly(ch2, sr1 // g, sr2 // g)
    # Trim to same length
    n = min(len(ch1), len(ch2))
    return AudioData(
        ch1=ch1[:n], ch2=ch2[:n], sample_rate=sr1,
        duration_s=n / sr1,
        source_file=f"{Path(path1).name} + {Path(path2).name}",
        n_channels_original=1,
    )


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def bandpass(signal: np.ndarray, sr: int,
             low_hz: float, high_hz: float) -> np.ndarray:
    """4th-order Butterworth bandpass filter."""
    nyq = sr / 2.0
    low = max(low_hz / nyq, 1e-4)
    high = min(high_hz / nyq, 0.9999)
    sos = butter(4, [low, high], btype="band", output="sos")
    return sosfilt(sos, signal)


def trim_window(audio: AudioData,
                start_s: float, end_s: float) -> AudioData:
    """Return a time-trimmed copy."""
    sr = audio.sample_rate
    i0 = int(start_s * sr)
    i1 = int(end_s   * sr)
    i0 = max(0, i0); i1 = min(len(audio.ch1), i1)
    return AudioData(
        ch1=audio.ch1[i0:i1], ch2=audio.ch2[i0:i1],
        sample_rate=sr,
        duration_s=(i1 - i0) / sr,
        source_file=audio.source_file,
        n_channels_original=audio.n_channels_original,
    )


def normalise(sig: np.ndarray) -> np.ndarray:
    """Zero-mean, unit RMS normalisation."""
    s = sig - sig.mean()
    rms = np.sqrt(np.mean(s ** 2))
    return s / rms if rms > 1e-12 else s


# ---------------------------------------------------------------------------
# Cross-correlation & TDOA
# ---------------------------------------------------------------------------

def compute_tdoa(audio: AudioData,
                 low_hz: float = 100.0,
                 high_hz: float = 8000.0,
                 do_filter: bool = True,
                 use_phat: bool = True,
                 max_lag_samples: int | None = None) -> TDOAResult:
    """
    GCC (Generalised Cross Correlation) with parabolic sub-sample interpolation.

    use_phat=True  — GCC-PHAT: divides the cross-spectrum by its magnitude
                     before IFFT.  Collapses sidelobes into a single sharp peak;
                     robust to large amplitude differences between channels.
    use_phat=False — Plain GCC: time-domain cross-correlation via scipy.

    Returns a TDOAResult with the full correlation array and the estimated
    fractional delay in samples and µs.
    """
    sr = audio.sample_rate
    ch1 = audio.ch1.copy()
    ch2 = audio.ch2.copy()

    if do_filter:
        ch1 = bandpass(ch1, sr, low_hz, high_hz)
        ch2 = bandpass(ch2, sr, low_hz, high_hz)

    ch1 = normalise(ch1)
    ch2 = normalise(ch2)

    n = len(ch1)

    if use_phat:
        # GCC-PHAT via FFT
        n_fft = int(2 ** np.ceil(np.log2(2 * n - 1)))  # next power of 2 ≥ 2n-1
        X1 = np.fft.rfft(ch1, n=n_fft)
        X2 = np.fft.rfft(ch2, n=n_fft)
        Gxy = X1 * np.conj(X2)
        Gxy /= (np.abs(Gxy) + 1e-10)               # PHAT weighting
        cc = np.fft.irfft(Gxy, n=n_fft)
        # Rearrange circular → linear lags -(n-1) … (n-1)
        corr = np.concatenate([cc[n_fft - n + 1:], cc[:n]])
        lags = np.arange(-(n - 1), n)
        if max_lag_samples is not None:
            mask = np.abs(lags) <= max_lag_samples
            corr = corr[mask]
            lags = lags[mask]
        # Normalise to ±1 (PHAT absolute scale is arbitrary)
        peak_abs = np.max(np.abs(corr))
        corr_norm = corr / (peak_abs + 1e-30)
    else:
        # Plain GCC — time-domain cross-correlation
        corr = correlate(ch1, ch2, mode="full")
        lags = np.arange(-(n - 1), n)
        if max_lag_samples is not None:
            mask = np.abs(lags) <= max_lag_samples
            corr = corr[mask]
            lags = lags[mask]
        norm = np.sqrt(np.sum(ch1 ** 2) * np.sum(ch2 ** 2))
        corr_norm = corr / (norm + 1e-30)

    peak_idx = int(np.argmax(corr_norm))
    peak_val = corr_norm[peak_idx]

    # Parabolic sub-sample interpolation
    frac = 0.0
    if 0 < peak_idx < len(corr_norm) - 1:
        y1, y2, y3 = corr_norm[peak_idx - 1], peak_val, corr_norm[peak_idx + 1]
        denom = y1 - 2 * y2 + y3
        if abs(denom) > 1e-12:
            frac = 0.5 * (y1 - y3) / denom

    delay_samples = float(lags[peak_idx]) + frac
    delay_us = delay_samples / sr * 1e6

    # SNR: peak vs median absolute correlation
    snr_db = 0.0
    med = np.median(np.abs(corr_norm))
    if med > 1e-12:
        snr_db = 20 * np.log10(abs(peak_val) / med)

    return TDOAResult(
        delay_samples=delay_samples,
        delay_us=delay_us,
        correlation=corr_norm,
        lags_samples=lags.astype(float),
        lags_us=lags / sr * 1e6,
        peak_index=peak_idx,
        peak_value=float(peak_val),
        snr_db=snr_db,
    )


# ---------------------------------------------------------------------------
# Speed of sound
# ---------------------------------------------------------------------------

def theoretical_speed(temp_c: float) -> float:
    """Speed of sound in dry air at given Celsius temperature (m/s)."""
    return 331.3 * np.sqrt(1.0 + temp_c / 273.15)


def compute_speed(tdoa: TDOAResult,
                  separation_m: float,
                  temp_c: float = 20.0) -> SpeedResult:
    """
    Given a TDOA result and the physical microphone separation,
    derive the speed of sound.

    Only valid when the sound source is on the mic axis (direct path).
    """
    delay_s = abs(tdoa.delay_us) * 1e-6
    if delay_s < 1e-9:
        speed = 0.0
    else:
        speed = separation_m / delay_s

    theory = theoretical_speed(temp_c)
    err_pct = (speed - theory) / theory * 100.0 if theory > 0 else 0.0

    return SpeedResult(
        speed_ms=speed,
        separation_m=separation_m,
        delay_us=tdoa.delay_us,
        theoretical_ms=theory,
        error_pct=err_pct,
    )


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------

def waveform_envelope(signal: np.ndarray, times: np.ndarray,
                       n_bins: int = 2000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a min/max display envelope — the same technique Audacity uses.

    Divides the signal into `n_bins` equal-width columns and returns the
    min, max, and RMS for each column.  This reduces millions of samples to
    a fixed number of points while faithfully representing the waveform shape
    at any zoom level.

    Returns (t_bins, env_min, env_max) — all length n_bins.
    """
    n = len(signal)
    n_bins = min(n_bins, n)
    idx = np.linspace(0, n, n_bins + 1, dtype=int)

    env_min = np.empty(n_bins)
    env_max = np.empty(n_bins)

    for i in range(n_bins):
        chunk = signal[idx[i]: idx[i + 1]]
        if len(chunk) == 0:
            env_min[i] = env_max[i] = 0.0
        else:
            env_min[i] = chunk.min()
            env_max[i] = chunk.max()

    t_bins = (times[idx[:-1]] + times[np.minimum(idx[1:], n - 1)]) / 2.0
    return t_bins, env_min, env_max


def detect_events(signal: np.ndarray, sr: int,
                  threshold_rms_mult: float = 5.0,
                  min_gap_s: float = 0.05) -> list[float]:
    """
    Return timestamps (seconds) of transient events in the signal.
    Uses a short-window RMS envelope with a rising-edge threshold.
    """
    window = max(1, int(sr * 0.002))   # 2 ms window
    rms_env = np.array([
        np.sqrt(np.mean(signal[i:i+window] ** 2))
        for i in range(0, len(signal) - window, window // 2)
    ])
    times_env = np.arange(len(rms_env)) * (window // 2) / sr

    threshold = threshold_rms_mult * np.median(rms_env)
    above = rms_env > threshold

    events = []
    in_event = False
    min_gap_env = min_gap_s / (window / 2 * sr)  # in envelope samples
    last_event = -1e9

    for i, val in enumerate(above):
        if val and not in_event:
            t = float(times_env[i])
            if t - last_event >= min_gap_s:
                events.append(t)
                last_event = t
            in_event = True
        elif not val:
            in_event = False

    return events
