"""Spectrogram computation and enhancement.

Aircraft tones (fan / propeller blade-passing lines and their harmonics) are
narrow-band lines that drift slowly with Doppler. The enhancement steps here
are designed to make those lines stand out over:

* stationary hum and machinery (removed by per-frequency median subtraction),
* broadband transients such as wind gusts, speech, cars (per-frame median
  subtraction / spectral whitening),
* speckle noise (small 2-D median filter along time).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import butter, sosfiltfilt, stft


@dataclass
class Spectrogram:
    times: np.ndarray  # seconds from recording start, shape (T,)
    freqs: np.ndarray  # Hz, shape (F,)
    power_db: np.ndarray  # shape (F, T)

    def band(self, fmin: float, fmax: float):
        m = (self.freqs >= fmin) & (self.freqs <= fmax)
        return Spectrogram(self.times, self.freqs[m], self.power_db[m])

    def window(self, tmin: float, tmax: float):
        m = (self.times >= tmin) & (self.times <= tmax)
        return Spectrogram(self.times[m], self.freqs, self.power_db[:, m])


def compute_spectrogram(x: np.ndarray, sr: int, window_s: float = 1.0, hop_s: float = 0.25,
                        fmin: float = 20.0, fmax: float | None = None) -> Spectrogram:
    """STFT power in dB. Long windows give the fine frequency resolution
    needed to see sub-percent Doppler shifts (1 s window -> 1 Hz bins)."""
    nper = int(round(window_s * sr))
    hop = max(1, int(round(hop_s * sr)))
    nfft = int(2 ** np.ceil(np.log2(nper)))
    if fmin > 0:
        sos = butter(4, fmin, btype="highpass", fs=sr, output="sos")
        x = sosfiltfilt(sos, x)
    f, t, z = stft(x, fs=sr, window="hann", nperseg=nper, noverlap=nper - hop, nfft=nfft,
                   boundary=None, padded=False)
    p = 10 * np.log10(np.abs(z) ** 2 + 1e-20)
    fmax = fmax or sr / 2
    m = (f >= fmin) & (f <= fmax)
    return Spectrogram(t, f[m], p[m])


def enhance(spec: Spectrogram, hum_window_s: float | None = None, whiten_hz: float = 40.0,
            despeckle: tuple[int, int] = (1, 5), floor_db: float = 0.0) -> Spectrogram:
    """Return an enhanced copy where tonal lines read as positive dB above local background.

    1. Remove stationary components: subtract each frequency row's median
       over time (or over a sliding window of ``hum_window_s`` seconds).
    2. Whiten: subtract a running median across frequency in each frame,
       which flattens broadband engine/wind noise and leaves narrow lines.
    3. Despeckle with a small median filter (freq bins x time frames).
    4. Clip below ``floor_db``.
    """
    p = spec.power_db.copy()
    dt = np.median(np.diff(spec.times)) if len(spec.times) > 1 else 1.0
    df = np.median(np.diff(spec.freqs)) if len(spec.freqs) > 1 else 1.0

    if hum_window_s is None:
        p -= np.median(p, axis=1, keepdims=True)
    else:
        w = max(3, int(round(hum_window_s / dt)) | 1)
        p -= median_filter(p, size=(1, w), mode="nearest")

    wf = max(3, int(round(whiten_hz / df)) | 1)
    p -= median_filter(p, size=(wf, 1), mode="nearest")

    if despeckle and max(despeckle) > 1:
        p = median_filter(p, size=despeckle, mode="nearest")
    p = np.maximum(p, floor_db)
    return Spectrogram(spec.times, spec.freqs, p)


def smooth_time(spec: Spectrogram, seconds: float) -> Spectrogram:
    dt = np.median(np.diff(spec.times))
    n = max(1, int(round(seconds / dt)))
    return Spectrogram(spec.times, spec.freqs, uniform_filter1d(spec.power_db, n, axis=1))
