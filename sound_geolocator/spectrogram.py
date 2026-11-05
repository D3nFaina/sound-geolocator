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
from scipy.signal import butter, sosfiltfilt


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
                        fmin: float = 20.0, fmax: float | None = None, chunk_s: float = 300.0,
                        progress=None) -> Spectrogram:
    """STFT power in dB (float32). Long windows give the fine frequency resolution
    needed to see sub-percent Doppler shifts (1 s window -> 1 Hz bins).

    Long recordings are processed in chunks of ``chunk_s`` seconds and only
    the fmin..fmax band is kept, so an hour of audio fits in memory.
    ``progress(fraction)`` is called after each chunk.
    """
    from scipy.signal import get_window

    nper = int(round(window_s * sr))
    hop = max(1, int(round(hop_s * sr)))
    nfft = int(2 ** np.ceil(np.log2(nper)))
    x = np.asarray(x, dtype=np.float32)
    if fmin > 0:
        sos = butter(4, fmin, btype="highpass", fs=sr, output="sos")
        x = sosfiltfilt(sos, x).astype(np.float32)
    fmax = min(fmax or sr / 2, sr / 2)
    f = np.fft.rfftfreq(nfft, 1 / sr)
    band = (f >= fmin) & (f <= fmax)
    win = get_window("hann", nper).astype(np.float32)
    scale = 1.0 / np.sum(win) ** 2
    n_frames = 1 + max(0, (len(x) - nper) // hop)
    out = np.empty((band.sum(), n_frames), dtype=np.float32)
    per_chunk = max(1, int(chunk_s * sr) // hop)
    for j0 in range(0, n_frames, per_chunk):
        j1 = min(n_frames, j0 + per_chunk)
        idx = (np.arange(j0, j1) * hop)[:, None] + np.arange(nper)[None, :]
        frames = x[idx] * win
        spec = np.fft.rfft(frames, n=nfft, axis=1)[:, band]
        out[:, j0:j1] = (10 * np.log10(np.abs(spec) ** 2 * scale + 1e-20)).T
        if progress:
            progress(j1 / n_frames)
    t = (np.arange(n_frames) * hop + nper / 2) / sr
    return Spectrogram(t, f[band], out)


def enhance(spec: Spectrogram, hum_window_s: float | None = None, whiten_hz: float = 40.0,
            despeckle: tuple[int, int] = (1, 5), floor_db: float = 0.0) -> Spectrogram:
    """Return an enhanced copy where tonal lines read as positive dB above local background.

    1. Remove stationary components: subtract each frequency row's median
       over time, or over a sliding window of ``hum_window_s`` seconds. The
       sliding version also removes lines that last only minutes (a passing
       car's whine, an engine left idling), which the whole-recording median
       keeps; 30 s suits aircraft sweeps.
    2. Whiten: subtract a running median across frequency in each frame,
       which flattens broadband engine/wind noise and leaves narrow lines.
    3. Despeckle with a small median filter (freq bins x time frames).
    4. Clip below ``floor_db``.
    """
    p = spec.power_db.astype(np.float32)
    dt = np.median(np.diff(spec.times)) if len(spec.times) > 1 else 1.0
    df = np.median(np.diff(spec.freqs)) if len(spec.freqs) > 1 else 1.0

    if hum_window_s is None:
        p -= np.median(p, axis=1, keepdims=True)
    else:
        p -= sliding_row_median(p, dt, hum_window_s)

    wf = max(3, int(round(whiten_hz / df)) | 1)
    p -= median_filter(p, size=(wf, 1), mode="nearest")

    if despeckle and max(despeckle) > 1:
        p = median_filter(p, size=despeckle, mode="nearest")
    p = np.maximum(p, floor_db)
    return Spectrogram(spec.times, spec.freqs, p)


def sliding_row_median(p: np.ndarray, dt: float, window_s: float, block_s: float = 2.0) -> np.ndarray:
    """Running median of every frequency row over ``window_s`` seconds (approximate, fast).

    Each row is first reduced to medians of short blocks (``block_s``), then a
    running median over the blocks spanning ``window_s`` is taken and
    interpolated back to every frame. A tone that stays in one bin longer than
    about half the window (hum, a whine, a parked engine) is removed; a Doppler
    sweep that crosses the bin in less time is kept.
    """
    F, T = p.shape
    nb = max(1, int(round(block_s / dt)))
    w = max(3, int(round(window_s / (nb * dt))) | 1)
    nblk = int(np.ceil(T / nb))
    if nblk < 3:
        return np.median(p, axis=1, keepdims=True) * np.ones((1, T), np.float32)
    pad = nblk * nb - T
    q = np.concatenate([p, p[:, -1:].repeat(pad, 1)], 1) if pad else p
    blk = np.median(q.reshape(F, nblk, nb), axis=2)
    run = median_filter(blk, size=(1, min(w, nblk | 1)), mode="nearest")
    tb = (np.arange(nblk) + 0.5) * nb - 0.5  # block centres, in frames
    j = np.clip(np.searchsorted(tb, np.arange(T)) - 1, 0, nblk - 2)
    fr = np.clip((np.arange(T) - tb[j]) / nb, 0.0, 1.0).astype(np.float32)
    return run[:, j] * (1 - fr) + run[:, j + 1] * fr


def smooth_time(spec: Spectrogram, seconds: float) -> Spectrogram:
    dt = np.median(np.diff(spec.times))
    n = max(1, int(round(seconds / dt)))
    return Spectrogram(spec.times, spec.freqs, uniform_filter1d(spec.power_db, n, axis=1))
