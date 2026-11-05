"""Doppler curve extraction from an (enhanced) spectrogram.

A curve is picked inside a user-chosen box (time range x frequency band),
because a real recording often holds several aircraft, harmonics and other
tonal sources. Inside the box the best continuous ridge is found with
dynamic programming (Viterbi): maximise summed line strength while
penalising frequency jumps between frames. Peaks are refined to sub-bin
accuracy with parabolic interpolation, and weak frames are dropped.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .spectrogram import Spectrogram


@dataclass
class DopplerCurve:
    """Observed tone frequency over time. ``t`` is seconds from the recording start."""

    t: np.ndarray
    f: np.ndarray
    strength: np.ndarray | None = None
    name: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        d["t"] = [float(x) for x in self.t]
        d["f"] = [None if not np.isfinite(x) else float(x) for x in self.f]
        d["strength"] = None if self.strength is None else [float(x) for x in self.strength]
        return d

    @classmethod
    def from_dict(cls, d):
        f = np.array([np.nan if x is None else x for x in d["f"]], float)
        s = None if d.get("strength") is None else np.asarray(d["strength"], float)
        return cls(np.asarray(d["t"], float), f, s, d.get("name", ""), d.get("meta", {}))


def save_curves(curves: list[DopplerCurve], path):
    Path(path).write_text(json.dumps([c.to_dict() for c in curves], indent=1))


def load_curves(path) -> list[DopplerCurve]:
    return [DopplerCurve.from_dict(d) for d in json.loads(Path(path).read_text())]


def _viterbi_slope(S, max_slope_bins, curv_penalty, seed=None, max_rise_bins=None):
    """Best path through S (F x T) with state (bin, slope).

    The slope (bins per frame) may change by at most 1 between frames, with
    ``curv_penalty`` per change. Penalising curvature instead of slope lets
    the path follow steep Doppler sweeps and carry straight through places
    where another line crosses. ``seed=(frame, lo_bin, hi_bin)`` forces the
    path through that frame inside [lo_bin, hi_bin]. ``max_rise_bins`` limits
    upward slopes (a single pass only ever falls in pitch).
    """
    F, T = S.shape
    K = max_slope_bins
    up = K if max_rise_bins is None else min(K, max_rise_bins)
    ns = K + up + 1  # slopes -K .. +up
    NEG = -1e18
    score = np.tile(S[:, 0][:, None], (1, ns))  # (F, ns)
    back_b = np.zeros((T, F, ns), dtype=np.int32)
    back_k = np.zeros((T, F, ns), dtype=np.int8)
    bins = np.arange(F)
    for j in range(1, T):
        if seed is not None and j - 1 == seed[0]:
            m = np.full(F, NEG)
            m[seed[1]:seed[2] + 1] = 0
            score = score + m[:, None]
        new = np.full((F, ns), NEG)
        nb = np.zeros((F, ns), dtype=np.int16)
        nk = np.zeros((F, ns), dtype=np.int8)
        for k in range(ns):  # new slope index
            slope = k - K
            src = bins - slope  # previous bin
            ok = (src >= 0) & (src < F)
            best = np.full(F, NEG)
            bk = np.zeros(F, dtype=np.int8)
            for dk in (-1, 0, 1):
                kp = k + dk
                if 0 <= kp < ns:
                    cand = np.full(F, NEG)
                    cand[ok] = score[src[ok], kp] - (curv_penalty if dk else 0.0)
                    upd = cand > best
                    best[upd] = cand[upd]
                    bk[upd] = kp
            new[:, k] = best + S[:, j]
            nb[:, k] = np.clip(src, 0, F - 1)
            nk[:, k] = bk
        score = new
        back_b[j], back_k[j] = nb, nk
    if seed is not None and seed[0] == T - 1:
        m = np.full(F, NEG)
        m[seed[1]:seed[2] + 1] = 0
        score = score + m[:, None]
    b, k = np.unravel_index(int(np.argmax(score)), score.shape)
    path = np.zeros(T, dtype=int)
    path[-1] = b
    for j in range(T - 1, 0, -1):
        b, k = back_b[j, b, k], back_k[j, b, k]
        path[j - 1] = b
    return path


def _refine(S, freqs, path, reach=2):
    F = S.shape[0]
    df = float(np.median(np.diff(freqs)))
    f = np.empty(len(path))
    strength = np.empty(len(path))
    for j, b in enumerate(path):
        lo, hi = max(0, b - reach), min(F, b + reach + 1)
        b = lo + int(np.argmax(S[lo:hi, j]))
        strength[j] = S[b, j]
        if 0 < b < F - 1:
            y0, y1, y2 = S[b - 1, j], S[b, j], S[b + 1, j]
            den = y0 - 2 * y1 + y2
            delta = 0.5 * (y0 - y2) / den if den < 0 else 0.0
            f[j] = freqs[b] + np.clip(delta, -0.5, 0.5) * df
        else:
            f[j] = freqs[b]
    return f, strength


def track_ridge(spec: Spectrogram, fmin: float, fmax: float, tmin: float | None = None,
                tmax: float | None = None, max_slope_hz_s: float = 8.0, curv_penalty_db: float = 4.0,
                saturate_db: float = 12.0, min_strength_db: float = 3.0, max_rise_hz_s: float | None = 0.5,
                seed: tuple[float, float] | None = None, name: str = "", **_ignored) -> DopplerCurve:
    """Extract the smoothest visible tonal ridge inside the box [tmin,tmax] x [fmin,fmax].

    The tracker works on a frequency grid fine enough that one step per frame
    equals a slope of 1 Hz/s, and penalises each 1 Hz/s change of slope by
    ``curv_penalty_db``. Line strength is capped at ``saturate_db`` so that a
    brighter line crossing ours is no reason to jump onto it.

    A single aircraft pass only ever falls in pitch, so by default the ridge
    may rise by at most ``max_rise_hz_s`` (set None to allow any direction,
    e.g. for a turning or climbing aircraft).

    seed=(t, f) forces the ridge through that point, to choose between
    several lines in the same box.
    """
    sub = spec.band(fmin, fmax)
    sub = sub.window(-np.inf if tmin is None else tmin, np.inf if tmax is None else tmax)
    S = sub.power_db
    F, T = S.shape
    if F < 3 or T < 3:
        raise ValueError("selection box is too small")
    df = float(np.median(np.diff(sub.freqs)))
    hop = float(np.median(np.diff(sub.times)))
    fine_df = min(df, hop * 1.0)  # 1 fine bin per frame == 1 Hz/s
    fine = np.arange(sub.freqs[0], sub.freqs[-1] + 1e-9, fine_df)
    E = np.empty((fine.size, T))
    for j in range(T):
        E[:, j] = np.interp(fine, sub.freqs, S[:, j])
    E = np.minimum(E, saturate_db)  # hard cap: every clearly visible line is worth the same
    K = max(1, int(np.ceil(max_slope_hz_s * hop / fine_df)))
    if fine.size * T * (2 * K + 1) > 4e8:
        raise ValueError("selection too large to trace: zoom in or draw a smaller box around one pass")
    pen = curv_penalty_db * (fine_df / hop)  # per fine-slope step
    sd = None
    if seed is not None:
        j0 = int(np.argmin(np.abs(sub.times - seed[0])))
        m = np.abs(sub.freqs - seed[1]) <= 5.0
        if not m.any():
            raise ValueError("seed frequency outside the band")
        fpk = sub.freqs[np.where(m)[0][np.argmax(S[m, j0])]]
        lo = int(np.searchsorted(fine, fpk - df))
        hi = int(np.searchsorted(fine, fpk + df))
        sd = (j0, max(0, lo), min(fine.size - 1, hi))
    up = None if max_rise_hz_s is None else int(np.ceil(max_rise_hz_s * hop / fine_df))
    path = _viterbi_slope(E, K, pen, sd, up)
    coarse = np.clip(np.round((fine[path] - sub.freqs[0]) / df).astype(int), 0, F - 1)
    f, strength = _refine(S, sub.freqs, coarse)
    f[strength < min_strength_db] = np.nan
    return DopplerCurve(sub.times.copy(), f, strength, name,
                        {"fmin": fmin, "fmax": fmax, "tmin": tmin, "tmax": tmax,
                         "seed": None if seed is None else list(seed)})


def clean_curve(curve: DopplerCurve, max_dev_hz: float = 2.0, window: int = 9, min_points: int = 10) -> DopplerCurve:
    """Drop outliers relative to a running median; keep the curve's time grid (NaN = dropped)."""
    from scipy.ndimage import median_filter

    f = curve.f.copy()
    ok = np.isfinite(f)
    if ok.sum() < min_points:
        return curve
    filled = np.interp(curve.t, curve.t[ok], f[ok])
    med = median_filter(filled, size=window, mode="nearest")
    f[np.abs(f - med) > max_dev_hz] = np.nan
    return DopplerCurve(curve.t, f, curve.strength, curve.name, curve.meta)


def trace_from_seed(spec: Spectrogram, t_seed: float, f_seed: float, tmin: float | None = None,
                    tmax: float | None = None, max_ratio: float = 2.5, min_strength_db: float = 3.0,
                    max_missing_s: float = 5.0, name: str = "", **kw) -> DopplerCurve:
    """Follow the line through one clicked point (t_seed, f_seed).

    Runs the curvature-penalised ridge tracker in a band f_seed/max_ratio ..
    f_seed*max_ratio, forced through the seed. The default 2.5 covers a whole
    sweep, approach to departure ((c+v)/(c-v)), for aircraft up to about
    150 m/s, typical when low enough to be heard well; raise it for faster
    aircraft. Then keeps the stretch around the seed where the line is
    visible, allowing gaps up to ``max_missing_s``.
    """
    c = track_ridge(spec, f_seed / max_ratio, f_seed * max_ratio, tmin, tmax, seed=(t_seed, f_seed),
                    min_strength_db=min_strength_db, name=name, **kw)
    ok = np.isfinite(c.f)
    j0 = int(np.argmin(np.abs(c.t - t_seed)))
    hop = float(np.median(np.diff(c.t)))
    gap = max(1, int(round(max_missing_s / hop)))

    def walk(step):
        j, last = j0, j0
        while 0 <= j < len(c.t):
            if ok[j]:
                last = j
            elif abs(j - last) > gap:
                break
            j += step
        return last

    a, b = walk(-1), walk(1)
    sl = slice(a, b + 1)
    meta = dict(c.meta, seed=[t_seed, f_seed])
    return DopplerCurve(c.t[sl], c.f[sl], c.strength[sl], name, meta)
