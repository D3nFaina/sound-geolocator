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


def digitise_along(spec: Spectrogram, guide_t, guide_f, halfwidth_hz: float | None = None,
                   rel_width: float = 0.04, min_strength_db: float = 3.0, clean_hz: float = 2.0,
                   name: str = "") -> DopplerCurve:
    """Digitise one Doppler line inside a corridor around a hand-drawn guide.

    ``guide_t``/``guide_f`` are a few points clicked roughly along the line
    (at least 2). The guide is interpolated smoothly (monotone PCHIP) over
    every spectrogram frame between its first and last point. In each frame
    only the bins within ``halfwidth_hz`` of the guide (default
    ``rel_width`` × frequency, at least 2 bins) are looked at, and the
    strongest one is taken, refined to sub-bin accuracy. Frames where the
    peak is weak, or sits on the edge of the corridor (meaning the line left
    it), are left empty, and then points more than ``clean_hz`` off a
    running median are dropped. Nothing outside the corridor can be picked,
    so the result can't wander onto another line.
    """
    from scipy.interpolate import PchipInterpolator

    gt = np.asarray(guide_t, float)
    gf = np.asarray(guide_f, float)
    o = np.argsort(gt)
    gt, gf = gt[o], gf[o]
    keep = np.concatenate([[True], np.diff(gt) > 1e-6])
    gt, gf = gt[keep], gf[keep]
    if len(gt) < 2:
        raise ValueError("click at least two points along the line")
    sub = spec.window(gt[0], gt[-1])
    times, freqs, S = sub.times, sub.freqs, sub.power_db
    if len(times) < 3:
        raise ValueError("the guide is too short")
    fg = PchipInterpolator(gt, gf)(times) if len(gt) > 2 else np.interp(times, gt, gf)
    df = float(np.median(np.diff(freqs)))
    hw = np.full_like(fg, halfwidth_hz) if halfwidth_hz else rel_width * fg
    hw = np.maximum(hw, 2 * df)
    f = np.full(len(times), np.nan)
    strength = np.full(len(times), np.nan)
    for j in range(len(times)):
        lo = int(np.searchsorted(freqs, fg[j] - hw[j]))
        hi = int(np.searchsorted(freqs, fg[j] + hw[j], side="right"))
        if hi - lo < 3:
            continue
        col = S[lo:hi, j]
        b = int(np.argmax(col))
        strength[j] = col[b]
        if b == 0 or b == len(col) - 1 or col[b] < min_strength_db:
            continue
        y0, y1, y2 = col[b - 1], col[b], col[b + 1]
        den = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / den if den < 0 else 0.0
        f[j] = freqs[lo + b] + np.clip(delta, -0.5, 0.5) * df
    cv = DopplerCurve(times.copy(), f, strength, name,
                      {"guide_t": [float(x) for x in gt], "guide_f": [float(x) for x in gf],
                       "halfwidth_hz": halfwidth_hz, "rel_width": rel_width, "method": "guide"})
    return clean_curve(cv, max_dev_hz=clean_hz) if clean_hz else cv


def _corridor_peaks(S, freqs, centre, hw, min_strength_db):
    """Strongest bin per frame inside centre±hw, refined; NaN where weak or on the edge."""
    df = float(np.median(np.diff(freqs)))
    f = np.full(S.shape[1], np.nan)
    st = np.full(S.shape[1], np.nan)
    for j in range(S.shape[1]):
        lo = int(np.searchsorted(freqs, centre[j] - hw[j]))
        hi = int(np.searchsorted(freqs, centre[j] + hw[j], side="right"))
        if hi - lo < 3:
            continue
        col = S[lo:hi, j]
        b = int(np.argmax(col))
        st[j] = col[b]
        if b == 0 or b == len(col) - 1 or col[b] < min_strength_db:
            continue
        y0, y1, y2 = col[b - 1], col[b], col[b + 1]
        den = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / den if den < 0 else 0.0
        f[j] = freqs[lo + b] + np.clip(delta, -0.5, 0.5) * df
    return f, st


def isolate_curve(raw: Spectrogram, guide_t, guide_f, line_window_s: float = 30.0, corridor_rel: float = 0.08,
                  min_strength_db: float = 3.0, max_slope_hz_s: float | None = None, curv_penalty_db: float = 4.0,
                  c: float | None = None, name: str = "", progress=None, debug=False) -> DopplerCurve:
    """Isolate one Doppler pass from a noisy spectrogram and fit it, from a few rough clicks.

    This is the procedure used by hand on the real recording, automated:

    1. Clean a local piece of the *raw* spectrogram (guide span ± ``line_window_s``,
       guide band ± 30 %): subtract each frequency row's running median over
       ``line_window_s`` seconds. Hum, whines and any tone that sits still for
       more than about half that long vanish, while the sweep, which crosses
       each bin quickly, stays. Then whiten across frequency and despeckle.
    2. Trace the smoothest ridge (curvature-penalised Viterbi, falling pitch
       only) inside a wide corridor: between two neighbouring clicks, every
       frequency from the lower click to the higher one, plus a margin of
       ``corridor_rel`` × f (at least 4 Hz). So two clicks, at the start and
       the end of the pass, are enough when the line is clear.
    3. Fit the straight-flyover Doppler model to that trace (robust loss), and
       use the model curve, not the clicks, as the centre of ever narrower
       corridors (6 %, 4 %, 2.5 %, 1.5 % of f; at least 4, 3, 2, 1.2 Hz). In
       each, trace the ridge again inside the corridor and refit. The model bridges places where the
       line is faint or crossed by another line, so the trace can't step
       onto a horizontal line any more.
    4. Drop points more than 3 robust sigmas off the final model and fit
       again, with error bars.

    Returns the curve on the frame grid between the first and last click
    (NaN where nothing reliable was found) with the fit in ``meta["fit"]``.
    """
    from .doppler import DEFAULT_SPEED_OF_SOUND, fit_flyover, straight_flyover_ratio
    from .spectrogram import enhance

    c = DEFAULT_SPEED_OF_SOUND if c is None else c
    gt = np.asarray(guide_t, float)
    gf = np.asarray(guide_f, float)
    o = np.argsort(gt)
    gt, gf = gt[o], gf[o]
    keep = np.concatenate([[True], np.diff(gt) > 1e-6])
    gt, gf = gt[keep], gf[keep]
    if len(gt) < 2:
        raise ValueError("click at least two points along the line")
    say = progress or (lambda *_: None)

    say(0.05, "cleaning the spectrogram")
    fmin, fmax = gf.min() / 1.3, gf.max() * 1.3
    loc = raw.band(fmin, fmax).window(gt[0] - line_window_s, gt[-1] + line_window_s)
    loc = enhance(Spectrogram(loc.times, loc.freqs, loc.power_db.astype(np.float32)),
                  hum_window_s=line_window_s, whiten_hz=min(40.0, 0.5 * (fmax - fmin)))
    sub = loc.window(gt[0], gt[-1])
    times, freqs, S = sub.times, sub.freqs, np.asarray(sub.power_db, float)
    if len(times) < 8:
        raise ValueError("the guide is too short")
    df = float(np.median(np.diff(freqs)))
    hop = float(np.median(np.diff(times)))
    fine_df = min(df, hop)  # one fine bin per frame == 1 Hz/s
    fine = np.arange(freqs[0], freqs[-1] + 1e-9, fine_df)
    E0 = np.empty((fine.size, len(times)))
    for j in range(len(times)):
        E0[:, j] = np.interp(fine, freqs, S[:, j])
    E0 = np.minimum(E0, 12.0)  # every clearly visible line is worth the same
    if max_slope_hz_s is None:  # steepest click-to-click slope, with room to spare
        max_slope_hz_s = max(8.0, 1.5 * float(np.max(np.abs(np.diff(gf) / np.diff(gt)))))
    K = max(1, int(np.ceil(max_slope_hz_s * hop / fine_df)))
    if fine.size * len(times) * (2 * K + 1) > 4e8:
        raise ValueError("pass too long or too wide to trace at once: click the start and end closer")
    pen = curv_penalty_db * (fine_df / hop)
    rise = 0  # a straight pass only ever falls in pitch

    def ridge(lo_f, hi_f):
        """Smoothest falling ridge inside [lo_f, hi_f] per frame, then the refined peak next to it."""
        E = E0.copy()
        E[(fine[:, None] < lo_f[None, :]) | (fine[:, None] > hi_f[None, :])] = -20.0
        path = fine[_viterbi_slope(E, K, pen, None, rise)]
        f, st = _corridor_peaks(S, freqs, path, np.full(len(times), 1.5 * df), min_strength_db)
        f[(f < lo_f) | (f > hi_f)] = np.nan
        return f, st

    def fit(ff):
        ok = np.isfinite(ff)
        if ok.sum() < 8:
            raise ValueError("the line is too faint here to fit: add clicks or widen the guide")
        return fit_flyover(times[ok], ff[ok], c)

    def model(p):
        return p["f0"] * straight_flyover_ratio(times, p["t_cpa"], p["speed"], p["d_cpa"], c)

    # 2. wide corridor: between neighbouring clicks, from the lower to the higher one, plus a margin
    say(0.2, "tracing the ridge")
    k = np.clip(np.searchsorted(gt, times) - 1, 0, len(gt) - 2)
    lo_f, hi_f = np.minimum(gf[k], gf[k + 1]), np.maximum(gf[k], gf[k + 1])
    lo_f = lo_f - np.maximum(corridor_rel * lo_f, 4.0)
    hi_f = hi_f + np.maximum(corridor_rel * hi_f, 4.0)
    f, st = ridge(lo_f, hi_f)
    p = fit(f)
    stages = [f]
    # 3. corridors around the model, wide first (to recover stretches the first trace missed), then narrow
    for k, (rel, floor) in enumerate(((0.06, 4.0), (0.04, 3.0), (0.025, 2.0), (0.015, 1.2))):
        say(0.35 + 0.15 * k, f"refining ({k + 1}/4)")
        m = model(p)
        hw = np.maximum(rel * m, floor)
        f, st = ridge(m - hw, m + hw)
        p = fit(f)
        stages.append(f)
    # 4. outliers
    r = f - model(p)
    sig = max(1.4826 * np.nanmedian(np.abs(r)), 0.3)
    f[np.abs(r) > 3 * sig] = np.nan
    p = fit(f)
    say(1.0, "done")
    meta = {"guide_t": [float(x) for x in gt], "guide_f": [float(x) for x in gf], "method": "isolate",
            "line_window_s": line_window_s, "fit": {k: float(v) for k, v in p.items()}, "fitted": True,
            "points": int(np.isfinite(f).sum())}
    cv = DopplerCurve(times.copy(), f, st, name, meta)
    if debug:
        return cv, stages, Spectrogram(times, freqs, S)
    return cv
