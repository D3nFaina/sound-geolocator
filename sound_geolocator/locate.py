"""Find the recording location (and start time) that best explains the Doppler curves.

Unknowns: observer position (lat, lon) and the UTC time the recording started.
All curves come from the same recording, so the delays between them are
known; that shared clock is what makes two or more passes pin down a point.

For every candidate position on a grid we simulate, for every candidate
flight, the Doppler ratio r(t) the observer would have heard. Each curve is
then slid along the time axis (one shared offset for all curves) and the
source tone frequency is fitted in closed form, f_src = sum(f r) / sum(r^2).
The cost of a curve is its relative RMS misfit against the best-matching
flight; the cost of a position is the mean squared misfit over curves at the
best shared start time. A coarse grid is refined around the minimum.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import fftconvolve

from . import DEFAULT_SPEED_OF_SOUND
from .doppler import doppler_ratio
from .extract import DopplerCurve
from .flights import Track
from .geo import enu_offsets_to_lla, lla_to_ecef, lla_to_enu_offsets

BAD = 1.0  # cost assigned when a flight does not cover a curve


@dataclass
class Assignment:
    curve: str
    flight: str
    f_src: float
    rel_rms: float
    rms_hz: float


@dataclass
class LocateResult:
    lat: float
    lon: float
    t_start: float  # unix seconds of recording start
    cost: float
    assignments: list[Assignment]
    grid_lat: np.ndarray = field(repr=False, default=None)
    grid_lon: np.ndarray = field(repr=False, default=None)
    cost_map: np.ndarray = field(repr=False, default=None)  # (ny, nx) best cost per cell
    refine_maps: list = field(repr=False, default_factory=list)


def _resample(curve: DopplerCurve, dt: float):
    ok = np.isfinite(curve.f)
    t0 = curve.t[ok][0]
    n = int(np.floor((curve.t[ok][-1] - t0) / dt)) + 1
    tg = t0 + dt * np.arange(n)
    # nearest original sample, only if within dt
    idx = np.clip(np.searchsorted(curve.t, tg), 1, len(curve.t) - 1)
    near = np.where(np.abs(curve.t[idx] - tg) < np.abs(curve.t[idx - 1] - tg), idx, idx - 1)
    f = curve.f[near]
    w = (np.isfinite(f) & (np.abs(curve.t[near] - tg) <= dt)).astype(float)
    return tg, np.nan_to_num(f), w


def _prefilter(tracks, lat0, lon0, radius_m, t_lo, t_hi, audible_m):
    keep = []
    for tr in tracks:
        if tr.t[-1] < t_lo - 120 or tr.t[0] > t_hi + 120:
            continue
        m = (tr.t >= t_lo - 120) & (tr.t <= t_hi + 120)
        if m.sum() < 2:
            continue
        e, n = lla_to_enu_offsets(lat0, lon0, tr.lat[m], tr.lon[m])
        horiz = np.hypot(e, n) - radius_m
        if np.min(np.hypot(np.maximum(horiz, 0), tr.alt[m])) <= audible_m:
            keep.append(tr)
    return keep


def _score_grid(obs_ecef, curves_rs, tracks, abs_t, n_shift, c, min_coverage=0.8):
    """Return per-curve cost (C, G, S) and flight index (C, G, S) and f_src."""
    G = obs_ecef.shape[0]
    C = len(curves_rs)
    best = np.full((C, G, n_shift), BAD)
    who = np.full((C, G, n_shift), -1, dtype=np.int32)
    fsrc = np.zeros((C, G, n_shift))
    for j, tr in enumerate(tracks):
        R, tau, _, _ = doppler_ratio(tr, obs_ecef[:, None, :], abs_t[None, :], c, return_extra=True)
        cov = tr.covered(tau).astype(float)
        R = np.where(cov > 0, R, 0.0)
        for i, (k0, f, w) in enumerate(curves_rs):
            L = len(f)
            seg = slice(k0, k0 + n_shift + L - 1)
            Rs, Cs = R[:, seg], cov[:, seg]
            kf = (w * f)[::-1][None, :]
            kw = w[::-1][None, :]
            sfr = fftconvolve(Rs, kf, mode="valid", axes=1)
            srr = fftconvolve(Rs * Rs, kw, mode="valid", axes=1)
            scv = fftconvolve(Cs, kw, mode="valid", axes=1) / w.sum()
            # f^2 summed only where the flight has data, so a partly covered
            # curve is judged on the part we can simulate
            sff = fftconvolve(Cs, (w * f * f)[::-1][None, :], mode="valid", axes=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                ss = np.maximum(sff - sfr * sfr / srr, 0)
                rel = np.sqrt(ss / sff)
                f0 = sfr / srr
            rel = np.where((scv >= min_coverage) & (srr > 0), rel, BAD)
            better = rel < best[i]
            best[i] = np.where(better, rel, best[i])
            who[i] = np.where(better, j, who[i])
            fsrc[i] = np.where(better, f0, fsrc[i])
    return best, who, fsrc


def locate(curves: list[DopplerCurve], tracks: list[Track], center: tuple[float, float],
           t_start_guess: float, t_uncertainty: float = 300.0, radius_km: float = 20.0,
           grid_step_m: float = 1000.0, refine_levels: int = 2, observer_alt_m: float = 0.0,
           dt: float = 0.5, c: float = DEFAULT_SPEED_OF_SOUND, audible_km: float = 15.0,
           min_coverage: float = 0.8, verbose: bool = True) -> LocateResult:
    """Grid-search the observer position and recording start time.

    center: (lat, lon) to search around; radius_km: half-width of the square
    search area; t_start_guess: approximate unix time of the recording start
    (e.g. from file metadata); t_uncertainty: +- seconds to search.
    min_coverage: fraction of a curve the flight track must cover to be a candidate.
    """
    if len(curves) < 1:
        raise ValueError("need at least one curve (two or more with known delay to fix a location)")
    lat0, lon0 = center
    rel0 = min(cv.t[np.isfinite(cv.f)][0] for cv in curves)
    rel1 = max(cv.t[np.isfinite(cv.f)][-1] for cv in curves)
    t_lo = t_start_guess - t_uncertainty
    t_hi = t_start_guess + t_uncertainty
    n_shift = int(np.floor((t_hi - t_lo) / dt)) + 1
    abs_t = t_lo + rel0 + dt * np.arange(n_shift + int(np.ceil((rel1 - rel0) / dt)) + 2)

    curves_rs = []
    for cv in curves:
        tg, f, w = _resample(cv, dt)
        curves_rs.append((int(round((tg[0] - rel0) / dt)), f, w))

    tracks = _prefilter(tracks, lat0, lon0, radius_km * 1000, t_lo + rel0, t_hi + rel1, audible_km * 1000)
    if verbose:
        print(f"[locate] {len(tracks)} candidate flights, {len(curves)} curves, {n_shift} start-time offsets")
    if not tracks:
        raise ValueError("no flights near the search area during the recording window")

    half = radius_km * 1000
    step = grid_step_m
    ce, cn = 0.0, 0.0
    maps = []
    result = None
    for level in range(refine_levels + 1):
        es = np.arange(ce - half, ce + half + 1e-6, step)
        ns = np.arange(cn - half, cn + half + 1e-6, step)
        E, N = np.meshgrid(es, ns)
        glat, glon = enu_offsets_to_lla(lat0, lon0, E, N)
        obs = lla_to_ecef(glat.ravel(), glon.ravel(), np.full(E.size, observer_alt_m))
        best, who, fsrc = _score_grid(obs, curves_rs, tracks, abs_t, n_shift, c, min_coverage)
        total = np.mean(best ** 2, axis=0)  # (G, S)
        s_best = np.argmin(total, axis=1)
        cell = total[np.arange(total.shape[0]), s_best]
        g = int(np.argmin(cell))
        s = int(s_best[g])
        cmap = cell.reshape(E.shape)
        maps.append((glat, glon, cmap))
        assigns = []
        for i, cv in enumerate(curves):
            j = who[i, g, s]
            _, f, w = curves_rs[i]
            rel = best[i, g, s]
            assigns.append(Assignment(cv.name or f"curve{i}", tracks[j].label if j >= 0 else "-",
                                      float(fsrc[i, g, s]), float(rel),
                                      float(rel * np.sqrt(np.sum(w * f * f) / w.sum()))))
        result = LocateResult(float(glat.ravel()[g]), float(glon.ravel()[g]), float(t_lo + s * dt),
                              float(cell[g]), assigns)
        if verbose:
            print(f"[locate] level {level}: step {step:.0f} m -> {result.lat:.5f}, {result.lon:.5f} "
                  f"start={result.t_start:.1f} cost={result.cost:.2e}")
        ce, cn = E.ravel()[g], N.ravel()[g]
        half = step * 2.5
        step = step / 5
    result.grid_lat, result.grid_lon, result.cost_map = maps[0]
    result.refine_maps = maps[1:]
    return result
