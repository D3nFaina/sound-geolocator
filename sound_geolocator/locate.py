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

Refinements (after the method described in colsto's video):
* locked multi-event scoring with a small per-curve timing slack, so
  measurement error on each closest-approach time is absorbed while the
  physics-derived delay between passes is kept;
* a speed / closest-approach-distance gate from a flyover fit of each curve;
* an optional silence check: a flight that would have passed close during
  the recording but matches no curve makes that position less likely;
* terrain height for each candidate from a DEM (``observer_alt_m`` callable).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import fftconvolve

from . import DEFAULT_SPEED_OF_SOUND
from .doppler import doppler_ratio, fit_flyover
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
    time_shift_s: float = 0.0  # per-curve timing correction used (CPA slack)


@dataclass
class LocateResult:
    lat: float
    lon: float
    t_start: float  # unix seconds of recording start
    cost: float
    assignments: list[Assignment]
    observer_alt: float = 0.0
    grid_lat: np.ndarray = field(repr=False, default=None)
    grid_lon: np.ndarray = field(repr=False, default=None)
    cost_map: np.ndarray = field(repr=False, default=None)  # (ny, nx) best cost per cell
    refine_maps: list = field(repr=False, default_factory=list)
    candidates: list = field(default_factory=list)  # (lat, lon, cost) of refined coarse minima, best first


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


def _window_min(a, k, step_penalty=0.0):
    """Min of a over a +-k window along the last axis, and the offset that achieved it.

    Each step away from the centre costs ``step_penalty``, so shifts are
    only used when they really improve the fit (keeps the shared start
    time from drifting when all curves shift together).
    """
    if k <= 0:
        return a, np.zeros(a.shape, dtype=np.int32)
    pad = np.pad(a, [(0, 0)] * (a.ndim - 1) + [(k, k)], constant_values=np.inf)
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * k + 1, axis=-1)
    win = win + step_penalty * np.abs(np.arange(-k, k + 1))
    idx = np.argmin(win, axis=-1)
    return np.take_along_axis(win, idx[..., None], -1)[..., 0], (idx - k).astype(np.int32)


def _track_speed(tr, t_lo, t_hi):
    m = (tr.t >= t_lo) & (tr.t <= t_hi)
    t = tr.t[m] if m.sum() >= 2 else tr.t
    return float(np.median(np.linalg.norm(tr.velocity(t), axis=-1)))


def _score_grid(obs_ecef, curves_rs, tracks, abs_t, n_shift, c, min_coverage=0.8, gates=None):
    """Per-curve misfit (C, G, S), best flight index and fitted source tone.

    Also returns, per flight, the closest slant distance to each grid point
    (G,) and the abs_t index at which that closest approach is *heard*,
    which the silence check uses.
    """
    G = obs_ecef.shape[0]
    C = len(curves_rs)
    best = np.full((C, G, n_shift), BAD)
    who = np.full((C, G, n_shift), -1, dtype=np.int32)
    fsrc = np.zeros((C, G, n_shift))
    dmin = np.zeros((len(tracks), G))
    t_at = np.zeros((len(tracks), G), dtype=np.int64)
    for j, tr in enumerate(tracks):
        R, tau, rng, _ = doppler_ratio(tr, obs_ecef[:, None, :], abs_t[None, :], c, return_extra=True)
        cov = tr.covered(tau).astype(float)
        R = np.where(cov > 0, R, 0.0)
        rng_c = np.where(cov > 0, rng, np.inf)
        t_at[j] = np.argmin(rng_c, axis=1)
        dmin[j] = rng_c[np.arange(G), t_at[j]]
        for i, (k0, f, w) in enumerate(curves_rs):
            gate = None
            if gates is not None and gates[i] is not None:
                ok_speed, d_lo, d_hi = gates[i][j]
                if not ok_speed:
                    continue
                gate = (dmin[j] >= d_lo) & (dmin[j] <= d_hi)
                if not gate.any():
                    continue
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
            ok = (scv >= min_coverage) & (srr > 0)
            if gate is not None:
                ok &= gate[:, None]
            rel = np.where(ok, rel, BAD)
            better = rel < best[i]
            best[i] = np.where(better, rel, best[i])
            who[i] = np.where(better, j, who[i])
            fsrc[i] = np.where(better, f0, fsrc[i])
    return best, who, fsrc, dmin, t_at


def _fit_gates(curves, tracks, t_lo, t_hi, speed_tol, dist_factor, max_fit_rms_hz, c, verbose):
    """Per curve, per flight: (speed plausible?, min CPA slant m, max CPA slant m), from a flyover fit."""
    gates = []
    speeds = [_track_speed(tr, t_lo, t_hi) for tr in tracks]
    for cv in curves:
        try:
            fit = fit_flyover(cv.t, cv.f, c)
        except Exception:
            fit = None
        if fit is None or fit["rms_hz"] > max_fit_rms_hz:
            gates.append(None)
            if verbose:
                print(f"[locate] {cv.name}: flyover fit too loose, no speed/distance gate")
            continue
        g = []
        for v in speeds:
            ok = abs(fit["speed"] - v) <= speed_tol * v
            g.append((ok, fit["d_cpa"] / dist_factor, fit["d_cpa"] * dist_factor))
        if verbose:
            n = sum(x[0] for x in g)
            print(f"[locate] {cv.name}: fit {fit['speed']:.0f} m/s, CPA {fit['d_cpa']:.0f} m -> {n}/{len(g)} flights pass the speed gate")
        gates.append(g)
    return gates


def locate(curves: list[DopplerCurve], tracks: list[Track], center: tuple[float, float],
           t_start_guess: float, t_uncertainty: float = 300.0, radius_km: float = 20.0,
           grid_step_m: float = 1000.0, refine_levels: int = 2, observer_alt_m=0.0,
           dt: float = 0.5, c: float = DEFAULT_SPEED_OF_SOUND, audible_km: float = 15.0,
           min_coverage: float = 0.8, cpa_slack_s: float = 2.0, fit_gate: bool = True,
           speed_tol: float = 0.35, dist_factor: float = 3.0, max_fit_rms_hz: float = 1.5,
           recording_s: float | None = None, silence_km: float | None = None,
           silence_penalty: float = 0.01, n_candidates: int = 3, slack_penalty_per_s: float = 5e-4, verbose: bool = True) -> LocateResult:
    """Grid-search the observer position and recording start time.

    center: (lat, lon) to search around; radius_km: half-width of the square
    search area; t_start_guess: approximate unix time of the recording start
    (e.g. from file metadata); t_uncertainty: +- seconds to search.

    observer_alt_m: metres, or a function (lat, lon) -> metres (e.g. a DEM,
    see :mod:`sound_geolocator.elevation`), on the same datum as the tracks.

    Locked multi-event scoring: all curves share one start time, so the
    physical delays between passes are kept. Each curve may still slide by
    up to ``cpa_slack_s`` to absorb measurement error in its timing, and its
    source tone is fitted freely.

    fit_gate: fit a straight flyover to each curve and only match it to
    flights whose ground speed agrees within ``speed_tol`` and whose closest
    approach to the candidate point is within ``dist_factor`` of the fitted
    slant range. Curves whose fit is poor (rms > max_fit_rms_hz) are not gated.

    n_candidates: how many separated coarse-grid minima to refine.

    silence_km: if set, penalise positions where a flight that matches none
    of the curves would have passed within this slant distance during the
    recording (``recording_s`` long) - you would have heard it.
    """
    if len(curves) < 1:
        raise ValueError("need at least one curve (two or more with known delay to fix a location)")
    lat0, lon0 = center
    rel1 = max(cv.t[np.isfinite(cv.f)][-1] for cv in curves)
    rec_len = max(rel1, recording_s or 0.0)
    t_lo = t_start_guess - t_uncertainty
    t_hi = t_start_guess + t_uncertainty
    n_shift = int(np.floor((t_hi - t_lo) / dt)) + 1
    n_rec = int(np.ceil(rec_len / dt)) + 2
    abs_t = t_lo + dt * np.arange(n_shift + n_rec)
    slack = int(round(cpa_slack_s / dt))

    curves_rs = []
    for cv in curves:
        tg, f, w = _resample(cv, dt)
        curves_rs.append((int(round(tg[0] / dt)), f, w))

    tracks = _prefilter(tracks, lat0, lon0, radius_km * 1000, t_lo, t_hi + rec_len, audible_km * 1000)
    if verbose:
        print(f"[locate] {len(tracks)} candidate flights, {len(curves)} curves, {n_shift} start-time offsets")
    if not tracks:
        raise ValueError("no flights near the search area during the recording window")
    gates = _fit_gates(curves, tracks, t_lo - 300, t_hi + rec_len + 300, speed_tol, dist_factor,
                       max_fit_rms_hz, c, verbose) if fit_gate else None

    def level_search(ce, cn, half, step):
        es = np.arange(ce - half, ce + half + 1e-6, step)
        ns = np.arange(cn - half, cn + half + 1e-6, step)
        E, N = np.meshgrid(es, ns)
        glat, glon = enu_offsets_to_lla(lat0, lon0, E, N)
        alt = observer_alt_m(glat.ravel(), glon.ravel()) if callable(observer_alt_m) \
            else np.full(E.size, float(observer_alt_m))
        obs = lla_to_ecef(glat.ravel(), glon.ravel(), alt)
        best, who, fsrc, dmin, t_at = _score_grid(obs, curves_rs, tracks, abs_t, n_shift, c, min_coverage, gates)
        # per-curve timing slack around the shared start time
        best_s, off = _window_min(best, slack, slack_penalty_per_s * dt)
        total = np.mean(best_s ** 2, axis=0)  # (G, S)
        if silence_km:
            sidx = np.arange(n_shift)
            wins = [np.take_along_axis(who[i], np.clip(sidx[None, :] + off[i], 0, n_shift - 1), 1)
                    for i in range(len(curves))]
            for j in range(len(tracks)):
                loud = (dmin[j] <= silence_km * 1000)[:, None] & \
                       (t_at[j][:, None] >= sidx[None, :]) & (t_at[j][:, None] <= sidx[None, :] + n_rec - 2)
                if not loud.any():
                    continue
                used = np.zeros_like(loud)
                for wi in wins:
                    used |= wi == j
                total = total + silence_penalty * (loud & ~used)
        s_best = np.argmin(total, axis=1)
        cell = total[np.arange(total.shape[0]), s_best]

        def result_at(g):
            s = int(s_best[g])
            assigns = []
            for i, cv in enumerate(curves):
                si = int(np.clip(s + off[i, g, s], 0, n_shift - 1))
                j = who[i, g, si]
                _, f, w = curves_rs[i]
                rel = best[i, g, si]
                assigns.append(Assignment(cv.name or f"curve{i}", tracks[j].label if j >= 0 else "-",
                                          float(fsrc[i, g, si]), float(rel),
                                          float(rel * np.sqrt(np.sum(w * f * f) / w.sum())),
                                          float((si - s) * dt)))
            return LocateResult(float(glat.ravel()[g]), float(glon.ravel()[g]), float(t_lo + s * dt),
                                float(cell[g]), assigns, observer_alt=float(alt[g])), E.ravel()[g], N.ravel()[g]

        return cell.reshape(E.shape), glat, glon, result_at

    # coarse level: keep several separated local minima, refine each, keep the best
    cmap, glat, glon, result_at = level_search(0.0, 0.0, radius_km * 1000, grid_step_m)
    maps = [(glat, glon, cmap)]
    from scipy.ndimage import minimum_filter

    is_min = cmap == minimum_filter(cmap, size=5, mode="nearest")
    order = [g for g in np.argsort(cmap, axis=None) if is_min.ravel()[g]][:n_candidates]
    finals = []
    for rank, g in enumerate(order):
        res, ce, cn = result_at(int(g))
        if verbose:
            print(f"[locate] candidate {rank + 1}: {res.lat:.5f}, {res.lon:.5f} cost={res.cost:.2e}")
        half, step = grid_step_m * 2.5, grid_step_m / 5
        for level in range(refine_levels):
            m, gl, go, ra = level_search(ce, cn, half, step)
            res, ce, cn = ra(int(np.argmin(m)))
            if level == 0:
                first_refine = (gl, go, m)
            half, step = step * 2.5, step / 5
        if verbose and refine_levels:
            print(f"[locate]   refined -> {res.lat:.5f}, {res.lon:.5f} start={res.t_start:.1f} cost={res.cost:.2e}")
        finals.append((res, first_refine if refine_levels else None))
    result, fr = min(finals, key=lambda x: x[0].cost)
    result.candidates = [(f[0].lat, f[0].lon, f[0].cost) for f in sorted(finals, key=lambda x: x[0].cost)]
    if fr is not None:
        maps.append(fr)
    result.grid_lat, result.grid_lon, result.cost_map = maps[0]
    result.refine_maps = maps[1:]
    return result
