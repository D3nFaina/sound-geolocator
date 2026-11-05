"""Rank place/time candidates by simulating the real aircraft's Doppler curves.

For each candidate (a place, a recording start time and one aircraft per
pass, as found by :meth:`DayArchive.search_world`), the recorded position
track of each aircraft is put on a 1 s grid in a local east/north/up frame
around the candidate, and the Doppler curve it would produce is simulated with
the retarded-time model. A small fit then adjusts what the search could only
know roughly:

* the listener's position, within +-3 km of the candidate cell,
* the listener's height, within -100/+150 m of the ground (terrain tiles),
* the recording start time, within +-40 s (shared by every pass),
* each pass's tone f0 (solved exactly, it is a scale factor).

The result is the rms misfit in Hz of every pass, so candidates can be
compared pass against pass (e.g. fit on B vs fit on C).
"""

from __future__ import annotations

import numpy as np

R_EARTH = 6371000.0
PAD_S = 70.0


def _enu_rotation(lat0, lon0):
    la, lo = np.radians(lat0), np.radians(lon0)
    return np.array([[-np.sin(lo), np.cos(lo), 0.0],
                     [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                     [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])


def _track_grid(archive, idx, ac_id, lat0, lon0, t_start, tobs):
    """One aircraft on a 1 s grid: east, north, up (m, around lat0/lon0 at sea level) and their velocities.

    Uses the same smoothing-spline track as the curve match in ``locate``.
    """
    from .flights import Track
    from .geo import lla_to_ecef

    i = idx[ac_id]
    s, n = int(archive.start[i]), int(archive.n[i])
    sl = slice(s, s + n)
    tt = np.asarray(archive.t[sl], float) + archive.day0
    g0 = t_start + tobs[0] - 40 - PAD_S
    g = g0 + np.arange(int(tobs[-1] - tobs[0]) + 80 + 2 * int(PAD_S))
    m = (tt >= g[0] - 120) & (tt <= g[-1] + 120)
    if m.sum() < 4:
        return g0 - t_start, np.full((len(g), 6), np.nan), 999.0
    w = (tt >= g[0]) & (tt <= g[-1])
    gap = float(np.max(np.diff(tt[w]))) if w.sum() > 1 else 999.0
    tr = Track(ac_id, tt[m], np.asarray(archive.lat[sl], float)[m], np.asarray(archive.lon[sl], float)[m],
               np.asarray(archive.alt[sl], float)[m])
    gc = np.clip(g, tt[m][0], tt[m][-1])
    Rm = _enu_rotation(lat0, lon0)
    P = (tr.position(gc) - lla_to_ecef(lat0, lon0, 0.0)) @ Rm.T
    V = tr.velocity(gc) @ Rm.T
    return g0 - t_start, np.concatenate([P, V], 1), gap


def _pass_residuals(x, G, g0rel, tobs, fobs, c):
    """x (N,4) = east, north, height, start shift. Returns residuals (N, M) and f0 (N,)."""
    O = x[:, None, :3]
    u = (x[:, 3:4] + tobs[None, :]) - g0rel[:, None]
    L = G.shape[1]
    ar = np.arange(G.shape[0])[:, None]
    tau = u.copy()
    for _ in range(4):
        i0 = np.clip(np.floor(tau).astype(int), 0, L - 2)
        fr = (tau - i0)[..., None]
        P = G[ar, i0, :3] * (1 - fr) + G[ar, i0 + 1, :3] * fr
        tau = u - np.linalg.norm(P - O, axis=-1) / c
    i0 = np.clip(np.floor(tau).astype(int), 0, L - 2)
    fr = (tau - i0)[..., None]
    PV = G[ar, i0] * (1 - fr) + G[ar, i0 + 1] * fr
    P, V = PV[..., :3], PV[..., 3:]
    d = O - P
    vr = np.sum(V * d, -1) / np.linalg.norm(d, axis=-1)
    r = c / (c - vr)
    f0 = np.sum(fobs * r, 1) / np.sum(r * r, 1)
    return fobs[None, :] - f0[:, None] * r, f0


def _fit_batch(Gs, g0s, obs, x0, lo, hi, c, iters=25):
    """Batched Levenberg-Marquardt over all passes together."""
    x = x0.copy()
    N = len(x)
    lam = np.full(N, 1e-2)

    def allres(x):
        out = [_pass_residuals(x, Gs[p], g0s[p], obs[p][0], obs[p][1], c) for p in range(len(obs))]
        return np.concatenate([o[0] for o in out], 1), [o[1] for o in out]

    R, _ = allres(x)
    cost = np.sum(R ** 2, 1)
    steps = np.array([20.0, 20.0, 20.0, 0.2])
    for _ in range(iters):
        J = np.empty(R.shape + (4,))
        for k in range(4):
            xp = x.copy()
            xp[:, k] += steps[k]
            J[..., k] = (allres(xp)[0] - R) / steps[k]
        JtJ = np.einsum("nmi,nmj->nij", J, J)
        Jtr = np.einsum("nmi,nm->ni", J, R)
        D = np.einsum("nii->ni", JtJ) + 1e-9
        A = JtJ + lam[:, None, None] * np.einsum("ni,ij->nij", D, np.eye(4))
        dx = -np.linalg.solve(A, Jtr[..., None])[..., 0]
        xn = np.clip(x + dx, lo, hi)
        Rn, _ = allres(xn)
        cn = np.sum(Rn ** 2, 1)
        better = cn < cost
        x[better], R[better], cost[better] = xn[better], Rn[better], cn[better]
        lam = np.where(better, lam * 0.3, lam * 10)
    R, f0s = allres(x)
    edges = np.cumsum([0] + [len(o[0]) for o in obs])
    rms = [np.sqrt(np.mean(R[:, edges[p]:edges[p + 1]] ** 2, 1)) for p in range(len(obs))]
    return x, rms, f0s


def rank_candidates(archive, cands, curves, ground_m=None, c=340.0, step_s=1.0, chunk=4000, progress=None):
    """Simulate and fit every candidate. ``curves`` are in the same order as each candidate's aircraft.

    ``ground_m``: ground height per candidate (m above sea level); fetched from
    the terrain tiles when None. Returns a dict of arrays: ``rms`` (N, passes),
    ``f0`` (N, passes), ``gap`` (N, passes, the longest hole in the ADS-B track
    around the pass, s), ``lat``/``lon`` (fitted listener), ``h``, ``t_start``
    (fitted), ``ground``.
    """
    from .elevation import terrarium_elevation

    cands = list(cands)
    N, P = len(cands), len(curves)
    lat0 = np.array([cd.lat for cd in cands])
    lon0 = np.array([cd.lon for cd in cands])
    if ground_m is None:
        ground_m = terrarium_elevation(lat0, lon0, cache_dir="~/.cache/sound_geolocator/terrarium",
                                       progress=(lambda f: progress(0.1 * f)) if progress else None)
    ground = np.nan_to_num(np.asarray(ground_m, float), nan=0.0)
    el = np.maximum(ground, 0.0)  # over water: sea level
    obs = []
    for cv in curves:
        t, f = np.asarray(cv.t, float), np.asarray(cv.f, float)
        k = max(1, int(round(step_s / max(np.median(np.diff(t)), 1e-3))))
        t, f = t[::k], f[::k]
        m = np.isfinite(f)
        obs.append((t[m], f[m]))
    idx = {m[0]: i for i, m in enumerate(archive.meta)}
    out = {"rms": np.full((N, P), np.nan), "f0": np.full((N, P), np.nan), "gap": np.full((N, P), np.nan),
           "east": np.full(N, np.nan), "north": np.full(N, np.nan), "h": np.full(N, np.nan),
           "shift": np.full(N, np.nan), "ground": ground}
    for c0 in range(0, N, chunk):
        if progress:
            progress(0.1 + 0.9 * c0 / N)
        sub = cands[c0:c0 + chunk]
        n = len(sub)
        Gs = [[] for _ in range(P)]
        g0s = [np.empty(n) for _ in range(P)]
        for k, cd in enumerate(sub):
            for p in range(P):
                g0, G, gap = _track_grid(archive, idx, cd.aircraft[p], cd.lat, cd.lon, cd.t_start, obs[p][0])
                Gs[p].append(G)
                g0s[p][k] = g0
                out["gap"][c0 + k, p] = gap
        Gs = [np.stack(G) for G in Gs]
        e = el[c0:c0 + n]
        lo = np.stack([np.full(n, -3000.0), np.full(n, -3000.0), e - 100, np.full(n, -40.0)], 1)
        hi = np.stack([np.full(n, 3000.0), np.full(n, 3000.0), e + 150, np.full(n, 40.0)], 1)
        best = None
        for e0 in (-1000.0, 1000.0):
            x0 = np.zeros((n, 4))
            x0[:, 0], x0[:, 2] = e0, e + 20
            x, rms, f0s = _fit_batch(Gs, g0s, obs, x0, lo, hi, c)
            tot = sum(r ** 2 for r in rms)
            if best is None:
                best = [x, np.stack(rms, 1), np.stack(f0s, 1), tot]
            else:
                m = tot < best[3]
                best[0][m], best[1][m], best[2][m], best[3][m] = x[m], np.stack(rms, 1)[m], np.stack(f0s, 1)[m], tot[m]
        sl = slice(c0, c0 + n)
        out["east"][sl], out["north"][sl], out["h"][sl], out["shift"][sl] = best[0].T
        out["rms"][sl], out["f0"][sl] = best[1], best[2]
    out["lat"] = lat0 + np.degrees(out["north"] / R_EARTH)
    out["lon"] = lon0 + np.degrees(out["east"] / (R_EARTH * np.cos(np.radians(lat0))))
    out["t_start"] = np.array([cd.t_start for cd in cands]) + out["shift"]
    if progress:
        progress(1.0)
    return out


def rank_locked(archive, cands, anchor_curve, other_curve, ground_m, c=340.0, pos_box_m=600.0, shift_box_s=3.0,
                step_s=1.0, chunk=4000, scan_s=10.0, progress=None):
    """Fit the anchor pass only, lock listener and start time, then score the other pass.

    For the other pass only its tone f0 is free: ``rms_locked`` is its misfit
    with the anchor's start time, and ``dt_other`` is how far its curve would
    have to slide in time to fit best (``rms_slid``): a real match has
    dt_other close to 0.
    """
    cands = list(cands)
    N = len(cands)
    ground = np.nan_to_num(np.asarray(ground_m, float), nan=0.0)
    el = np.maximum(ground, 0.0)
    obs = []
    for cv in (anchor_curve, other_curve):
        t, f = np.asarray(cv.t, float), np.asarray(cv.f, float)
        k = max(1, int(round(step_s / max(np.median(np.diff(t)), 1e-3))))
        t, f = t[::k], f[::k]
        m = np.isfinite(f)
        obs.append((t[m], f[m]))
    # the other pass's grid must reach the slid times too
    ext = (np.concatenate([[obs[1][0][0] - scan_s], obs[1][0], [obs[1][0][-1] + scan_s]]))
    idx = {m[0]: i for i, m in enumerate(archive.meta)}
    keys = ("rms_anchor", "rms_locked", "rms_slid", "dt_other", "f0_anchor", "f0_other", "east", "north", "h", "shift")
    out = {k: np.full(N, np.nan) for k in keys}
    shifts = np.arange(-scan_s, scan_s + 1e-9, 0.25)
    for c0 in range(0, N, chunk):
        if progress:
            progress(c0 / N)
        sub = cands[c0:c0 + chunk]
        n = len(sub)
        GA, GB, gA, gB = [], [], np.empty(n), np.empty(n)
        for k, cd in enumerate(sub):
            g0, G, _ = _track_grid(archive, idx, cd.aircraft[0], cd.lat, cd.lon, cd.t_start, obs[0][0])
            GA.append(G); gA[k] = g0
            g0, G, _ = _track_grid(archive, idx, cd.aircraft[1], cd.lat, cd.lon, cd.t_start, ext)
            GB.append(G); gB[k] = g0
        GA, GB = np.stack(GA), np.stack(GB)
        e = el[c0:c0 + n]
        lo = np.stack([np.full(n, -pos_box_m), np.full(n, -pos_box_m), e - 100, np.full(n, -shift_box_s)], 1)
        hi = np.stack([np.full(n, pos_box_m), np.full(n, pos_box_m), e + 150, np.full(n, shift_box_s)], 1)
        x0 = np.zeros((n, 4))
        x0[:, 2] = e + 20
        x, rms, f0s = _fit_batch([GA], [gA], obs[:1], x0, lo, hi, c)
        rB, f0B = _pass_residuals(x, GB, gB, obs[1][0], obs[1][1], c)
        best = np.full(n, np.inf)
        bdt = np.zeros(n)
        for s in shifts:
            xs = x.copy()
            xs[:, 3] += s
            r, _ = _pass_residuals(xs, GB, gB, obs[1][0], obs[1][1], c)
            v = np.sqrt(np.mean(r ** 2, 1))
            better = v < best
            best[better], bdt[better] = v[better], s
        sl = slice(c0, c0 + n)
        out["rms_anchor"][sl], out["f0_anchor"][sl] = rms[0], f0s[0]
        out["rms_locked"][sl], out["f0_other"][sl] = np.sqrt(np.mean(rB ** 2, 1)), f0B
        out["rms_slid"][sl], out["dt_other"][sl] = best, bdt
        out["east"][sl], out["north"][sl], out["h"][sl], out["shift"][sl] = x.T
    lat0 = np.array([cd.lat for cd in cands])
    lon0 = np.array([cd.lon for cd in cands])
    out["lat"] = lat0 + np.degrees(out["north"] / R_EARTH)
    out["lon"] = lon0 + np.degrees(out["east"] / (R_EARTH * np.cos(np.radians(lat0))))
    out["t_start"] = np.array([cd.t_start for cd in cands]) + out["shift"]
    out["ground"] = ground
    if progress:
        progress(1.0)
    return out
