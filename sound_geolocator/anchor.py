"""Anchor on one pass, lock its time, and test the next pass against it.

The best-fitted pass (the *anchor*, e.g. B) pins the geometry hard: the
listener sits abeam of the aircraft at its fitted slant distance when it is at
closest approach, and the recording start is then fixed to the second by the
fitted t_cpa. Every aircraft with the right ground speed gives, at every
second of its track (about 140 m at 140 m/s), two such listener positions,
one on each side, each with its own start time.

The second pass (e.g. C) is turned into listener positions and start times the
same way. A place survives only where an anchor position and a second-pass
position agree within the fit-derived tolerances, both in space and in start
time. With the start time locked by the anchor, a second-pass aircraft at the
right distance but at the wrong moment is rejected.

Tolerances come from the fits: distance 3 sigma + a systematic share
(speed of sound and wind, a few %), start time 3 sigma of both t_cpa plus the
1 s stepping. The ground speed window keeps a wind allowance because ADS-B
reports speed over the ground while the Doppler shift follows the airspeed.
"""

from __future__ import annotations

import numpy as np

R_EARTH = 6371000.0


def pass_window(fit, sys_frac=0.03, wind=15.0, nsig=3.0):
    """Search window for one fitted pass (dict from ``fit_flyover`` with errors)."""
    v, d, tc = fit["speed"], fit["d_cpa"], fit["t_cpa"]
    se, de, te = (fit.get(k) or 0.0 for k in ("speed_err", "d_cpa_err", "t_cpa_err"))
    dv = nsig * se + sys_frac * v + wind
    dd = nsig * de + sys_frac * d
    # emission-time t_cpa: c error moves it by about d/c * sys_frac
    dt = nsig * te + sys_frac * d / 340.0
    return dict(t_cpa=tc, speed=v, d_cpa=d, speed_lo=v - dv, speed_hi=v + dv, d_tol=dd, t_tol=dt)


def listener_points(archive, win, bbox=None, step_s=1.0, max_sub=6, ground_fn=None, t_lo=None, t_hi=None):
    """Listener positions and start times for one pass, from every matching aircraft.

    Returns dict of arrays: lat, lon (listener), t0 (recording start, unix),
    ac (aircraft index), side (-1 left, +1 right), gap (s to the aircraft's
    next sample).
    """
    t = np.asarray(archive.t, np.float64)
    gs = np.asarray(archive.gs, np.float32)
    trk = np.asarray(archive.trk, np.float32)
    alt = np.asarray(archive.alt, np.float32)
    m = (gs >= win["speed_lo"]) & (gs <= win["speed_hi"]) & np.isfinite(trk)
    m &= alt <= win["d_cpa"] + win["d_tol"] + 3500.0
    if bbox is not None:
        la, lo = np.asarray(archive.lat), np.asarray(archive.lon)
        m &= (la >= bbox[0] - 0.2) & (la <= bbox[1] + 0.2) & (lo >= bbox[2] - 0.3) & (lo <= bbox[3] + 0.3)
    if t_lo is not None:
        tt = t + archive.day0 - win["t_cpa"]
        m &= (tt >= t_lo - 60) & (tt <= t_hi + 60)
    idx = np.nonzero(m)[0]
    del m
    starts = np.asarray(archive.start)
    ac = np.searchsorted(starts, idx, side="right") - 1
    last = (starts[ac] + np.asarray(archive.n)[ac] - 1) == idx
    nxt = np.minimum(idx + 1, len(t) - 1)
    dtn = np.where(last, step_s, t[nxt] - t[idx])
    la0 = np.asarray(archive.lat)[idx].astype(np.float64)
    lo0 = np.asarray(archive.lon)[idx].astype(np.float64)
    a0 = alt[idx].astype(np.float64)
    th = np.radians(trk[idx].astype(np.float64))
    v = gs[idx].astype(np.float64)
    ve, vn = v * np.sin(th), v * np.cos(th)
    out = {k: [] for k in ("lat", "lon", "t0", "ac", "side", "gap")}
    for k in range(max_sub):
        dt = k * step_s
        use = dt < np.minimum(dtn, max_sub * step_s)
        if not use.any():
            break
        coslat = np.cos(np.radians(la0[use]))
        pla = la0[use] + np.degrees(vn[use] * dt / R_EARTH)
        plo = lo0[use] + np.degrees(ve[use] * dt / (R_EARTH * coslat))
        g = ground_fn(pla, plo) if ground_fn else np.zeros(len(pla))
        g = np.maximum(np.nan_to_num(g), 0.0)
        dz = a0[use] - g
        h2 = win["d_cpa"] ** 2 - dz ** 2
        ok = (dz > 0) & (h2 > 0)
        h = np.sqrt(np.where(ok, h2, 0.0))
        pn, pe = -np.sin(th[use]), np.cos(th[use])  # unit vector to the right of the track
        for side in (-1.0, 1.0):
            out["lat"].append((pla + np.degrees(side * h * pn / R_EARTH))[ok])
            out["lon"].append((plo + np.degrees(side * h * pe / (R_EARTH * coslat)))[ok])
            out["t0"].append((t[idx][use] + dt + archive.day0 - win["t_cpa"])[ok])
            out["ac"].append(ac[use][ok])
            out["side"].append(np.full(ok.sum(), side, np.int8))
            out["gap"].append(dtn[use][ok])
    res = {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}
    if bbox is not None:
        b = (res["lat"] >= bbox[0]) & (res["lat"] <= bbox[1]) & (res["lon"] >= bbox[2]) & (res["lon"] <= bbox[3])
        res = {k: v[b] for k, v in res.items()}
    return res


def _keys(lat, lon, t0, cell_deg, t_bin):
    a = np.floor(lat / cell_deg).astype(np.int64) + (1 << 14)
    b = np.floor(((lon + 180.0) % 360.0) / cell_deg).astype(np.int64)
    c = np.floor((t0 - 1.7e9) / t_bin).astype(np.int64)
    return (a << 46) | (b << 26) | c


def join(A, B, pos_tol_m, t_tol_s, distinct=True, chunk=2_000_000):
    """Pairs (i in A, j in B) with listener positions within pos_tol and start times within t_tol."""
    cell_deg = pos_tol_m / 111200.0
    kB = _keys(B["lat"], B["lon"], B["t0"], cell_deg, t_tol_s)
    o = np.argsort(kB)
    kB = kB[o]
    pairs_i, pairs_j = [], []
    neigh = [(da << 46) + (db << 26) + dc for da in (-1, 0, 1) for db in (-2, -1, 0, 1, 2) for dc in (-1, 0, 1)]
    for c0 in range(0, len(A["lat"]), chunk):
        sl = slice(c0, c0 + chunk)
        kA = _keys(A["lat"][sl], A["lon"][sl], A["t0"][sl], cell_deg, t_tol_s)
        for dk in neigh:
            q = kA + dk
            lo = np.searchsorted(kB, q, side="left")
            hi = np.searchsorted(kB, q, side="right")
            cnt = hi - lo
            has = np.nonzero(cnt)[0]
            if not len(has):
                continue
            rep = np.repeat(has, cnt[has])
            off = np.arange(len(rep)) - np.repeat(np.cumsum(cnt[has]) - cnt[has], cnt[has])
            jj = o[np.repeat(lo[has], cnt[has]) + off]
            ii = rep + c0
            dn = np.radians(B["lat"][jj] - A["lat"][ii]) * R_EARTH
            de = np.radians(B["lon"][jj] - A["lon"][ii]) * R_EARTH * np.cos(np.radians(A["lat"][ii]))
            good = (dn * dn + de * de <= pos_tol_m ** 2) & (np.abs(B["t0"][jj] - A["t0"][ii]) <= t_tol_s)
            if distinct:
                good &= A["ac"][ii] != B["ac"][jj]
            pairs_i.append(ii[good])
            pairs_j.append(jj[good])
    if not pairs_i:
        return np.zeros(0, int), np.zeros(0, int)
    i = np.concatenate(pairs_i)
    j = np.concatenate(pairs_j)
    u = np.unique(np.stack([i, j], 1), axis=0)
    return u[:, 0], u[:, 1]


def anchor_search(archive, fit_anchor, fit_other, bbox=None, sys_frac=0.03, wind=15.0, ground_fn=None,
                  block_s=3600.0, progress=None):
    """Candidates (WorldCandidate list) where the anchor pass and the other pass agree.

    Each candidate's position and start time are the anchor's (locked); the
    other pass only confirms. ``aircraft`` is [anchor aircraft, other aircraft].
    Also returns per candidate the start-time and position mismatch of the
    other pass.
    """
    from .adsb_day import WorldCandidate

    wa = pass_window(fit_anchor, sys_frac, wind)
    wb = pass_window(fit_other, sys_frac, wind)
    pos_tol = wa["d_tol"] + wb["d_tol"] + 150.0  # both cross-track tolerances + 1 s stepping
    t_tol = wa["t_tol"] + wb["t_tol"] + 1.0
    day0 = archive.day0
    out, dt_mis, dpos = [], [], []
    seen = set()
    nblk = int(np.ceil((86400 + 7200) / block_s))
    for kb in range(nblk):
        if progress:
            progress(kb / nblk)
        lo_t = day0 - 3600 + kb * block_s
        A = listener_points(archive, wa, bbox, ground_fn=ground_fn, t_lo=lo_t, t_hi=lo_t + block_s)
        A = {k: v[(A["t0"] >= lo_t) & (A["t0"] < lo_t + block_s)] for k, v in A.items()}
        if not len(A["lat"]):
            continue
        B = listener_points(archive, wb, bbox, ground_fn=ground_fn, t_lo=lo_t - t_tol, t_hi=lo_t + block_s + t_tol)
        i, j = join(A, B, pos_tol, t_tol)
        if not len(i):
            continue
        ia, ib = A["ac"][i].astype(np.int64), B["ac"][j].astype(np.int64)
        # one candidate per aircraft pair, side and 3 s of start time (neighbouring seconds converge in the fit)
        key = np.stack([ia, ib, np.floor(A["t0"][i] / 3.0).astype(np.int64), A["side"][i].astype(np.int64)], 1)
        _, first = np.unique(key, axis=0, return_index=True)
        for f in first:
            a, b = i[f], j[f]
            k4 = tuple(int(x) for x in key[f])
            if k4 in seen:
                continue
            seen.add(k4)
            out.append(WorldCandidate(float(A["lat"][a]), float(A["lon"][a]), float(A["t0"][a]), 2, 1,
                                      [archive.meta[int(ia[f])][0], archive.meta[int(ib[f])][0]], 0))
            dt_mis.append(float(B["t0"][b] - A["t0"][a]))
            dn = np.radians(B["lat"][b] - A["lat"][a]) * R_EARTH
            de = np.radians(B["lon"][b] - A["lon"][a]) * R_EARTH * np.cos(np.radians(A["lat"][a]))
            dpos.append(float(np.hypot(dn, de)))
    if progress:
        progress(1.0)
    return out, np.array(dt_mis), np.array(dpos), dict(anchor=wa, other=wb, pos_tol=pos_tol, t_tol=t_tol)
