"""Predicted Doppler curves of real flights for one listener position and start time.

Used to put candidate flights on top of the spectrogram: if the listener
really was at (lat, lon) and the recording started at ``t_start``, every
flight nearby draws a line whose shape is fixed by geometry; only its tone is
unknown. Each flight is compared with the picked passes it overlaps in time,
and its tone is set to the value that fits that pass best (closed form).
"""

from __future__ import annotations

import numpy as np

from . import DEFAULT_SPEED_OF_SOUND
from .doppler import doppler_ratio
from .geo import lla_to_ecef


def predict_flights(tracks, curves, lat, lon, alt, t_start, duration_s, max_km=15.0, dt=1.0,
                    c=DEFAULT_SPEED_OF_SOUND, min_overlap=0.5):
    """For each flight that comes within ``max_km``: its predicted curve and best-matching pass.

    Returns a list of dicts sorted by closest distance:
    ident, label, typecode, category, t (s into the recording), r (frequency
    ratio), d_min (m), t_cpa (s, heard), match (curve name or None), f0 (Hz),
    rms_hz, rel (rms / f0).
    """
    obs = lla_to_ecef(lat, lon, alt)
    t_rel = np.arange(0.0, duration_s + dt, dt)
    abs_t = t_start + t_rel
    out = []
    for tr in tracks:
        if tr.t[-1] < abs_t[0] - 120 or tr.t[0] > abs_t[-1] + 60:
            continue
        with np.errstate(all="ignore"):
            r, tau, rng, _ = doppler_ratio(tr, obs, abs_t, c, return_extra=True)
        ok = tr.covered(tau) & np.isfinite(r) & (rng < max_km * 1000)
        if ok.sum() < 5:
            continue
        k = int(np.argmin(np.where(ok, rng, np.inf)))
        item = {"ident": tr.ident, "label": tr.label, "type": tr.typecode, "category": tr.category,
                "t": t_rel[ok], "r": r[ok], "d_min": float(rng[k]), "t_cpa": float(t_rel[k]),
                "match": None, "f0": None, "rms_hz": None, "rel": None}
        best = None
        for cv in curves:
            good = np.isfinite(cv.f)
            ct, cf = cv.t[good], cv.f[good]
            if len(ct) < 5:
                continue
            idx = np.round(ct / dt).astype(int)
            inside = (idx >= 0) & (idx < len(t_rel))
            idx = idx[inside]
            covd = np.zeros(len(t_rel), bool)
            covd[np.nonzero(ok)[0]] = True
            m = covd[idx]
            if m.sum() < min_overlap * len(ct):
                continue
            rr = r[idx[m]]
            ff = cf[inside][m]
            f0 = float(np.sum(ff * rr) / np.sum(rr * rr))
            rms = float(np.sqrt(np.mean((ff - f0 * rr) ** 2)))
            if best is None or rms / f0 < best[2] / best[1]:
                best = (cv.name, f0, rms)
        if best:
            item.update(match=best[0], f0=best[1], rms_hz=best[2], rel=best[2] / best[1])
        out.append(item)
    out.sort(key=lambda x: x["d_min"])
    return out
