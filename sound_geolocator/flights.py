"""Aircraft tracks: containers, loaders (CSV / JSON / OpenSky) and interpolation."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.interpolate import UnivariateSpline

from .geo import lla_to_ecef

FT = 0.3048


@dataclass
class Track:
    """One aircraft's trajectory. ``t`` is unix seconds (UTC), altitude in metres MSL."""

    ident: str
    t: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    callsign: str = ""
    typecode: str = ""  # ICAO aircraft type designator, e.g. "AT76", "A320"
    position_noise_m: float = 15.0
    max_gap_s: float = 60.0
    _splines: list = field(default=None, repr=False)

    def __post_init__(self):
        order = np.argsort(self.t)
        self.t = np.asarray(self.t, float)[order]
        self.lat = np.asarray(self.lat, float)[order]
        self.lon = np.asarray(self.lon, float)[order]
        self.alt = np.asarray(self.alt, float)[order]
        keep = np.concatenate([[True], np.diff(self.t) > 0])
        self.t, self.lat, self.lon, self.alt = (a[keep] for a in (self.t, self.lat, self.lon, self.alt))

    @property
    def label(self):
        return f"{self.callsign.strip() or self.ident}"

    def _fit(self):
        if self._splines is None:
            xyz = lla_to_ecef(self.lat, self.lon, self.alt)
            k = min(3, len(self.t) - 1)
            s = len(self.t) * self.position_noise_m ** 2
            self._splines = [UnivariateSpline(self.t, xyz[:, i], k=k, s=s) for i in range(3)]
        return self._splines

    def covered(self, t):
        """True where ``t`` lies inside the track and not inside a data gap."""
        t = np.asarray(t, float)
        ok = (t >= self.t[0]) & (t <= self.t[-1])
        idx = np.clip(np.searchsorted(self.t, t), 1, len(self.t) - 1)
        gap = self.t[idx] - self.t[idx - 1]
        return ok & (gap <= self.max_gap_s)

    def position(self, t):
        sp = self._fit()
        return np.stack([s(t) for s in sp], axis=-1)

    def velocity(self, t):
        sp = self._fit()
        return np.stack([s.derivative()(t) for s in sp], axis=-1)

    def slice(self, t0, t1):
        m = (self.t >= t0) & (self.t <= t1)
        return Track(self.ident, self.t[m], self.lat[m], self.lon[m], self.alt[m], self.callsign,
                     self.typecode, self.position_noise_m, self.max_gap_s)

    @property
    def category(self):
        from .aircraft_types import classify

        return classify(self.typecode)


# ----------------------------------------------------------------------------- loaders

def _parse_time(v: str) -> float:
    v = v.strip()
    try:
        return float(v)
    except ValueError:
        d = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.timestamp()


_ALIASES = {
    "time": ["time", "timestamp", "t", "unix_time", "ts", "utc"],
    "ident": ["icao24", "hex", "icao", "ident", "id", "flight_id", "aircraft"],
    "callsign": ["callsign", "flight", "call_sign"],
    "typecode": ["typecode", "type", "aircraft_type", "icao_type", "actype", "model"],
    "lat": ["lat", "latitude"],
    "lon": ["lon", "lng", "long", "longitude"],
    "alt_m": ["alt_m", "geoaltitude", "geo_altitude", "altitude_m", "baroaltitude", "baro_altitude"],
    "alt_ft": ["alt_ft", "altitude", "alt", "alt_baro", "alt_geom", "altitude_ft"],
}


def _col(header, key):
    low = {h.lower().strip(): h for h in header}
    for a in _ALIASES[key]:
        if a in low:
            return low[a]
    return None


def load_tracks_csv(path: str | Path) -> list[Track]:
    """Load tracks from a CSV of position reports.

    Recognised columns (case-insensitive): time (unix s or ISO 8601), icao24/hex,
    callsign, lat, lon, and either alt_m / geoaltitude / baroaltitude (metres,
    OpenSky style) or altitude / alt_baro (feet, ADS-B Exchange / FR24 style).
    """
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []
    header = rows[0].keys()
    ct, ci, cc = _col(header, "time"), _col(header, "ident"), _col(header, "callsign")
    cla, clo = _col(header, "lat"), _col(header, "lon")
    cam, caf = _col(header, "alt_m"), _col(header, "alt_ft")
    cty = _col(header, "typecode")
    if not (ct and cla and clo and (cam or caf)):
        raise ValueError(f"{path}: need time, lat, lon and altitude columns, got {list(header)}")
    groups: dict[str, dict] = {}
    for r in rows:
        try:
            lat, lon = float(r[cla]), float(r[clo])
            alt = float(r[cam]) if cam and r[cam] not in ("", None) else float(r[caf]) * FT
        except (TypeError, ValueError):
            continue  # skip rows without a position (e.g. "ground" altitude)
        key = (r[ci] if ci else "aircraft").strip().lower()
        g = groups.setdefault(key, {"t": [], "lat": [], "lon": [], "alt": [], "cs": "", "ty": ""})
        g["t"].append(_parse_time(r[ct])); g["lat"].append(lat); g["lon"].append(lon); g["alt"].append(alt)
        if cc and r[cc]:
            g["cs"] = r[cc].strip()
        if cty and r[cty]:
            g["ty"] = r[cty].strip().upper()
    return [Track(k, g["t"], g["lat"], g["lon"], g["alt"], g["cs"], g["ty"])
            for k, g in groups.items() if len(g["t"]) >= 4]


def save_tracks_csv(tracks: list[Track], path: str | Path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "icao24", "callsign", "typecode", "lat", "lon", "alt_m"])
        for tr in tracks:
            for i in range(len(tr.t)):
                w.writerow([f"{tr.t[i]:.2f}", tr.ident, tr.callsign, tr.typecode, f"{tr.lat[i]:.6f}",
                            f"{tr.lon[i]:.6f}", f"{tr.alt[i]:.1f}"])


# ----------------------------------------------------------------------------- OpenSky

OPENSKY_API = "https://opensky-network.org/api"
OPENSKY_TOKEN = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"


def _opensky_session():
    import requests

    s = requests.Session()
    cid, secret = os.environ.get("OPENSKY_CLIENT_ID"), os.environ.get("OPENSKY_CLIENT_SECRET")
    if cid and secret:
        r = requests.post(OPENSKY_TOKEN, data={"grant_type": "client_credentials", "client_id": cid,
                                               "client_secret": secret}, timeout=30)
        r.raise_for_status()
        s.headers["Authorization"] = f"Bearer {r.json()['access_token']}"
    return s


def fetch_opensky_states(lat, lon, radius_km, t_start, t_end, step_s=10, session=None) -> list[Track]:
    """Poll OpenSky ``/states/all`` over a bounding box and time span.

    The REST API only serves recent data (roughly the last hour, and the
    ``time`` parameter needs an account). For older recordings use the
    OpenSky historical database (``pyopensky`` / Trino) or an ADS-B Exchange
    export and :func:`load_tracks_csv`. Set OPENSKY_CLIENT_ID and
    OPENSKY_CLIENT_SECRET for authenticated access.
    """
    s = session or _opensky_session()
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * np.cos(np.radians(lat)))
    params = dict(lamin=lat - dlat, lamax=lat + dlat, lomin=lon - dlon, lomax=lon + dlon)
    rows = []
    for t in np.arange(t_start, t_end + step_s, step_s):
        r = s.get(f"{OPENSKY_API}/states/all", params={**params, "time": int(t)}, timeout=30)
        r.raise_for_status()
        for sv in r.json().get("states") or []:
            icao, cs, tpos, lo, la, baro, geo = sv[0], sv[1], sv[3], sv[5], sv[6], sv[7], sv[13]
            alt = geo if geo is not None else baro
            if None in (tpos, lo, la, alt) or sv[8]:  # sv[8] = on_ground
                continue
            rows.append((icao, cs or "", float(tpos), la, lo, float(alt)))
    return _rows_to_tracks(rows)


def fetch_opensky_track(icao24: str, time: float, session=None) -> Track | None:
    """``/tracks/all``: waypoints of one flight (sparse; best for the last 30 days)."""
    s = session or _opensky_session()
    r = s.get(f"{OPENSKY_API}/tracks/all", params={"icao24": icao24.lower(), "time": int(time)}, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    j = r.json()
    pts = [p for p in j.get("path", []) if p[1] is not None and p[3] is not None and not p[5]]
    if len(pts) < 4:
        return None
    a = np.array([[p[0], p[1], p[2], p[3]] for p in pts], float)
    return Track(icao24, a[:, 0], a[:, 1], a[:, 2], a[:, 3], j.get("callsign") or "", max_gap_s=600)


def _rows_to_tracks(rows):
    groups: dict[str, list] = {}
    for icao, cs, t, la, lo, alt in rows:
        groups.setdefault(icao, []).append((t, la, lo, alt, cs))
    out = []
    for icao, pts in groups.items():
        pts = sorted(set(pts))
        if len(pts) >= 4:
            a = np.array([p[:4] for p in pts], float)
            out.append(Track(icao, a[:, 0], a[:, 1], a[:, 2], a[:, 3], pts[-1][4]))
    return out


def load_tracks(path: str | Path) -> list[Track]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        return [Track(d["ident"], d["t"], d["lat"], d["lon"], d["alt"], d.get("callsign", ""), d.get("typecode", ""))
                for d in data]
    return load_tracks_csv(path)
