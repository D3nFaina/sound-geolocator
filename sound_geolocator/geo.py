"""Geodesy helpers: WGS84 lat/lon/alt <-> ECEF, local ENU grids."""

from __future__ import annotations

import numpy as np

_A = 6378137.0
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)


def lla_to_ecef(lat, lon, alt):
    """Degrees, degrees, metres -> ECEF metres. Broadcasts; last axis is xyz."""
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))
    alt = np.asarray(alt, dtype=float)
    s = np.sin(lat)
    n = _A / np.sqrt(1 - _E2 * s * s)
    x = (n + alt) * np.cos(lat) * np.cos(lon)
    y = (n + alt) * np.cos(lat) * np.sin(lon)
    z = (n * (1 - _E2) + alt) * s
    return np.stack(np.broadcast_arrays(x, y, z), axis=-1)


def enu_offsets_to_lla(lat0, lon0, east_m, north_m):
    """Small-offset (tens of km) local tangent plane -> lat/lon degrees."""
    lat0r = np.radians(lat0)
    s = np.sin(lat0r)
    n = _A / np.sqrt(1 - _E2 * s * s)
    m = _A * (1 - _E2) / (1 - _E2 * s * s) ** 1.5
    lat = lat0 + np.degrees(np.asarray(north_m) / m)
    lon = lon0 + np.degrees(np.asarray(east_m) / (n * np.cos(lat0r)))
    return lat, lon


def lla_to_enu_offsets(lat0, lon0, lat, lon):
    """Inverse of :func:`enu_offsets_to_lla` (small offsets)."""
    lat0r = np.radians(lat0)
    s = np.sin(lat0r)
    n = _A / np.sqrt(1 - _E2 * s * s)
    m = _A * (1 - _E2) / (1 - _E2 * s * s) ** 1.5
    north = np.radians(np.asarray(lat) - lat0) * m
    east = np.radians(np.asarray(lon) - lon0) * n * np.cos(lat0r)
    return east, north


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(a))
