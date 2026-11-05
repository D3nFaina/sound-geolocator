"""Ground elevation for candidate listener positions (digital elevation model).

The listener's height changes the slant distance to each aircraft, which
matters in hilly terrain. A :class:`DEM` is a callable ``(lat, lon) ->
metres`` that :func:`sound_geolocator.locate.locate` accepts as
``observer_alt_m``.

Datums: SRTM / Copernicus / EU-DEM heights are above mean sea level (the
geoid), while ADS-B geometric altitude (OpenSky ``geoaltitude``) is above the
WGS84 ellipsoid. Pass ``geoid_offset_m`` (the local geoid height, about +45 m
in northern Italy, look it up for your area) to put the DEM on the same datum
as the tracks. Barometric altitudes are pressure-based and differ again.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .geo import enu_offsets_to_lla

OPENTOPODATA = "https://api.opentopodata.org/v1/{dataset}"


class DEM:
    def __init__(self, lats, lons, heights, height_above_ground_m=1.5, geoid_offset_m=0.0):
        self.lats = np.asarray(lats, float)
        self.lons = np.asarray(lons, float)
        self.heights = np.asarray(heights, float)
        self.height_above_ground_m = height_above_ground_m
        self.geoid_offset_m = geoid_offset_m
        self._interp = RegularGridInterpolator((self.lats, self.lons), self.heights, bounds_error=False,
                                               fill_value=None)

    def __call__(self, lat, lon):
        lat, lon = np.broadcast_arrays(np.asarray(lat, float), np.asarray(lon, float))
        h = self._interp(np.stack([lat.ravel(), lon.ravel()], axis=-1)).reshape(lat.shape)
        return h + self.height_above_ground_m + self.geoid_offset_m

    # ------------------------------------------------------------------ I/O
    def save(self, path):
        np.savez_compressed(path, lats=self.lats, lons=self.lons, heights=self.heights)

    @classmethod
    def load(cls, path, **kw):
        d = np.load(path)
        return cls(d["lats"], d["lons"], d["heights"], **kw)

    @classmethod
    def from_geotiff(cls, path, **kw):
        """Read a GeoTIFF in geographic (lat/lon) coordinates. Needs ``rasterio``."""
        import rasterio

        with rasterio.open(path) as ds:
            h = ds.read(1).astype(float)
            if ds.nodata is not None:
                h[h == ds.nodata] = np.nan
            tr = ds.transform
            lons = tr.c + tr.a * (np.arange(ds.width) + 0.5)
            lats = tr.f + tr.e * (np.arange(ds.height) + 0.5)
        if lats[0] > lats[-1]:
            lats, h = lats[::-1], h[::-1]
        return cls(lats, lons, np.nan_to_num(h, nan=float(np.nanmedian(h))), **kw)

    @classmethod
    def from_opentopodata(cls, lat0, lon0, radius_km, step_m=500.0, dataset="srtm30m", cache=None,
                          url=OPENTOPODATA, session=None, **kw):
        """Sample an area from the free OpenTopoData API (100 points per request, 1 request/s).

        ``dataset`` e.g. "srtm30m" (worldwide), "eudem25m" (Europe), "ned10m" (USA).
        ``cache`` is an .npz path to reuse between runs.
        """
        if cache and Path(cache).exists():
            return cls.load(cache, **kw)
        import requests

        s = session or requests.Session()
        half = radius_km * 1000 + 2 * step_m
        offs = np.arange(-half, half + 1e-6, step_m)
        lats, _ = enu_offsets_to_lla(lat0, lon0, 0 * offs, offs)
        _, lons = enu_offsets_to_lla(lat0, lon0, offs, 0 * offs)
        LA, LO = np.meshgrid(lats, lons, indexing="ij")
        pts = np.stack([LA.ravel(), LO.ravel()], -1)
        out = np.empty(len(pts))
        for k in range(0, len(pts), 100):
            chunk = pts[k:k + 100]
            q = "|".join(f"{a:.6f},{b:.6f}" for a, b in chunk)
            r = s.get(url.format(dataset=dataset), params={"locations": q}, timeout=60)
            r.raise_for_status()
            out[k:k + len(chunk)] = [x["elevation"] if x["elevation"] is not None else np.nan for x in r.json()["results"]]
            time.sleep(1.05)
        h = out.reshape(LA.shape)
        h = np.where(np.isfinite(h), h, np.nanmedian(h))
        dem = cls(lats, lons, h, **kw)
        if cache:
            dem.save(cache)
        return dem
