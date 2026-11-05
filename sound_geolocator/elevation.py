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


TERRARIUM = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"


def terrarium_elevation(lat, lon, zoom=9, cache_dir=None, progress=None):
    """Ground height (m above sea level) at many points, worldwide, from the free AWS terrain tiles.

    Uses the Mapzen/AWS "terrarium" PNG tiles (SRTM and others; about 300 m
    per pixel at zoom 9 in mid latitudes, good enough for a listener height).
    Only the tiles that contain points are downloaded; ``cache_dir`` keeps
    them for next time.
    """
    import io

    import requests
    from PIL import Image

    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    n = 2 ** zoom
    xf = (lon + 180.0) / 360.0 * n
    la = np.radians(np.clip(lat, -85.05, 85.05))
    yf = (1.0 - np.log(np.tan(la) + 1.0 / np.cos(la)) / np.pi) / 2.0 * n
    tx, ty = np.floor(xf).astype(int) % n, np.clip(np.floor(yf).astype(int), 0, n - 1)
    px = np.clip(((xf - np.floor(xf)) * 256).astype(int), 0, 255)
    py = np.clip(((yf - np.floor(yf)) * 256).astype(int), 0, 255)
    out = np.full(lat.shape, np.nan)
    key = (tx.astype(np.int64) * n + ty).ravel()
    order = np.argsort(key, kind="stable")
    tiles, first, counts = np.unique(key[order], return_index=True, return_counts=True)
    flat = out.ravel()
    pxf, pyf = px.ravel(), py.ravel()
    cache = Path(cache_dir).expanduser() if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    for k, kk in enumerate(tiles):
        if progress:
            progress(k / len(tiles))
        x, y = int(kk // n), int(kk % n)
        f = cache / f"terrarium_{zoom}_{x}_{y}.png" if cache else None
        if f is not None and f.exists():
            raw = f.read_bytes()
        else:
            raw = None
            for attempt in range(4):
                try:
                    r = s.get(TERRARIUM.format(z=zoom, x=x, y=y), timeout=30)
                    r.raise_for_status()
                    raw = r.content
                    break
                except Exception:
                    time.sleep(2 ** attempt)
            if raw is None:
                continue
            if f is not None:
                f.write_bytes(raw)
        im = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), float)
        h = im[..., 0] * 256 + im[..., 1] + im[..., 2] / 256 - 32768
        m = order[first[k]:first[k] + counts[k]]
        flat[m] = h[pyf[m], pxf[m]]
    return flat.reshape(lat.shape)


class TerrainTiles:
    """Ground height lookups from terrarium tiles, decoded once and kept in memory: ``tiles(lat, lon)``."""

    def __init__(self, zoom=8, cache_dir="~/.cache/sound_geolocator/terrarium"):
        self.zoom, self.cache_dir, self.mem = zoom, cache_dir, {}

    def __call__(self, lat, lon):
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        n = 2 ** self.zoom
        xf = (lon + 180.0) / 360.0 * n
        la = np.radians(np.clip(lat, -85.05, 85.05))
        yf = (1.0 - np.log(np.tan(la) + 1.0 / np.cos(la)) / np.pi) / 2.0 * n
        tx, ty = np.floor(xf).astype(np.int64) % n, np.clip(np.floor(yf).astype(np.int64), 0, n - 1)
        px = np.clip(((xf - np.floor(xf)) * 256).astype(int), 0, 255)
        py = np.clip(((yf - np.floor(yf)) * 256).astype(int), 0, 255)
        key = tx * n + ty
        out = np.full(lat.shape, np.nan)
        order = np.argsort(key, kind="stable")
        tiles, first, counts = np.unique(key[order], return_index=True, return_counts=True)
        for k, kk in enumerate(tiles):
            h = self.mem.get(int(kk))
            if h is None:
                x, y = int(kk // n), int(kk % n)
                # fetch the whole tile through terrarium_elevation (cached on disk), as a 256 x 256 grid
                ii, jj = np.mgrid[0:256, 0:256]
                yy = (y + (ii + 0.5) / 256) / n
                xx = (x + (jj + 0.5) / 256) / n
                glat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * yy))))
                glon = xx * 360.0 - 180.0
                h = terrarium_elevation(glat, glon, zoom=self.zoom, cache_dir=self.cache_dir)
                self.mem[int(kk)] = h
            m = order[first[k]:first[k] + counts[k]]
            out[m] = h[py[m], px[m]]
        return out
