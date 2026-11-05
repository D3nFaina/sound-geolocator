"""A whole day of worldwide ADS-B in a compact, quickly searchable form.

The free daily archives of adsb.lol (github.com/adsblol/globe_history_YYYY,
release ``vYYYY.MM.DD-planes-readsb-prod-0``) are a tar split into parts
(``.tar.aa``, ``.tar.ab`` …) holding one gzip JSON trace per aircraft. That
is about 4 GB and far too slow to scan for every query, so :func:`build_day`
streams it once and keeps only what a sound search can use:

* airborne points below ``max_alt_m`` (an aircraft much higher than that is
  not heard as a clear Doppler pass),
* thinned to at most one point every ``min_dt`` seconds,
* time, lat, lon, altitude (geometric when present), ground speed, track.

The result is a folder ``<name>.sgday`` of plain ``.npy`` arrays sorted by
aircraft, opened memory-mapped by :class:`DayArchive`. It answers two
questions:

* :meth:`DayArchive.tracks`: the flights near a place and time, as
  :class:`~sound_geolocator.flights.Track` objects for :func:`locate`.
* :meth:`DayArchive.search_world`: where on Earth and when could the
  recording have been made, given the fitted passes (speed, closest
  distance, time of closest approach in the clip). Each aircraft sample
  that is fast and low enough "votes" for the places abeam of it at the
  right distance and for the clip start time that would put the pass at the
  right moment; places and times where every pass gets a vote are the
  candidates. No assumption about the country is made.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .aircraft_types import classify

FT = 0.3048

KT = 0.514444


# ----------------------------------------------------------------------------- reading the tar parts

class _Chain(io.RawIOBase):
    """Read several files back to back, as one seekable stream (split tar parts)."""

    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]
        self.sizes = [p.stat().st_size for p in self.paths]
        self.total = sum(self.sizes)
        self.pos = 0
        self.k = 0
        self.f = open(self.paths[0], "rb")

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        if whence == 1:
            off += self.pos
        elif whence == 2:
            off += self.total
        k, base = 0, 0
        while k < len(self.sizes) - 1 and off >= base + self.sizes[k]:
            base += self.sizes[k]
            k += 1
        if k != self.k:
            self.f.close()
            self.k = k
            self.f = open(self.paths[k], "rb")
        self.f.seek(off - base)
        self.pos = off
        return off

    def readinto(self, b):
        while True:
            n = self.f.readinto(b)
            if n:
                self.pos += n
                return n
            if self.k + 1 >= len(self.paths):
                return 0
            self.f.close()
            self.k += 1
            self.f = open(self.paths[self.k], "rb")

    def close(self):
        try:
            self.f.close()
        finally:
            super().close()


def find_parts(path) -> list[Path]:
    """``x.tar.aa`` -> [x.tar.aa, x.tar.ab, …]; a plain .tar -> [it]."""
    p = Path(path).expanduser()
    if p.suffix[1:].isalpha() and len(p.suffix) == 3 and p.stem.endswith(".tar"):
        parts = sorted(p.parent.glob(p.stem + ".??"))
        return [q for q in parts if len(q.suffix) == 3 and q.suffix[1:].isalpha()]
    if p.suffix == ".tar" and not p.exists():
        parts = sorted(p.parent.glob(p.name + ".??"))
        if parts:
            return parts
    return [p]


def _parse_trace(raw: bytes, max_alt_m: float, min_dt: float):
    """One trace_full JSON -> (meta, array[n, 6]) or None."""
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    d = json.loads(raw)
    base = float(d.get("timestamp", 0))
    rows = []
    last = -1e18
    cs = ""
    for r in d.get("trace", ()):
        if not cs and len(r) > 8 and isinstance(r[8], dict) and r[8].get("flight"):
            cs = r[8]["flight"].strip()
        la, lo, baro = r[1], r[2], r[3]
        if la is None or lo is None:
            continue
        geo = r[10] if len(r) > 10 else None
        alt = geo if isinstance(geo, (int, float)) else baro
        if not isinstance(alt, (int, float)):
            continue  # on the ground
        alt *= FT
        if alt > max_alt_m:
            continue
        t = base + float(r[0])
        if t - last < min_dt:
            continue
        last = t
        gs = r[4] if isinstance(r[4], (int, float)) else np.nan
        trk = r[5] if isinstance(r[5], (int, float)) else np.nan
        rows.append((t, la, lo, alt, gs * KT, trk))
    if len(rows) < 4:
        return None
    meta = (str(d.get("icao", "")).lower(), str(d.get("t", "") or "").upper(), cs)
    return meta, np.asarray(rows, float)


def _parse_member(args):
    raw, max_alt_m, min_dt = args
    try:
        return _parse_trace(raw, max_alt_m, min_dt)
    except Exception:
        return None


_COLS = (("t", np.float64), ("lat", np.float32), ("lon", np.float32), ("alt", np.int16), ("gs", np.float16),
         ("trk", np.float16))


def _write_shard(path, metas, arrays):
    allp = np.concatenate(arrays) if arrays else np.zeros((0, 6))
    cols = {k: (np.round(allp[:, i]) if k == "alt" else allp[:, i]).astype(dt) for i, (k, dt) in enumerate(_COLS)}
    np.savez(path, n=np.array([len(a) for a in arrays], np.int64), **cols)
    Path(str(path) + ".json").write_text(json.dumps(metas))


def build_day(source, out=None, max_alt_m=7000.0, min_dt=3.0, workers=None, progress=None,
              time_budget_s=None, chunk=4000) -> Path | None:
    """Convert one adsb.lol / readsb globe_history day into a ``.sgday`` folder.

    ``source``: the first tar part (``….tar.aa``), a whole ``.tar``, or an
    extracted folder containing ``traces/``. Returns the output folder.

    The conversion is resumable: work is written to ``<out>/_shards`` as it
    goes. With ``time_budget_s`` it stops after about that long and returns
    None; calling it again carries on where it stopped.
    """
    import time
    from concurrent.futures import ProcessPoolExecutor

    t_begin = time.time()
    source = Path(source).expanduser()
    if out is None:
        stem = source.name.split(".tar")[0] if not source.is_dir() else source.name
        out = source.parent / f"{stem}.sgday"
    out = Path(out)
    shards = out / "_shards"
    shards.mkdir(parents=True, exist_ok=True)
    state_p = out / "_build.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {"offset": 0, "shard": 0, "file": 0,
                                                                         "aircraft": 0, "finished": False}
    workers = workers or max(1, (os.cpu_count() or 2) - 1)

    def save_state():
        state_p.write_text(json.dumps(state))

    def flush(metas, arrays, next_offset=None, next_file=None):
        _write_shard(shards / f"shard_{state['shard']:05d}.npz", metas, arrays)
        state["shard"] += 1
        state["aircraft"] += len(metas)
        if next_offset is not None:
            state["offset"] = next_offset
        if next_file is not None:
            state["file"] = next_file
        save_state()

    def run(raws, ex):
        jobs = [(r, max_alt_m, min_dt) for r in raws]
        res = ex.map(_parse_member, jobs, chunksize=16) if ex else map(_parse_member, jobs)
        metas, arrays = [], []
        for r in res:
            if r is not None:
                metas.append(r[0])
                arrays.append(r[1])
        return metas, arrays

    ex = ProcessPoolExecutor(workers) if workers > 1 else None
    try:
        if not state["finished"]:
            if source.is_dir():
                files = sorted(source.rglob("trace_full_*"))
                k = state["file"]
                while k < len(files):
                    part = files[k:k + chunk]
                    metas, arrays = run([p.read_bytes() for p in part], ex)
                    k += len(part)
                    flush(metas, arrays, next_file=k)
                    if progress:
                        progress(k / len(files), state["aircraft"])
                    if time_budget_s and time.time() - t_begin > time_budget_s:
                        return None
            else:
                chain = _Chain(find_parts(source))
                state["total"] = chain.total
                base = state["offset"]
                chain.seek(base)
                tf = tarfile.open(fileobj=chain, mode="r|")
                raws, last_report = [], 0.0
                for m in tf:
                    if m.isfile() and "trace_full_" in m.name:
                        raws.append(tf.extractfile(m).read())
                    if len(raws) >= chunk:
                        # every member up to and including m is read; the next header starts after its data
                        nxt = base + m.offset_data + ((m.size + 511) // 512) * 512
                        metas, arrays = run(raws, ex)
                        raws = []
                        flush(metas, arrays, next_offset=nxt)
                        if progress:
                            progress(nxt / chain.total, state["aircraft"])
                        if time_budget_s and time.time() - t_begin > time_budget_s:
                            return None
                if raws:
                    metas, arrays = run(raws, ex)
                    flush(metas, arrays)
            state["finished"] = True
            save_state()
    finally:
        if ex:
            ex.shutdown()

    _merge_shards(out, source.name, max_alt_m, min_dt)
    try:  # the shards are a copy of the data: free the space if we may delete
        import shutil

        shutil.rmtree(shards)
    except OSError:
        pass
    if progress:
        progress(1.0, state["aircraft"])
    return out


def _merge_shards(out, source_name, max_alt_m, min_dt):
    """Concatenate the shards column by column (keeps memory low) and write the archive."""
    shards = sorted((out / "_shards").glob("shard_*.npz"))
    metas, ns = [], []
    for p in shards:
        metas += json.loads(Path(str(p) + ".json").read_text())
        with np.load(p) as z:
            ns.append(z["n"])
    if not metas:
        raise ValueError("no aircraft traces found")
    n = np.concatenate(ns)
    start = np.concatenate([[0], np.cumsum(n)[:-1]]).astype(np.int64)
    tmin = min(float(np.load(p)["t"].min()) for p in shards if np.load(p)["t"].size)
    day0 = float(np.floor(tmin / 86400) * 86400)
    if tmin - day0 > 80000:  # traces start just before midnight: belongs to the next day
        day0 += 86400
    for k, dt in _COLS:
        parts = []
        for p in shards:
            with np.load(p) as z:
                parts.append(z[k])
        col = np.concatenate(parts)
        if k == "t":
            col = (col - day0).astype(np.float32)
        np.save(out / f"{k}.npy", col)
        del col, parts
    t = np.load(out / "t.npy", mmap_mode="r")
    lat = np.load(out / "lat.npy", mmap_mode="r")
    lon = np.load(out / "lon.npy", mmap_mode="r")
    box = np.stack([np.minimum.reduceat(lat, start), np.maximum.reduceat(lat, start),
                    np.minimum.reduceat(lon, start), np.maximum.reduceat(lon, start),
                    np.minimum.reduceat(t, start), np.maximum.reduceat(t, start)], 1).astype(np.float32)
    np.save(out / "ac_box.npy", box)
    np.save(out / "ac_start.npy", start)
    np.save(out / "ac_n.npy", n.astype(np.int32))
    (out / "aircraft.json").write_text(json.dumps({"day0": day0, "max_alt_m": max_alt_m, "min_dt": min_dt,
                                                   "source": source_name, "aircraft": metas}))


# ----------------------------------------------------------------------------- using it

@dataclass
class WorldCandidate:
    lat: float
    lon: float
    t_start: float  # unix time of the recording start
    n_passes: int  # how many passes found a matching aircraft here
    votes: int
    aircraft: list  # per pass: icao of one matching aircraft (or "")
    busy: int = 0  # low aircraft passing nearby during the recording (fewer = more plausible)

    def to_json(self):
        return {"lat": self.lat, "lon": self.lon, "t_start": self.t_start, "n_passes": self.n_passes,
                "votes": self.votes, "aircraft": self.aircraft, "busy": self.busy}


class DayArchive:
    def __init__(self, folder):
        self.folder = Path(folder).expanduser()
        info = json.loads((self.folder / "aircraft.json").read_text())
        self.day0 = info["day0"]
        self.info = info
        self.meta = info["aircraft"]
        ld = lambda k: np.load(self.folder / f"{k}.npy", mmap_mode="r")  # noqa: E731
        self.t, self.lat, self.lon, self.alt = ld("t"), ld("lat"), ld("lon"), ld("alt")
        self.gs, self.trk = ld("gs"), ld("trk")
        self.start, self.n, self.box = ld("ac_start"), ld("ac_n"), np.asarray(ld("ac_box"))
        self.category = np.array([classify(m[1]) for m in self.meta])

    def __len__(self):
        return len(self.meta)

    @property
    def n_points(self):
        return int(self.t.shape[0])

    def summary(self):
        from datetime import datetime, timezone

        day = datetime.fromtimestamp(self.day0, timezone.utc).strftime("%Y-%m-%d")
        cats = {c: int((self.category == c).sum()) for c in ("prop", "jet", "heli", "unknown")}
        return {"day": day, "aircraft": len(self), "points": self.n_points, "categories": cats,
                "max_alt_m": self.info["max_alt_m"], "folder": str(self.folder)}

    def _points(self, i):
        s, n = int(self.start[i]), int(self.n[i])
        return slice(s, s + n)

    def tracks(self, t0, t1, lat, lon, radius_km=40.0, keep=None) -> list:
        """Flights within ``radius_km`` of (lat, lon) between unix times t0 and t1."""
        from .flights import Track
        from .geo import lla_to_enu_offsets

        a, b = t0 - self.day0, t1 - self.day0
        dlat = radius_km / 111.2
        dlon = radius_km / (111.2 * max(np.cos(np.radians(lat)), 0.05))
        bx = self.box
        lon_w = ((bx[:, 2] - lon + 180) % 360) - 180, ((bx[:, 3] - lon + 180) % 360) - 180
        sel = np.where((bx[:, 4] <= b) & (bx[:, 5] >= a) & (bx[:, 0] <= lat + dlat) & (bx[:, 1] >= lat - dlat)
                       & (lon_w[0] <= dlon) & (lon_w[1] >= -dlon))[0]
        out = []
        for i in sel:
            if keep and self.category[i] not in keep:
                continue
            sl = self._points(i)
            t = np.asarray(self.t[sl], float)
            m = (t >= a - 120) & (t <= b + 120)
            if m.sum() < 4:
                continue
            la, lo = np.asarray(self.lat[sl], float)[m], np.asarray(self.lon[sl], float)[m]
            e, nn = lla_to_enu_offsets(lat, lon, la, lo)
            if np.min(np.hypot(e, nn)) > radius_km * 1000:
                continue
            icao, typ, cs = self.meta[i]
            out.append(Track(icao, t[m] + self.day0, la, lo, np.asarray(self.alt[sl], float)[m], cs, typ,
                             max_gap_s=120))
        return out

    def _low_index(self, max_alt_m=4500.0, cell_km=3.0, t_bin=300.0):
        """Sorted (cell, 5-min bin) keys of low aircraft, with how many distinct aircraft each holds.

        Altitudes are above sea level and the ground height is unknown, so "low" is generous
        (4.5 km): it still catches the approach traffic of a high airport such as Denver.
        """
        key = (max_alt_m, cell_km, t_bin)
        if getattr(self, "_low", None) is not None and self._low[0] == key:
            return self._low
        alt = np.asarray(self.alt)
        idx = np.nonzero(alt <= max_alt_m)[0]
        ac = np.searchsorted(np.asarray(self.start), idx, side="right") - 1
        k = _key(np.asarray(self.lat)[idx].astype(np.float64), np.asarray(self.lon)[idx].astype(np.float64),
                 np.asarray(self.t)[idx].astype(np.float64), cell_km / 111.2, t_bin)
        pair = np.unique(np.stack([k, ac], 1), axis=0)  # one entry per aircraft per cell and bin
        uk, cnt = np.unique(pair[:, 0], return_counts=True)
        self._low = (key, uk, cnt)
        return self._low

    def low_traffic(self, keys, cell_deg, t_bin, recording_s=3600.0):
        """For search keys (place + start time): low aircraft within ~1 cell during the recording.

        Counts (aircraft, 3 km cell, 5 min) entries in the 3 x 3 cells around
        the place, so one aircraft crossing the area counts about 2 to 4 times.
        """
        (_, cell_km, tb), uk, cnt = self._low_index(cell_km=cell_deg * 111.2)
        keys = np.asarray(keys, np.int64)
        a = (keys >> _SH_LAT)
        b = (keys >> _SH_LON) & ((1 << (_SH_LAT - _SH_LON)) - 1)
        c = (keys & ((1 << _SH_LON) - 1)) - (1 << 18)
        t0 = (c + 0.5) * t_bin
        b0 = np.floor(t0 / tb).astype(np.int64)
        nb = int(np.ceil((recording_s or 3600.0) / tb)) + 1
        total = np.zeros(len(keys), np.int64)
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                for j in range(nb):
                    q = ((a + da) << _SH_LAT) | ((b + db) << _SH_LON) | (b0 + j + (1 << 18))
                    pos = np.minimum(np.searchsorted(uk, q), len(uk) - 1)
                    total += np.where(uk[pos] == q, cnt[pos], 0)
        return total

    def search_world(self, passes, t_lo=None, t_hi=None, speed_tol=0.4, dist_factor=1.4, cell_km=3.0, t_bin=20.0,
                     c=340.0, max_ground_m=1800.0, n_best=30, distinct=True,
                     recording_s=3600.0, progress=None) -> list[WorldCandidate]:
        """Places and start times where aircraft match every fitted pass.

        ``passes``: list of dicts with ``t_cpa`` (s into the recording),
        ``speed`` (m/s), ``d_cpa`` (m, slant distance at closest approach), as
        returned by :func:`~sound_geolocator.doppler.fit_flyover` (``t_cpa`` is
        when the aircraft *emitted* the closest-approach sound),
        and optionally ``types`` (e.g. ["prop"]). ``t_lo``/``t_hi`` bound the
        recording start (unix); default: the whole day.

        For every aircraft sample whose ground speed matches a pass and
        which is low enough to be ``d_cpa`` away from someone on the ground
        (ground height unknown, up to ``max_ground_m``), the listener could
        be abeam of it on either side, at the horizontal offset that puts it
        between ``d_cpa / dist_factor`` and ``d_cpa * dist_factor`` away (one
        fitted curve pins speed and distance only loosely); and the recording
        would then have started at
        t_sample − t_cpa. Those (place, start time) cells get a
        vote per pass; cells voted by every pass are returned, best first.
        """
        import heapq

        if t_lo is None:
            t_lo, t_hi = self.day0 - 3600, self.day0 + 86400 + 3600
        lo_t, hi_t = t_lo - self.day0, t_hi - self.day0
        cell_deg = cell_km / 111.2
        gs_all = np.asarray(self.gs, np.float32)
        alt_all = np.asarray(self.alt, np.float32)
        trk_ok = np.isfinite(np.asarray(self.trk, np.float32))
        t_all = np.asarray(self.t, np.float32)
        starts = np.asarray(self.start)
        ac_of_pt = None
        # 1) per pass: which samples can vote, and for which start time
        sel = []
        for k, p in enumerate(passes):
            v, d, tc = float(p["speed"]), float(p["d_cpa"]), float(p["t_cpa"])
            m = (gs_all >= v * (1 - speed_tol)) & (gs_all <= v * (1 + speed_tol))
            m &= alt_all <= dist_factor * d + max_ground_m  # can't be further above the listener than d
            m &= trk_ok
            ts = t_all - np.float32(tc)
            m &= (ts >= lo_t - 2 * t_bin) & (ts <= hi_t + 2 * t_bin)
            if p.get("types"):
                if ac_of_pt is None:
                    ac_of_pt = np.repeat(np.arange(len(self), dtype=np.int32), np.asarray(self.n))
                m &= np.isin(self.category, list(p["types"]))[ac_of_pt]
            idx = np.nonzero(m)[0]
            sel.append((idx, ts[idx], d))
            del m, ts
        del gs_all, alt_all, trk_ok
        ref = int(np.argmin([len(x[0]) for x in sel]))

        def votes_for(k, w0, w1):
            idx, ts, d = sel[k]
            j = np.nonzero((ts >= w0) & (ts < w1))[0]
            idx, ts = idx[j], ts[j]
            la = np.asarray(self.lat)[idx].astype(np.float64)
            lo = np.asarray(self.lon)[idx].astype(np.float64)
            th = np.radians(np.asarray(self.trk)[idx].astype(np.float64))
            pn, pe = -np.sin(th), np.cos(th)  # right of track, (north, east)
            coslat = np.maximum(np.cos(np.radians(la)), 0.05)
            # horizontal offset of the listener: slant distance d (fit uncertain by dist_factor)
            # and height above the listener alt - ground, ground between 0 and max_ground_m
            alt = np.asarray(self.alt)[idx].astype(np.float64)
            dlo, dhi = d / dist_factor, d * dist_factor
            hmin = np.sqrt(np.maximum(dlo ** 2 - alt ** 2, 0.0))
            hmax = np.sqrt(np.maximum(dhi ** 2 - np.maximum(alt - max_ground_m, 0.0) ** 2, 0.0))
            nf = int(np.clip(np.ceil(dhi / (cell_km * 1000)), 1, 6))
            us = np.linspace(0, 1, nf + 1) if dhi > 0.3 * cell_km * 1000 else np.array([0.5])
            keys = []
            for side in (-1.0, 1.0):
                for u in us:
                    h = side * (hmin + u * (hmax - hmin))
                    keys.append(_key(la + h * pn / 111200.0, lo + h * pe / (111200.0 * coslat), ts, cell_deg, t_bin))
            fracs = keys  # one block of votes per offset
            return np.concatenate(keys), np.tile(idx, len(fracs))

        # 2) in blocks of start time (keeps memory small): cells voted by every pass
        block = 3600.0
        heap = []  # (score, key, hits, votes, who per pass)
        w = lo_t
        nblocks = int(np.ceil((hi_t - lo_t) / block)) or 1
        kb = 0
        while w < hi_t:
            kb += 1
            if progress:
                progress(kb / (nblocks + 1))
            rk_all, rwho_all = votes_for(ref, w, w + block)
            w_next = w + block
            if len(rk_all):
                rk, first, inv = np.unique(rk_all, return_index=True, return_inverse=True)
                votes = np.bincount(inv, minlength=len(rk))
                rwho = rwho_all[first]
                ref_ac = np.searchsorted(starts, rwho, side="right")
                hits = np.ones(len(rk), np.int32)
                match_who = {ref: rwho}
                for k in range(len(passes)):
                    if k == ref:
                        continue
                    kk, ww = votes_for(k, w - 2 * t_bin, w_next + 2 * t_bin)
                    o = np.argsort(kk)
                    ks, ws = kk[o], ww[o]
                    got = np.zeros(len(rk), bool)
                    mw = np.full(len(rk), -1, np.int64)
                    if len(ks):
                        for dk in _NEIGH:
                            q = rk + dk
                            pos = np.minimum(np.searchsorted(ks, q), len(ks) - 1)
                            ok = ks[pos] == q
                            if distinct:  # one aircraft can't be two different passes
                                ok &= np.searchsorted(starts, ws[pos], side="right") != ref_ac
                            new = ok & ~got
                            mw[new] = ws[pos[new]]
                            got |= ok
                    hits += got
                    match_who[k] = mw
                good = np.nonzero(hits >= min(2, len(passes)))[0]
                # silence: a place where low aircraft pass all the time (an airport) explains any
                # set of passes; the recording only has a few, so prefer quiet places
                busy = self.low_traffic(rk[good], cell_deg, t_bin, recording_s)
                score = (hits[good].astype(np.int64) * 10_000_000 - np.minimum(busy, 9_999) * 1000
                         + np.minimum(votes[good], 999))
                for jj in np.argsort(-score)[: n_best * 20]:
                    j = good[jj]
                    item = (int(score[jj]), int(rk[j]), int(hits[j]), int(votes[j]),
                            [int(match_who[k][j]) for k in range(len(passes))], int(busy[jj]))
                    if len(heap) < n_best * 20:
                        heapq.heappush(heap, item)
                    elif item > heap[0]:
                        heapq.heapreplace(heap, item)
            w = w_next
        # 3) best first, one candidate per neighbourhood
        out, seen = [], []
        for score, key, hits, votes, who, busy in sorted(heap, reverse=True):
            if len(out) >= n_best:
                break
            la, lo, tb = _unkey(key, cell_deg, t_bin)
            if any(abs(la - s0) < 3 * cell_deg and abs(lo - s1) < 3 * cell_deg / max(np.cos(np.radians(la)), .05)
                   and abs(tb - s2) < 3 * t_bin for s0, s1, s2 in seen):
                continue
            seen.append((la, lo, tb))
            ac = [self.meta[int(np.searchsorted(starts, wv, side="right") - 1)][0] if wv >= 0 else "" for wv in who]
            out.append(WorldCandidate(round(la, 4), round(lo, 4), tb + self.day0, hits, votes, ac, busy))
        if progress:
            progress(1.0)
        return out


_LAT_OFF = 1 << 14
_SH_LON = 22
_SH_LAT = 44


def _key(lat, lon, tstart, cell_deg, t_bin):
    a = np.floor(lat / cell_deg).astype(np.int64) + _LAT_OFF
    lonw = ((lon + 180.0) % 360.0)
    b = np.floor(lonw / cell_deg).astype(np.int64)
    c = np.floor(tstart / t_bin).astype(np.int64) + (1 << 18)
    return (a << _SH_LAT) | (b << _SH_LON) | c


def _unkey(k, cell_deg, t_bin):
    k = int(k)
    c = (k & ((1 << _SH_LON) - 1)) - (1 << 18)
    b = (k >> _SH_LON) & ((1 << (_SH_LAT - _SH_LON)) - 1)
    a = (k >> _SH_LAT) - _LAT_OFF
    lat = (a + 0.5) * cell_deg
    lon = (b + 0.5) * cell_deg - 180.0
    return lat, lon, (c + 0.5) * t_bin


_NEIGH = np.array([(da << _SH_LAT) + (db << _SH_LON) + dc for da in (-1, 0, 1) for db in (-1, 0, 1)
                   for dc in (-1, 0, 1)], np.int64)


def open_day(path, progress=None, **build_kw) -> DayArchive:
    """Open a ``.sgday`` folder, converting the tar parts first if needed."""
    p = Path(path).expanduser()
    if p.is_dir() and (p / "aircraft.json").exists():
        return DayArchive(p)
    if p.name.endswith(".sgday"):
        raise ValueError(f"not a converted day: {p}")
    out = sgday_path(p)
    if not (out / "aircraft.json").exists():
        build_day(p, out, progress=progress, **build_kw)
    return DayArchive(out)


def refine_world(archive: DayArchive, curves, candidates, n=10, radius_km=5.0, grid_step_m=500.0,
                 t_uncertainty=60.0, c=340.0, recording_s=None, only_matched=True, progress=None, **locate_kw):
    """Check the best world candidates with the full Doppler-curve match.

    Returns a list of (candidate, LocateResult or None) sorted by match cost:
    the vote only says "aircraft with about the right speed and distance were
    there at the right times"; the curve match checks the whole shape of every
    pass. With ``only_matched`` (default, fast) only the aircraft that voted
    for the candidate are used; otherwise every flight nearby (slow).
    """
    from .locate import locate

    out = []
    m = min(n, len(candidates))
    dur = recording_s or 3600
    for k, cd in enumerate(candidates[:n]):
        if progress:
            progress(k / max(m, 1))
        tracks = archive.tracks(cd.t_start - t_uncertainty - 300, cd.t_start + t_uncertainty + dur + 300,
                                cd.lat, cd.lon, radius_km + 25)
        if only_matched:
            want = {a for a in cd.aircraft if a}
            tracks = [t for t in tracks if t.ident in want]
        if not tracks:
            out.append((cd, None))
            continue
        try:
            res = locate(curves, tracks, (cd.lat, cd.lon), cd.t_start, t_uncertainty, radius_km, grid_step_m,
                         c=c, recording_s=recording_s, verbose=False, fit_gate=False,
                         **{"dt": 1.0, **locate_kw})
        except Exception:
            res = None
        out.append((cd, res))
    out.sort(key=lambda x: np.inf if x[1] is None else x[1].cost)
    if progress:
        progress(1.0)
    return out


def build_progress(out) -> dict:
    """How far a conversion is (reads ``_build.json``; works from another process)."""
    out = Path(out)
    if (out / "aircraft.json").exists():
        return {"fraction": 1.0, "aircraft": None, "done": True}
    st = out / "_build.json"
    if not st.exists():
        return {"fraction": 0.0, "aircraft": 0, "done": False}
    s = json.loads(st.read_text())
    total = s.get("total") or 1
    return {"fraction": min(s.get("offset", 0) / total, 0.999), "aircraft": s.get("aircraft", 0), "done": False}


def sgday_path(source) -> Path:
    p = Path(source).expanduser()
    if p.name.endswith(".sgday"):
        return p
    stem = p.name.split(".tar")[0] if not p.is_dir() else p.name
    return p.parent / f"{stem}.sgday"


def main(argv=None):
    import argparse
    import time

    ap = argparse.ArgumentParser(prog="python -m sound_geolocator.adsb_day",
                                 description="Convert / inspect a whole-day ADS-B archive (adsb.lol globe_history).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="convert the tar parts (give the .tar.aa) into a .sgday folder")
    b.add_argument("source")
    b.add_argument("--max-alt", type=float, default=7000.0)
    b.add_argument("--min-dt", type=float, default=3.0)
    b.add_argument("--workers", type=int, default=None)
    b.add_argument("--budget", type=float, default=None, help="stop after about this many seconds (resumable)")
    i = sub.add_parser("info", help="summary of a converted day")
    i.add_argument("path")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        t = time.time()
        r = build_day(a.source, sgday_path(a.source), a.max_alt, a.min_dt, a.workers, time_budget_s=a.budget,
                      progress=lambda f, n: print(f"{f * 100:5.1f}%  {n} aircraft  {time.time() - t:.0f} s",
                                                  flush=True))
        print("done:" if r else "paused (run again to continue):", r or sgday_path(a.source))
    else:
        print(json.dumps(DayArchive(sgday_path(a.path)).summary(), indent=1))


if __name__ == "__main__":
    main()
