# Sound Geolocator

Estimate where (and when) a recording was made from the sound of aircraft passing overhead.

Planes are almost always within earshot, and every one of them broadcasts its position over ADS-B. When a plane passes, the pitch of its engine tones slides down: the Doppler effect. The shape of that slide depends on where the listener stands relative to the flight path. One pass narrows the listener down to a curve on the map. Two or more passes in the same recording share one clock, so the delay between them is known, and together they pin down a single point.

```
audio/video ──► spectrogram ──► enhance ──► pick Doppler curves ──┐
                                                                  ├──► grid search (position × start time) ──► location
ADS-B tracks (OpenSky / CSV) ──► simulate Doppler at each candidate position ──┘
```

## How it works

1. **Load audio** from any audio or video file (ffmpeg decodes video).
2. **Spectrogram** with a long window (1 s → 1 Hz bins), since Doppler shifts of a few percent on a 100 Hz tone are only a few Hz.
3. **Enhance**: remove stationary hum (per-frequency median over time), whiten broadband noise like wind and traffic (running median across frequency), and despeckle. Tonal aircraft lines (fan / propeller blade-passing tones and harmonics) remain.
4. **Pick curves**: click one point on a line (`--seed T,F`) or draw a box around it (`--box T0,T1,F0,F1`). A Viterbi ridge tracker follows the line. It penalises changes of slope rather than slope itself, so it handles steep sweeps and carries straight through places where other lines cross. Line strength is capped so a brighter crossing line is no reason to jump. By default the pitch may only fall, as it does in a single pass.
5. **Simulate**: for a candidate listener position, solve the retarded-time equation `t − τ = |P(τ) − O| / c` along each flight's smoothed ADS-B track, and compute the received/emitted frequency ratio `c / (c − v·n)`.
6. **Search**: on a grid of positions, slide all curves together along time (one shared unknown start time) against every candidate flight, fitting the unknown source tone frequency in closed form. Each curve takes its best-matching flight. The position and start time with the lowest total misfit win. The best few separated spots on the coarse grid are each refined, and the lowest cost wins.

The search follows the method described in colsto's video "This background sound leaks your exact location":

* **Locked multi-event scoring.** The passes share one clock, so the physical delay between them stays fixed. Each pass may still slide by a small timing slack (`--cpa-slack`, default 2 s) to absorb error in measuring its closest-approach time, and its tone frequency is fitted freely.
* **Speed and distance gate.** A straight-flyover fit of each curve gives the aircraft's speed and closest slant distance. A curve is only matched to flights with a similar ground speed, and only at spots where that flight's closest approach is consistent with the fitted distance.
* **Aircraft type filter** (`--types prop,heli,unknown`). ICAO type codes are classed as propeller, jet or helicopter. Propellers and helicopters give the clearest tones.
* **Terrain** (`--dem srtm30m`, a GeoTIFF, or .npz). Each candidate's height comes from a DEM.
* **Silence check** (`--silence-km 2 --recording-s 360`). A spot is penalised if a flight that matches none of your curves would have passed within that distance during the recording, because you would have heard it.

`sgeo fit` also fits a straight-flyover model to each curve on its own (tone frequency, speed, closest-approach time and slant range). It is a quick sanity check that a picked line really is an aircraft pass, and needs no flight data.

## GUI

```bash
sgeo gui        # opens http://127.0.0.1:8765
```

Open a recording, click on the falling lines to trace them, load flight tracks, click the map to set the search centre, then press **Find location**. The demo button loads a synthetic case with a known answer. Everything runs locally, and the map tiles are the only thing loaded from the internet.

![GUI](docs/gui.png)

## Install

```bash
pip install -e .            # needs ffmpeg on PATH for video files
pip install -e '.[dev]' && pytest
```

## Quick start (synthetic demo with a known answer)

```bash
sgeo demo --out demo_out
# estimate 45.46322, 9.18834  true 45.46420, 9.19000  error 169 m
# start time error -1.0 s
```

`demo_out/spectrogram.png` shows the picked curves, and `demo_out/cost_map.png` shows the search:

![picked curves](docs/spectrogram.png)
![cost map](docs/cost_map.png)

## Real recordings

```bash
# 1. Look at the enhanced spectrogram and find descending tonal lines
sgeo spectrogram clip.mp4 --fmax 1500 --out spec.png

# 2. Pick at least two passes (seconds, Hz), then sanity-check them
sgeo extract clip.mp4 --fmax 1500 --seed 42,310 --seed 118,265 --out curves.json --plot picked.png
sgeo fit curves.json

# 3. Get ADS-B tracks for the area and time (see below), then search
sgeo locate curves.json tracks.csv --center 45.46,9.19 --radius 25 \
     --start 2025-06-01T14:03:00Z --uncertainty 300 --alt 120 --plot map.png
```

`--start` is your best guess of the recording's UTC start time (file metadata, upload time, clocks in frame), and `--uncertainty` is how far off it might be. `--center/--radius` define the search area.

### Flight data

`sgeo locate` reads a CSV of position reports. Column names are matched loosely: `time` (unix or ISO 8601), `icao24`/`hex`, `callsign`, `lat`, `lon`, and altitude as `alt_m`/`geoaltitude` (metres) or `altitude`/`alt_baro` (feet).

* **OpenSky REST API, recent data only** (`sgeo fetch-opensky --center LAT,LON --start ... --out tracks.csv`). Set `OPENSKY_CLIENT_ID` / `OPENSKY_CLIENT_SECRET` for authenticated access.
* **OpenSky historical database** (Trino, via `pyopensky`) for older recordings. Export state vectors to CSV.
* **ADS-B Exchange / adsb.lol** history, or any CSV with the columns above.

Use tracks covering a few minutes before and after the recording. The sound heard at time *t* was emitted up to a minute earlier.

## Accuracy and limits

* **Speed of sound** (`--c`, default 340 m/s) varies with temperature, and wind shifts the curves. Both bias the result by hundreds of metres. Set `--c` from the local temperature when you know it (331.3 + 0.606·T°C).
* **Observer altitude** (`--alt` or `--dem`) should be on the same datum as the track altitudes. OpenSky `geoaltitude` is height above the WGS84 ellipsoid, while DEM heights are above sea level. Add the local geoid height with `--geoid-offset` (about +45 m in northern Italy).
* **Geometry matters.** Two passes on crossing tracks give a sharp fix. Two passes along the same route leave a long ambiguous strip.
* The ADS-B track has to cover each picked curve (80% by default). Aircraft without ADS-B (many small planes and helicopters) cannot be matched.

## Layout

| module | what it does |
|---|---|
| `audio.py` | load audio/video, resample |
| `spectrogram.py` | STFT, hum removal, whitening, despeckling |
| `extract.py` | ridge tracking (box or seed), curve cleaning, JSON I/O |
| `doppler.py` | retarded-time Doppler simulation, straight-flyover fit |
| `flights.py` | track model (smoothing spline in ECEF), CSV/OpenSky loaders |
| `locate.py` | position × start-time search: locked multi-event scoring, gates, silence check |
| `aircraft_types.py` | ICAO type code → propeller / jet / helicopter |
| `elevation.py` | DEM from OpenTopoData, GeoTIFF or .npz |
| `gui/` | local web interface (`sgeo gui`) |
| `synth.py` | synthetic flights and recordings for tests and demo |
| `cli.py` | `sgeo` command |

## Responsible use

This tool can reveal where a recording was made. Use it on your own recordings, or for research and verification work where locating the recording is legitimate. Don't use it to track or expose private individuals.
