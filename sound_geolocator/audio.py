"""Load audio from any audio or video file (via ffmpeg when needed)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio(path: str | Path, sr: int | None = 16000, channel: str = "mix"):
    """Return (samples float32 mono, sample_rate).

    Files soundfile cannot read directly (mp4, mkv, mov, m4a, webm, ...) are
    decoded with ffmpeg. ``sr=None`` keeps the native rate.
    """
    path = Path(path)
    try:
        data, rate = sf.read(str(path), always_2d=True, dtype="float32")
    except Exception:
        # let ffmpeg downmix and resample: a 1-hour 48 kHz stereo track would
        # otherwise need ~700 MB of temporary WAV
        data, rate = _ffmpeg_decode(path, sr=sr, mono=channel == "mix")
    if channel == "mix":
        x = data.mean(axis=1)
    else:
        x = data[:, int(channel)]
    if sr is not None and sr != rate:
        g = np.gcd(int(sr), int(rate))
        x = resample_poly(x, sr // g, rate // g).astype(np.float32)
        rate = sr
    return x.astype(np.float32), int(rate)


def _ffmpeg_exe():
    """ffmpeg on PATH, else the binary bundled with the imageio-ffmpeg package."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _ffmpeg_decode(path: Path, sr: int | None = None, mono: bool = False):
    exe = _ffmpeg_exe()
    if exe is None:
        raise RuntimeError(f"Cannot read {path}: install ffmpeg (or `pip install imageio-ffmpeg`) "
                           "to load video/compressed audio")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        cmd = [exe, "-nostdin", "-loglevel", "error", "-y", "-i", str(path), "-vn", "-acodec", "pcm_s16le"]
        if mono:
            cmd += ["-ac", "1"]
        if sr:
            cmd += ["-ar", str(int(sr))]
        cmd.append(str(wav))
        subprocess.run(cmd, check=True)
        data, rate = sf.read(str(wav), always_2d=True, dtype="float32")
    return data, rate


def save_audio(path: str | Path, x: np.ndarray, sr: int):
    sf.write(str(path), np.asarray(x, dtype=np.float32), sr)
