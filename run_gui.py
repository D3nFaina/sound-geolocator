"""Open the Sound Geolocator GUI from Spyder (or any Python IDE).

1. Open this file in Spyder.
2. Set CLIP below to your recording (or leave it as None and open it from the GUI).
3. Press Run (F5). The first run installs any missing packages, then the GUI
   opens in your browser at http://127.0.0.1:8765

To stop the GUI: click in the Spyder console and press Ctrl+C
(or use "Restart kernel"). Run again with F5 to restart it.
"""

import os
import subprocess
import sys

# ---------------------------------------------------------------- settings
CLIP = os.path.expanduser("~/Downloads/Unknown location - raw audio.mp4")
PORT = 8765
# --------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

NEEDED = {"numpy": "numpy", "scipy": "scipy", "soundfile": "soundfile", "matplotlib": "matplotlib",
          "requests": "requests", "flask": "flask", "imageio_ffmpeg": "imageio-ffmpeg"}
missing = []
for module, package in NEEDED.items():
    try:
        __import__(module)
    except ImportError:
        missing.append(package)
if missing:
    print("Installing:", ", ".join(missing), "(one time only)…")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

from sound_geolocator.gui.app import main  # noqa: E402

clip = CLIP if CLIP and os.path.exists(CLIP) else None
if CLIP and clip is None:
    print(f"Clip not found: {CLIP} (you can open a file from the GUI instead)")
main(port=PORT, open_browser=True, preload=clip)
