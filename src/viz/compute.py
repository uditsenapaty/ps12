"""Compute backends for the dashboard — chosen per-run in the web UI.

- LocalBackend:    interpolation runs on this machine (CPU/GPU). Works with uploaded + local files.
- LightningBackend: connects to the ps12 Lightning Studio (T4) and runs the interpolation THERE
                    (where the trained model + INSAT data + GPU live), returning a small preview.
                    Works with files already on the Studio (no flaky uploads).

All heavy imports are lazy so this module imports cleanly even without torch/streamlit/lightning_sdk.
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _ds(a: np.ndarray, k: int = 4) -> np.ndarray:
    """Downsample for a lightweight preview transfer."""
    return a[::k, ::k]


class LocalBackend:
    name = "Local (CPU/GPU)"
    remote = False

    def connect(self) -> str:
        return "Local backend ready."

    def available(self) -> bool:
        return True

    def list_insat(self) -> list[str]:
        out: list[str] = []
        for d in (ROOT / "samples" / "insat", ROOT / "data" / "insat"):
            if d.exists():
                out += [str(p) for p in d.rglob("*.h5")]
        return sorted(out)

    def interpolate_bt(self, source: str, p0, p2, model_name: str, kwargs: dict, t: float = 0.5) -> np.ndarray:
        sys.path.insert(0, str(ROOT))
        from src.data.readers import read_frame
        from src.infer.interpolate import interpolate_pair_bt
        from src.models.factory import get_model
        m = get_model(model_name, **(kwargs or {}))
        bt0 = read_frame(p0, source, with_lonlat=False).bt
        bt2 = read_frame(p2, source, with_lonlat=False).bt
        return interpolate_pair_bt(bt0, bt2, m, t)


class LightningBackend:
    name = "Lightning.ai (T4)"
    remote = True

    def __init__(self):
        self._studio = None

    def connect(self) -> str:
        sys.path.insert(0, str(ROOT))
        from cloud.lightning_exec import get_studio
        from lightning_sdk import Machine
        self._studio = get_studio()
        try:
            self._studio.start(Machine.T4, interruptible=False)   # cheapest GPU, on-demand
        except TypeError:
            try:
                self._studio.start(Machine.T4)
            except Exception:
                pass
        except Exception:
            pass
        return self._studio.run("cd ~/ps12 2>/dev/null; echo CONNECTED $(nvidia-smi -L 2>/dev/null | head -1)")

    def available(self) -> bool:
        return self._studio is not None

    def ensure_data(self) -> str:
        """Make sure INSAT data exists on the Studio; report what's there (download is a separate action)."""
        return self._studio.run(
            "cd ~/ps12 && echo INSAT_FILES=$(ls samples/insat/*.h5 data/insat/*.h5 2>/dev/null | wc -l) "
            "GOES_FILES=$(ls data/goes19/*.nc 2>/dev/null | wc -l)")

    def download_insat(self, user: str, pwd: str) -> str:
        return self._studio.run(
            f"cd ~/ps12 && MOSDAC_USERNAME='{user}' MOSDAC_PASSWORD='{pwd}' "
            f"python data_setup.py --download insat --sample 2>&1 | tail -5")

    def list_insat(self) -> list[str]:
        out = self._studio.run("cd ~/ps12 && ls samples/insat/*.h5 data/insat/*.h5 2>/dev/null")
        return [ln.strip() for ln in out.splitlines() if ln.strip().endswith(".h5")]

    def interpolate_bt(self, source: str, p0, p2, model_name: str, kwargs: dict, t: float = 0.5) -> np.ndarray:
        """Run interpolation on the Studio; return a downsampled BT preview via base64 .npy."""
        wkw = json.dumps(kwargs or {})
        cmd = (
            "cd ~/ps12 && python - <<'PY'\n"
            "import numpy as np, base64, io, json\n"
            "from src.data.readers import read_frame\n"
            "from src.infer.interpolate import interpolate_pair_bt\n"
            "from src.models.factory import get_model\n"
            f"m = get_model('{model_name}', **json.loads(r'''{wkw}'''))\n"
            f"bt0 = read_frame(r'''{p0}''', '{source}', with_lonlat=False).bt\n"
            f"bt2 = read_frame(r'''{p2}''', '{source}', with_lonlat=False).bt\n"
            f"pred = interpolate_pair_bt(bt0, bt2, m, {t})\n"
            "buf = io.BytesIO(); np.save(buf, pred[::4, ::4].astype('float32'))\n"
            "print('B64:' + base64.b64encode(buf.getvalue()).decode())\n"
            "PY"
        )
        out = self._studio.run(cmd)
        for line in out.splitlines():
            if line.startswith("B64:"):
                return np.load(io.BytesIO(base64.b64decode(line[4:].strip())))
        raise RuntimeError("no result from Studio:\n" + out[-600:])


def get_backend(kind: str):
    return LightningBackend() if str(kind).lower().startswith("lightning") else LocalBackend()
