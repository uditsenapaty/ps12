"""FILM (Frame Interpolation for Large Motion) — large-displacement specialist.

30-min cloud motion produces large displacements; FILM is designed for exactly that. We use a Torch
port loadable as a TorchScript module (weights/film/film_net.pt), vendored/fetched by data_setup.py.
Real inference when present; honestly gated otherwise.
"""
from __future__ import annotations

import numpy as np

from .base import Interpolator
from .torch_utils import device_auto, gray_to_tensor, pad_to_multiple, tensor_to_gray, unpad
from .vendor import has_weights, torch_available, weights_path


class FilmInterpolator(Interpolator):
    name = "film"
    requires_weights = True

    def __init__(self, device: str | None = None):
        self._device = device
        self._model = None

    def available(self) -> bool:
        return torch_available() and (has_weights("film") and any(weights_path("film").rglob("*.pt")))

    def _load(self):
        if self._model is not None:
            return
        import torch
        pts = list(weights_path("film").rglob("*.pt"))
        if not pts:
            raise RuntimeError(
                "FILM TorchScript weights not found in weights/film (expected *.pt). "
                "Fetch the Torch FILM port (e.g. dajes/frame-interpolation-pytorch) — see walkthrough.md."
            )
        self._device = device_auto(self._device)
        self._model = torch.jit.load(str(pts[0]), map_location=self._device).eval()
        torch.set_grad_enabled(False)

    def interpolate(self, frame0: np.ndarray, frame1: np.ndarray, t: float = 0.5) -> np.ndarray:
        self._check_pair(frame0, frame1)
        self.ensure_available()
        self._load()
        import torch
        p0, orig = pad_to_multiple(np.nan_to_num(frame0), 64)
        p1, _ = pad_to_multiple(np.nan_to_num(frame1), 64)
        x0 = gray_to_tensor(p0).to(self._device)
        x1 = gray_to_tensor(p1).to(self._device)
        dt = torch.full((1, 1), float(t), device=self._device)
        out = self._model(x0, x1, dt)
        if isinstance(out, dict):
            out = out.get("image", next(iter(out.values())))
        return np.clip(unpad(tensor_to_gray(out), orig), 0.0, 1.0).astype(np.float32)
