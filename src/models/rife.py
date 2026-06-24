"""RIFE (Real-Time Intermediate Flow Estimation) — primary deep interpolator.

Wraps the official RIFE / Practical-RIFE repo (vendored under src/models/vendor/rife, weights in
weights/rife). RIFE estimates the *intermediate* flow directly and supports an arbitrary timestep
t in (0, 1), which gives the 15-min and 7.5-min products. Native grayscale-friendly and T4-light.

This wrapper does REAL inference when the repo + weights are present; otherwise `available()` is False
and `ensure_available()` raises a clear "vendor repo + fetch weights, run on server" error — never a
fake frame.
"""
from __future__ import annotations

import numpy as np

from .base import Interpolator
from .torch_utils import device_auto, gray_to_tensor, pad_to_multiple, tensor_to_gray, unpad
from .vendor import add_to_path, has_vendor, has_weights, torch_available, weights_path


class RifeInterpolator(Interpolator):
    name = "rife"
    requires_weights = True

    def __init__(self, device: str | None = None, weights_dir: str | None = None):
        from pathlib import Path
        self._device = device
        self._model = None
        self._weights_dir = Path(weights_dir) if weights_dir else weights_path("rife")

    def available(self) -> bool:
        has_w = self._weights_dir.exists() and any(self._weights_dir.rglob("*"))
        return torch_available() and has_vendor("rife") and has_w

    def _load(self):
        if self._model is not None:
            return
        import torch
        add_to_path("rife")
        Model = None
        for imp in ("train_log.RIFE_HDv3", "RIFE_HDv3", "model.RIFE_HDv3", "train_log.RIFE_HD"):
            try:
                Model = __import__(imp, fromlist=["Model"]).Model
                break
            except Exception:
                continue
        if Model is None:
            raise RuntimeError(
                "RIFE implementation not found under src/models/vendor/rife. "
                "Clone hzwer/Practical-RIFE there and place the checkpoint in weights/rife "
                "(see walkthrough.md)."
            )
        self._device = device_auto(self._device)
        m = Model()
        m.load_model(str(self._weights_dir), -1)
        m.eval()
        m.device()
        self._model = m
        torch.set_grad_enabled(False)

    def interpolate(self, frame0: np.ndarray, frame1: np.ndarray, t: float = 0.5) -> np.ndarray:
        self._check_pair(frame0, frame1)
        self.ensure_available()
        self._load()
        p0, orig = pad_to_multiple(np.nan_to_num(frame0), 32)
        p1, _ = pad_to_multiple(np.nan_to_num(frame1), 32)
        i0 = gray_to_tensor(p0).to(self._device)
        i1 = gray_to_tensor(p1).to(self._device)
        try:
            mid = self._model.inference(i0, i1, timestep=t)
        except TypeError:
            mid = self._model.inference(i0, i1)  # older API: midpoint only (t=0.5)
        out = unpad(tensor_to_gray(mid), orig)
        return np.clip(out, 0.0, 1.0).astype(np.float32)
