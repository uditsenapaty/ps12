"""AI optical-flow interpolation using RAFT (torchvision pretrained weights).

RAFT is the explicit motion-vector estimator the PS asks for (objective 1). torchvision ships a
pretrained `raft_large`, so this runs with NO manual weight vendoring (weights auto-download on first
use). The intermediate frame is synthesised by Super-SloMo's linear intermediate-flow approximation on
top of the RAFT flow fields, then bilinear backward-warping. Real and runnable on CPU or GPU.
"""
from __future__ import annotations

import numpy as np

from .base import Interpolator
from .torch_utils import device_auto, pad_to_multiple, unpad


class RaftFlowInterpolator(Interpolator):
    name = "raft"
    requires_weights = True  # fetched automatically by torchvision

    def __init__(self, device: str | None = None, iters: int = 12):
        self._device = device
        self._iters = iters
        self._model = None
        self._tf = None

    def available(self) -> bool:
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
            return True
        except Exception:
            return False

    def _load(self):
        if self._model is not None:
            return
        import torch
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        self._device = device_auto(self._device)
        weights = Raft_Large_Weights.DEFAULT
        self._tf = weights.transforms()
        self._model = raft_large(weights=weights, progress=True).to(self._device).eval()
        torch.set_grad_enabled(False)

    def flow(self, img0: np.ndarray, img1: np.ndarray) -> np.ndarray:
        """Dense RAFT flow img0 -> img1, shape (H, W, 2) in pixels. Inputs (H, W) in [0, 1]."""
        self.ensure_available()
        self._load()
        import torch
        p0, orig = pad_to_multiple(np.nan_to_num(img0), 8)
        p1, _ = pad_to_multiple(np.nan_to_num(img1), 8)

        def to3(a):
            t = torch.from_numpy(a.astype(np.float32))[None, None].clamp(0, 1).repeat(1, 3, 1, 1)
            return t
        a, b = self._tf(to3(p0), to3(p1))
        a, b = a.to(self._device), b.to(self._device)
        flow = self._model(a, b, num_flow_updates=self._iters)[-1][0].cpu().numpy()  # (2, H, W)
        flow = np.transpose(flow, (1, 2, 0))  # (H, W, 2): [..,0]=u(x), [..,1]=v(y)
        return unpad(flow, orig)

    def interpolate(self, frame0: np.ndarray, frame1: np.ndarray, t: float = 0.5) -> np.ndarray:
        self._check_pair(frame0, frame1)
        from .classical import _warp  # reuse the bilinear backward-warp
        f01 = self.flow(frame0, frame1)
        f10 = self.flow(frame1, frame0)
        f_t0 = -(1.0 - t) * t * f01 + (t ** 2) * f10
        f_t1 = ((1.0 - t) ** 2) * f01 - t * (1.0 - t) * f10
        warp0 = _warp(np.nan_to_num(frame0), f_t0)
        warp1 = _warp(np.nan_to_num(frame1), f_t1)
        out = (1.0 - t) * warp0 + t * warp1
        return np.clip(out, 0.0, 1.0).astype(np.float32)
