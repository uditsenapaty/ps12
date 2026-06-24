"""Classical optical-flow frame interpolation (the 'traditional' baseline the PS contrasts against).

Implements the Super-SloMo *linear intermediate-flow* approximation on top of a dense optical-flow
field (TV-L1 if opencv-contrib is present, else Farneback). This is a REAL, fully CPU-runnable
interpolator — it is the end-to-end `.nc -> .nc` path the deterministic battery verifies without any
GPU or learned weights. It also makes the "blur / ghosting on fast non-linear cloud motion" failure
visible, motivating the deep models.
"""
from __future__ import annotations

import numpy as np

from .base import Interpolator


def _dense_flow(img0: np.ndarray, img1: np.ndarray) -> np.ndarray:
    """Dense optical flow img0 -> img1 (H, W, 2), inputs in [0, 1]. TV-L1 if available, else Farneback."""
    import cv2
    a = np.clip(np.nan_to_num(img0) * 255.0, 0, 255).astype(np.uint8)
    b = np.clip(np.nan_to_num(img1) * 255.0, 0, 255).astype(np.uint8)
    if hasattr(cv2, "optflow") and hasattr(cv2.optflow, "DualTVL1OpticalFlow_create"):
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        return tvl1.calc(a, b, None)
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 25, 3, 7, 1.5, 0)


def _warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Backward-warp `img` by `flow`: out(x) = img(x + flow(x)) via bilinear remap."""
    import cv2
    h, w = img.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = (gx + flow[..., 0]).astype(np.float32)
    map_y = (gy + flow[..., 1]).astype(np.float32)
    return cv2.remap(img.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class ClassicalFlowInterpolator(Interpolator):
    name = "classical"

    def interpolate(self, frame0: np.ndarray, frame1: np.ndarray, t: float = 0.5) -> np.ndarray:
        self._check_pair(frame0, frame1)
        f01 = _dense_flow(frame0, frame1)   # 0 -> 1
        f10 = _dense_flow(frame1, frame0)   # 1 -> 0
        # Super-SloMo linear approximation of the flow from the intermediate time t to each endpoint.
        f_t0 = -(1.0 - t) * t * f01 + (t ** 2) * f10
        f_t1 = ((1.0 - t) ** 2) * f01 - t * (1.0 - t) * f10
        warp0 = _warp(np.nan_to_num(frame0), f_t0)
        warp1 = _warp(np.nan_to_num(frame1), f_t1)
        out = (1.0 - t) * warp0 + t * warp1
        return np.clip(out, 0.0, 1.0).astype(np.float32)


class LinearBlendInterpolator(Interpolator):
    """Trivial temporal average — reference floor (no motion compensation)."""
    name = "linear"

    def interpolate(self, frame0: np.ndarray, frame1: np.ndarray, t: float = 0.5) -> np.ndarray:
        self._check_pair(frame0, frame1)
        out = (1.0 - t) * np.nan_to_num(frame0) + t * np.nan_to_num(frame1)
        return np.clip(out, 0.0, 1.0).astype(np.float32)
