"""Super-SloMo — the PS-named baseline and the architecture closest to the satellite anchor paper
(Vandal & Nemani). Arbitrary-time synthesis t in (0, 1) -> supports the 7.5-min product.

Wraps avinashpaliwal/Super-SloMo (vendored under src/models/vendor/superslomo; checkpoint in
weights/superslomo/SuperSloMo.ckpt). Implements the official two-network intermediate synthesis
(flow computation + arbitrary-time flow interpolation + visibility blending). Real when present;
honestly gated otherwise.
"""
from __future__ import annotations

import numpy as np

from .base import Interpolator
from .torch_utils import device_auto, gray_to_tensor, pad_to_multiple, tensor_to_gray, unpad
from .vendor import add_to_path, has_vendor, has_weights, torch_available, weights_path


class SuperSloMoInterpolator(Interpolator):
    name = "superslomo"
    requires_weights = True

    def __init__(self, device: str | None = None):
        self._device = device
        self._fc = None     # flow computation UNet
        self._at = None     # arbitrary-time flow interpolation UNet
        self._ssm = None    # vendored module
        self._warp_cache: dict = {}

    def available(self) -> bool:
        return torch_available() and has_vendor("superslomo") and has_weights("superslomo")

    def _ckpt(self):
        cks = list(weights_path("superslomo").rglob("*.ckpt")) + list(weights_path("superslomo").rglob("*.pth"))
        if not cks:
            raise RuntimeError("Super-SloMo checkpoint (*.ckpt) not found in weights/superslomo (see walkthrough.md).")
        return cks[0]

    def _load(self):
        if self._fc is not None:
            return
        import torch
        add_to_path("superslomo")
        try:
            import model as ssm  # vendored Super-SloMo model.py (UNet, backWarp)
        except Exception as e:
            raise RuntimeError(
                "Super-SloMo 'model.py' not found under src/models/vendor/superslomo "
                "(clone avinashpaliwal/Super-SloMo). " + str(e)
            )
        self._ssm = ssm
        self._device = device_auto(self._device)
        dev = torch.device(self._device)
        self._fc = ssm.UNet(6, 4).to(dev).eval()
        self._at = ssm.UNet(20, 5).to(dev).eval()
        state = torch.load(self._ckpt(), map_location=dev)
        self._fc.load_state_dict(state["state_dictFC"])
        self._at.load_state_dict(state["state_dictAT"])
        torch.set_grad_enabled(False)

    def _backwarp(self, w: int, h: int):
        key = (w, h)
        if key not in self._warp_cache:
            import torch
            self._warp_cache[key] = self._ssm.backWarp(w, h, torch.device(self._device)).to(self._device)
        return self._warp_cache[key]

    def interpolate(self, frame0: np.ndarray, frame1: np.ndarray, t: float = 0.5) -> np.ndarray:
        self._check_pair(frame0, frame1)
        self.ensure_available()
        self._load()
        import torch
        p0, orig = pad_to_multiple(np.nan_to_num(frame0), 32)
        p1, _ = pad_to_multiple(np.nan_to_num(frame1), 32)
        I0 = gray_to_tensor(p0).to(self._device)
        I1 = gray_to_tensor(p1).to(self._device)
        H, W = p0.shape
        back = self._backwarp(W, H)

        flow_out = self._fc(torch.cat((I0, I1), dim=1))
        F_0_1, F_1_0 = flow_out[:, :2], flow_out[:, 2:]
        C00 = C11 = -(1 - t) * t
        C01 = t * t
        C10 = (1 - t) * (1 - t)
        F_t_0 = C00 * F_0_1 + C01 * F_1_0
        F_t_1 = C10 * F_0_1 + C11 * F_1_0
        g_I0 = back(I0, F_t_0)
        g_I1 = back(I1, F_t_1)
        intrp = self._at(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1, g_I0), dim=1))
        F_t_0_f = intrp[:, :2] + F_t_0
        F_t_1_f = intrp[:, 2:4] + F_t_1
        V_t_0 = torch.sigmoid(intrp[:, 4:5])
        V_t_1 = 1 - V_t_0
        g_I0_f = back(I0, F_t_0_f)
        g_I1_f = back(I1, F_t_1_f)
        w0, w1 = (1 - t), t
        Ft = (w0 * V_t_0 * g_I0_f + w1 * V_t_1 * g_I1_f) / (w0 * V_t_0 + w1 * V_t_1)
        return np.clip(unpad(tensor_to_gray(Ft), orig), 0.0, 1.0).astype(np.float32)
