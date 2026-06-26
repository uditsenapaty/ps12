"""UNetVFI — a compact, fully-trainable flow-based interpolator we own end-to-end.

Why this exists: RIFE/FILM/Super-SloMo are used as strong *pretrained* references, but the PS wants a
model *trained on satellite data*. UNetVFI is a small encoder-decoder that predicts bidirectional
flow + a blend mask, backward-warps the two inputs to time t, and fuses them. It trains from scratch
on GOES/Himawari triplets in hours on a single T4 and self-supervises on INSAT — a genuinely runnable
training path (no fragile dependence on an external repo's train loop).

Single-channel in/out (satellite IR is 1-band) — no 3-channel hack needed. torch is imported lazily so
this module is importable without torch installed.
"""
from __future__ import annotations

import numpy as np

from .base import Interpolator
from .torch_utils import device_auto, pad_to_multiple, unpad
from .vendor import has_weights, torch_available, weights_path


def build_net():
    """Construct the UNetVFI torch nn.Module (imported lazily)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def conv(ci, co, k=3, s=1):
        return nn.Sequential(nn.Conv2d(ci, co, k, s, k // 2), nn.PReLU(co))

    class Down(nn.Module):
        def __init__(self, ci, co):
            super().__init__()
            self.c = nn.Sequential(conv(ci, co, 3, 2), conv(co, co))

        def forward(self, x):
            return self.c(x)

    class Up(nn.Module):
        def __init__(self, ci, co):
            super().__init__()
            self.c = nn.Sequential(conv(ci, co), conv(co, co))

        def forward(self, x, skip):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            return self.c(torch.cat([x, skip], 1))

    class UNetVFINet(nn.Module):
        """Predicts (flow_t0[2], flow_t1[2], mask[1]) from concat(I0, I1, t-plane) and warps to time t.

        t-conditioned: the target time t is fed in as a constant input channel (a "t-plane"), so the
        network *sees* which intermediate time it must synthesise — this is what lets one model handle
        multiple granularities (t=0.25 / 0.5 / 0.75 for 30→15→7.5) rather than assuming linear motion.
        """
        def __init__(self, base: int = 32):
            super().__init__()
            self.inc = nn.Sequential(conv(3, base), conv(base, base))   # I0, I1, t-plane (granularity-aware)
            self.d1 = Down(base, base * 2)
            self.d2 = Down(base * 2, base * 4)
            self.d3 = Down(base * 4, base * 8)
            self.u3 = Up(base * 8 + base * 4, base * 4)
            self.u2 = Up(base * 4 + base * 2, base * 2)
            self.u1 = Up(base * 2 + base, base)
            self.head = nn.Conv2d(base, 5, 3, 1, 1)            # flow_t0(2), flow_t1(2), mask(1)
            self.src_head = nn.Conv2d(base, 1, 3, 1, 1)        # PINN source term S(x) (cloud growth/decay)

        @staticmethod
        def _warp(img, flow):
            B, C, H, W = img.shape
            ys, xs = torch.meshgrid(torch.arange(H, device=img.device), torch.arange(W, device=img.device), indexing="ij")
            grid = torch.stack((xs, ys), 0).float()[None].repeat(B, 1, 1, 1)
            vgrid = grid + flow
            vgrid[:, 0] = 2.0 * vgrid[:, 0] / max(W - 1, 1) - 1.0
            vgrid[:, 1] = 2.0 * vgrid[:, 1] / max(H - 1, 1) - 1.0
            return F.grid_sample(img, vgrid.permute(0, 2, 3, 1), mode="bilinear", padding_mode="border", align_corners=True)

        @staticmethod
        def _t_tensor(t, ref):
            """t -> (B,1,1,1) tensor on ref's device/dtype. Accepts a python float (same t for the whole
            batch) or a per-sample tensor of shape (B,) — so multi-granularity batches mix different t."""
            B = ref.shape[0]
            if torch.is_tensor(t):
                return t.to(device=ref.device, dtype=ref.dtype).reshape(B, 1, 1, 1)
            return torch.full((B, 1, 1, 1), float(t), device=ref.device, dtype=ref.dtype)

        def forward(self, i0, i1, t: float = 0.5, return_aux: bool = False):
            tt = self._t_tensor(t, i0)                                    # (B,1,1,1)
            tplane = tt.expand(i0.shape[0], 1, i0.shape[-2], i0.shape[-1])  # t as a full-res input feature
            x0 = self.inc(torch.cat([i0, i1, tplane], 1))
            x1 = self.d1(x0)
            x2 = self.d2(x1)
            x3 = self.d3(x2)
            y = self.u3(x3, x2)
            y = self.u2(y, x1)
            y = self.u1(y, x0)
            out = self.head(y)
            f_t0 = out[:, 0:2] * tt
            f_t1 = out[:, 2:4] * (1 - tt)
            mask = torch.sigmoid(out[:, 4:5])
            w0 = self._warp(i0, f_t0)
            w1 = self._warp(i1, f_t1)
            pred = (mask * w0 + (1 - mask) * w1).clamp(0, 1)
            if return_aux:                 # for the PINN physics loss
                source = self.src_head(y)
                return pred, f_t0, f_t1, mask, source
            return pred

    return UNetVFINet


class UNetVFIInterpolator(Interpolator):
    """Our custom model. `weights_dir` selects a checkpoint folder so different trained/fine-tuned
    variants (e.g. weights/unet vs weights/unet_insat) are independently selectable in the dashboard."""
    name = "unet"
    requires_weights = True

    def __init__(self, device: str | None = None, base: int = 32, weights_dir: str | None = None):
        from pathlib import Path
        self._device = device
        self._base = base
        self._weights_dir = Path(weights_dir) if weights_dir else weights_path("unet")
        self._net = None

    def _ckpts(self):
        return sorted(self._weights_dir.rglob("best.pt")) + sorted(self._weights_dir.rglob("*.pt"))

    def available(self) -> bool:
        return torch_available() and bool(self._ckpts())

    def _load(self):
        if self._net is not None:
            return
        import torch
        self._device = device_auto(self._device)
        cks = self._ckpts()
        if not cks:
            raise RuntimeError(
                f"UNetVFI weights not found in {self._weights_dir} "
                f"(train first: python -m src.train.finetune)."
            )
        state = torch.load(cks[0], map_location=self._device)
        base = state.get("base", self._base)
        net = build_net()(base).to(self._device).eval()
        net.load_state_dict(state["model"], strict=False)  # tolerate checkpoints w/o the PINN source head
        self._net = net
        torch.set_grad_enabled(False)

    def interpolate(self, frame0: np.ndarray, frame1: np.ndarray, t: float = 0.5) -> np.ndarray:
        self._check_pair(frame0, frame1)
        self.ensure_available()
        self._load()
        import torch
        p0, orig = pad_to_multiple(np.nan_to_num(frame0), 8)
        p1, _ = pad_to_multiple(np.nan_to_num(frame1), 8)
        i0 = torch.from_numpy(p0).float()[None, None].to(self._device)
        i1 = torch.from_numpy(p1).float()[None, None].to(self._device)
        pred = self._net(i0, i1, t)[0, 0].cpu().numpy()
        return np.clip(unpad(pred, orig), 0.0, 1.0).astype(np.float32)
