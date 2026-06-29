"""UNetVFI — our custom, fully-trainable flow-based interpolator (we own it end-to-end).

Iteration-1 architecture: **FeatSynthVFI**. The pretrained baselines that are hardest to beat (FILM,
classical TV-L1) win because of multi-scale *feature-domain synthesis* + good flow; our edge is that we
train ON satellite thermal-IR (FILM/Super-SloMo are pretrained on natural RGB video). So this network
fuses the strong ideas and trains them on IR:

  * Siamese (weight-tied) multi-scale ENCODER applied to each 1-band frame -> per-frame feature pyramids.
  * A coarse->fine FLOW DECODER fuses both pyramids and predicts a bidirectional intermediate flow
    (flow_a, flow_b) + a blend mask. Time t SCALES the flow (f_{t0}=t·flow_a, f_{t1}=(1−t)·flow_b) — the
    linear-motion intermediate-flow formulation, kept so t stays implicit and per-sample t works.
  * A SYNTHESIS DECODER warps the encoder features to time t and predicts a bounded residual on top of
    the warp-blend (FILM/SoftSplat-style) — this fixes warp artifacts/blur that hurt SSIM/edge-SSIM/LPIPS.
    The residual is zero-initialised (res_scale) so early training is the proven warp-blend and the
    synthesis branch ramps in gradually (overfit guard on a small dataset).
  * A PINN source head S models brightness growth/decay (advection physics) — what optical flow can't.

Interface is unchanged: build_net()(base); forward(i0,i1,t,return_aux) -> pred (B,1,H,W) in [0,1], and
with return_aux -> (pred, f_t0, f_t1, mask, source) so the advection PINN loss (u = f_t1 − f_t0) and the
training/eval/inference code keep working. Single IR band in/out. torch imported lazily.
"""
from __future__ import annotations

import numpy as np

from .base import Interpolator
from .torch_utils import device_auto, pad_to_multiple, unpad
from .vendor import has_weights, torch_available, weights_path


def build_net():
    """Construct the FeatSynthVFI torch nn.Module (imported lazily)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def conv(ci, co, k=3, s=1):
        return nn.Sequential(nn.Conv2d(ci, co, k, s, k // 2), nn.PReLU(co))

    def _warp(img, flow):
        """Backward bilinear warp: out(x) = img(x + flow(x)). flow=(B,2,H,W) in pixels."""
        B, C, H, W = img.shape
        ys, xs = torch.meshgrid(torch.arange(H, device=img.device), torch.arange(W, device=img.device), indexing="ij")
        grid = torch.stack((xs, ys), 0).float()[None].repeat(B, 1, 1, 1)
        vg = grid + flow
        vgx = 2.0 * vg[:, 0] / max(W - 1, 1) - 1.0
        vgy = 2.0 * vg[:, 1] / max(H - 1, 1) - 1.0
        vgrid = torch.stack((vgx, vgy), dim=-1)
        return F.grid_sample(img, vgrid, mode="bilinear", padding_mode="border", align_corners=True)

    def warp_to(feat, flow_full):
        """Warp a (possibly downscaled) feature map by a full-res flow, rescaling flow res + magnitude."""
        if feat.shape[-2:] != flow_full.shape[-2:]:
            sx = feat.shape[-1] / flow_full.shape[-1]
            sy = feat.shape[-2] / flow_full.shape[-2]
            f = F.interpolate(flow_full, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            f = torch.stack((f[:, 0] * sx, f[:, 1] * sy), dim=1)
        else:
            f = flow_full
        return _warp(feat, f)

    def up(x, skip):
        return F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def up_flow(flow4, ref):
        """Upscale a 4-channel flow (flow_a xy, flow_b xy) to ref's resolution, scaling magnitude by the
        resolution ratio so the displacement stays consistent across scales."""
        sx = ref.shape[-1] / flow4.shape[-1]
        sy = ref.shape[-2] / flow4.shape[-2]
        f = F.interpolate(flow4, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return torch.stack((f[:, 0] * sx, f[:, 1] * sy, f[:, 2] * sx, f[:, 3] * sy), dim=1)

    class Enc(nn.Module):
        """Siamese per-frame multi-scale encoder (1 IR band in)."""
        def __init__(self, base):
            super().__init__()
            self.c0 = nn.Sequential(conv(1, base), conv(base, base))
            self.c1 = nn.Sequential(conv(base, 2 * base, 3, 2), conv(2 * base, 2 * base))
            self.c2 = nn.Sequential(conv(2 * base, 4 * base, 3, 2), conv(4 * base, 4 * base))
            self.c3 = nn.Sequential(conv(4 * base, 8 * base, 3, 2), conv(8 * base, 8 * base))

        def forward(self, x):
            e0 = self.c0(x)
            e1 = self.c1(e0)
            e2 = self.c2(e1)
            e3 = self.c3(e2)
            return e0, e1, e2, e3

    class FeatSynthVFINet(nn.Module):
        """iter-3: COARSE-TO-FINE intermediate flow (RIFE/IFRNet style). The flow is estimated at 1/8 and
        residually refined at 1/4, 1/2, 1/1; at each finer scale the encoder features are WARPED by the
        current (time-scaled) flow before predicting the residual, so the network aligns features as it
        sharpens the motion. Then the synthesis decoder warps the pyramids to time t and adds a smooth
        (1/2-res) residual. t stays implicit (it scales the flow), and return_aux still yields
        (pred, f_t0, f_t1, mask, source)."""
        def __init__(self, base: int = 32):
            super().__init__()
            self.base = base
            self.enc = Enc(base)
            # coarse->fine flow decoder: fuse pyramids, predict + residually refine a 4-ch flow per scale
            self.fb = conv(16 * base, 8 * base)
            self.flow3 = nn.Conv2d(8 * base, 4, 3, 1, 1)                 # initial flow @ 1/8
            self.f2 = nn.Sequential(conv(8 * base + 4 * base + 4 * base, 4 * base), conv(4 * base, 4 * base))
            self.flow2 = nn.Conv2d(4 * base, 4, 3, 1, 1)
            self.f1 = nn.Sequential(conv(4 * base + 2 * base + 2 * base, 2 * base), conv(2 * base, 2 * base))
            self.flow1 = nn.Conv2d(2 * base, 4, 3, 1, 1)
            self.f0 = nn.Sequential(conv(2 * base + base + base, base), conv(base, base))
            self.flow0 = nn.Conv2d(base, 4, 3, 1, 1)
            self.mask_head = nn.Conv2d(base, 1, 3, 1, 1)
            self.src_head = nn.Conv2d(base, 1, 3, 1, 1)                  # PINN source S
            # synthesis decoder over warped features -> smooth (1/2-res) bounded residual on the blend
            self.sb = conv(16 * base, 4 * base)
            self.s2 = nn.Sequential(conv(4 * base + 8 * base, 2 * base), conv(2 * base, 2 * base))
            self.s1 = nn.Sequential(conv(2 * base + 4 * base, base), conv(base, base))
            self.res_head = nn.Conv2d(base, 1, 3, 1, 1)
            self.res_scale = nn.Parameter(torch.zeros(1))

        @staticmethod
        def _t_tensor(t, ref):
            B = ref.shape[0]
            if torch.is_tensor(t):
                return t.to(device=ref.device, dtype=ref.dtype).reshape(B, 1, 1, 1)
            return torch.full((B, 1, 1, 1), float(t), device=ref.device, dtype=ref.dtype)

        def forward(self, i0, i1, t: float = 0.5, return_aux: bool = False):
            a0, a1, a2, a3 = self.enc(i0)
            b0, b1, b2, b3 = self.enc(i1)
            tt = self._t_tensor(t, i0)
            # coarse-to-fine flow: predict @ 1/8, refine residually @ 1/4, 1/2, 1/1 with feature warping
            d = self.fb(torch.cat([a3, b3], 1))
            fl = self.flow3(d)
            fl = up_flow(fl, a2)
            wa, wb = _warp(a2, fl[:, 0:2] * tt), _warp(b2, fl[:, 2:4] * (1 - tt))
            d = self.f2(torch.cat([up(d, a2), wa, wb], 1))
            fl = fl + self.flow2(d)
            fl = up_flow(fl, a1)
            wa, wb = _warp(a1, fl[:, 0:2] * tt), _warp(b1, fl[:, 2:4] * (1 - tt))
            d = self.f1(torch.cat([up(d, a1), wa, wb], 1))
            fl = fl + self.flow1(d)
            fl = up_flow(fl, a0)
            wa, wb = _warp(a0, fl[:, 0:2] * tt), _warp(b0, fl[:, 2:4] * (1 - tt))
            g = self.f0(torch.cat([up(d, a0), wa, wb], 1))
            fl = fl + self.flow0(g)                                   # final full-res flow
            f_t0 = fl[:, 0:2] * tt
            f_t1 = fl[:, 2:4] * (1 - tt)
            mask = torch.sigmoid(self.mask_head(g))
            w0 = _warp(i0, f_t0)
            w1 = _warp(i1, f_t1)
            blend = mask * w0 + (1 - mask) * w1
            # synthesis: warp the feature pyramids to time t, predict a smooth bounded residual
            wa3, wb3 = warp_to(a3, f_t0), warp_to(b3, f_t1)
            wa2, wb2 = warp_to(a2, f_t0), warp_to(b2, f_t1)
            wa1, wb1 = warp_to(a1, f_t0), warp_to(b1, f_t1)
            s = self.sb(torch.cat([wa3, wb3], 1))
            s = self.s2(torch.cat([up(s, wa2), wa2, wb2], 1))
            s = self.s1(torch.cat([up(s, wa1), wa1, wb1], 1))         # base ch @ 1/2 res
            residual_lr = torch.tanh(self.res_head(s)) * self.res_scale
            residual = F.interpolate(residual_lr, size=blend.shape[-2:], mode="bilinear", align_corners=False)
            pred = (blend + residual).clamp(0, 1)
            if return_aux:
                source = self.src_head(g)
                return pred, f_t0, f_t1, mask, source
            return pred

    return FeatSynthVFINet


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
