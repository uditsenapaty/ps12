"""Training losses for frame interpolation.

Charbonnier (robust L1) photometric loss + a census/structure term that is robust to the smooth BT
gradients and brightness shifts typical of thermal-IR cloud fields. Real torch implementations.
"""
from __future__ import annotations


def charbonnier(pred, target, eps: float = 1e-3):
    import torch
    return torch.sqrt((pred - target) ** 2 + eps ** 2).mean()


def gradient_loss(pred, target):
    """L1 on spatial gradients — sharpens cloud edges, penalises blur."""
    import torch
    def grad(x):
        gx = x[..., :, 1:] - x[..., :, :-1]
        gy = x[..., 1:, :] - x[..., :-1, :]
        return gx, gy
    pgx, pgy = grad(pred)
    tgx, tgy = grad(target)
    return (pgx - tgx).abs().mean() + (pgy - tgy).abs().mean()


def census_loss(pred, target, eps: float = 1e-2):
    """Soft census transform difference (illumination-robust structural matching)."""
    import torch
    import torch.nn.functional as F

    def census(x):
        # 3x3 local mean-normalised comparison
        pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
        patches = F.unfold(pad, kernel_size=3).view(x.shape[0], x.shape[1], 9, x.shape[2], x.shape[3])
        center = patches[:, :, 4:5]
        return torch.tanh((patches - center) / eps)
    cp, ct = census(pred), census(target)
    return (cp - ct).abs().mean()


def combined_loss(pred, target, weights=(1.0, 0.1, 0.1)):
    w_ch, w_gr, w_ce = weights
    total = w_ch * charbonnier(pred, target)
    if w_gr:
        total = total + w_gr * gradient_loss(pred, target)
    if w_ce:
        total = total + w_ce * census_loss(pred, target)
    return total
