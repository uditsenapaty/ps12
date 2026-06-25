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


def _grid_warp(img, flow):
    """Backward-warp `img` by `flow` (B,2,H,W) via bilinear grid_sample. out(x)=img(x+flow(x))."""
    import torch
    import torch.nn.functional as F
    B, C, H, W = img.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=img.device), torch.arange(W, device=img.device), indexing="ij")
    grid = torch.stack((xs, ys), 0).float()[None].repeat(B, 1, 1, 1) + flow
    grid[:, 0] = 2.0 * grid[:, 0] / max(W - 1, 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / max(H - 1, 1) - 1.0
    return F.grid_sample(img, grid.permute(0, 2, 3, 1), mode="bilinear", padding_mode="border", align_corners=True)


def advection_physics_loss(i0, i2, f_t0, f_t1, source, w_div: float = 0.05, w_src: float = 0.05):
    """PINN loss: brightness-transport (advection) with a learned source term.

    Physics: ∂I/∂t + u·∇I = S.  In integral/warp form between the two real input frames,
        I₂(x) ≈ I₀(x − u(x)) + S(x),     u = frame0→frame2 displacement = f_{t→1} − f_{t→0}.
    The source S models cloud growth/dissipation — exactly what classical optical flow (S≡0) can't.
    Adds a small low-divergence prior on u and an L1 sparsity prior on S.
    """
    u = f_t1 - f_t0                                  # frame0 -> frame2 displacement
    warped = _grid_warp(i0, -u)                       # transport I0 to frame2's grid
    l_adv = ((warped + source - i2) ** 2).mean()
    ux, uy = u[:, 0:1], u[:, 1:2]
    div = (ux[..., :, 1:] - ux[..., :, :-1]).abs().mean() + (uy[..., 1:, :] - uy[..., :-1, :]).abs().mean()
    l_src = source.abs().mean()
    return l_adv + w_div * div + w_src * l_src


def combined_loss(pred, target, weights=(1.0, 0.1, 0.1)):
    w_ch, w_gr, w_ce = weights
    total = w_ch * charbonnier(pred, target)
    if w_gr:
        total = total + w_gr * gradient_loss(pred, target)
    if w_ce:
        total = total + w_ce * census_loss(pred, target)
    return total
