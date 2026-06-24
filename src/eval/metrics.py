"""Image-quality and cloud-motion metrics.

PS-named: MSE, PSNR, SSIM, FSIM (+ MAE in Kelvin). Perceptual: LPIPS. Cloud-motion-aware: optical-flow
endpoint error (EPE) and temporal warping error. Inputs are normalized [0, 1] frames unless noted;
invalid (NaN) pixels are excluded from MSE/PSNR/MAE and zero-filled for structural metrics.

Core metrics use numpy/scikit-image (always available). FSIM/LPIPS use `piq` (torch) when present and
degrade gracefully otherwise — they are never faked.
"""
from __future__ import annotations

import numpy as np


def _valid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & np.isfinite(b)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    m = _valid(a, b)
    if not m.any():
        return float("nan")
    d = a[m] - b[m]
    return float(np.mean(d * d))


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    e = mse(a, b)
    if e <= 0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / e))


def mae_kelvin(a_bt: np.ndarray, b_bt: np.ndarray) -> float:
    """Mean absolute error in Kelvin (physical interpretability)."""
    m = _valid(a_bt, b_bt)
    return float(np.mean(np.abs(a_bt[m] - b_bt[m]))) if m.any() else float("nan")


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    from skimage.metrics import structural_similarity
    a0 = np.nan_to_num(a).astype(np.float64)
    b0 = np.nan_to_num(b).astype(np.float64)
    return float(structural_similarity(a0, b0, data_range=data_range))


def edge_ssim(a: np.ndarray, b: np.ndarray, data_range: float = 1.0, percentile: float = 80.0) -> float:
    """SSIM restricted to high-gradient (cloud-edge) regions — sensitive to cloud-structure motion."""
    from skimage.metrics import structural_similarity
    a0, b0 = np.nan_to_num(a).astype(np.float64), np.nan_to_num(b).astype(np.float64)
    gy, gx = np.gradient(a0)
    grad = np.hypot(gx, gy)
    thr = np.percentile(grad, percentile)
    mask = grad >= thr
    full, smap = structural_similarity(a0, b0, data_range=data_range, full=True)
    return float(smap[mask].mean()) if mask.any() else float(full)


def fsim(a: np.ndarray, b: np.ndarray) -> float | None:
    """Feature Similarity Index via piq (torch). Returns None if piq/torch unavailable."""
    try:
        import torch
        import piq
    except Exception:
        return None
    ta = torch.from_numpy(np.nan_to_num(a)).float()[None, None].clamp(0, 1).repeat(1, 3, 1, 1)
    tb = torch.from_numpy(np.nan_to_num(b)).float()[None, None].clamp(0, 1).repeat(1, 3, 1, 1)
    with torch.no_grad():
        return float(piq.fsim(ta, tb, data_range=1.0))


def lpips(a: np.ndarray, b: np.ndarray) -> float | None:
    """Learned perceptual similarity via piq (torch). Returns None if unavailable."""
    try:
        import torch
        import piq
    except Exception:
        return None
    ta = torch.from_numpy(np.nan_to_num(a)).float()[None, None].clamp(0, 1).repeat(1, 3, 1, 1)
    tb = torch.from_numpy(np.nan_to_num(b)).float()[None, None].clamp(0, 1).repeat(1, 3, 1, 1)
    with torch.no_grad():
        return float(piq.LPIPS()(ta, tb))


def flow_epe(pred_flow: np.ndarray, gt_flow: np.ndarray) -> float:
    """Average optical-flow endpoint error (cloud-motion fidelity)."""
    d = pred_flow - gt_flow
    return float(np.mean(np.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2)))


def temporal_warp_error(prev: np.ndarray, curr: np.ndarray) -> float:
    """Consistency: MSE between curr and prev motion-compensated toward curr (lower = smoother)."""
    from ..models.classical import _dense_flow, _warp
    flow = _dense_flow(prev, curr)
    warped = _warp(np.nan_to_num(prev), flow)
    return mse(warped, np.nan_to_num(curr))


def compute_all(pred: np.ndarray, gt: np.ndarray, *, bt_min: float, bt_max: float) -> dict[str, float]:
    """Full metric suite for a predicted vs ground-truth normalized frame pair."""
    span = bt_max - bt_min
    pred_bt = pred * span + bt_min
    gt_bt = gt * span + bt_min
    out: dict[str, float] = {
        "mse": mse(pred, gt),
        "psnr": psnr(pred, gt, data_range=1.0),
        "ssim": ssim(pred, gt),
        "edge_ssim": edge_ssim(pred, gt),
        "mae_kelvin": mae_kelvin(pred_bt, gt_bt),
    }
    f = fsim(pred, gt)
    if f is not None:
        out["fsim"] = f
    lp = lpips(pred, gt)
    if lp is not None:
        out["lpips"] = lp
    return out
