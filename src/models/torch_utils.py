"""Shared torch helpers for the deep interpolators.

Satellite frames are single-channel [0, 1]. Pretrained VFI/flow nets expect 3-channel input and
spatial dims divisible by a stride (8/32). These helpers replicate the channel, pad/un-pad, and
convert between numpy (H, W) and torch (1, 3, H, W) — keeping everything deterministic.
"""
from __future__ import annotations

import numpy as np


def pad_to_multiple(arr: np.ndarray, m: int = 32) -> tuple[np.ndarray, tuple[int, int]]:
    """Reflect-pad a 2-D array so H, W are multiples of `m`. Returns (padded, (orig_h, orig_w))."""
    h, w = arr.shape
    ph, pw = (-h) % m, (-w) % m
    padded = np.pad(arr, ((0, ph), (0, pw)), mode="reflect") if (ph or pw) else arr
    return padded, (h, w)


def unpad(arr: np.ndarray, orig: tuple[int, int]) -> np.ndarray:
    h, w = orig
    return arr[:h, :w]


def gray_to_tensor(arr: np.ndarray):
    """(H, W) float [0,1] -> torch (1, 3, H, W)."""
    import torch
    a = np.nan_to_num(arr).astype(np.float32)
    t = torch.from_numpy(a)[None, None].clamp(0, 1)
    return t.repeat(1, 3, 1, 1)


def tensor_to_gray(t) -> np.ndarray:
    """torch (1, C, H, W) -> (H, W) float by averaging channels."""
    arr = t.detach().cpu().float().clamp(0, 1).numpy()[0]
    return arr.mean(axis=0).astype(np.float32)


def device_auto(prefer: str | None = None) -> str:
    import torch
    if prefer:
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"
