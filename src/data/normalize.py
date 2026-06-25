"""Invertible brightness-temperature (BT) normalization.

Satellite TIR inputs are calibrated BT in Kelvin. VFI networks expect ~[0, 1] inputs. We map BT to
[0, 1] with a FIXED physical range so the transform is deterministic and exactly invertible — model
outputs round-trip back to physical Kelvin for the `.nc` writer.

The clean-IR window (GOES Ch13 10.3 µm, Himawari B13 10.4 µm, INSAT TIR1 10.8 µm) spans roughly
180 K (cold convective cloud tops) to 330 K (hot land/sea surface). NaN/space/fill pixels are
preserved as NaN through the round-trip; use `fill_invalid` to get a model-ready array + mask.
"""
from __future__ import annotations

import numpy as np

BT_MIN_DEFAULT = 180.0  # K — coldest realistic cloud-top in the 10 µm window
BT_MAX_DEFAULT = 330.0  # K — hottest realistic surface


def bt_to_norm(bt: np.ndarray, bt_min: float = BT_MIN_DEFAULT, bt_max: float = BT_MAX_DEFAULT) -> np.ndarray:
    """Kelvin -> [0, 1] (clipped). NaNs are preserved as NaN."""
    if bt_max <= bt_min:
        raise ValueError(f"bt_max ({bt_max}) must exceed bt_min ({bt_min})")
    x = (bt.astype(np.float32) - bt_min) / (bt_max - bt_min)
    return np.clip(x, 0.0, 1.0)


def norm_to_bt(x: np.ndarray, bt_min: float = BT_MIN_DEFAULT, bt_max: float = BT_MAX_DEFAULT) -> np.ndarray:
    """[0, 1] -> Kelvin (exact inverse of bt_to_norm modulo the clip). NaNs preserved."""
    return x.astype(np.float32) * (bt_max - bt_min) + bt_min


def fill_invalid(x: np.ndarray, fill: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Replace NaN with `fill` for model input; return (filled, valid_mask).

    `valid_mask` is True where the pixel was a real observation. Restore NaNs after inference with
    `restore_invalid`.
    """
    mask = np.isfinite(x)
    filled = np.where(mask, x, fill).astype(np.float32)
    return filled, mask


def restore_invalid(x: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Set pixels that were invalid in the input back to NaN."""
    out = x.astype(np.float32).copy()
    out[~valid_mask] = np.nan
    return out
