"""Tiled `.nc -> .nc` frame interpolation — the PS I/O contract.

Pipeline: read BT -> normalize -> tile (with overlap) -> model per tile -> feather-blend untile ->
denormalize -> restore off-disk NaNs -> write CF NetCDF at the synthetic timestamp.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np

from ..data.ncio import write_nc
from ..data.normalize import BT_MAX_DEFAULT, BT_MIN_DEFAULT, bt_to_norm, fill_invalid, norm_to_bt
from ..data.readers import SatFrame, read_frame
from ..data.tiling import tile_image, untile_image
from ..models.base import Interpolator


def interpolate_pair_bt(
    bt0: np.ndarray,
    bt1: np.ndarray,
    model: Interpolator,
    t: float = 0.5,
    *,
    bt_min: float = BT_MIN_DEFAULT,
    bt_max: float = BT_MAX_DEFAULT,
    tile: int = 512,
    overlap: int = 64,
) -> np.ndarray:
    """Interpolate the intermediate BT field at fractional time `t` from two BT frames (Kelvin)."""
    model.ensure_available()
    if bt0.shape != bt1.shape:
        raise ValueError(f"frame shapes differ: {bt0.shape} vs {bt1.shape}")
    valid = np.isfinite(bt0) & np.isfinite(bt1)
    n0, _ = fill_invalid(bt_to_norm(bt0, bt_min, bt_max))
    n1, _ = fill_invalid(bt_to_norm(bt1, bt_min, bt_max))

    tiles0, positions, shape = tile_image(n0, tile, overlap)
    tiles1, _, _ = tile_image(n1, tile, overlap)
    out_tiles = [model.interpolate(a, b, t) for a, b in zip(tiles0, tiles1)]
    pred_norm = untile_image(out_tiles, positions, shape, overlap)

    pred_bt = norm_to_bt(pred_norm, bt_min, bt_max)
    pred_bt = np.where(valid, pred_bt, np.nan).astype(np.float32)
    return pred_bt


def interpolate_nc(
    path0: str | Path,
    path1: str | Path,
    source: str,
    model: Interpolator,
    out_path: str | Path,
    t: float = 0.5,
    *,
    bt_min: float = BT_MIN_DEFAULT,
    bt_max: float = BT_MAX_DEFAULT,
    with_lonlat: bool = True,
    tile: int = 512,
    overlap: int = 64,
) -> tuple[Path, SatFrame, SatFrame]:
    """Read two frames, interpolate the middle, write it as `.nc`. Returns (out_path, frame0, frame1)."""
    fr0 = read_frame(path0, source, with_lonlat=with_lonlat)
    fr1 = read_frame(path1, source, with_lonlat=False)
    pred_bt = interpolate_pair_bt(fr0.bt, fr1.bt, model, t, bt_min=bt_min, bt_max=bt_max,
                                  tile=tile, overlap=overlap)
    mid_time = fr0.time + timedelta(seconds=(fr1.time - fr0.time).total_seconds() * t)
    out = write_nc(
        out_path, pred_bt, mid_time, source=source, band=fr0.band,
        lats=fr0.lats, lons=fr0.lons, synthetic=True,
        extra_attrs={"model": model.name, "t": t, "parent0": Path(path0).name, "parent1": Path(path1).name},
    )
    return out, fr0, fr1
