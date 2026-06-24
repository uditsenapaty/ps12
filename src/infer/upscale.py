"""Recursive temporal upscaling of a frame sequence (the INSAT 30→15→7.5 min product).

`levels=1` halves the interval (30→15), `levels=2` halves again (30→15→7.5). Each level inserts a
synthetic midpoint between every pair of consecutive frames, so a sequence of N frames at cadence T
becomes (N-1)·2^levels + 1 frames at cadence T/2^levels. Outputs are written as `.nc` (PS I/O contract).
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np

from ..data.ncio import write_nc
from ..data.normalize import BT_MAX_DEFAULT, BT_MIN_DEFAULT
from ..data.readers import read_frame
from ..models.base import Interpolator
from .interpolate import interpolate_pair_bt


def upscale_total_steps(n_frames: int, levels: int) -> int:
    """Total number of midpoint interpolations for a progress bar."""
    total, cur = 0, n_frames
    for _ in range(levels):
        total += cur - 1
        cur = 2 * cur - 1
    return total


def upscale_once_bt(frames: list[np.ndarray], model: Interpolator, progress=None, **kw) -> list[np.ndarray]:
    """Insert one synthetic midpoint between each consecutive pair (cadence -> cadence/2).

    `progress` is an optional no-arg callable invoked after each interpolation (for UI progress bars).
    """
    out: list[np.ndarray] = []
    for i in range(len(frames) - 1):
        out.append(frames[i])
        out.append(interpolate_pair_bt(frames[i], frames[i + 1], model, 0.5, **kw))
        if progress:
            progress()
    out.append(frames[-1])
    return out


def temporal_upscale_bt(frames: list[np.ndarray], model: Interpolator, levels: int = 2,
                        progress=None, **kw) -> list[np.ndarray]:
    """Apply `upscale_once_bt` `levels` times (30→15→7.5 for levels=2)."""
    seq = list(frames)
    for _ in range(levels):
        seq = upscale_once_bt(seq, model, progress=progress, **kw)
    return seq


def temporal_upscale_nc(
    paths: list[str | Path],
    source: str,
    model: Interpolator,
    out_dir: str | Path,
    levels: int = 2,
    base_step_min: float = 30.0,
    *,
    bt_min: float = BT_MIN_DEFAULT,
    bt_max: float = BT_MAX_DEFAULT,
    tile: int = 512,
    overlap: int = 64,
) -> tuple[list[Path], list[np.ndarray]]:
    """Read a sequence of `.nc/.h5` frames, upscale temporally, and write the dense `.nc` sequence."""
    model.ensure_available()
    frames = [read_frame(p, source, with_lonlat=(i == 0)) for i, p in enumerate(paths)]
    bts = [f.bt for f in frames]
    up = temporal_upscale_bt(bts, model, levels, bt_min=bt_min, bt_max=bt_max, tile=tile, overlap=overlap)

    t0 = frames[0].time
    step = timedelta(minutes=base_step_min / (2 ** levels))
    factor = 2 ** levels
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for i, bt in enumerate(up):
        t = t0 + i * step
        is_orig = (i % factor == 0)
        op = out_dir / f"{source}_{t.strftime('%Y%m%d_%H%M%S')}_{'orig' if is_orig else 'synth'}.nc"
        write_nc(op, bt, t, source=source, band=frames[0].band,
                 lats=frames[0].lats, lons=frames[0].lons, synthetic=not is_orig,
                 extra_attrs={"model": model.name, "upscale_levels": levels, "cadence_min": base_step_min / factor})
        out_paths.append(op)
    return out_paths, up
