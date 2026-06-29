"""Himawari-9 AHI reader via satpy (`ahi_hsd`).

AHI Full-Disk Band 13 (10.4 µm) is delivered as ~10 HSD segment files per timestep. `read_himawari`
accepts a directory, a glob, or the list of segments for one timestep and returns calibrated
brightness temperature (Kelvin). Gated: raises clearly if satpy isn't installed (server has it).
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

from .readers import SatFrame
from .triplets import parse_timestamp


def _segments(path: str | Path) -> list[str]:
    p = Path(path)
    if p.is_dir():
        return sorted(glob.glob(str(p / "*.DAT")) + glob.glob(str(p / "*.DAT.bz2")))
    if any(ch in str(p) for ch in "*?["):
        return sorted(glob.glob(str(p)))
    return [str(p)]


def read_himawari(path: str | Path, source: str = "himawari9", band: str = "B13",
                  with_lonlat: bool = False) -> SatFrame:
    # Fast path (added): a pre-decoded BT NetCDF produced by scripts/prep_himawari.py. Reading the raw
    # HSD .DAT.bz2 segments through satpy is slow (bz2 decompress + calibrate per read), so for training
    # we convert each timestep once to a central-cropped BT .nc and read that here. Routed via read_frame
    # exactly like the segment path; documented in explanation.md.
    if Path(path).suffix.lower() == ".nc":
        from .ncio import read_nc
        bt, t, attrs = read_nc(path)
        return SatFrame(bt=bt.astype(np.float32), time=t, source=source,
                        band=attrs.get("band", band), meta={"file": Path(path).name, "cached_nc": True})

    try:
        from satpy import Scene
    except Exception as e:  # pragma: no cover - server has satpy
        raise RuntimeError("satpy not installed — needed to read Himawari AHI HSD (pip install satpy).") from e

    files = _segments(path)
    if not files:
        raise FileNotFoundError(f"no AHI segment files found at {path}")
    scn = Scene(reader="ahi_hsd", filenames=files)
    scn.load([band])
    da = scn[band]                                   # brightness temperature (K)
    bt = np.asarray(da.values, dtype=np.float32)
    bt[bt <= 0] = np.nan

    t = da.attrs.get("start_time")
    if t is None:
        t = parse_timestamp(Path(files[0]).name, source)

    lats = lons = None
    if with_lonlat and "area" in da.attrs:
        lon2d, lat2d = da.attrs["area"].get_lonlats()
        lats = np.asarray(lat2d, dtype=np.float32)
        lons = np.asarray(lon2d, dtype=np.float32)
        lats[~np.isfinite(lats)] = np.nan
        lons[~np.isfinite(lons)] = np.nan

    return SatFrame(bt=bt, time=t, source=source, band=band, lats=lats, lons=lons,
                    meta={"file": Path(files[0]).name, "n_segments": len(files), "resolution_km": 2.0})
