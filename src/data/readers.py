"""Readers that turn raw satellite files into a common `SatFrame` (calibrated BT in Kelvin).

- GOES-19 ABI L1b RadF (.nc): radiance -> brightness temperature via the file's Planck constants;
  optional GOES-R fixed-grid -> geodetic (lat/lon).
- INSAT-3DR/3DS L1C_SGP (.h5): implemented in `readers_insat.py` (calibrated against the real file
  structure documented in docs/data-structure.md).
- Himawari-9 AHI: read via satpy `ahi_hsd` in `readers_himawari.py`.

All readers return NaN for off-disk / fill pixels so normalization and metrics handle them uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from .triplets import parse_timestamp


@dataclass
class SatFrame:
    bt: np.ndarray                      # (H, W) float32 Kelvin, NaN where invalid
    time: datetime
    source: str                         # 'goes19' | 'himawari9' | 'insat3dr' | 'insat3ds'
    band: str                           # 'C13' | 'B13' | 'TIR1'
    lats: np.ndarray | None = None      # (H, W) degrees_north
    lons: np.ndarray | None = None      # (H, W) degrees_east
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.bt.shape


def radiance_to_bt_goes(rad: np.ndarray, fk1: float, fk2: float, bc1: float, bc2: float) -> np.ndarray:
    """GOES-R ABI emissive-band brightness temperature (Kelvin) from spectral radiance."""
    with np.errstate(invalid="ignore", divide="ignore"):
        bt = (fk2 / np.log((fk1 / rad) + 1.0) - bc1) / bc2
    bt = np.where(rad > 0, bt, np.nan)
    return bt.astype(np.float32)


def goes_fixed_grid_lonlat(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """GOES-R fixed-grid scan angles (x, y) -> (lat, lon) degrees (PUG geometry)."""
    proj = ds["goes_imager_projection"]
    r_eq = float(proj.attrs["semi_major_axis"])
    r_pol = float(proj.attrs["semi_minor_axis"])
    h_sat = float(proj.attrs["perspective_point_height"]) + r_eq
    lon0 = np.deg2rad(float(proj.attrs["longitude_of_projection_origin"]))

    x = ds["x"].values.astype(np.float64)
    y = ds["y"].values.astype(np.float64)
    xx, yy = np.meshgrid(x, y)
    sin_x, cos_x = np.sin(xx), np.cos(xx)
    sin_y, cos_y = np.sin(yy), np.cos(yy)

    a = sin_x ** 2 + cos_x ** 2 * (cos_y ** 2 + (r_eq ** 2 / r_pol ** 2) * sin_y ** 2)
    b = -2.0 * h_sat * cos_x * cos_y
    c = h_sat ** 2 - r_eq ** 2
    disc = b ** 2 - 4.0 * a * c
    with np.errstate(invalid="ignore"):
        r_s = (-b - np.sqrt(disc)) / (2.0 * a)
        s_x = r_s * cos_x * cos_y
        s_y = -r_s * sin_x
        s_z = r_s * cos_x * sin_y
        lat = np.rad2deg(np.arctan((r_eq ** 2 / r_pol ** 2) * s_z / np.sqrt((h_sat - s_x) ** 2 + s_y ** 2)))
        lon = np.rad2deg(lon0 - np.arctan(s_y / (h_sat - s_x)))
    off_disk = disc < 0
    lat[off_disk] = np.nan
    lon[off_disk] = np.nan
    return lat.astype(np.float32), lon.astype(np.float32)


def read_goes_nc(path: str | Path, source: str = "goes19", with_lonlat: bool = False,
                 crop_frac: float | None = None) -> SatFrame:
    """Read a GOES ABI L1b Radiances file (single channel, e.g. C13) into a SatFrame.

    crop_frac (0<f<=1): read only the central fraction of the disk (sliced lazily before loading) —
    a big speed-up for training, since the full disk is ~5424² and mostly off-limb space.
    """
    path = Path(path)
    with xr.open_dataset(path, mask_and_scale=True) as ds:
        rad_da = ds["Rad"]
        if crop_frac and crop_frac < 1.0:
            h, w = rad_da.shape
            ch, cw = int(h * crop_frac), int(w * crop_frac)
            y0, x0 = (h - ch) // 2, (w - cw) // 2
            rad = rad_da[y0:y0 + ch, x0:x0 + cw].values.astype(np.float32)
        else:
            rad = rad_da.values.astype(np.float32)
        fk1 = float(ds["planck_fk1"].values)
        fk2 = float(ds["planck_fk2"].values)
        bc1 = float(ds["planck_bc1"].values)
        bc2 = float(ds["planck_bc2"].values)
        bt = radiance_to_bt_goes(rad, fk1, fk2, bc1, bc2)
        band_id = int(ds["band_id"].values) if "band_id" in ds else -1
        lats = lons = None
        if with_lonlat:
            lats, lons = goes_fixed_grid_lonlat(ds)
        meta = {"file": path.name, "band_id": band_id, "resolution_km": 2.0}
    return SatFrame(bt=bt, time=parse_timestamp(path.name, source), source=source,
                    band=f"C{band_id:02d}" if band_id > 0 else "C13", lats=lats, lons=lons, meta=meta)


def read_frame(path: str | Path, source: str, **kwargs) -> SatFrame:
    """Dispatch to the correct reader by source."""
    src = source.lower()
    if src.startswith("goes"):
        return read_goes_nc(path, source=src, **kwargs)
    if src.startswith("insat"):
        from .readers_insat import read_insat_h5
        return read_insat_h5(path, source=src, **kwargs)
    if src.startswith("himawari"):
        from .readers_himawari import read_himawari
        return read_himawari(path, source=src, **kwargs)
    raise ValueError(f"unknown source '{source}'")
