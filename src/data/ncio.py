"""NetCDF (.nc) writer for interpolated frames — the PS I/O contract: output files are `.nc`.

Rebuilds a CF-1.8 NetCDF holding the (denormalized) brightness temperature in Kelvin, the geolocation
(2-D lat/lon when available), and the synthetic timestamp. Round-trips with `read_nc` (asserted by the
deterministic battery).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def write_nc(
    path: str | Path,
    bt: np.ndarray,
    time: datetime,
    *,
    source: str,
    band: str,
    lats: np.ndarray | None = None,
    lons: np.ndarray | None = None,
    synthetic: bool = True,
    extra_attrs: dict | None = None,
) -> Path:
    """Write a 2-D BT field (Kelvin) to a CF NetCDF file."""
    if bt.ndim != 2:
        raise ValueError(f"BT must be 2-D, got {bt.shape}")
    h, w = bt.shape
    coords: dict = {"y": np.arange(h, dtype=np.int32), "x": np.arange(w, dtype=np.int32)}
    data_vars = {
        "BT": (
            ("y", "x"),
            bt.astype(np.float32),
            {
                "long_name": "brightness_temperature",
                "standard_name": "toa_brightness_temperature",
                "units": "K",
                "_FillValue": np.float32(np.nan),
            },
        )
    }
    if lats is not None and lons is not None:
        data_vars["lat"] = (("y", "x"), lats.astype(np.float32), {"units": "degrees_north", "standard_name": "latitude"})
        data_vars["lon"] = (("y", "x"), lons.astype(np.float32), {"units": "degrees_east", "standard_name": "longitude"})

    if time.tzinfo is None:
        time = time.replace(tzinfo=timezone.utc)
    seconds = (time - _EPOCH).total_seconds()

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds["time"] = ((), np.float64(seconds), {"units": "seconds since 1970-01-01T00:00:00Z", "standard_name": "time"})
    ds.attrs.update(
        {
            "Conventions": "CF-1.8",
            "title": "PS12 interpolated thermal-IR frame",
            "source_satellite": source,
            "band": band,
            "frame_type": "synthetic_interpolated" if synthetic else "observed",
            "history": f"created by ps12 frame-interpolation at {datetime.now(timezone.utc).isoformat()}",
        }
    )
    if extra_attrs:
        ds.attrs.update({k: str(v) for k, v in extra_attrs.items()})

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {"BT": {"zlib": True, "complevel": 4}}
    ds.to_netcdf(path, encoding=encoding)
    return path


def read_nc(path: str | Path) -> tuple[np.ndarray, datetime, dict]:
    """Read a BT NetCDF written by `write_nc`. Returns (bt_kelvin, time, attrs)."""
    # decode_times=False keeps `time` as the raw float seconds we wrote (xarray would otherwise
    # auto-decode the CF units to datetime64, and float() of that overflows).
    with xr.open_dataset(path, decode_times=False) as ds:
        bt = ds["BT"].values.astype(np.float32)
        secs = float(ds["time"].values)
        t = datetime.fromtimestamp(secs, tz=timezone.utc)
        attrs = dict(ds.attrs)
    return bt, t, attrs
