"""INSAT-3DR / 3DS Imager L1C_SGP reader (HDF5 from MOSDAC).

Verified against a real `3RIMG_..._L1C_SGP_V01R00.h5`. The file stores each band as raw 10-bit counts
(`IMG_TIR1`, shape (1,H,W) uint16) plus a 1024-entry **count → brightness-temperature lookup table**
(`IMG_TIR1_TEMP`, Kelvin). So calibrated BT = `LUT[count]`. TIR1 ≈ 10.8 µm is the PS-required ~10 µm
band; TIR2/MIR/WV are also present (future multi-band work). Off-disk/space pixels (count 0) → NaN.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .readers import SatFrame
from .triplets import parse_timestamp

# band -> (count image dataset, count->BT lookup-table dataset)
_BANDS = {
    "TIR1": ("IMG_TIR1", "IMG_TIR1_TEMP"),   # ~10.8 µm  (default, PS-required)
    "TIR2": ("IMG_TIR2", "IMG_TIR2_TEMP"),   # ~12.0 µm
    "MIR":  ("IMG_MIR", "IMG_MIR_TEMP"),     # ~3.9 µm
    "WV":   ("IMG_WV", "IMG_WV_TEMP"),       # ~6.8 µm (water vapour)
}


def read_insat_h5(path: str | Path, source: str = "insat3dr", band: str = "TIR1",
                  with_lonlat: bool = False, crop_frac: float | None = None) -> SatFrame:
    """Read one INSAT L1C_SGP band into a SatFrame of calibrated brightness temperature (Kelvin)."""
    import h5py
    path = Path(path)
    img_ds, lut_ds = _BANDS.get(band.upper(), _BANDS["TIR1"])
    with h5py.File(path, "r") as h:
        if img_ds not in h:
            raise KeyError(f"{img_ds} not in {path.name}; available: {list(h.keys())[:12]}")
        arr = h[img_ds][0]                                  # (H, W) uint16 counts
        if crop_frac and crop_frac < 1.0:
            H, W = arr.shape
            ch, cw = int(H * crop_frac), int(W * crop_frac)
            y0, x0 = (H - ch) // 2, (W - cw) // 2
            arr = arr[y0:y0 + ch, x0:x0 + cw]
        counts = arr.astype(np.int32)
        lut = h[lut_ds][:].astype(np.float32)               # count -> BT (K)
        bt = lut[np.clip(counts, 0, len(lut) - 1)]
        # off-disk/space pixels are count 0; also drop non-physical LUT entries
        bt = np.where((counts > 0) & (bt > 100.0) & (bt < 360.0), bt, np.nan).astype(np.float32)
        sat = h.attrs.get("Satellite_Name", b"")
        sat = sat.decode() if isinstance(sat, bytes) else str(sat)

    return SatFrame(
        bt=bt, time=parse_timestamp(path.name, source), source=source,
        band=f"{band.upper()}", lats=None, lons=None,
        meta={"file": path.name, "satellite": sat, "resolution_km": 4.0, "img_ds": img_ds},
    )
