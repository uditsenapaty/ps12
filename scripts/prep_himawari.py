#!/usr/bin/env python
"""One-time Himawari-9 prep: assemble each AHI B13 timestep's 10 HSD segments via satpy, central-crop,
and write a fast BT NetCDF the project pipeline can read directly (read_himawari .nc fast-path).

This avoids re-running bz2 decompression + satpy calibration on every training read. Output filenames
keep the HS_H09_<YYYYMMDD>_<HHMM>_B13 token so the existing himawari timestamp regex + index builder
pick them up unchanged.

  python scripts/prep_himawari.py [N_TIMESTEPS] [CROP]
"""
from __future__ import annotations

import glob
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.ncio import write_nc  # noqa: E402

SEGDIR = ROOT / "data" / "himawari9"
OUTDIR = ROOT / "data" / "himawari9_nc"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 72
CROP = int(sys.argv[2]) if len(sys.argv) > 2 else 2048


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    segs = sorted(glob.glob(str(SEGDIR / "HS_H09_*_B13_FLDK_*.DAT*")))
    groups: dict[str, list[str]] = {}
    for s in segs:
        m = re.search(r"HS_H09_(\d{8}_\d{4})_B13", Path(s).name)
        if m:
            groups.setdefault(m.group(1), []).append(s)
    ts_sorted = sorted(groups)

    from satpy import Scene
    done = 0
    for ts in ts_sorted:
        files = sorted(groups[ts])
        if len(files) < 10:                       # need the full disk (all 10 segments)
            continue
        out = OUTDIR / f"HS_H09_{ts}_B13.nc"
        if out.exists():
            done += 1
            if done >= LIMIT:
                break
            continue
        try:
            scn = Scene(reader="ahi_hsd", filenames=files)
            scn.load(["B13"])
            bt = np.asarray(scn["B13"].values, dtype=np.float32)
            bt[bt <= 0] = np.nan
            h, w = bt.shape
            half = CROP // 2
            cy, cx = h // 2, w // 2
            crop = bt[cy - half:cy + half, cx - half:cx + half]
            t = datetime.strptime(ts, "%Y%m%d_%H%M")
            write_nc(out, crop, t, source="himawari9", band="B13", synthetic=False)
            done += 1
            print(f"[prep] {ts} -> {out.name}  crop{crop.shape} "
                  f"BT[{np.nanmin(crop):.1f},{np.nanmax(crop):.1f}]", flush=True)
        except Exception as e:
            print(f"[prep] {ts} FAILED {type(e).__name__}: {str(e)[:140]}", flush=True)
        if done >= LIMIT:
            break
    print(f"[prep] done: {done} himawari timesteps -> {OUTDIR}")


if __name__ == "__main__":
    main()
