"""Parse acquisition timestamps from satellite filenames and build interpolation triplets.

A triplet (t0, t1, t2) has t1 as the ground-truth middle frame. The model learns t1 = f(t0, t2).
For 10-min sources (GOES/Himawari) the natural triplet spans 20 min (predict the +10 min frame).
For INSAT (30-min) the leave-one-out triplet spans 60 min (predict the real +30 min frame).
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

# GOES ABI: ...sYYYYJJJHHMMSSS_e..._c....nc  (start-time token, JJJ = day-of-year)
_GOES_RE = re.compile(r"_s(?P<y>\d{4})(?P<doy>\d{3})(?P<h>\d{2})(?P<m>\d{2})(?P<s>\d{2})\d?")
# INSAT MOSDAC: 3RIMG_24JUN2026_0014_L1C_SGP_V01R00.h5  /  3DIMG_...
_INSAT_RE = re.compile(r"3[RD]IMG_(?P<d>\d{2})(?P<mon>[A-Z]{3})(?P<y>\d{4})_(?P<h>\d{2})(?P<m>\d{2})", re.I)
# Himawari HSD: HS_H09_20260624_0010_B13_FLDK_R20_S0110.DAT(.bz2)
_HIMA_RE = re.compile(r"HS_H\d{2}_(?P<y>\d{4})(?P<mon>\d{2})(?P<d>\d{2})_(?P<h>\d{2})(?P<m>\d{2})")

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
)}


def parse_timestamp(name: str, source: str) -> datetime:
    """Extract the acquisition start time from a filename. Raises if it can't be parsed."""
    name = Path(name).name
    src = source.lower()
    if src.startswith("goes"):
        m = _GOES_RE.search(name)
        if m:
            base = datetime(int(m["y"]), 1, 1) + timedelta(days=int(m["doy"]) - 1)
            return base.replace(hour=int(m["h"]), minute=int(m["m"]), second=int(m["s"]))
    elif src.startswith("insat"):
        m = _INSAT_RE.search(name)
        if m:
            return datetime(int(m["y"]), _MONTHS[m["mon"].upper()], int(m["d"]), int(m["h"]), int(m["m"]))
    elif src.startswith("himawari"):
        m = _HIMA_RE.search(name)
        if m:
            return datetime(int(m["y"]), int(m["mon"]), int(m["d"]), int(m["h"]), int(m["m"]))
    raise ValueError(f"cannot parse timestamp from '{name}' for source '{source}'")


def index_frames(files: list[str | Path], source: str, dedup_minutes: float = 3.0) -> list[tuple[datetime, Path]]:
    """Return (timestamp, path) sorted by time, dropping near-duplicate slots (e.g. INSAT :14/:15)."""
    stamped: list[tuple[datetime, Path]] = []
    for f in files:
        try:
            stamped.append((parse_timestamp(str(f), source), Path(f)))
        except ValueError:
            continue
    stamped.sort(key=lambda x: x[0])
    deduped: list[tuple[datetime, Path]] = []
    for ts, p in stamped:
        if deduped and abs((ts - deduped[-1][0]).total_seconds()) < dedup_minutes * 60:
            continue
        deduped.append((ts, p))
    return deduped


def build_triplets(
    indexed: list[tuple[datetime, Path]],
    step_minutes: float,
    tol_minutes: float = 2.0,
) -> list[tuple[Path, Path, Path]]:
    """Form (t0, t1, t2) where t1-t0 ≈ t2-t1 ≈ step_minutes (consecutive equal-spaced frames)."""
    triplets: list[tuple[Path, Path, Path]] = []
    tol = tol_minutes * 60
    step = step_minutes * 60
    for i in range(len(indexed) - 2):
        (a, pa), (b, pb), (c, pc) = indexed[i], indexed[i + 1], indexed[i + 2]
        d1 = (b - a).total_seconds()
        d2 = (c - b).total_seconds()
        if abs(d1 - step) <= tol and abs(d2 - step) <= tol:
            triplets.append((pa, pb, pc))
    return triplets


def build_leave_one_out(
    indexed: list[tuple[datetime, Path]],
    step_minutes: float,
    tol_minutes: float = 2.0,
) -> list[tuple[Path, Path, Path]]:
    """INSAT eval: (t0, t_mid, t2) where t0,t2 are 2*step apart and t_mid is the held-out real frame.

    e.g. step=30 -> input 00:00 & 01:00, ground truth 00:30.
    """
    triplets: list[tuple[Path, Path, Path]] = []
    tol = tol_minutes * 60
    step = step_minutes * 60
    for i in range(len(indexed) - 2):
        (a, pa), (b, pb), (c, pc) = indexed[i], indexed[i + 1], indexed[i + 2]
        if abs((b - a).total_seconds() - step) <= tol and abs((c - b).total_seconds() - step) <= tol:
            triplets.append((pa, pb, pc))  # pb is the held-out GT for the 2*step input gap
    return triplets
