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


def build_multigran(
    indexed: list[tuple[datetime, Path]],
    base_gap_minutes: float,
    levels: int = 3,
    tol_minutes: float = 2.0,
) -> list[tuple[Path, Path, Path, float]]:
    """Variable-gap, variable-t samples (t0, t2, gt, t) for granularity-aware training.

    For each anchor frame and each level L in 1..levels, the input span is L*base_gap_minutes; EVERY
    real interior frame becomes a ground-truth target at its true fraction t = (t_gt - t0) / span.
    Dense 10-min GOES/Himawari therefore teach the model many gaps (motion magnitudes) AND many
    times t — exactly what the 30→15→7.5 product needs (t=0.25 / 0.5 / 0.75), instead of only the
    t=0.5 midpoint with a linear-motion assumption.

    e.g. base_gap=20, levels=3 on a 10-min sequence -> spans 20/40/60 min:
      span 20 -> t=0.5 ; span 40 -> t=0.25,0.5,0.75 ; span 60 -> t=1/6..5/6.
    """
    samples: list[tuple[Path, Path, Path, float]] = []
    tol = tol_minutes * 60
    n = len(indexed)
    for i in range(n):
        t0, p0 = indexed[i]
        for L in range(1, int(levels) + 1):
            span = base_gap_minutes * 60 * L
            j = None
            for k in range(i + 1, n):
                dt = (indexed[k][0] - t0).total_seconds()
                if abs(dt - span) <= tol:
                    j = k
                    break
                if dt > span + tol:
                    break
            if j is None:
                continue
            t2, p2 = indexed[j]
            actual = (t2 - t0).total_seconds()
            for g in range(i + 1, j):
                tg = (indexed[g][0] - t0).total_seconds()
                if tg <= 0 or tg >= actual:
                    continue
                samples.append((p0, p2, indexed[g][1], round(tg / actual, 4)))
    return samples


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
