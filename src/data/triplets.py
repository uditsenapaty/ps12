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


def build_anytime_samples(
    indexed: list[tuple[datetime, Path]],
    cadence_minutes: float,
    time_step: float = 0.5,
    gap_levels: int = 3,
    tol_minutes: float = 2.0,
) -> list[tuple[Path, Path, Path, float]]:
    """Arbitrary-time samples (t0, t2, gt, t) on a CONFIGURABLE t-grid across multiple gap sizes.

    This is *arbitrary-time* (continuous-time, Super-SloMo style) training with multi-rate (variable-
    gap) augmentation — NOT "multi-granularity" in the multi-scale sense: every (gap, t) row is an
    INDEPENDENT sample with its own loss. A multi-scale loss (α·L_fine + β·L_coarse) is a separate idea.

    Configurable cost vs. coverage (variable-t training is expensive, so it is opt-in / tunable):
      time_step (dt): spacing of the t-grid in (0,1). MUST evenly divide 1 (1/dt an integer), so the
        grid {dt, 2·dt, … 1−dt} tiles [0,1]. dt=0.5 → ONLY the midpoint t=0.5 (cheapest); dt=0.25 →
        t∈{0.25,0.5,0.75}; dt=0.2 → {0.2,0.4,0.6,0.8}. Smaller dt ⇒ many more samples ⇒ slower training.
      gap_levels: number of gap sizes (granules). The base span is (1/dt)·cadence so EVERY grid point
        lands exactly on a real frame; spans = base, 2·base, …, gap_levels·base. Bigger spans = larger
        motion magnitudes (covers the GOES 20-min → INSAT 60-min transfer).

    A sample is emitted only when a real frame exists at the required grid time (within tol). This is
    why some positions are skipped: e.g. with dt=0.5 a 40-min span needs the frame at +20 (t=0.5) — the
    frame at +10 is off-grid and is NOT used; near the sequence start/end, spans that don't fit are
    dropped. Each gap level contributes the same (1/dt − 1) grid points per anchor → balanced t coverage.
    """
    n_sub = round(1.0 / time_step)
    if n_sub < 2 or abs(n_sub * time_step - 1.0) > 1e-6:
        raise ValueError(f"time_step must evenly divide 1 (1/dt an integer >= 2); got {time_step}. "
                         "Use 0.5, 0.25, 0.2, 0.125, 0.1 ...")
    base_min = n_sub * cadence_minutes            # span where grid points coincide with real frames
    tol = tol_minutes * 60
    times = [t for t, _ in indexed]
    paths = [p for _, p in indexed]
    n = len(indexed)

    def _find(anchor: int, offset_min: float):
        """Index of a frame at times[anchor] + offset_min (within tol), else None. Sorted -> early-out."""
        target = times[anchor] + timedelta(minutes=offset_min)
        for k in range(anchor + 1, n):
            d = (times[k] - target).total_seconds()
            if abs(d) <= tol:
                return k
            if d > tol:
                break
        return None

    samples: list[tuple[Path, Path, Path, float]] = []
    for i in range(n):
        for m in range(1, int(gap_levels) + 1):
            span_min = base_min * m
            j = _find(i, span_min)                       # far input frame I2 at +span
            if j is None:
                continue
            for s in range(1, n_sub):                    # grid fraction s·dt = s/n_sub
                g = _find(i, cadence_minutes * m * s)    # grid time = (s/n_sub)·span = cadence·m·s
                if g is None:
                    continue
                samples.append((paths[i], paths[j], paths[g], round(s / n_sub, 4)))
    return samples


def build_multigap_groups(
    indexed: list[tuple[datetime, Path]],
    cadence_minutes: float,
    max_level: int = 2,
    tol_minutes: float = 2.0,
) -> list[tuple[Path, list[tuple[Path, Path]]]]:
    """Symmetric temporal multi-granularity groups: ONE target frame, several SYMMETRIC brackets.

    For target frame `g`, granularity level L uses the symmetric bracket (frame[g−L·cadence],
    frame[g+L·cadence]) whose midpoint is exactly `g` (so the interpolation time is always t=0.5).
    Levels run L = 1 .. max_level, bounded by the sequence ends:
        frame@10 -> only (0,20)            (no frame before 0, so level 1 only)
        frame@20 -> (10,30) and (0,40)     (levels 1 and 2)
        frame@30 -> (20,40) and (10,50)
    The training step reconstructs the ONE target from every bracket and sums the losses, so the model
    renders it consistently whether the gap is small (±1·cadence) or large (±max_level·cadence). This is
    multi-granularity in TIME — a combined multi-gap loss at the midpoint — controlled by `max_level`.

    Returns groups: (target_path, [(left_path, right_path), … up to max_level]).
    """
    tol = tol_minutes * 60
    times = [t for t, _ in indexed]
    paths = [p for _, p in indexed]
    n = len(indexed)

    def _at(g: int, offset_min: float):
        """Index of a frame at times[g] + offset_min (offset may be negative), within tol, else None."""
        target = times[g] + timedelta(minutes=offset_min)
        for k in range(n):
            if abs((times[k] - target).total_seconds()) <= tol:
                return k
        return None

    groups: list[tuple[Path, list[tuple[Path, Path]]]] = []
    for g in range(n):
        views: list[tuple[Path, Path]] = []
        for L in range(1, int(max_level) + 1):
            left = _at(g, -L * cadence_minutes)
            right = _at(g, +L * cadence_minutes)
            if left is None or right is None:
                break                                   # symmetric bracket doesn't fit -> stop widening
            views.append((paths[left], paths[right]))
        if views:                                       # any target with >=1 symmetric bracket (e.g. frame@10)
            groups.append((paths[g], views))
    return groups


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
