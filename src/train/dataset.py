"""SatTripletDataset — yields normalized (I0, GT, I1) patches from interpolation triplets.

Reads a triplet index (built by data_setup.py), loads the three frames, converts to normalized BT,
and samples a random patch with enough valid (non-space) pixels. Geometry-only augmentation (flip /
rot90) preserves physical BT values. Fully real and CPU-testable on the sample data.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..data.normalize import BT_MAX_DEFAULT, BT_MIN_DEFAULT, bt_to_norm, fill_invalid
from ..data.readers import read_frame


@lru_cache(maxsize=512)  # hold all frames in memory (cropped) so reads hit cache after warmup
def _read_norm(path: str, source: str, bt_min: float, bt_max: float, crop_frac: float) -> np.ndarray:
    """Cached read -> normalized [0,1] (NaN-filled with 0). Cache avoids re-reading shared frames.
    `crop_frac` reads only the central fraction of the disk (big training speed-up for GOES full disk)."""
    kw: dict = {"with_lonlat": False}
    if crop_frac and source.lower().startswith("goes"):
        kw["crop_frac"] = crop_frac
    fr = read_frame(path, source, **kw)
    n, _ = fill_invalid(bt_to_norm(fr.bt, bt_min, bt_max))
    return n.astype(np.float32)


class SatTripletDataset:
    """Indexable triplet dataset. Use with torch.utils.data.DataLoader (returns torch tensors if torch
    is installed, else numpy). Each item: dict(I0, GT, I1, t)."""

    def __init__(
        self,
        index_json: str | Path | None = None,
        triplets: list[tuple[str, str, str]] | None = None,
        samples: list[tuple[str, str, str, float]] | None = None,
        source: str = "goes19",
        patch: int = 256,
        bt_min: float = BT_MIN_DEFAULT,
        bt_max: float = BT_MAX_DEFAULT,
        augment: bool = True,
        min_valid_frac: float = 0.5,
        seed: int = 1234,
        crop_frac: float = 0.35,
        anytime: bool = False,
    ):
        if samples is None and triplets is None:
            if index_json is None:
                raise ValueError("provide index_json, triplets, or samples")
            data = json.loads(Path(index_json).read_text(encoding="utf-8"))
            source = data.get("source", source)
            if anytime and data.get("anytime"):
                samples = data["anytime"]
            else:
                triplets = data["triplets"]
        # Canonical sample = (I0_path, I1_path, GT_path, t). Triplets (t0, t_mid, t2) map to t=0.5 with
        # I0=t0, I1=t2, GT=t_mid; arbitrary-time rows are already (t0, t2, gt, t).
        if samples is not None:
            self.samples = [(str(s[0]), str(s[1]), str(s[2]), float(s[3])) for s in samples]
        else:
            self.samples = [(str(t[0]), str(t[2]), str(t[1]), 0.5) for t in triplets]
        self.source = source
        self.patch = patch
        self.bt_min, self.bt_max = bt_min, bt_max
        self.augment = augment
        self.min_valid_frac = min_valid_frac
        self.crop_frac = crop_frac
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_patch(self, a: np.ndarray, b: np.ndarray, c: np.ndarray):
        h, w = a.shape
        p = self.patch
        if h <= p or w <= p:
            return a, b, c
        for _ in range(10):
            y = int(self.rng.integers(0, h - p))
            x = int(self.rng.integers(0, w - p))
            pb = b[y:y + p, x:x + p]
            if np.mean(pb > 0) >= self.min_valid_frac:  # enough real (non-space) signal
                return a[y:y + p, x:x + p], pb, c[y:y + p, x:x + p]
        return a[:p, :p], b[:p, :p], c[:p, :p]

    def _augment(self, *arrs):
        if not self.augment:
            return arrs
        k = int(self.rng.integers(0, 4))
        flip = bool(self.rng.integers(0, 2))
        out = []
        for a in arrs:
            a = np.rot90(a, k)
            if flip:
                a = np.fliplr(a)
            out.append(np.ascontiguousarray(a))
        return tuple(out)

    def __getitem__(self, i: int):
        p0, p2, pgt, t = self.samples[i]            # I0, I1 (other end), GT (interior frame), time t
        a = _read_norm(p0, self.source, self.bt_min, self.bt_max, self.crop_frac)   # I0
        c = _read_norm(p2, self.source, self.bt_min, self.bt_max, self.crop_frac)   # I1
        b = _read_norm(pgt, self.source, self.bt_min, self.bt_max, self.crop_frac)  # GT
        a, b, c = self._sample_patch(a, b, c)       # validity is checked on the GT (b)
        a, b, c = self._augment(a, b, c)
        try:
            import torch
            to = lambda x: torch.from_numpy(x).float()[None]  # (1, H, W)
            return {"I0": to(a), "GT": to(b), "I1": to(c), "t": float(t)}
        except Exception:
            return {"I0": a[None], "GT": b[None], "I1": c[None], "t": float(t)}
