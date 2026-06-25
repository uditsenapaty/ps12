"""Overlapped tiling + feather-blended reconstruction for full-disk inference.

Full-disk frames are large (GOES ~5424², INSAT ~2816²). Models run on tiles. We tile with overlap and
reconstruct with a weighted (feathered) blend so tile seams disappear.

Reconstruction identity: because the blend is a weighted average with full coverage
(`out = Σ tile·w / Σ w`), `untile(tile(x)) == x` exactly for any positive weight window — this is
asserted by the deterministic battery.
"""
from __future__ import annotations

import numpy as np

Position = tuple[int, int]  # (y0, x0) top-left of a tile in the full image


def _starts(length: int, tile: int, stride: int) -> list[int]:
    """Tile start offsets covering [0, length), with the last tile flush to the edge."""
    if tile >= length:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def tile_image(img: np.ndarray, tile: int = 256, overlap: int = 32) -> tuple[list[np.ndarray], list[Position], tuple[int, int]]:
    """Split a 2-D image into (possibly overlapping) tiles.

    Returns (tiles, positions, original_shape). `img` may contain NaNs.
    """
    if img.ndim != 2:
        raise ValueError(f"expected 2-D image, got shape {img.shape}")
    h, w = img.shape
    stride = max(1, tile - overlap)
    ys, xs = _starts(h, tile, stride), _starts(w, tile, stride)
    tiles: list[np.ndarray] = []
    positions: list[Position] = []
    for y0 in ys:
        for x0 in xs:
            th, tw = min(tile, h - y0), min(tile, w - x0)
            tiles.append(img[y0 : y0 + th, x0 : x0 + tw].copy())
            positions.append((y0, x0))
    return tiles, positions, (h, w)


def _feather_window(h: int, w: int, overlap: int) -> np.ndarray:
    """Smooth 2-D weight window (1 in the centre, tapering to ~0 at edges)."""
    def ramp(n: int) -> np.ndarray:
        v = np.ones(n, dtype=np.float64)
        r = min(overlap, n // 2)
        if r > 0:
            edge = (np.arange(1, r + 1) / (r + 1))
            v[:r] = edge
            v[-r:] = edge[::-1]
        return v
    wy, wx = ramp(h), ramp(w)
    return np.outer(wy, wx) + 1e-6  # strictly positive -> safe division


def untile_image(
    tiles: list[np.ndarray],
    positions: list[Position],
    shape: tuple[int, int],
    overlap: int = 32,
) -> np.ndarray:
    """Reconstruct a full image from tiles via feather-blended weighted averaging."""
    h, w = shape
    acc = np.zeros((h, w), dtype=np.float64)
    wsum = np.zeros((h, w), dtype=np.float64)
    for t, (y0, x0) in zip(tiles, positions):
        th, tw = t.shape
        win = _feather_window(th, tw, overlap)
        acc[y0 : y0 + th, x0 : x0 + tw] += np.nan_to_num(t, nan=0.0) * win
        wsum[y0 : y0 + th, x0 : x0 + tw] += win
    out = acc / np.maximum(wsum, 1e-12)
    return out.astype(np.float32)
