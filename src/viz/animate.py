"""Render brightness-temperature frames into colormapped images, animations, and motion overlays."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def bt_to_rgb(bt: np.ndarray, bt_min: float = 180.0, bt_max: float = 330.0, cmap: str = "Greys") -> np.ndarray:
    """Map a BT (K) field to an RGB uint8 image. 'Greys' shows cold cloud tops bright (IR convention)."""
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=bt_min, vmax=bt_max)
    x = norm(np.nan_to_num(bt, nan=bt_max))           # space/off-disk -> warm end
    rgba = cm.get_cmap(cmap)(1.0 - x)                 # invert: cold = bright
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return rgb


def frames_to_gif(frames_bt: list[np.ndarray], out_path: str | Path, *, fps: int = 6,
                  bt_min: float = 180.0, bt_max: float = 330.0, cmap: str = "Greys") -> Path:
    """Write a time-lapse GIF/MP4 from a list of BT frames (extension decides the format)."""
    import imageio.v2 as imageio
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [bt_to_rgb(f, bt_min, bt_max, cmap) for f in frames_bt]
    if out_path.suffix.lower() in (".mp4", ".m4v"):
        imageio.mimsave(out_path, imgs, fps=fps)
    else:
        imageio.mimsave(out_path, imgs, duration=1.0 / fps)
    return out_path


def flow_overlay_rgb(bt: np.ndarray, flow: np.ndarray, step: int = 24,
                     bt_min: float = 180.0, bt_max: float = 330.0) -> np.ndarray:
    """Draw subsampled motion-vector arrows over the BT image (RGB uint8)."""
    import cv2
    rgb = bt_to_rgb(bt, bt_min, bt_max).copy()
    h, w = bt.shape
    for y in range(step, h - step, step):
        for x in range(step, w - step, step):
            u, v = float(flow[y, x, 0]), float(flow[y, x, 1])
            p1, p2 = (x, y), (int(x + u), int(y + v))
            cv2.arrowedLine(rgb, p1, p2, (220, 30, 30), 1, tipLength=0.3)
    return rgb


def write_side_by_side(left_bt: list[np.ndarray], right_bt: list[np.ndarray], out_path: str | Path,
                       *, fps: int = 6, bt_min: float = 180.0, bt_max: float = 330.0) -> Path:
    """Original (left) vs interpolated (right) time-lapse, concatenated horizontally."""
    import imageio.v2 as imageio
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    n = min(len(left_bt), len(right_bt))
    frames = []
    for i in range(n):
        l = bt_to_rgb(left_bt[i], bt_min, bt_max)
        r = bt_to_rgb(right_bt[i], bt_min, bt_max)
        sep = np.full((l.shape[0], 4, 3), 255, np.uint8)
        frames.append(np.concatenate([l, sep, r], axis=1))
    imageio.mimsave(out_path, frames, duration=1.0 / fps)
    return out_path
