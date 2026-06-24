"""Locate cloned model repos and their weights.

Deep backbones (RIFE / FILM / Super-SloMo) are git-cloned under `referred_clones/<name>/` (their nested
`.git` is stripped by data_setup.py so the code is committable to THIS repo); checkpoints go in
`weights/<name>/`. RAFT needs no clone — it ships in torchvision. These helpers report whether a model
is runnable here so wrappers gate honestly instead of faking output.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_DIR = PROJECT_ROOT / "referred_clones"
WEIGHTS_DIR = PROJECT_ROOT / "weights"


def vendor_path(name: str) -> Path:
    return VENDOR_DIR / name


def weights_path(name: str) -> Path:
    return WEIGHTS_DIR / name


def has_weights(name: str) -> bool:
    p = weights_path(name)
    return p.exists() and any(p.rglob("*"))


def has_vendor(name: str) -> bool:
    p = vendor_path(name)
    return p.exists() and any(p.rglob("*.py"))


def add_to_path(name: str) -> Path:
    p = vendor_path(name)
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False
