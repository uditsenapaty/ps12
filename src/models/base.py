"""Common interface for all interpolators.

Every model consumes two normalized single-channel frames in [0, 1] (shape (H, W)) and returns the
predicted frame at fractional time `t` in (0, 1). This uniform contract lets `infer/` and the
dashboard swap models freely, and lets the deterministic battery exercise any model the same way.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Interpolator(ABC):
    name: str = "base"
    #: True if the model can run right now on this machine (weights present / deps importable).
    requires_gpu: bool = False
    requires_weights: bool = False

    @abstractmethod
    def interpolate(self, frame0: np.ndarray, frame1: np.ndarray, t: float = 0.5) -> np.ndarray:
        """Return the predicted intermediate frame (H, W) in [0, 1]."""

    def available(self) -> bool:
        """Whether `interpolate` can run now. Overridden by deep models that need weights/GPU."""
        return True

    def ensure_available(self) -> None:
        if not self.available():
            raise RuntimeError(
                f"Model '{self.name}' is not runnable here "
                f"(requires_weights={self.requires_weights}, requires_gpu={self.requires_gpu}). "
                f"Run on the GPU server after fetching weights — see walkthrough.md."
            )

    @staticmethod
    def _check_pair(frame0: np.ndarray, frame1: np.ndarray) -> None:
        if frame0.shape != frame1.shape:
            raise ValueError(f"frame shapes differ: {frame0.shape} vs {frame1.shape}")
        if frame0.ndim != 2:
            raise ValueError(f"frames must be 2-D (H, W), got {frame0.shape}")
