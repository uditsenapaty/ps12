"""Inference: tiled `.nc -> .nc` frame interpolation + temporal upscaling."""
from .interpolate import interpolate_pair_bt, interpolate_nc  # noqa: F401
from .upscale import temporal_upscale_bt, temporal_upscale_nc  # noqa: F401
