"""Model registry/factory — select any interpolator by name and report what's runnable here."""
from __future__ import annotations

from .base import Interpolator
from .classical import ClassicalFlowInterpolator, LinearBlendInterpolator

_REGISTRY = {
    "classical": ClassicalFlowInterpolator,
    "linear": LinearBlendInterpolator,
}


def _lazy(name: str):
    if name == "raft":
        from .raft import RaftFlowInterpolator
        return RaftFlowInterpolator
    if name == "rife":
        from .rife import RifeInterpolator
        return RifeInterpolator
    if name == "film":
        from .film import FilmInterpolator
        return FilmInterpolator
    if name == "superslomo":
        from .superslomo import SuperSloMoInterpolator
        return SuperSloMoInterpolator
    if name == "unet":
        from .unet_vfi import UNetVFIInterpolator
        return UNetVFIInterpolator
    raise KeyError(name)


ALL_MODELS = ["classical", "linear", "raft", "unet", "rife", "film", "superslomo"]


def get_model(name: str, **kwargs) -> Interpolator:
    name = name.lower()
    if name in _REGISTRY:
        return _REGISTRY[name](**kwargs)
    if name in ("rife_ft", "rife-ft"):  # fine-tuned RIFE variant -> weights/rife_ft
        from .vendor import PROJECT_ROOT
        kwargs.setdefault("weights_dir", str(PROJECT_ROOT / "weights" / "rife_ft"))
        return _lazy("rife")(**kwargs)
    return _lazy(name)(**kwargs)


def available_models(**kwargs) -> dict[str, bool]:
    """Map model name -> whether it can run on this machine right now."""
    out: dict[str, bool] = {}
    for name in ALL_MODELS:
        try:
            out[name] = get_model(name, **kwargs).available()
        except Exception:
            out[name] = False
    return out


def discover_models() -> list[dict]:
    """Enumerate selectable interpolators for the dashboard: always-on baselines + any pretrained
    deep models with weights + every trained/fine-tuned checkpoint folder under weights/.

    Returns dicts: {label, name, kwargs, available}. Distinguishes the custom UNetVFI variants
    (e.g. GOES/Himawari-trained vs INSAT-adapted) so the web user can pick finetuned vs custom.
    """
    from pathlib import Path

    from .vendor import PROJECT_ROOT

    items: list[dict] = []

    def add(label, name, kwargs=None):
        try:
            ok = get_model(name, **(kwargs or {})).available()
        except Exception:
            ok = False
        items.append({"label": label, "name": name, "kwargs": kwargs or {}, "available": ok})

    add("Classical TV-L1 (traditional baseline)", "classical")
    add("Linear blend (floor)", "linear")
    add("RAFT optical-flow (pretrained)", "raft")
    add("RIFE (pretrained)", "rife")
    add("FILM (pretrained, large motion)", "film")
    add("Super-SloMo (pretrained)", "superslomo")

    from .vendor import PROJECT_ROOT as _PR
    if (_PR / "weights" / "rife_ft").exists():
        add("RIFE (fine-tuned on satellite)", "rife_ft")

    # custom UNetVFI: one entry per trained checkpoint folder under weights/
    wroot = PROJECT_ROOT / "weights"
    if wroot.exists():
        for sub in sorted(wroot.glob("unet*")):
            if sub.is_dir() and (any(sub.rglob("best.pt")) or any(sub.rglob("*.pt"))):
                tag = sub.name.replace("unet", "UNetVFI").replace("_", " ")
                add(f"{tag} (ours, custom)", "unet", {"weights_dir": str(sub)})
    if not any(it["name"] == "unet" for it in items):
        add("UNetVFI (ours — train to enable)", "unet")
    return items
