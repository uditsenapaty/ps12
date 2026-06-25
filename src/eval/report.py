"""Evaluation harness + comparison report.

For each evaluation triplet (t0, GT, t2) and each model, predict the middle frame from (t0, t2) at
t=0.5 and score it against the real GT with the full metric suite. Produces a Markdown report with a
per-model table and PSNR/SSIM bar plots — the PS "report comparing results with ground truth".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data.normalize import BT_MAX_DEFAULT, BT_MIN_DEFAULT, bt_to_norm, fill_invalid
from ..data.readers import read_frame
from ..infer.interpolate import interpolate_pair_bt
from ..models.factory import get_model
from . import metrics


def _read_bt(path, source: str, crop_frac: float | None, with_lonlat: bool = False):
    """Read a frame, cropping the central region for GOES (keeps eval fast + within memory)."""
    kw: dict = {"with_lonlat": with_lonlat}
    if crop_frac and source.lower().startswith("goes"):
        kw["crop_frac"] = crop_frac
    return read_frame(path, source, **kw)


def _norm_gt(path: str, source: str, bt_min: float, bt_max: float, crop_frac: float | None = None) -> np.ndarray:
    fr = _read_bt(path, source, crop_frac)
    n, _ = fill_invalid(bt_to_norm(fr.bt, bt_min, bt_max))
    return n


def run_eval(
    triplets: list[tuple[str, str, str]],
    source: str,
    model_names: list[str],
    out_dir: str | Path,
    *,
    t: float = 0.5,
    bt_min: float = BT_MIN_DEFAULT,
    bt_max: float = BT_MAX_DEFAULT,
    tile: int = 256,
    overlap: int = 32,
    crop_frac: float | None = 0.3,
    max_triplets: int | None = None,
) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    trips = triplets[:max_triplets] if max_triplets else triplets
    for name in model_names:
        model = get_model(name)
        if not model.available():
            print(f"[eval] skip '{name}' — not runnable here (needs weights/GPU). Run on server.")
            continue
        for k, (p0, pgt, p2) in enumerate(trips):
            fr0 = _read_bt(p0, source, crop_frac)
            fr2 = _read_bt(p2, source, crop_frac)
            pred_bt = interpolate_pair_bt(fr0.bt, fr2.bt, model, t, bt_min=bt_min, bt_max=bt_max,
                                          tile=tile, overlap=overlap)
            pred_n = bt_to_norm(pred_bt, bt_min, bt_max)
            gt_n = _norm_gt(pgt, source, bt_min, bt_max, crop_frac)
            m = metrics.compute_all(np.nan_to_num(pred_n), gt_n, bt_min=bt_min, bt_max=bt_max)
            m.update({"model": name, "triplet": k})
            rows.append(m)
            print(f"[eval] {name} triplet {k}: psnr {m['psnr']:.2f} ssim {m['ssim']:.4f}")
    (out_dir / "metrics_raw.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summarize_and_report(rows, out_dir)
    if trips:
        try:
            save_comparison_png(trips[0], source, model_names, out_dir, bt_min=bt_min, bt_max=bt_max,
                                t=t, crop_frac=crop_frac, tile=tile, overlap=overlap)
        except Exception as e:
            print(f"[eval] qualitative panel skipped: {e}")
        try:
            save_timelapse_gif(trips[0], source, model_names, out_dir, bt_min=bt_min, bt_max=bt_max,
                               crop_frac=crop_frac, tile=tile, overlap=overlap)
        except Exception as e:
            print(f"[eval] timelapse gif skipped: {e}")
    return rows


def save_comparison_png(triplet, source, model_names, out_dir, *, bt_min, bt_max, t=0.5,
                        crop_frac=None, tile=256, overlap=32):
    """Qualitative panel: inputs, GT, each available model's prediction + |pred-GT| error maps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..viz.animate import bt_to_rgb
    out_dir = Path(out_dir)
    p0, pgt, p2 = triplet
    fr0 = _read_bt(p0, source, crop_frac)
    fr2 = _read_bt(p2, source, crop_frac)
    gt = _read_bt(pgt, source, crop_frac).bt
    preds: dict[str, np.ndarray] = {}
    for name in model_names:
        m = get_model(name)
        if m.available():
            preds[name] = interpolate_pair_bt(fr0.bt, fr2.bt, m, t, bt_min=bt_min, bt_max=bt_max,
                                              tile=tile, overlap=overlap)
    panels = [("input t0", fr0.bt, None), ("ground truth", gt, None)]
    for name, pr in preds.items():
        panels.append((name, pr, np.abs(np.nan_to_num(pr) - np.nan_to_num(gt))))
    cols = len(panels)
    fig, axes = plt.subplots(2, cols, figsize=(3 * cols, 6.2), squeeze=False)
    for j, (label, img, err) in enumerate(panels):
        axes[0][j].imshow(bt_to_rgb(img, bt_min, bt_max)); axes[0][j].set_title(label, fontsize=9)
        axes[0][j].axis("off")
        if err is None:
            axes[1][j].axis("off")
        else:
            im = axes[1][j].imshow(err, cmap="magma", vmin=0, vmax=max(1e-3, np.nanpercentile(err, 99)))
            axes[1][j].set_title("|pred-GT| (K)", fontsize=8); axes[1][j].axis("off")
            fig.colorbar(im, ax=axes[1][j], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_triplet0.png", dpi=120)
    plt.close(fig)
    print(f"[eval] qualitative panel -> {out_dir/'comparison_triplet0.png'}")


def save_timelapse_gif(triplet, source, model_names, out_dir, *, bt_min, bt_max,
                       crop_frac=None, tile=256, overlap=32):
    """Original (t0,GT,t2) vs interpolated (t0,pred,t2) side-by-side GIF for the best available model."""
    from ..viz.animate import write_side_by_side
    out_dir = Path(out_dir)
    p0, pgt, p2 = triplet
    fr0 = _read_bt(p0, source, crop_frac)
    fr2 = _read_bt(p2, source, crop_frac)
    gt = _read_bt(pgt, source, crop_frac).bt
    chosen = next((n for n in model_names if get_model(n).available()), None)
    if chosen is None:
        return
    pred = interpolate_pair_bt(fr0.bt, fr2.bt, get_model(chosen), 0.5, bt_min=bt_min, bt_max=bt_max,
                               tile=tile, overlap=overlap)
    write_side_by_side([fr0.bt, gt, fr2.bt], [fr0.bt, pred, fr2.bt],
                       out_dir / f"timelapse_{chosen}.gif", fps=2, bt_min=bt_min, bt_max=bt_max)
    print(f"[eval] timelapse -> {out_dir}/timelapse_{chosen}.gif")


def summarize(rows: list[dict]) -> dict[str, dict]:
    keys = ["mse", "psnr", "ssim", "edge_ssim", "mae_kelvin", "fsim", "lpips"]
    out: dict[str, dict] = {}
    for name in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == name]
        out[name] = {k: float(np.nanmean([r[k] for r in sub if k in r])) for k in keys if any(k in r for r in sub)}
    return out


def summarize_and_report(rows: list[dict], out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    agg = summarize(rows)
    cols = ["psnr", "ssim", "fsim", "edge_ssim", "mse", "mae_kelvin", "lpips"]
    present = [c for c in cols if any(c in v for v in agg.values())]
    lines = ["# Validation report — interpolated vs ground truth", "",
             f"Validated on {len({r['triplet'] for r in rows})} triplets: input frames 20 min apart, "
             "predict the held-out real 10-min middle, score against ground truth "
             "(INSAT: leave-one-out at 30 min).", "",
             "| model | " + " | ".join(present) + " |",
             "|" + "---|" * (len(present) + 1)]
    for name, v in agg.items():
        cells = [f"{v.get(c, float('nan')):.4f}" for c in present]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "Higher PSNR/SSIM/FSIM/edge_SSIM = better; lower MSE/MAE(K)/LPIPS = better.",
              "MAE is in Kelvin (physical). edge_SSIM targets cloud-edge structure (motion fidelity)."]
    rep = out_dir / "report.md"
    rep.write_text("\n".join(lines), encoding="utf-8")
    try:
        _plot(agg, present, out_dir)
    except Exception as e:  # plotting is non-essential; numbers stand on their own
        print(f"[eval] plotting skipped: {e}")
    print(f"[eval] report -> {rep}")
    return rep


def _plot(agg: dict, present: list[str], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(agg.keys())
    for metric in [m for m in ("psnr", "ssim") if m in present]:
        vals = [agg[m_].get(metric, np.nan) for m_ in models]
        plt.figure(figsize=(6, 3.5))
        plt.bar(models, vals, color="#3b7dd8")
        plt.ylabel(metric.upper()); plt.title(f"{metric.upper()} by model"); plt.xticks(rotation=20)
        plt.tight_layout(); plt.savefig(out_dir / f"{metric}_by_model.png", dpi=120); plt.close()
