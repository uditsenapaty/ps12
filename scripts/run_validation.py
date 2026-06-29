#!/usr/bin/env python
"""Validation driver for the architecture-improvement loop (orchestration only — calls the project's
unchanged src.eval.report pipeline; does not modify it).

Key efficiency: the baselines (classical, raft, film, superslomo) are DETERMINISTIC and FIXED — only the
custom 'unet' (and an optionally fine-tuned 'rife_ft') change between iterations. So fixed-baseline
per-triplet metrics are computed ONCE and cached; each iteration recomputes only the changing model(s)
and merges with the cache. Eval protocol matches the PS / validation_report README: input frames
2*cadence apart, predict the held-out real middle frame (GOES 10-min -> 20-min gap; INSAT 30-min ->
fixed 60-min gap).

  # one-time: populate the baseline cache + report the bar
  python scripts/run_validation.py --index data/index/goes19_triplets.json --source goes19 \
    --models classical,raft,film,superslomo,unet --out validation_report/baseline_goes19 \
    --max-triplets 12 --crop-frac 0.25 --baseline-cache validation_report/_cache/goes19.json

  # each iteration: only 'unet' is recomputed (baselines reused from cache)
  python scripts/run_validation.py --index ... --models classical,raft,film,superslomo,unet \
    --out validation_report/iter1 --max-triplets 12 --crop-frac 0.25 --baseline-cache validation_report/_cache/goes19.json
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

# --- orchestration-only env shim (does NOT touch pipeline source) -------------------------------
# torch>=2.6 defaults torch.load(weights_only=True), which rejects the legacy pickled Super-SloMo
# checkpoint. Restore pre-2.6 behaviour here in the driver so trusted baseline checkpoints load with
# their real weights. Loading plumbing only — it changes no metric or model.
import torch as _torch  # noqa: E402
_ORIG_LOAD = _torch.load
def _compat_load(*a, **k):  # noqa: E306
    k.setdefault("weights_only", False)
    return _ORIG_LOAD(*a, **k)
_torch.load = _compat_load

from src.eval.report import run_eval, summarize, summarize_and_report  # noqa: E402
from src.models.factory import get_model  # noqa: E402

CHANGING = {"unet", "rife_ft"}            # recomputed every iteration; everything else is cacheable
HIGHER = {"psnr", "ssim", "fsim", "edge_ssim"}
LOWER = {"mse", "mae_kelvin", "lpips"}
METRIC_ORDER = ["psnr", "ssim", "fsim", "edge_ssim", "mse", "mae_kelvin", "lpips"]


def _runnable(models: list[str]) -> list[str]:
    """Keep only models that load AND complete one interpolation (run_eval has no per-model guard)."""
    a = np.random.RandomState(0).rand(256, 256).astype("float32")
    b = np.random.RandomState(1).rand(256, 256).astype("float32")
    ok: list[str] = []
    for name in models:
        try:
            m = get_model(name)
            if not m.available():
                print(f"[preflight] drop '{name}': available=False (weights/repo missing)"); continue
            m.interpolate(a, b, 0.5)
            ok.append(name)
        except Exception as e:
            print(f"[preflight] drop '{name}': {type(e).__name__}: {str(e)[:100]}")
    return ok


def rank_models(agg: dict[str, dict]) -> dict:
    """Borda-style aggregate rank across all present metrics (rank 1 = best on a metric)."""
    models = list(agg.keys())
    metrics = [m for m in METRIC_ORDER if any(m in agg[mm] for mm in models)]
    avg_rank = {m: 0.0 for m in models}
    wins = {m: 0 for m in models}
    per_metric_winner = {}
    for metric in metrics:
        vals = [(mm, agg[mm].get(metric)) for mm in models if metric in agg[mm]]
        order = sorted(vals, key=lambda kv: kv[1], reverse=(metric in HIGHER))
        per_metric_winner[metric] = order[0][0]
        wins[order[0][0]] += 1
        for rank, (mm, _) in enumerate(order, start=1):
            avg_rank[mm] += rank
    n = max(len(metrics), 1)
    avg_rank = {m: avg_rank[m] / n for m in models}
    ranking = sorted(avg_rank.items(), key=lambda kv: kv[1])
    return {"metrics": metrics, "ranking": ranking, "per_metric_winner": per_metric_winner,
            "wins": wins, "n_metrics": len(metrics)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--source", default="goes19")
    ap.add_argument("--models", default="classical,raft,film,superslomo,unet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-triplets", type=int, default=12)
    ap.add_argument("--crop-frac", type=float, default=0.25)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--baseline-cache", default=None, help="json cache of fixed-baseline per-triplet rows")
    a = ap.parse_args()

    idx = json.loads(Path(a.index).read_text())
    trips = [tuple(t) for t in idx["triplets"]][: a.max_triplets]  # (t0, GT_mid, t2)
    if not trips:
        raise SystemExit(f"no eval triplets in {a.index}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = {"source": a.source, "crop_frac": a.crop_frac, "tile": a.tile,
           "triplets": ["|".join(t) for t in trips]}

    requested = _runnable([m.strip() for m in a.models.split(",") if m.strip()])
    if "unet" not in requested:
        print("[warn] 'unet' (custom model) not runnable — train it first.")
    print(f"[validation] runnable models: {requested}")

    # Load cache (only valid if config matches exactly).
    cache_rows: dict[str, list] = {}
    cache_path = Path(a.baseline_cache) if a.baseline_cache else None
    if cache_path and cache_path.exists():
        try:
            c = json.loads(cache_path.read_text())
            if c.get("config") == cfg:
                cache_rows = c.get("rows", {})
        except Exception:
            cache_rows = {}

    fixed = [m for m in requested if m not in CHANGING]
    changing = [m for m in requested if m in CHANGING]
    to_compute = changing + [m for m in fixed if m not in cache_rows]
    reused = [m for m in fixed if m in cache_rows]
    if reused:
        print(f"[validation] reuse cached baselines: {reused}")
    print(f"[validation] compute now: {to_compute}")

    rows: list[dict] = []
    for m in reused:
        rows.extend(cache_rows[m])
    if to_compute:
        with tempfile.TemporaryDirectory() as td:
            new_rows = run_eval(trips, a.source, to_compute, td, crop_frac=a.crop_frac,
                                tile=a.tile, max_triplets=None)
        rows.extend(new_rows)
        # update cache with freshly computed FIXED baselines
        if cache_path:
            for m in fixed:
                if m in to_compute:
                    cache_rows[m] = [r for r in new_rows if r["model"] == m]
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"config": cfg, "rows": cache_rows}, indent=2))

    # Final report + plots from merged rows (via the unchanged pipeline summariser).
    (out / "metrics_raw.json").write_text(json.dumps(rows, indent=2))
    summarize_and_report(rows, out)
    agg = summarize(rows)
    rk = rank_models(agg)
    (out / "ranking.json").write_text(json.dumps(
        {"source": a.source, "n_triplets": len({r["triplet"] for r in rows}),
         "aggregate": agg, "ranking": rk}, indent=2))

    print("\n================ AGGREGATE RANKING ================")
    print(f"source={a.source}  triplets={len({r['triplet'] for r in rows})}  metrics={rk['metrics']}")
    for mm, ar in rk["ranking"]:
        flag = "  <-- WINNER" if mm == rk["ranking"][0][0] else ""
        print(f"  {mm:12s} avg_rank={ar:.2f}  metric_wins={rk['wins'][mm]}/{rk['n_metrics']}{flag}")
    print("per-metric winner:", rk["per_metric_winner"])
    uw = rk["wins"].get("unet", 0)
    print(f"\nBEST OVERALL: {rk['ranking'][0][0]}   unet: avg_rank="
          f"{dict(rk['ranking']).get('unet', float('nan')):.2f}, metric_wins={uw}/{rk['n_metrics']}")


if __name__ == "__main__":
    main()
