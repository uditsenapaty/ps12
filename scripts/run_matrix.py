#!/usr/bin/env python
"""scripts/run_matrix.py — 3 training setups x validate on {GOES, Himawari, INSAT} = 9 tables.

Runs ON the Lightning Studio (cwd ~/ps12, CUDA). For each setup it trains/fine-tunes OUR custom model
(`--steps`, default 5000), then evaluates EVERY runnable model (classical + pretrained zero-shot +
finetuned-if-any + our model) on a held-out chronological split of each satellite, writing
validation_report/matrix/<setup>_<source>/report.md (+ a SUMMARY.md).

Setups
  1. GOES held-out      train custom on GOES (20-min triplets)                  -> weights/unet_goes
  2. HIMAWARI held-out  train custom on Himawari (20-min triplets)              -> weights/unet_hima
  3. COMBINATION        train custom on GOES+Himawari multigap (gap 20/40,      -> weights/unet_mix
                        level 2, combined loss), THEN self-supervise on INSAT
                        (60-min leave-one-out), warm-started from unet_mix      -> weights/unet_mix_insat

Held-out = leakage-free CHRONOLOGICAL split per source (train = earlier frames, eval = later frames,
with a buffer gap dropped so no eval triplet shares a frame with training). The eval set for each source
is FIXED across all three setups, so classical + pretrained baselines (which never train) are computed
ONCE per source and reused. RIFE / rife_ft are included automatically iff runnable (weights present),
else listed as N/A in the footnote (no faking).

  python scripts/run_matrix.py --steps 5000 --max-eval 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

# torch>=2.6 defaults torch.load(weights_only=True), which rejects the legacy pickled Super-SloMo
# checkpoint. Restore pre-2.6 behaviour so the trusted baseline weights load. Plumbing only.
import torch as _torch  # noqa: E402
_ORIG_LOAD = _torch.load
def _compat_load(*a, **k):  # noqa: E306
    k.setdefault("weights_only", False)
    return _ORIG_LOAD(*a, **k)
_torch.load = _compat_load

from src.data.normalize import BT_MAX_DEFAULT, BT_MIN_DEFAULT, bt_to_norm, fill_invalid  # noqa: E402
from src.data.triplets import (build_leave_one_out, build_multigap_groups,  # noqa: E402
                               build_triplets, index_frames)
from src.eval import metrics  # noqa: E402
from src.eval.report import _read_bt, summarize  # noqa: E402
from src.infer.interpolate import interpolate_pair_bt  # noqa: E402
from src.models.factory import get_model  # noqa: E402
from src.train.finetune import train  # noqa: E402
from src.train.rife_finetune import finetune as rife_finetune  # noqa: E402

DATA = ROOT / "data"
IDXD = DATA / "index_matrix"
WD = ROOT / "weights"
REP = ROOT / "validation_report" / "matrix"

SRC_DIR = {"goes19": DATA / "goes19", "himawari9": DATA / "himawari9_nc", "insat3dr": DATA / "insat"}
CADENCE = {"goes19": 10.0, "himawari9": 10.0, "insat3dr": 30.0}     # min between frames; gap = 2*cadence
GAP_TXT = {"goes19": "20-min", "himawari9": "20-min", "insat3dr": "60-min"}
TRAIN_FRAC = {"goes19": 0.8, "himawari9": 0.8, "insat3dr": 0.55}    # INSAT is thin -> keep more for eval
COLS = ["psnr", "ssim", "fsim", "edge_ssim", "mse", "mae_kelvin", "lpips"]
HIGHER = {"psnr", "ssim", "fsim", "edge_ssim"}
CROP = 0.25                                                          # central crop for GOES/INSAT eval


# ----------------------------------------------------------------------------- data splits / indices
def indexed(source: str):
    d = SRC_DIR[source]
    files = [p for p in d.rglob("*") if p.suffix.lower() in (".nc", ".h5", ".hdf", ".hdf5")]
    return index_frames(files, source)                              # [(ts, path)] sorted + deduped


def split(idx, frac: float, buf: int = 2):
    """Chronological split: train = first `frac`, eval = remainder after a `buf`-frame gap (no leakage)."""
    n = len(idx)
    k = max(1, int(round(n * frac)))
    return idx[:k], idx[k + buf:]


def write_triplet_index(path: Path, source: str, frames) -> int:
    cad = CADENCE[source]
    trips = build_triplets(frames, cad)                             # (t0,t1,t2) @ cadence -> span 2*cad
    path.write_text(json.dumps({"source": source, "step_min": cad,
                                "triplets": [[str(a), str(b), str(c)] for a, b, c in trips],
                                "anytime": []}), encoding="utf-8")
    return len(trips)


def write_multigap_index(path: Path, source: str, frames, max_level: int = 2) -> int:
    cad = CADENCE[source]
    mg = build_multigap_groups(frames, cadence_minutes=cad, max_level=max_level)
    path.write_text(json.dumps({"source": source, "step_min": cad, "multigap_levels": max_level,
                                "multigap": [[str(t), [[str(l), str(r)] for l, r in v]] for t, v in mg],
                                "triplets": [], "anytime": []}), encoding="utf-8")
    return len(mg)


def eval_triplets(source: str, frames, max_eval: int):
    loo = build_leave_one_out(frames, CADENCE[source])
    return [(str(a), str(b), str(c)) for a, b, c in loo][:max_eval]


# ----------------------------------------------------------------------------- evaluation
def eval_rows(label: str, model, source: str, trips):
    rows = []
    crop = CROP if source.startswith(("goes", "insat")) else None
    for k, (p0, pgt, p2) in enumerate(trips):
        fr0 = _read_bt(p0, source, crop)
        fr2 = _read_bt(p2, source, crop)
        gt = _read_bt(pgt, source, crop)
        pred = interpolate_pair_bt(fr0.bt, fr2.bt, model, 0.5, bt_min=BT_MIN_DEFAULT, bt_max=BT_MAX_DEFAULT,
                                   tile=256, overlap=32)
        gtn, _ = fill_invalid(bt_to_norm(gt.bt, BT_MIN_DEFAULT, BT_MAX_DEFAULT))
        m = metrics.compute_all(np.nan_to_num(bt_to_norm(pred, BT_MIN_DEFAULT, BT_MAX_DEFAULT)), gtn,
                                bt_min=BT_MIN_DEFAULT, bt_max=BT_MAX_DEFAULT)
        m.update({"model": label, "triplet": k})
        rows.append(m)
    print(f"    [eval] {label:32s} {source:10s} n={len(rows)}"
          + (f"  psnr={np.mean([r['psnr'] for r in rows]):.2f}" if rows else "  (no triplets)"), flush=True)
    return rows


def rank(agg):
    mets = [m for m in COLS if any(m in v for v in agg.values())]
    models = list(agg)
    ar = {m: 0.0 for m in models}
    for met in mets:
        order = sorted([(mm, agg[mm][met]) for mm in models if met in agg[mm]],
                       key=lambda kv: kv[1], reverse=met in HIGHER)
        for r, (mm, _) in enumerate(order, 1):
            ar[mm] += r
    ar = {m: ar[m] / max(len(mets), 1) for m in models}
    return sorted(ar.items(), key=lambda kv: kv[1])


def write_report(out_dir: Path, title: str, protocol: str, rows, footnotes):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_raw.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    agg = summarize(rows)
    present = [c for c in COLS if any(c in v for v in agg.values())]
    rk = rank(agg)
    lines = [f"# {title}", "", protocol, "",
             "| model | " + " | ".join(present) + " |",
             "|" + "---|" * (len(present) + 1)]
    for name, _ in rk:                                              # best-ranked first
        v = agg[name]
        lines.append(f"| {name} | " + " | ".join(f"{v.get(c, float('nan')):.4f}" for c in present) + " |")
    lines += ["", "**Rank (Borda avg across metrics, 1 = best):** "
              + ", ".join(f"{m} ({r:.2f})" for m, r in rk),
              "", "Higher PSNR/SSIM/FSIM/edge_SSIM = better; lower MSE/MAE(K)/LPIPS = better. MAE in Kelvin."]
    if footnotes:
        lines += [""] + [f"- {f}" for f in footnotes]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  [report] -> {out_dir/'report.md'}", flush=True)
    return agg, rk


# ----------------------------------------------------------------------------- model helpers
def maybe(name, **kw):
    """Return a runnable interpolator or None (honest gating — never fake)."""
    try:
        m = get_model(name, **kw)
        return m if m.available() else None
    except Exception as e:
        print(f"    [skip] {name}: {type(e).__name__}: {str(e)[:80]}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-eval", type=int, default=12, help="eval triplets per source")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--multigap-level", type=int, default=2, help="setup-3 GOES+Himawari gap levels (20/40)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true", help="retrain even if a checkpoint exists (else reuse)")
    a = ap.parse_args()
    t_start = time.time()
    IDXD.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)

    # ---- 1. splits + indices -----------------------------------------------------------------------
    idx = {s: indexed(s) for s in SRC_DIR}
    trn, evl = {}, {}
    for s in SRC_DIR:
        trn[s], evl[s] = split(idx[s], TRAIN_FRAC[s])
        print(f"[split] {s}: {len(idx[s])} frames -> train {len(trn[s])}, eval {len(evl[s])}", flush=True)
    eval_sets = {s: eval_triplets(s, evl[s], a.max_eval) for s in SRC_DIR}
    for s in SRC_DIR:
        print(f"[eval-set] {s}: {len(eval_sets[s])} held-out triplets ({GAP_TXT[s]} gap)", flush=True)

    i_goes_tri = IDXD / "goes_train_triplets.json"
    i_hima_tri = IDXD / "hima_train_triplets.json"
    i_goes_mg = IDXD / "goes_train_multigap.json"
    i_hima_mg = IDXD / "hima_train_multigap.json"
    i_insat_tri = IDXD / "insat_train_triplets.json"
    n_gt = write_triplet_index(i_goes_tri, "goes19", trn["goes19"])
    n_ht = write_triplet_index(i_hima_tri, "himawari9", trn["himawari9"])
    n_gmg = write_multigap_index(i_goes_mg, "goes19", trn["goes19"], a.multigap_level)
    n_hmg = write_multigap_index(i_hima_mg, "himawari9", trn["himawari9"], a.multigap_level)
    n_it = write_triplet_index(i_insat_tri, "insat3dr", trn["insat3dr"])
    print(f"[index] train triplets: goes={n_gt} hima={n_ht} insat={n_it}; "
          f"multigap groups: goes={n_gmg} hima={n_hmg}", flush=True)

    # ---- 2. train the three custom models (skip if already trained, unless --force) ----------------
    def need(outdir, ckpt="best.pt"):
        ex = (Path(outdir) / ckpt).exists()
        if ex and not a.force:
            print(f"[skip-train] {Path(outdir).name} ({ckpt} exists — reusing)", flush=True)
        return a.force or not ex

    if need(WD / "unet_goes"):
        print("\n===== SETUP 1: train custom on GOES (20-min) =====", flush=True)
        train(str(i_goes_tri), out=str(WD / "unet_goes"), steps=a.steps, batch=a.batch, device=a.device, ema_decay=a.ema)
    if need(WD / "unet_hima"):
        print("\n===== SETUP 2: train custom on Himawari (20-min) =====", flush=True)
        train(str(i_hima_tri), out=str(WD / "unet_hima"), steps=a.steps, batch=a.batch, device=a.device, ema_decay=a.ema)
    if need(WD / "unet_mix"):
        print("\n===== SETUP 3a: train custom on GOES+Himawari multigap (gap 20/40) =====", flush=True)
        train(f"{i_goes_mg},{i_hima_mg}", out=str(WD / "unet_mix"), steps=a.steps, batch=a.batch,
              device=a.device, multigap=True, ema_decay=a.ema)
    if need(WD / "unet_mix_insat"):
        print("\n===== SETUP 3b: self-supervise custom on INSAT (60-min), warm-start from unet_mix =====", flush=True)
        train(str(i_insat_tri), out=str(WD / "unet_mix_insat"), steps=a.steps, batch=max(2, a.batch // 2),
              device=a.device, init_weights=str(WD / "unet_mix" / "best.pt"), ema_decay=a.ema)

    # ---- 2b. finetuned-baseline RIFE: fine-tuned the SAME 3 ways (only if pretrained RIFE is runnable)
    rife_ok = maybe("rife") is not None
    rife_pre = WD / "rife"
    RIFE_FT = {"setup1": WD / "rife_goes", "setup2": WD / "rife_hima", "setup3": WD / "rife_mix_insat"}
    if rife_ok:
        def _ft(idx, w, out, bs, tag):
            """Fine-tune RIFE; NON-FATAL (a failure leaves that setup's rife_ft simply absent)."""
            if not need(Path(out), "flownet.pkl"):
                return True
            print(f"\n===== RIFE-ft {tag} =====", flush=True)
            try:
                rife_finetune(idx, weights=str(w), out=str(out), steps=a.steps, batch=bs, device=a.device)
                return (Path(out) / "flownet.pkl").exists()
            except Exception as e:
                print(f"[rife-ft] FAILED {tag}: {type(e).__name__}: {str(e)[:160]}", flush=True)
                return False

        _ft(str(i_goes_tri), rife_pre, RIFE_FT["setup1"], a.batch, "1: GOES")
        _ft(str(i_hima_tri), rife_pre, RIFE_FT["setup2"], a.batch, "2: Himawari")
        mix_ok = _ft(f"{i_goes_tri},{i_hima_tri}", rife_pre, WD / "rife_mix", a.batch, "3a: GOES+Himawari")
        if mix_ok:
            _ft(str(i_insat_tri), WD / "rife_mix", RIFE_FT["setup3"], max(2, a.batch // 2), "3b: INSAT self-sup")
        else:
            print("[rife-ft] skip 3b (INSAT self-sup) — 3a (rife_mix) unavailable", flush=True)
    else:
        print("[rife-ft] pretrained RIFE not runnable -> skipping finetuned-RIFE baseline", flush=True)

    # ---- 3. zero-shot baselines (setup-independent) -> compute once per source ----------------------
    print("\n===== baselines (zero-shot, computed once per source) =====", flush=True)
    BASE = [("classical", "classical (zero-shot)"), ("raft", "raft (zero-shot)"),
            ("film", "film (zero-shot)"), ("superslomo", "superslomo (zero-shot)"),
            ("rife", "rife (zero-shot)")]
    base_rows = {s: [] for s in SRC_DIR}
    base_have, base_missing = set(), set()
    for s in SRC_DIR:
        for nm, label in BASE:
            mdl = maybe(nm)
            if mdl is None:
                base_missing.add(nm)
                continue
            base_have.add(nm)
            base_rows[s] += eval_rows(label, mdl, s, eval_sets[s])

    # finetuned baseline = RIFE fine-tuned the same 3 ways (the per-setup model is picked in the loop).
    ft_note = ("Finetuned baseline = RIFE fine-tuned on satellite IR the SAME 3 ways as our model "
               "(GOES / Himawari / GOES+Himawari → INSAT self-sup). FILM/Super-SloMo have no fine-tune "
               "loop in this repo, so a 'finetuned FILM/SloMo' would be fabricated and is omitted."
               if rife_ok else
               "Finetuned baseline N/A: pretrained RIFE not runnable on this box; FILM/Super-SloMo have "
               "no fine-tune loop here (a finetuned row would be fabricated).")

    base_footnote = ("Zero-shot = classical (algorithmic) + pretrained nets used frozen (no training on "
                     "satellite IR). Runnable: " + (", ".join(sorted(base_have)) or "none")
                     + (". Not runnable: " + ", ".join(sorted(base_missing)) if base_missing else "."))

    # ---- 4. assemble the 9 reports -----------------------------------------------------------------
    SETUPS = [
        ("setup1", "GOES held-out — custom trained on GOES (20-min triplets)", WD / "unet_goes",
         "unet (ours · GOES-trained)"),
        ("setup2", "HIMAWARI held-out — custom trained on Himawari (20-min triplets)", WD / "unet_hima",
         "unet (ours · Himawari-trained)"),
        ("setup3", "COMBINATION — GOES+Himawari multigap (20/40) then INSAT self-sup (60-min)",
         WD / "unet_mix_insat", "unet (ours · GOES+Hima → INSAT self-sup)"),
    ]
    summary = ["# Validation matrix — 3 setups x {GOES, Himawari, INSAT}", "",
               "Each cell trains OUR custom model one way, then scores every runnable model on a held-out "
               "chronological split of each satellite. Classical + pretrained rows are zero-shot (identical "
               "across setups). " + ft_note, ""]
    for skey, sdesc, wdir, ulabel in SETUPS:
        custom = maybe("unet", weights_dir=str(wdir))
        rife_ft_model = maybe("rife", weights_dir=str(RIFE_FT[skey])) if rife_ok else None
        summary.append(f"## {skey}: {sdesc}")
        for s in SRC_DIR:
            rows = list(base_rows[s])
            if rife_ft_model is not None:
                rows += eval_rows(f"rife_ft (finetuned · {skey})", rife_ft_model, s, eval_sets[s])
            if custom is not None:
                rows += eval_rows(ulabel, custom, s, eval_sets[s])
            else:
                print(f"  [warn] custom model for {skey} not runnable ({wdir})", flush=True)
            title = f"{skey.upper()} → validate on {s.upper()} ({GAP_TXT[s]} gap, leave-one-out)"
            protocol = (f"Held-out chronological split: trained on the earlier frames, scored on "
                        f"{len(eval_sets[s])} later {s.upper()} triplets it never saw "
                        f"(input frames {GAP_TXT[s]} apart, predict the real middle).")
            out_dir = REP / f"{skey}_{s}"
            agg, rk = write_report(out_dir, title, protocol, rows, [base_footnote, ft_note])
            best = rk[0][0]
            ours = next((r for m, r in rk if m == ulabel), None)
            summary.append(f"- **{s.upper()}** ({GAP_TXT[s]}): best = `{best}`; "
                           f"ours rank = {ours:.2f}" if ours is not None else f"- **{s.upper()}**: ours N/A")
        summary.append("")

    (REP / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"\n[done] wrote 9 reports + SUMMARY under {REP}  ({(time.time()-t_start)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
