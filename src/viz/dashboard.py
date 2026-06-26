"""PS12 Streamlit dashboard — professional UI over the real interpolation / upscaling / eval code.

Run:  streamlit run src/viz/dashboard.py

Tabs
  ▶ Interpolate         two consecutive frames -> intermediate frame(s); model picker across
                        pretrained / fine-tuned (rife_ft) / custom (unet); motion overlay; metrics vs GT.
  🔼 Temporal Upscaling  INSAT-3DS/3DR 30 -> 15 -> 7.5 min (recursive midpoint insertion) as a time-lapse.
  📊 Validation Report   committed validation_report/ (tables, plots, qualitative panels, GIFs).
Nothing is mocked — every panel calls the real infer/eval code.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

# credentials: Streamlit Cloud secrets (st.secrets) or local .env.local -> environment
try:
    for _k, _v in dict(st.secrets).items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass
from src.data.env import load_env  # noqa: E402

load_env()

from src.data.normalize import BT_MAX_DEFAULT, BT_MIN_DEFAULT, bt_to_norm  # noqa: E402
from src.data.readers import read_frame  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.infer.interpolate import interpolate_pair_bt  # noqa: E402
from src.infer.upscale import (  # noqa: E402
    continuous_total_steps, temporal_upscale_bt, upscale_continuous_bt, upscale_total_steps)
from src.models.factory import discover_models, get_model  # noqa: E402
from src.viz.animate import bt_to_rgb, flow_overlay_rgb, frames_to_gif  # noqa: E402
from src.viz.compute import get_backend  # noqa: E402

VALIDATION_DIR = ROOT / "validation_report"
SAMPLES_DIR = ROOT / "samples"
SOURCES = ["goes19", "himawari9", "insat3dr", "insat3ds"]

st.set_page_config(page_title="PS12 · Satellite Frame Interpolation", page_icon="🛰️", layout="wide")

st.markdown(
    """
    <style>
      #MainMenu, footer {visibility: hidden;}
      .hero {background: linear-gradient(90deg,#0b2c4a 0%,#10243b 60%,#0E1117 100%);
             padding:18px 24px;border-radius:12px;border:1px solid #1f2c3d;margin-bottom:8px;}
      .hero h1 {margin:0;font-size:1.55rem;color:#E6EDF3;}
      .hero p {margin:.25rem 0 0;color:#8aa0b6;font-size:.92rem;}
      .badge {display:inline-block;padding:2px 9px;border-radius:11px;font-size:.74rem;margin:2px 4px 2px 0;}
      .ok {background:#13351f;color:#5fd38a;border:1px solid #1d5b34;}
      .no {background:#3a1c1c;color:#e08585;border:1px solid #5b2424;}
      .card {background:#161B22;border:1px solid #222b38;border-radius:10px;padding:14px 16px;}
      .cap {color:#8aa0b6;font-size:.8rem;text-align:center;}
      .stImage img {border-radius:8px;border:1px solid #222b38;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero"><h1>🛰️ Fill in the Frames — Satellite Temporal Super-Resolution</h1>'
    '<p>AI optical-flow frame interpolation for INSAT-3DS/3DR · GOES-19 · Himawari — thermal IR (~10 µm). '
    'Enhance temporal resolution 30 → 15 → 7.5 min and validate against real ground truth.</p></div>',
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _models():
    return discover_models()


def _load_bt(file_or_path, source: str) -> np.ndarray:
    return read_frame(file_or_path, source, with_lonlat=False).bt


def _model_picker(models, key: str):
    labels = [m["label"] + ("" if m["available"] else "  ·  needs weights/GPU") for m in models]
    i = st.selectbox("Model", range(len(models)), format_func=lambda j: labels[j], key=key)
    return models[i]


def _render_upscale_result(bts, dense, base_step, out_cad, caption):
    """Shared output panel for both upscaling tabs: metrics + time-lapse gif (or a frame grid)."""
    m1, m2, m3 = st.columns(3)
    m1.metric("Input frames", len(bts))
    m2.metric("Output frames", len(dense))
    m3.metric("Cadence", f"{out_cad:g} min", delta=f"-{base_step - out_cad:g} min", delta_color="inverse")
    gif = Path(tempfile.gettempdir()) / "ps12_upscaled.gif"
    try:
        frames_to_gif(dense, gif, fps=4)
        st.image(str(gif), caption=caption)
    except Exception as e:
        st.info(f"Animation unavailable ({e}); showing frames.")
        for row in range(0, len(dense), 6):
            for col, fr in zip(st.columns(6), dense[row:row + 6]):
                col.image(bt_to_rgb(fr, BT_MIN_DEFAULT, BT_MAX_DEFAULT), use_column_width=True)
    st.caption("Write the dense `.nc` sequence with `src.infer.upscale.temporal_upscale_nc(...)`.")


models = _models()
sample_files = sorted([p for p in SAMPLES_DIR.rglob("*") if p.suffix.lower() in (".nc", ".h5")])
if "backend" not in st.session_state:   # default to Local so local INSAT data shows on first load
    st.session_state["backend"] = get_backend("Local")
    st.session_state["backend_msg"] = "local CPU/GPU"

# ---------------------------------------------------------------- sidebar (status)
with st.sidebar:
    st.markdown("### PS12 · Frame Interpolation")
    st.caption("Optical-flow + deep VFI for geostationary IR")
    st.divider()
    st.markdown("**Models runnable here**")
    badges = "".join(
        f'<span class="badge {"ok" if m["available"] else "no"}">{m["label"].split(" (")[0]}</span>'
        for m in models
    )
    st.markdown(badges, unsafe_allow_html=True)
    st.divider()
    st.markdown("**System status**")
    st.markdown(f"- Sample frames found: **{len(sample_files)}**")
    st.markdown(f"- Validation experiments: **{len(list(VALIDATION_DIR.glob('*/'))) if VALIDATION_DIR.exists() else 0}**")
    st.divider()
    st.markdown("**Compute backend**")
    bk_kind = st.radio("backend", ["Local (CPU/GPU)", "Lightning.ai (T4)"], key="bk_kind",
                       label_visibility="collapsed")
    if st.button("🔌 Connect", key="btn_connect", use_container_width=True):
        try:
            b = get_backend(bk_kind)
            with st.spinner(f"Connecting to {b.name}…"):
                msg = b.connect()
            st.session_state["backend"] = b
            st.session_state["backend_msg"] = msg
            st.success(f"Connected · {b.name}")
        except Exception as e:
            st.session_state.pop("backend", None)
            st.error(f"Connect failed: {e}")
    _bk = st.session_state.get("backend")
    if _bk is not None:
        st.caption(f"● **{_bk.name}** — {str(st.session_state.get('backend_msg',''))[:55]}")
        try:
            _insat = _bk.list_insat()
            st.caption(f"INSAT files available: **{len(_insat)}**")
            if getattr(_bk, "remote", False) and len(_insat) == 0:
                if st.button("⬇ Download INSAT sample (on Studio)", key="dl_insat", use_container_width=True):
                    u, p = os.environ.get("MOSDAC_USERNAME", ""), os.environ.get("MOSDAC_PASSWORD", "")
                    with st.spinner("Ordering already done on MOSDAC — pulling via SFTP on the Studio…"):
                        st.code(_bk.download_insat(u, p)); st.rerun()
        except Exception as e:
            st.caption(f"data check: {e}")
    else:
        st.caption("Not connected — local CPU is used. Pick **Lightning.ai** + Connect for the T4 + "
                   "server-side INSAT files.")

    # --- live Lightning GPU credit rate + account balance (no connect/boot needed) ---
    st.divider()
    if st.button("💳 GPU rate & balance", key="btn_rates", use_container_width=True):
        try:
            import importlib
            import cloud.lightning_exec as _lx
            importlib.reload(_lx)  # Streamlit caches submodules; reload picks up rates_and_balance live
            with st.spinner("Querying Lightning billing…"):
                st.session_state["rates"] = _lx.rates_and_balance("T4")
        except Exception as e:
            st.session_state["rates"] = {"error": str(e), "machines": [], "balance": None, "total_spent": None}
    _rates = st.session_state.get("rates")
    if _rates:
        _bal = _rates.get("balance")
        if _bal is not None:
            _spent = _rates.get("total_spent")
            st.markdown(f"- Balance: **{_bal:.2f} credits**"
                        + (f" · spent to date: {_spent:.1f}" if _spent is not None else ""))
            if _bal <= 0:
                st.warning("0 credits — top up or switch account in `.env.local` to start a T4.")
        _rows = _rates.get("machines") or []
        if _rows:
            _tbl = ["| T4 machine | cloud | on-demand/hr | spot/hr |", "|---|---|--:|--:|"]
            for _m in _rows:
                _tbl.append(f"| `{_m['slug']}` | {_m['provider']} | {_m['on_demand']} | {_m['spot']} |")
            st.markdown("\n".join(_tbl))
            _top = _rows[0]
            st.caption(f"Inference default: **{_top['slug']}** ≈ **{_top['on_demand']} cr/hr** (cheapest). "
                       "1 credit ≈ $1.")
        if _rates.get("error"):
            st.caption(f"note: {_rates['error']}")

tab_interp, tab_upscale, tab_valid = st.tabs(["▶  Interpolate", "🔼  Temporal Upscaling", "📊  Validation Report"])

# ---------------------------------------------------------------- Interpolate
with tab_interp:
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown("#### Configuration")
        source = st.selectbox("Satellite source", SOURCES, key="src_i")
        chosen = _model_picker(models, "mdl_i")
        factor = st.radio("Output cadence", [2, 4], horizontal=True, key="fac_i",
                          format_func=lambda f: "2× (e.g. 30→15)" if f == 2 else "4× (→7.5)")
        show_overlay = st.checkbox("Motion-vector overlay", value=True, key="ov_i")
        _bk_i = st.session_state.get("backend")
        on_server = st.checkbox("Use server (Lightning) files", value=False, key="srv_i",
                                help="Pick INSAT files already on the Studio; inference runs on the T4.")
    use_remote = bool(on_server and _bk_i is not None and getattr(_bk_i, "remote", False))
    with right:
        st.markdown("#### Inputs — two consecutive frames (+ optional ground-truth middle)")
        sp0 = sp2 = up0 = up1 = upgt = pick0 = pick1 = pickgt = None
        if use_remote:
            try:
                srv = _bk_i.list_insat()
            except Exception as e:
                srv = []; st.warning(f"Could not list server files: {e}")
            if not srv:
                st.info("No INSAT files on the Studio yet — use the sidebar **Download INSAT sample** first.")
            sc1, sc2 = st.columns(2)
            sp0 = sc1.selectbox("Server frame t0", [""] + srv, key="sp0")
            sp2 = sc2.selectbox("Server frame t2", [""] + srv, key="sp2")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.caption("**t0** · required")
                up0 = st.file_uploader("Frame t0", type=["nc", "h5"], key="f0")
                pick0 = st.selectbox("…or sample t0", [""] + [str(p) for p in sample_files], key="p0")
            with c2:
                st.caption("**t2** · required")
                up1 = st.file_uploader("Frame t2", type=["nc", "h5"], key="f1")
                pick1 = st.selectbox("…or sample t2", [""] + [str(p) for p in sample_files], key="p1")
            with c3:
                st.caption("**t1 (GT)** · optional — validation only")
                _gt_help = ("The REAL middle frame. Provide it only to score the prediction "
                            "(PSNR/SSIM/FSIM/…). Not needed to generate the in-between frames — "
                            "leave blank to just interpolate.")
                upgt = st.file_uploader("GT middle t1 (optional)", type=["nc", "h5"], key="fgt", help=_gt_help)
                pickgt = st.selectbox("…or sample t1 (GT)", [""] + [str(p) for p in sample_files],
                                      key="pgt", help=_gt_help)
    go = st.button("Interpolate", type="primary", key="btn_i", use_container_width=True)

    if go and use_remote:
        if not sp0 or not sp2:
            st.error("Pick both server frames."); st.stop()
        with st.status("Running interpolation on the Lightning T4…", expanded=True) as status:
            st.write(f"🛰️ {Path(sp0).name}  +  {Path(sp2).name}")
            st.write(f"🧠 {chosen['label'].split(' (')[0]} · factor {factor}× · on the T4")
            rkw = {k: v for k, v in (chosen["kwargs"] or {}).items() if k != "weights_dir"}  # studio resolves its own weights
            mids = []
            for j in range(1, factor):
                mids.append((j / factor, _bk_i.interpolate_bt(source, sp0, sp2, chosen["name"], rkw, j / factor)))
            status.update(label="Done on T4 ✓ (preview downsampled)", state="complete", expanded=False)
        st.markdown(f"#### Synthetic frame(s) — **{chosen['label'].split(' (')[0]}** on Lightning T4")
        for col, (t, fr) in zip(st.columns(max(1, len(mids))), mids):
            col.image(bt_to_rgb(fr, BT_MIN_DEFAULT, BT_MAX_DEFAULT), use_column_width=True)
            col.markdown(f'<div class="cap">t = {t:.2f} (synthetic · ¼-res preview)</div>', unsafe_allow_html=True)
        st.caption("Full-resolution .nc is written on the Studio; the preview is downsampled for transfer.")

    if go and not use_remote:
        src0, src2 = up0 or (pick0 or None), up1 or (pick1 or None)
        if not src0 or not src2:
            st.error("Provide both t0 and t2 (upload or pick a sample)."); st.stop()
        with st.status("Running interpolation (local)…", expanded=True) as status:
            st.write("📥 Loading frame **t0**…"); bt0 = _load_bt(src0, source)
            st.write("📥 Loading frame **t2**…"); bt2 = _load_bt(src2, source)
            st.write(f"🧠 Loading model **{chosen['label'].split(' (')[0]}**…")
            model = get_model(chosen["name"], **chosen["kwargs"])
            st.write(f"✨ Interpolating {factor - 1} frame(s) at {bt0.shape[0]}×{bt0.shape[1]}…")
            mids = [(j / factor, interpolate_pair_bt(bt0, bt2, model, j / factor)) for j in range(1, factor)]
            status.update(label="Interpolation complete ✓", state="complete", expanded=False)
        st.markdown(f"#### Result — **{chosen['label'].split(' (')[0]}**  ·  cold cloud tops shown bright")
        seq = [("t0  (input)", bt0)] + [(f"t = {t:.2f}  (synthetic)", m) for t, m in mids] + [("t2  (input)", bt2)]
        for col, (label, frame) in zip(st.columns(len(seq)), seq):
            col.image(bt_to_rgb(frame, BT_MIN_DEFAULT, BT_MAX_DEFAULT), use_column_width=True)
            col.markdown(f'<div class="cap">{label}</div>', unsafe_allow_html=True)

        if show_overlay and hasattr(model, "flow"):
            with st.expander("Estimated motion vectors (t0 → t2)", expanded=True):
                try:
                    flow = model.flow(bt_to_norm(bt0), bt_to_norm(bt2))
                    st.image(flow_overlay_rgb(bt0, flow), use_column_width=True)
                except Exception as e:
                    st.info(f"Overlay unavailable for this model: {e}")

        gt_src = upgt or (pickgt or None)
        if gt_src and mids:
            btgt = _load_bt(gt_src, source)
            pred_mid = mids[len(mids) // 2][1]
            if pred_mid.shape == btgt.shape:
                m = metrics.compute_all(bt_to_norm(pred_mid), bt_to_norm(btgt), bt_min=BT_MIN_DEFAULT, bt_max=BT_MAX_DEFAULT)
                st.markdown("#### Metrics vs ground truth")
                keys = [k for k in ["psnr", "ssim", "fsim", "edge_ssim", "mse", "mae_kelvin", "lpips"] if k in m]
                for col, k in zip(st.columns(len(keys)), keys):
                    col.metric(k.upper().replace("_", " "), f"{m[k]:.4f}")

# ---------------------------------------------------------------- Temporal Upscaling
def _load_sequence(seq_src, source, label):
    """Load a list of uploads/paths to BT arrays with a progress bar."""
    rbar = st.progress(0.0, text="📥 Loading frames…")
    bts = []
    for i, s in enumerate(seq_src):
        bts.append(_load_bt(s, source))
        rbar.progress((i + 1) / len(seq_src), text=f"📥 Loaded {i + 1}/{len(seq_src)} frames")
    return bts


with tab_upscale:
    st.info("**Temporal upscaling** densifies a sequence by inserting AI frames between consecutive frames. "
            "**Recursive** halves the gap repeatedly (×2/×4, power-of-2; reuses synthetic frames so error can "
            "compound). **Continuous** inserts any number of frames per gap, each computed *directly* from the "
            "two real frames (any cadence, no compounding; off-midpoint sharpness needs an `--anytime` model).")
    up_rec, up_cont = st.tabs(["🔁  Recursive (×2 / ×4)", "♾️  Continuous (any cadence)"])

    # ---- Recursive (power-of-2, midpoint, may compound) ----
    with up_rec:
        cfgL, cfgR = st.columns([1, 2], gap="large")
        with cfgL:
            source_u = st.selectbox("Satellite source", ["insat3ds", "insat3dr", "goes19", "himawari9"], key="src_u")
            chosen_u = _model_picker(models, "mdl_u")
            levels = st.radio("Upscaling", [1, 2], horizontal=True, key="lvl_u",
                              format_func=lambda L: "30→15 (×2)" if L == 1 else "30→15→7.5 (×4)")
            base_step = st.number_input("Source cadence (min)", value=30, min_value=1, key="step_u")
        with cfgR:
            ups = st.file_uploader("Upload time-ordered frames (.nc/.h5)", type=["nc", "h5"],
                                   accept_multiple_files=True, key="seq_u")
            picks = st.multiselect("…or pick samples (in time order)", [str(p) for p in sample_files], key="seqpick_u")
        if st.button("Upscale (recursive)", type="primary", key="btn_u", use_container_width=True):
            seq_src = list(ups) if ups else list(picks)
            if len(seq_src) < 2:
                st.error("Provide at least 2 frames in time order."); st.stop()
            model_u = get_model(chosen_u["name"], **chosen_u["kwargs"])
            with st.status(f"Upscaling ×{2 ** levels}…", expanded=True) as status:
                bts = _load_sequence(seq_src, source_u, "rec")
                total = max(1, upscale_total_steps(len(bts), levels))
                done = {"n": 0}
                ibar = st.progress(0.0, text="✨ Synthesising intermediate frames…")

                def _cb():
                    done["n"] += 1
                    ibar.progress(min(1.0, done["n"] / total), text=f"✨ Synthesised {done['n']}/{total} frames")

                dense = temporal_upscale_bt(bts, model_u, levels=levels, progress=_cb)
                status.update(label="Upscaling complete ✓", state="complete", expanded=False)
            out_cad = base_step / (2 ** levels)
            _render_upscale_result(bts, dense, base_step, out_cad,
                                   f"Recursive ×{2 ** levels} — {out_cad:g}-min cadence "
                                   "(originals preserved; intermediates synthetic; built recursively).")

    # ---- Continuous (arbitrary cadence, direct from real frames, no compounding) ----
    with up_cont:
        cfgL, cfgR = st.columns([1, 2], gap="large")
        with cfgL:
            source_uc = st.selectbox("Satellite source", ["insat3ds", "insat3dr", "goes19", "himawari9"], key="src_uc")
            chosen_uc = _model_picker(models, "mdl_uc")
            n_insert = st.slider("Frames to insert per gap (N)", 1, 9, 3, key="nins_uc",
                                 help="Inserts N frames at t = k/(N+1), each computed directly from the two "
                                      "real frames (no recursion). N=3 on 30-min input → 7.5-min cadence.")
            base_step_c = st.number_input("Source cadence (min)", value=30, min_value=1, key="step_uc")
            st.caption(f"→ output cadence ≈ **{base_step_c / (n_insert + 1):g} min** "
                       f"(t = {', '.join(f'{k/(n_insert+1):.2f}' for k in range(1, n_insert + 1))})")
        with cfgR:
            ups_c = st.file_uploader("Upload time-ordered frames (.nc/.h5)", type=["nc", "h5"],
                                     accept_multiple_files=True, key="seq_uc")
            picks_c = st.multiselect("…or pick samples (in time order)", [str(p) for p in sample_files], key="seqpick_uc")
        if st.button("Upscale (continuous)", type="primary", key="btn_uc", use_container_width=True):
            seq_src = list(ups_c) if ups_c else list(picks_c)
            if len(seq_src) < 2:
                st.error("Provide at least 2 frames in time order."); st.stop()
            model_uc = get_model(chosen_uc["name"], **chosen_uc["kwargs"])
            with st.status(f"Inserting {n_insert} frame(s)/gap…", expanded=True) as status:
                bts = _load_sequence(seq_src, source_uc, "cont")
                total = max(1, continuous_total_steps(len(bts), n_insert))
                done = {"n": 0}
                ibar = st.progress(0.0, text="✨ Synthesising intermediate frames…")

                def _cb_c():
                    done["n"] += 1
                    ibar.progress(min(1.0, done["n"] / total), text=f"✨ Synthesised {done['n']}/{total} frames")

                dense = upscale_continuous_bt(bts, model_uc, n_insert, progress=_cb_c)
                status.update(label="Upscaling complete ✓", state="complete", expanded=False)
            out_cad = base_step_c / (n_insert + 1)
            _render_upscale_result(bts, dense, base_step_c, out_cad,
                                   f"Continuous — {n_insert} frame(s)/gap → {out_cad:g}-min cadence; "
                                   "each frame direct from two real frames (no error compounding).")

# ---------------------------------------------------------------- Validation Report
with tab_valid:
    st.markdown("#### Validation — interpolated vs real ground truth")
    st.caption("Protocol: input frames 20 min apart → predict the held-out real 10-min middle → score "
               "vs GOES/Himawari ground truth (INSAT: leave-one-out at 30 min).")
    exps = sorted([d for d in VALIDATION_DIR.iterdir() if d.is_dir()]) if VALIDATION_DIR.exists() else []
    if not exps:
        st.info("No validation results yet. On the GPU server run `src.eval.report.run_eval(...)` "
                "(walkthrough.md Step 8) and commit `validation_report/` — results then appear here.")
    else:
        exp = st.selectbox("Experiment", [d.name for d in exps], key="exp_v")
        d = VALIDATION_DIR / exp
        if (d / "report.md").exists():
            st.markdown((d / "report.md").read_text(encoding="utf-8"))
        pngs = sorted(d.glob("*.png"))
        if pngs:
            cols = st.columns(2)
            for i, png in enumerate(pngs):
                cols[i % 2].image(str(png), caption=png.name, use_column_width=True)
        for gif in sorted(d.glob("*.gif")):
            st.image(str(gif), caption=gif.name)

st.divider()
st.caption("PS12 · open-source backbones (RIFE · FILM · Super-SloMo · RAFT) + custom UNetVFI · "
           "trained on GOES/Himawari, applied to INSAT-3DS/3DR · metrics: PSNR/SSIM/FSIM/MSE/MAE-K/LPIPS.")
