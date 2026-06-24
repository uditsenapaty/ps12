# Implementation Plan — PS12: Fill in the Frames Seamlessly

**AI/ML Optical-Flow Frame Interpolation for Satellite Imagery (INSAT-3DS/3DR · GOES-19 · Himawari)**

> Source-of-truth implementation plan. Build order, architecture, data, evaluation, and the
> local-vs-GPU-server split. Companion files: `walkthrough.md` (server runbook), `data_setup.py`
> (data bootstrap), `referred_papers/` (cited PDFs + `SOURCES.md`).

---

## 1. Context — why, what, success

Geostationary satellites observe at a fixed cadence: **INSAT-3DR/3DS ≈ 30 min**, **GOES-19 / Himawari
≈ 10 min**. Coarse temporal resolution hampers near-real-time tracking of fast, **non-linear**
phenomena — cyclones, thunderstorms, fire fronts, floods. Classical optical flow (Farnebäck / TV-L1)
assumes brightness constancy + ~linear motion, so on growing/dissipating convective cloud it yields
**blur, ghosting, halo artefacts**. PS12 asks for an **AI/ML optical-flow frame-interpolation** system
that synthesises physically-plausible intermediate frames, validates them with image-similarity
metrics, wraps them in a web dashboard, and finally **applies the best model to INSAT-3DS** (30→15 min,
and further to 7.5 min equivalent).

**Three reframes drive every decision:**

1. **Single-channel thermal IR, not RGB video.** Inputs are calibrated **brightness temperature (BT,
   Kelvin)** in the ~10 µm clean-IR window. RGB-pretrained VFI nets are adapted to 1-channel BT via a
   deterministic, **invertible** normalization so outputs round-trip back to physical `.nc`.
2. **Motion over a 30-min gap is large and non-rigid.** Model + training target large displacement
   (FILM-style coarse-to-fine flow), not just small interframe motion.
3. **INSAT has no 15-min ground truth** — that gap *is* the deliverable. So quantitative INSAT
   validation uses a **leave-one-frame-out** protocol, and INSAT adaptation is **self-supervised** on
   its own 30-min cadence. Steps 1–3 are validated on GOES-19 / Himawari where dense GT exists.

The clean-IR window is shared across all three sensors (**GOES-19 ABI Ch13 10.3 µm**, **Himawari AHI
B13 10.4 µm**, **INSAT Imager TIR1 10.8 µm**), which physically justifies **cross-satellite transfer**:
train where data is dense (GOES/Himawari), adapt where it is sparse (INSAT).

---

## 2. Engineering contract — NO fakes, NO shortcuts (hard constraint)

Every artifact is a **real, working implementation** — never mock data, placeholder returns, fake
metrics, hardcoded "expected" outputs, or `pass`-stubs that pretend to succeed.

- Local code **actually runs on CPU against real sample `.nc`/`.h5`** (the deterministic battery passes
  on real data, not synthetic fillers).
- Readers really decode satpy/xarray/h5py BT; metrics really compute the math; model wrappers really
  load real pretrained weights and run a real forward pass.
- `data_setup.py` and `walkthrough.md` are **real working scripts/runbooks** — download + bootstrap
  logic is genuinely functional; only *heavy execution* (bulk download, training, full-disk inference)
  is **deferred** to the GPU server, not faked.
- Anything that truly needs a GPU/credential is **clearly gated and labelled** (raises a clear
  "run on server" error), never silently stubbed to look done.
- No metric is reported unless computed on actual frames. No "TODO returns 0.99".

---

## 3. Locked decisions

| Topic | Decision |
|-------|----------|
| Train/validate source (Steps 1–3) | **GOES-19 + Himawari** (PS: "validate against Himawari/GOES-19") |
| Model roster | **RIFE** (primary) + **FILM** + **Super-SloMo** + **RAFT** (flow) + **classical TV-L1** baseline |
| Strategy | **Fine-tune** pretrained backbones (not from scratch) — budget-smart; from-scratch only if it underperforms |
| Target (Step 4) | INSAT-3DR `3RIMG_L1C_SGP` for dev/validation; **INSAT-3DS** for the final delivered animation |
| Dashboard | **Streamlit** |
| Server storage | **100 GB cap** → curated event subsets + cleanup |
| Optional quality ceiling | EMA-VFI / VFIMamba on an occasional stronger GPU |

All backbones open-source (RIFE / FILM Apache-2.0 / RAFT BSD / Super-SloMo / RAFT open). Note: some RIFE
releases carry a non-commercial research clause — fine for the hackathon, flagged in `referred_papers/`.

---

## 4. Research grounding (frontier)

| Work | Relevance | What we take | PDF |
|------|-----------|--------------|-----|
| **Vandal & Nemani**, *Temporal Interpolation of GEO Imagery with Optical Flow* (IEEE TGRS) | THE satellite anchor: Super-SloMo extended to multichannel GOES-R IR, 10→1 min | triplet construction, BT normalization, per-band flow, leave-one-out idea | NTRS |
| **RIFE** (IFNet) | estimates *intermediate* flow t→{0,1}; fast; native grayscale; T4-friendly | **primary workhorse** (fine-tune) | arXiv 2011.06294 |
| **FILM** (Google) | built for large motion = 30-min cloud displacement | large-motion comparison/fallback | arXiv 2202.04901 |
| **Super-SloMo** | PS-named; arbitrary-time t∈(0,1); matches anchor | named baseline + 7.5-min product | arXiv 1712.00080 |
| **RAFT** | explicit dense **motion-vector** estimation & viz | flow overlays + flow-EPE metric | arXiv 2003.12039 |
| **EMA-VFI / VFIMamba / BiM-VFI** (2024-25 SOTA ≈29.4 dB) | higher quality if stronger GPU available | optional quality ceiling | arXiv / NeurIPS / CVPR |
| **3D-UNet diffusion for IR BT nowcasting** (Nature Sci Rep 2025) | deterministic-vs-generative trade-off for IR | noted stretch; not core budget | nature s41598-025-34207-9 |

PDFs archived under `referred_papers/` with a `SOURCES.md` mapping file→citation→URL.

---

## 5. System design

```
.nc/.h5 in ─▶ satpy/xarray/h5py reader ─▶ BT(K) ─▶ normalize[180–330K]→[0,1] ─▶ tile (e.g. 256²)
                                                                                    │
                           ┌─────────────────────────────────────────────────────────┤
                           ▼                                                          ▼
              optical-flow core (RIFE IFNet / FILM / RAFT)               classical baseline (TV-L1)
                           │  predict frame at t∈(0,1)                                │
                           ▼                                                          ▼
              merge tiles (overlap feather-blend) ─▶ denormalize→BT(K) ─▶ write .nc ─▶ metrics + animation
```

- **Optical flow / motion (PS objective 1):** RIFE's IFNet yields *intermediate* flow directly (the
  motion vectors); RAFT provides explicit dense flow fields for the motion-vector deliverable +
  overlays. Classical TV-L1/Farnebäck kept as the "traditional fails" comparison the PS calls out.
- **Frame synthesis (PS objective 2):** flow-warp both neighbours to time *t* + learned
  fusion/refinement (RIFE/FILM/Super-SloMo heads). Arbitrary *t* gives 30→15→7.5 min.
- **Representation:** 1-channel BT; fixed physical range → `[0,1]` (invertible). Replicate to 3ch for
  RGB-pretrained weights *or* adapt first conv; chosen per-model at fine-tune. Full disk is huge
  (GOES F ≈ 5424², INSAT ≈ 2816²) → **train on tiles**, **infer with overlapped tiling + feather
  blend** to avoid seams. NaN/space-pixel masking handled in normalization.
- **Training triplets:** from dense GOES-19/Himawari 10-min sequences form `(t0, t1, t2)`; learn
  `t1 = f(t0, t2, τ=0.5)`. **Temporal** (not random) train/val split to prevent leakage.
- **INSAT adaptation (self-supervised):** INSAT's own 30-min frames give triplets `(00:00, 00:30,
  01:00)` → predict `00:30`. No external GT needed; adapts resolution (4 km) + slight band shift.
- **INSAT eval (leave-one-frame-out):** hold out real `00:30`, predict it from `00:00`+`01:00`, score
  with the full metric suite → real quantitative numbers on INSAT. The *delivered* 15-min product
  (`00:00,00:30 → 00:15`) has no GT → judged visually + temporal-consistency metrics.

---

## 6. Data

| Sensor | Band | λ | Res | Cadence | Source | Format | Access |
|--------|------|---|-----|---------|--------|--------|--------|
| GOES-19 ABI | Ch13 Clean IR | 10.3 µm | 2 km | 10 min (Full Disk) | `s3://noaa-goes19/ABI-L1b-RadF/` | netCDF4 `.nc` | **open / anonymous** |
| Himawari-9 AHI | B13 | 10.4 µm | 2 km | 10 min | `s3://noaa-himawari9/AHI-L1b-FLDK/` | HSD/netCDF | **open** |
| INSAT-3DR/3DS Imager | TIR1 | 10.8 µm | 4 km | 30 min | **MOSDAC** (SFTP) | HDF5 `.h5` | **login-gated** |

- **Unified I/O:** `satpy` (Pytroll) readers for ABI L1b, AHI HSD, and INSAT-3D give calibrated BT +
  optional resampling; `xarray`/`netCDF4`/`h5py` as fallback. INSAT L1C_SGP read directly via `h5py`
  (`IMG_TIR1` / `IMG_TIR1_TEMP` + `GeoX`/`GeoY` lat-lon) when the satpy reader doesn't cover L1C.
- **GOES download:** anonymous `boto3` (`UNSIGNED`) / `s3fs` / `goes2go`; path
  `ABI-L1b-RadF/{YYYY}/{DOY}/{HH}/OR_ABI-L1b-RadF-M6C13_G19_s…​.nc`.
- **Himawari download:** anonymous AWS `noaa-himawari9` (AHI L1b FLDK, B13) via satpy `ahi_hsd`.
- **INSAT download (MOSDAC — real recipe):** *Ordering* is a manual web step (Order Data → Archive →
  INSAT-3DR → Imager → `3RIMG_L1C_SGP` → date range → All Band / HDF / Products Only → cart). The order
  lands in `/Order/<batch>/3RIMG_DDMMMYYYY_HHMM_L1C_SGP_V01R00.h5` (~85 MB each, ~30-min cadence).
  **Download is fully scripted** from `sftp://download.mosdac.gov.in:22` (creds from env):
  - **Linux server:** `lftp` mirror — `set ftp:ssl-force true; set ssl:verify-certificate no;
    mirror --use-pget-n=10 /Order <local>`.
  - **Local Windows:** `paramiko` SFTP client (cross-platform).
  - Filenames carry the timestamp → parsed + sorted + **deduped** (some slots have :14/:15 twins) to
    build triplets. INSAT-3DR has the long archive (2016→); 3DS for the final deliverable once ordered.
- **I/O contract (PS requirement):** model input AND output are `.nc`. A thin writer rebuilds a CF `.nc`
  with denormalized BT, geolocation and the synthetic timestamp; INSAT `.h5` accepted on input.
- **Data budget (100 GB):** curated event windows (cyclone/convective days) over a regional crop, not
  full multi-week disks; `data_setup.py` enforces a size ceiling and deletes intermediates.
- **Secrets:** `.env.local` is gitignored; all creds (MOSDAC, HF, OpenAI) read from env, never written
  into code, logs, or commits.

---

## 7. Models, training & budget (T4-aware)

| Role | Model | Strategy | Fits T4? |
|------|-------|----------|----------|
| **Trained-on-satellite (ours)** | **UNetVFI** | train from scratch on IR triplets; self-sup INSAT | ✅ small net, ~hours on T4 |
| Primary pretrained | **RIFE** (fine-tune IFNet) | fine-tune pretrained on IR tiles, τ-aware | ✅ infer + fine-tune @256² tiles, small batch |
| Large-motion | **FILM** | fine-tune / zero-shot compare | ✅ infer; fine-tune if time |
| Named baseline | **Super-SloMo** | reproduce anchor; arbitrary-t (7.5 min) | ✅ |
| Flow viz / metric | **RAFT** | inference only (motion fields) | ✅ |
| Classical | **TV-L1 / Farnebäck** (OpenCV) | CPU baseline (artefact narrative) | CPU |
| Optional ceiling | **EMA-VFI / VFIMamba** | stronger GPU, briefly | needs >T4 for comfort |

**Fine-tune, not train from scratch** → order **hours**, not days. T4 (16 GB) handles RIFE/FILM/RAFT
inference and RIFE fine-tuning at 256² tiles with gradient accumulation. The "better GPU for less time"
is reserved for the optional EMA-VFI/VFIMamba run. GPU-hour estimates + the exact grid are pinned in
`walkthrough.md` **before** any run, gated behind a green CPU deterministic battery.

---

## 8. Evaluation & report

- **PS-named:** MSE, **PSNR**, **SSIM**, **FSIM** (+ MAE in Kelvin for physical interpretability).
- **Perceptual / structure:** LPIPS; gradient/edge-SSIM on high-BT-gradient (cloud-edge) regions.
- **Cloud-motion-aware (PS asks explicitly):** **flow endpoint error (EPE)** of predicted vs
  RAFT-derived GT flow; **temporal warping error** between consecutive outputs (consistency).
- Metrics via **`piq`** (FSIM/SSIM/PSNR/LPIPS) + custom flow/temporal metrics; all CPU-unit-tested
  (`SSIM(x,x)=1`, `MSE(x,x)=0`, …).
- **Validation report:** auto-generated — per-method metric tables, PSNR/SSIM plots, qualitative panel
  with difference maps, classical-vs-AI artefact comparison. Written to the **committed
  `validation_report/`** folder (`report.md` + `metrics_raw.json` + `*.png` + time-lapse GIF) so the
  validation (20-min input → predict the held-out 10-min middle → score vs GT) is reproducible and
  visible in the dashboard's **Validation Report** tab without re-running the GPU job. `OPENAI_API_KEY`
  optionally drafts the narrative; never required for the numbers.

---

## 9. Visualization dashboard (Streamlit — Web GUI is an explicit eval criterion)

Three tabs: (1) **Interpolate** — a model picker spanning pretrained, fine-tuned (`rife_ft`), and custom
(`unet`, `unet_insat`) variants (auto-discovered from `weights/`), side-by-side **time-lapse animations**
(original vs interpolated), factor slider (2×/4×), **motion-vector overlay**, live metrics, `.nc`
upload→interpolate→download. (2) **Temporal Upscaling** — INSAT-3DS/3DR **30→15→7.5 min** by recursive
midpoint insertion over a frame sequence, rendered as a densified time-lapse + downloadable `.nc`.
(3) **Validation Report** — renders the committed `validation_report/` (metric tables, PSNR/SSIM plots,
qualitative panels + error maps, time-lapse GIFs). `src/viz/` builds the animations; the dashboard calls
the real `infer`/`eval` code (no fakes), with `st.status`/`st.progress` loading + inference feedback.
The dashboard can be **served from the cloud GPU and tunnelled to your local browser**
(`connect.py --serve` / `cloud/serve_dashboard.py` via ngrok) so the UI runs on the T4 while you drive
it from your PC.

---

## 10. Local (no-GPU) vs GPU-server split

| Runs LOCAL now (this PC, CPU only) | Runs on GPU SERVER later (`data_setup.py` + `walkthrough.md`) |
|---|---|
| Repo scaffold, configs, `README`, `referred_papers/` | Install `requirements-gpu.txt` (torch+CUDA, model repos) |
| All `.nc`/`.h5` readers/writers (satpy/xarray/h5py) | **Bulk dataset download** (GOES/Himawari/INSAT) + triplet index |
| Normalization, tiling/untiling, masking | **Fine-tuning / training** (RIFE/FILM/Super-SloMo) |
| Metric implementations + unit tests | **Full-disk inference**, `.nc` generation |
| **Deterministic battery** (CPU) | INSAT self-sup adaptation + leave-one-out eval |
| Streamlit dashboard on a few sample frames | Full animations + report generation |
| CPU dry-run forward pass on tiny tensors | Optional EMA-VFI/VFIMamba ceiling run |
| Download a *handful* of small real frames to learn structure | — |

**`data_setup.py`** = server bootstrap CLI (argparse): `--env` (install GPU deps), `--clone`,
`--download {goes,himawari,insat} --start --end`, `--build-index`. **`connect.py`** = one-command
bridge to a cloud **T4 + persistent storage** (Lightning.ai Studio via `lightning_sdk` / Kaggle GPU
kernel / Colab+Drive); it prompts for the chosen provider's creds, and runs `data_setup.py` only if the
dataset isn't already in persistent storage. **`walkthrough.md`** = ordered server runbook: provision →
env → download → det-battery → fine-tune (custom + RIFE) → infer → validate → dashboard.

**Scope:** supervised training/fine-tuning uses **GOES-19 + Himawari only** (real 10-min GT); INSAT is
**inference** for the 15/7.5-min product, with an optional **self-supervised** INSAT adaptation on its
own 30-min cadence (no external labels; `selfsup_insat.enabled` in config).

---

## 11. Repo structure

```
ps12/
├── implementation-plan.md      # this document
├── walkthrough.md              # GPU-server runbook
├── data_setup.py               # real server bootstrap / heavy-data loader CLI
├── connect.py                  # connect to cloud T4 (Lightning.ai / Kaggle / Colab) + persistent storage
├── cloud/                      # connector docs + Colab notebook + generated Kaggle kernel
├── .env.local.example          # documented credential names (copy to gitignored .env.local)
├── README.md
├── requirements-local.txt      # CPU: satpy xarray netCDF4 h5py numpy opencv scikit-image
│                               #      piq imageio[ffmpeg] boto3 s3fs paramiko torch(cpu) streamlit pytest
├── requirements-gpu.txt        # torch+cuda torchvision + model repos (server only)
├── referred_papers/            # PDFs of every cited paper + SOURCES.md (file→citation→URL)
├── referred_clones/            # git-cloned model repos (RIFE/FILM/Super-SloMo), .git stripped
├── validation_report/          # COMMITTED validation results (tables, plots, panels, GIFs)
├── docs/                       # data-structure.md, model-choices.md (real findings + rationale)
├── configs/                    # yaml: data ranges, tiling, model, train, eval
├── src/
│   ├── data/                   # readers (goes/himawari/insat), normalize, tiling, triplets
│   ├── models/                 # unet_vfi (ours) + rife/film/superslomo/raft/classical wrappers (1 interface)
│   ├── train/                  # finetune (custom) + rife_finetune (existing) + self-sup INSAT
│   ├── infer/                  # .nc → .nc interpolation + temporal upscaling (30→15→7.5)
│   ├── eval/                   # metrics (mse/psnr/ssim/fsim/lpips/epe/temporal) + validation report
│   └── viz/                    # animation builder + Streamlit dashboard (3 tabs)
├── tests/                      # deterministic battery + unit tests (CPU)
├── notebooks/                  # structure exploration on sample .nc
└── data/ samples/ outputs/ weights/   # gitignored
```

---

## 12. Build order (autonomous, auto-mode)

1. **Repo + docs:** `git init`; `.gitignore`; this plan, `README.md`, `requirements-*.txt`,
   `walkthrough.md`.
2. **Papers → `referred_papers/` (PDF)** + `SOURCES.md`.
3. **Local deps** into `myenv` (CPU-only).
4. **Real sample data** (GOES anon-S3, INSAT-3DR paramiko-SFTP, Himawari anon-S3 best-effort) → inspect
   real structure → `docs/data-structure.md`.
5. **Real data pipeline:** readers + normalization + tiling + triplet builder + `data_setup.py`.
6. **Real models:** unified interface over RIFE/FILM/Super-SloMo/RAFT/classical; weight-load + CPU dry-run.
7. **Real train/infer/eval:** fine-tune + self-sup loops (server-runnable); `.nc`→`.nc` infer; metrics + report.
8. **Real Streamlit dashboard** wired to infer/eval.
9. **Verify** (Section 14).

---

## 13. Deterministic battery (gate before any GPU spend)

shapes & dtype; BT↔normalize round-trip; tile→untile identity; NaN/space mask preserved; `.nc`
write→read identity; metric sanity (`SSIM(x,x)=1`, `MSE(x,x)=0`); tiny-tensor model forward on CPU;
config schema validation. Seeded, sub-second, all CPU. **No full GPU job without a green battery.**

---

## 14. Verification (end-to-end)

1. CPU det-battery green (`pytest tests/`).
2. `infer` on two sample GOES `.nc` 20 min apart → produces the 10-min `.nc`; compare to the real
   10-min frame → metrics print.
3. Streamlit dashboard renders original-vs-interpolated animation + motion overlay on samples.
4. *(Server)* leave-one-out INSAT eval table + 15-min INSAT-3DS animation.

---

## 15. Risks & mitigations

- **INSAT access gated** → real MOSDAC SFTP downloader (creds from env); manual order step documented.
- **Domain gap GOES→INSAT** (res/band/geography) → self-sup INSAT fine-tune + cross-sat training.
- **Large non-linear motion** → FILM/coarse-to-fine + flow-EPE metric to catch failure.
- **Tiling seams** → overlap + feather blend; validated by reconstruct-identity test.
- **Grayscale↔RGB weight mismatch** → channel-replicate or first-conv adapt, A/B at fine-tune.
- **100 GB cap** → event-based curation + intermediate cleanup in `data_setup.py`.

---

## Sources
- Vandal & Nemani, *Temporal Interpolation of GEO Satellite Imagery With Optical Flow*, IEEE TGRS — ntrs.nasa.gov/citations/20210020625
- RIFE arXiv:2011.06294 · FILM arXiv:2202.04901 · RAFT arXiv:2003.12039 · Super-SloMo arXiv:1712.00080
- EMA-VFI arXiv:2303.00440 · VFIMamba (NeurIPS 2024) · BiM-VFI (CVPR 2025) · VFI rankings: github.com/AIVFI
- 3D-UNet diffusion IR nowcasting, Nature Sci Rep 2025 — nature.com/articles/s41598-025-34207-9
- NOAA GOES on AWS — registry.opendata.aws/noaa-goes · Himawari — registry.opendata.aws/noaa-himawari
- INSAT-3D/3DR/3DS — mosdac.gov.in
