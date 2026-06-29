# PS12 — Fill in the Frames Seamlessly

**AI/ML optical-flow frame interpolation for geostationary satellite imagery.** Generate synthetic
intermediate frames between consecutive thermal-IR (~10 µm) images to enhance temporal resolution
(INSAT-3DS/3DR 30 → 15 → 7.5 min), validated on high-cadence GOES-19 / Himawari, with a Streamlit
dashboard and a metrics report.

> Full technical plan: **[`implementation-plan.md`](implementation-plan.md)**
> GPU-server runbook: **[`walkthrough.md`](walkthrough.md)**
> Cited papers (PDF): **[`referred_papers/`](referred_papers/)**

## Datasets (TIR ~10 µm, clean-IR window)
| Sensor | Band | λ | Res | Cadence | Access |
|--------|------|---|-----|---------|--------|
| GOES-19 ABI | Ch13 | 10.3 µm | 2 km | 10 min | open AWS `noaa-goes19` |
| Himawari-9 AHI | B13 | 10.4 µm | 2 km | 10 min | open AWS `noaa-himawari9` |
| INSAT-3DR/3DS | TIR1 | 10.8 µm | 4 km | 30 min | MOSDAC (SFTP, login) |

## Models
**UNetVFI (our custom model, trained on satellite IR)** · RIFE (pretrained + fine-tuned) · FILM (large
motion) · Super-SloMo (baseline) · RAFT (flow/motion vectors) · classical TV-L1 (artefact comparison).
All open-source. Heavy nets used **pretrained**; the small UNetVFI is **trained** (cheap on a T4).

## Custom architecture — UNetVFI (**FeatSynthVFI**)
A flow-based interpolator we own end-to-end, fusing the strong baselines' best ideas and training them on
satellite IR. A **weight-tied Siamese encoder** builds a feature pyramid for each of the **two IR frames
(one band each)**; a **coarse-to-fine decoder** predicts the **bidirectional intermediate flow** (RIFE/
IFRNet-style) at 1/8 res and residually refines it (1/4→1/2→1/1) with feature warping, plus a **visibility
mask** (Super-SloMo). The two inputs are warped to time `t` and blended; a **feature-synthesis decoder**
(FILM/SoftSplat-style) then warps the pyramids to `t` and adds a smooth, bounded residual. A **PINN source
head** models brightness growth/decay. Time `t` is **implicit** — it only scales the flow, so one model
renders any intermediate time. ~4.25 M params → trains from scratch in ~30 min on one T4.

```
  frame t0 ─► Siamese encoder ─► pyramid {1/1,1/2,1/4,1/8}  ┐
  frame t2 ─► Siamese encoder ─► pyramid {1/1,1/2,1/4,1/8}  ┘
        │
        ▼  coarse→fine flow decoder (warp features by current flow, refine residually per scale)
   flow_{t→0}, flow_{t→1}  +  mask M  +  PINN source S
        │
        ▼  warp-blend:  M·warp(t0) + (1−M)·warp(t2)
        ▼  + synthesis residual rendered from the *warped feature pyramids* (smooth, bounded)
   pred(t)  →  t∈(0,1): 30→15 (t=½), 30→7.5 (t=¼,¾), …  →  denormalized → .nc
```

Trained **across GOES-19 + Himawari-9 + INSAT at once** (one `ConcatDataset`) with Charbonnier + edge/
gradient + soft-census + a small VGG **perceptual** loss + the **PINN advection** loss, using **EMA**.
See `src/models/unet_vfi.py`, `src/train/finetune.py`, and `explanation.md` §6/§11.

### Held-out results (24 GOES triplets, **separate-day** test — no leakage)
After 5 architecture iterations the custom model is a decisive **#2**: it **beats classical, RAFT and
Super-SloMo on every metric**, and **matches pretrained-SOTA FILM** to ~0.4% (FILM is frozen/pretrained on
millions of natural-video frames, so it cannot overfit). Full table + plots in `validation_report/goes19_heldout/`.

| model | PSNR | SSIM | FSIM | edge-SSIM | LPIPS |
|---|---|---|---|---|---|
| FILM (pretrained SOTA) | **38.91** | **0.962** | **0.9947** | **0.928** | **0.097** |
| **UNetVFI (ours)** | 38.76 | 0.957 | 0.9945 | 0.926 | 0.145 |
| classical | 38.30 | 0.951 | 0.9936 | 0.920 | 0.150 |
| RAFT | 37.00 | 0.944 | 0.9918 | 0.893 | 0.158 |
| Super-SloMo | 35.46 | 0.946 | 0.9881 | 0.889 | 0.173 |

## Repo layout
```
src/data   readers (goes/himawari/insat) · normalize · tiling · triplets
src/models unified interface over RIFE/FILM/Super-SloMo/RAFT/classical
src/train  fine-tune + self-supervised INSAT adaptation
src/infer  .nc -> .nc interpolation + temporal upscaling (30->15->7.5)
src/eval   MSE/PSNR/SSIM/FSIM/LPIPS + flow-EPE/temporal + validation report
src/viz    animations + Streamlit dashboard (Interpolate / Temporal Upscaling / Validation Report)
tests      deterministic battery (CPU, runs on real samples)
data_setup.py  server bootstrap: download + index + weights (100 GB-aware)
```

## Commands & arguments

Three commands cover the whole workflow. **All credentials are read from `.env.local`** (gitignored) —
edit that file, or set env vars inline, to switch accounts.

### 1 · Run the web app — `streamlit run src/viz/dashboard.py`
Local UI; pick the compute backend in the sidebar (**Local CPU**, or **Lightning.ai T4** via the in-page
🔌 Connect button) and check the 💳 GPU rate & balance. Tabs: **Interpolate** · **Temporal Upscaling**
(Recursive ×2/×4 · Continuous any-cadence) · **Validation Report**. No CLI args; one env override:
- `LIGHTNING_MACHINE=T4` — remote GPU tier (default `T4_SMALL`, the cheapest single T4).

### 2 · Set up data on the cloud T4 — `python connect.py --provider lightning --bootstrap --full-data`
Connects to the Lightning Studio, clones the model repos, downloads **GOES + Himawari + ≥1-day INSAT**
into 100 GB persistent storage, and builds the indices.

| arg | default | meaning |
|-----|---------|---------|
| `--provider {lightning,kaggle,colab}` | interactive menu | cloud target |
| `--bootstrap` | off | one-time setup: clone models + download data |
| `--full-data` | off | full event windows (mandatory GOES+Himawari + ≥1-day INSAT) instead of `--sample` size |
| `--train` | off | also launch training after connecting |
| `--serve` | off | serve the dashboard on the remote (Lightning plugin / ngrok) |

### 3 · Re-train + commit results — `python scripts/cloud_retrain.py --steps 8000 --pinn --source goes19 --commit`
Drives the Studio: sync → train with your args → validate → fetch `report.md` → git commit + push.

| arg | default | meaning |
|-----|---------|---------|
| `--steps N` | 8000 | training steps |
| `--batch B` | 8 | batch size (tiles) |
| `--source {goes19,himawari9,insat3dr}` | goes19 | which index to train on |
| `--pinn` | off | physics-informed advection loss (+ learned source term) |
| `--pinn-weight W` | 0.1 | PINN loss weight |
| `--anytime` | off | arbitrary-time training (variable t-grid → off-midpoint frames) |
| `--multigap` | off | temporal multi-granularity (one target from symmetric gaps, combined loss) |
| `--out DIR` | weights/unet | checkpoint directory |
| `--models a,b,c` | classical,raft,unet | models compared in the report |
| `--max-triplets N` | 20 | eval triplets |
| `--commit` | off | git commit + push the validation report |

### Underlying scripts (run these directly on the server)

**`python data_setup.py …`** — data + index bootstrap (no GPU touched):

| arg | default | meaning |
|-----|---------|---------|
| `--download {goes,himawari,insat}` | — | download a source |
| `--sample` | off | grab a few frames only (structure check) |
| `--start` / `--end YYYY-MM-DD` | last 2 days | download date range |
| `--max-gb F` | 50 | stop once the target dir exceeds this |
| `--build-index` | — | build the triplet / arbitrary-time / multigap index |
| `--source TAG` | goes19 | source tag for `--build-index` |
| `--step-min M` | 10 | frame cadence (10 = GOES/Himawari, 30 = INSAT) |
| `--time-step DT` | 0.5 | arbitrary-time t-grid spacing; **must divide 1** (0.5 = midpoint, 0.25 = quarters) |
| `--gap-levels N` | 3 | arbitrary-time gap sizes (spans = base, 2·base, …) |
| `--multigap-levels N` | 1 | multigap max symmetric bracket level (1 = plain midpoint, 2 = +wider bracket) |
| `--env` | — | install GPU requirements |
| `--clone [all\|rife\|film\|superslomo]` | all | clone deep-model repos into `referred_clones/` |

**`python -m src.train.finetune --index … `** — the training loop `cloud_retrain` calls:

| arg | default | meaning |
|-----|---------|---------|
| `--index PATH` | **required** | triplet index json (`data/index/<source>_triplets.json`) |
| `--val-index PATH` | = index | validation index |
| `--out DIR` | weights/unet | checkpoint directory |
| `--steps / --batch / --lr / --patch / --base` | 20000 / 8 / 1e-4 / 256 / 32 | core hyper-parameters |
| `--device {cuda,cpu}` | auto | compute device |
| `--workers N` | 0 | dataloader workers |
| `--val-every N` | 500 | validate + checkpoint interval |
| `--init PATH.pt` | — | warm-start from existing weights |
| `--pinn` / `--pinn-weight W` | off / 0.1 | physics-informed loss |
| `--anytime` | off | arbitrary-time samples (off-midpoint t) |
| `--multigap` | off | temporal multi-granularity (symmetric combined loss) |

**`python cloud/lightning_exec.py …`** — Studio helper: `--start` · `--stop` · `--whoami` · `--rates`
(balance + per-hour T4 credit rates) · `"<shell command>"` (run a command on the Studio).

> Note: `--anytime` / `--multigap` use the samples the index already contains, controlled at build time
> by `--time-step` / `--gap-levels` / `--multigap-levels`. The model input is the **two IR frames (2 ch);
> `t` is implicit** (it scales the flow). Re-train `weights/unet` after changing these training settings.

## Quickstart (local, CPU)
```bash
myenv/Scripts/python -m pip install -r requirements-local.txt
cp .env.local.example .env.local        # then fill the creds you need (gitignored)
#   MOSDAC_USERNAME/PASSWORD (INSAT), HF_TOKEN (weights), cloud keys (optional)
python data_setup.py --download goes  --sample          # a few real GOES frames
python data_setup.py --download insat --sample          # a few real INSAT frames (SFTP)
pytest tests/                                            # deterministic battery (green on real data)
streamlit run src/viz/dashboard.py                      # Interpolate / Temporal Upscaling / Validation Report
```

## Cloud GPU (one command)
```bash
python connect.py            # pick Lightning.ai (T4, 100 GB) / Kaggle / Colab; prompts for creds,
                             # sets up persistent storage, runs data_setup.py if data is missing
```
See **`cloud/README.md`**.

## On the GPU server (manual)
See **`walkthrough.md`** — `python data_setup.py --env`, `--clone`, bulk download, fine-tune (custom
UNetVFI + RIFE), infer, validate, dashboard. No full job runs without a green deterministic battery
first. Training uses **GOES-19 + Himawari**; INSAT is inference (+ optional self-supervised adaptation).

## Secrets
`.env.local` is gitignored. MOSDAC / HF / OpenAI credentials are read from environment variables and
are never written into code, logs, or commits.
