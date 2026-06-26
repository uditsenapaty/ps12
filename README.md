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

## Custom architecture — UNetVFI
A compact flow-based interpolator we own end-to-end: a U-Net takes the two IR frames **plus the target
time `t` as an input channel ("t-plane")** and predicts **bidirectional intermediate flow + a
visibility/occlusion mask** (RIFE-style intermediate flow ⊕ Super-SloMo-style visibility blending), then
backward-warps both inputs to time `t` and fuses them. Single IR band per frame (no RGB hack); the
t-conditioning lets one model render **any** intermediate time. ~2–5 M params → trains from scratch on
GOES/Himawari in hours on one T4; self-supervises on INSAT.

```
  frame t0 ─┐                      ┌──────────── U-Net encoder → decoder ────────────┐
  frame t2 ─┼─► concat (3 ch) ────►│  inc 32 ─Down→ 64 ─Down→ 128 ─Down→ 256 (bottle)│
  t-plane  ─┘   t0 · t2 · t        │     └─skip─┐  └─skip─┐  └─skip─┐                 │
                                   │      Up 32◄┘   Up 64◄┘  Up 128◄┘                 │
                                   │        │                                        │
                                   │     head → 5 channels                           │
                                   └────────┼────────────────────────────────────────┘
                                            ▼
                 ┌──────────────────────────┼──────────────────────────┐
                 ▼                          ▼                           ▼
          flow_{t→0} (2ch)           flow_{t→1} (2ch)             mask (1ch, σ)
            × t                        × (1−t)                         │
                 │                          │                         │
                 ▼                          ▼                         │
       backward-warp(t0) ──┐    ┌── backward-warp(t2)                 │
                           ▼    ▼                                     │
            pred(t) = mask · warp(t0)  +  (1 − mask) · warp(t2) ◄─────┘
                           │
                           ▼   t ∈ (0,1)  →  30→15 (t=½), 30→7.5 (t=¼,¾), …
                  synthetic intermediate frame  (denormalized → .nc)
```

Trained with a Charbonnier (robust L1) + edge/gradient + soft-census loss tuned for the smooth thermal-IR
gradients of moving cloud. See `src/models/unet_vfi.py` and `docs/model-choices.md`.

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
> by `--time-step` / `--gap-levels` / `--multigap-levels`. The UNetVFI input is **time-conditioned (3 ch)**,
> so after enabling these you must re-train `weights/unet` (old 2-ch checkpoints are skipped on load).

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
