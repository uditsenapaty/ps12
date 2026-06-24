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
A compact flow-based interpolator we own end-to-end: a U-Net predicts **bidirectional intermediate
flow + a visibility/occlusion mask** (RIFE-style intermediate flow ⊕ Super-SloMo-style visibility
blending), then backward-warps both inputs to time `t` and fuses them. Single-channel in/out (no RGB
hack). ~2–5 M params → trains from scratch on GOES/Himawari in hours on one T4; self-supervises on INSAT.

```
  frame t0 ─┐                      ┌──────────── U-Net encoder → decoder ────────────┐
            ├─► concat (2 ch) ────►│  inc 32 ─Down→ 64 ─Down→ 128 ─Down→ 256 (bottle)│
  frame t2 ─┘                      │     └─skip─┐  └─skip─┐  └─skip─┐                 │
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
