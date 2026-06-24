# Model & API choices — what we run, what's the backbone, and why

## Inference vs training — what runs where
- **Training (GPU server):** `UNetVFI` (ours) is trained on GOES-19/Himawari triplets, then
  self-supervised-adapted to INSAT. Optionally RIFE is fine-tuned via its repo for a pretrained-backbone
  comparison.
- **Inference (GPU server, also CPU for light models):** any model in the roster turns two `.nc`
  frames into intermediate `.nc` frame(s). RAFT + classical run on CPU; RIFE/FILM/Super-SloMo/UNetVFI
  want a GPU (or trained weights) and are honestly gated otherwise.

## The roster and the role of each (all open-source)
| Model | Role | Backbone? | Why |
|-------|------|-----------|-----|
| **UNetVFI** (ours, `src/models/unet_vfi.py`) | **the model trained on satellite data** | **yes — primary trainable** | small flow+synthesis U-Net; predicts bidirectional flow + blend mask; **trains from scratch on IR in hours on a T4** and self-supervises on INSAT. Full control, no fragile external train loop. |
| **RIFE** | primary *pretrained* deep interpolator (and fine-tune target) | yes (pretrained) | estimates the *intermediate* flow directly (the PS "motion vectors"), arbitrary-time t∈(0,1) → 15 & 7.5 min, grayscale-friendly, T4-light. |
| **RAFT** (torchvision) | explicit dense **motion-vector** estimator + flow-based interpolation | yes (pretrained) | SOTA optical flow with **pretrained weights shipped in torchvision** — no manual weight vendoring; drives the dashboard motion overlay and the flow-EPE metric. |
| **FILM** | large-motion reference | pretrained | 30-min cloud displacement is large; FILM is built for large motion. |
| **Super-SloMo** | PS-named baseline; closest to the satellite anchor (Vandal & Nemani) | pretrained | arbitrary-time synthesis; the literature anchor extends exactly this to GEO IR. |
| **Classical TV-L1 / Farnebäck** | the "traditional optical flow" baseline | no | makes the **blur/ghosting on fast non-linear motion** failure (which the PS calls out) measurable; the floor the AI models must beat. |

So: **UNetVFI is the satellite-trained backbone**; RIFE/FILM/Super-SloMo/RAFT are strong **pretrained**
references; classical is the contrast baseline. Every backbone is open-source — **closed-weight models
are never used as a backbone** (project rule + reproducibility).

### Two trained models we deliver
1. **Custom (`UNetVFI`)** — trained from scratch on GOES/Himawari (`src/train/finetune.py`), then
   self-supervised-adapted to INSAT (`src/train/insat_selfsup.py`). The "novel combination": RIFE-style
   bidirectional intermediate-flow + Super-SloMo-style visibility/occlusion blending + a census+gradient
   loss tuned for smooth IR gradients. Checkpoints: `weights/unet/`, `weights/unet_insat/`.
2. **Fine-tuned existing (`RIFE`)** — the pretrained RIFE fine-tuned on satellite triplets via its own
   training step (`src/train/rife_finetune.py`), warm-started from the public checkpoint. Checkpoint:
   `weights/rife_ft/`.

The dashboard lists pretrained, **fine-tuned (`rife_ft`)**, and **custom (`unet`, `unet_insat`)**
variants side-by-side (each trained checkpoint is auto-discovered from `weights/`); a **Temporal
Upscaling** tab densifies INSAT 30→15→7.5 min; and the **Validation Report** tab shows the committed
`validation_report/` comparison (tables + plots + qualitative panels) with proper metrics.

## Budget rationale (T4, or a stronger GPU briefly)
- **Don't train heavy nets from scratch.** RIFE/FILM/RAFT/Super-SloMo are used **pretrained**; only the
  small UNetVFI is trained (cheap: 256² tiles, hours on one T4). "Cheaper sufficient beats
  thorough-looking."
- Inference is tiled, so even full-disk frames fit in T4 memory.
- A stronger GPU (used briefly) is reserved only for an optional EMA-VFI/VFIMamba quality ceiling run.
- Estimated total: well under a modest T4 allocation (see `walkthrough.md` budget table).

## Why we do NOT use OpenAI / LLM APIs for the core
- **The task is pixel synthesis, not language.** Frame interpolation = estimate motion + warp + fuse
  brightness-temperature pixels. An LLM contributes nothing to that and would add latency, per-call
  cost, and non-determinism — the opposite of what a reproducible, metric-validated pipeline needs.
- **Determinism & reproducibility.** Seedable CPU/GPU math with a deterministic battery; no external
  black-box in the data→frame path.
- **Cost.** Per-frame API calls don't scale to time-lapses of full-disk imagery; the local models are
  free to run on the T4.
- **The only optional API use:** `OPENAI_API_KEY` may draft the *natural-language narrative* of the
  comparison report (a one-shot text summary), strictly optional and well under the $10 ceiling — the
  metrics, tables and plots are computed without it. If the key is absent, the report is generated
  without narrative. `HF_TOKEN` is used only to pull open pretrained weights from HuggingFace Hub.

## Secrets
All credentials (`MOSDAC_*`, `HF_TOKEN`, `OPENAI_API_KEY`) live in `.env.local` (gitignored) and are
read from the environment — never hardcoded, logged, or committed.
