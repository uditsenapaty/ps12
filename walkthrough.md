# walkthrough.md — GPU-server runbook

End-to-end steps to reproduce PS12 on the GPU server (T4, or a stronger GPU for less time). Local dev
is CPU-only; everything heavy (bulk download, training, full-disk inference) runs here. **No full job
runs without a green deterministic battery first.**

Storage budget: **100 GB**. Use event-based subsets + `--max-gb` + cleanup (Step 4).

> **Fast path — `python connect.py`** picks a cloud provider (Lightning.ai T4 / Kaggle / Colab+Drive),
> prompts for that provider's creds, provisions persistent storage, runs `data_setup.py` only if the
> data isn't already there, and can launch training. See `cloud/README.md`. The manual steps below are
> the equivalent done by hand on any box (incl. the Lightning Studio terminal).

---

## How you'll run this (intended end-to-end)
1. **Pick a GPU box with persistent storage** — Lightning.ai Studio (T4, 100 GB) *[recommended]*, or
   Colab+Drive / Kaggle.
2. **Complete this walkthrough ON that box** (Steps 0–9): env + data, deterministic battery, train
   UNetVFI + fine-tune RIFE (GOES/Himawari), self-supervise on INSAT, validate, build the INSAT product.
   → data + **trained checkpoints live in the box's persistent storage** (NOT committed to git — too big).
3. **Commit only `validation_report/`** (small: tables, plots, panels, GIFs) so the results are visible
   anywhere, including the dashboard's Validation Report tab.
4. **Use the dashboard** (next section). `connect.py` connects to that already-set-up box and
   **verifies** data + checkpoints exist — if not, it errors and points you back here.

## Serving / using the dashboard — three modes
**A. On the server (recommended — full trained models, GPU).** The dashboard runs where the data +
checkpoints + GPU are.
- Lightning: `streamlit run src/viz/dashboard.py` → click the **Streamlit plugin public-link** (no
  ngrok). Or `python connect.py --provider lightning --serve` (ngrok), or the **ports plugin** for 8501.
- Colab: the notebook's last cell serves via **ngrok** (needs your ngrok token).

**B. Local on your PC (debugging / no GPU).** `streamlit run src/viz/dashboard.py` — classical + RAFT
run on CPU and the **Validation Report** tab shows committed results. Trained UNetVFI/RIFE need their
checkpoints (on the server) → use mode A for those. **ngrok** is useful here for initial debugging /
sharing a local run: `NGROK_AUTHTOKEN=… python cloud/serve_dashboard.py`.

**C. Streamlit Community Cloud.** Deploy the repo; put creds in **`st.secrets`**. Good for the
Validation Report + CPU models; it *can* drive a Lightning Studio remotely (below) but mode A is simpler.

### Can a local / Streamlit-Cloud dashboard connect to Lightning.ai?
**Yes** — `lightning_sdk` runs from any box with internet + `LIGHTNING_USER_ID`/`LIGHTNING_API_KEY`
(env locally, or `st.secrets` on Streamlit Cloud), so it can start a Studio and run remote inference.
**But** that round-trips every request to the Studio (upload → run → download): more latency + moving
parts than just running the dashboard **on** the Studio. **Recommendation:** trained-model/heavy
inference in mode A; use local/Cloud (B/C) for the UI, Validation Report, and CPU models. ngrok is only
needed for B (local debugging) and Colab — Lightning has its own public-link plugin.

---

## 0. Provision & clone

```bash
git clone <this-repo> ps12 && cd ps12
python -m venv venv && source venv/bin/activate     # Linux server
pip install -r requirements-local.txt
# CUDA torch matched to the driver (example: CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python data_setup.py --env                          # remaining GPU deps
python data_setup.py --clone all                    # RIFE / FILM / Super-SloMo -> referred_clones/
```

## 1. Secrets (`.env.local`, never committed)

```
MOSDAC_USERNAME=uditsenapaty
MOSDAC_PASSWORD=********           # your MOSDAC password
HF_TOKEN=hf_********               # for pretrained weights on HF Hub
# OPENAI_API_KEY=sk-********       # OPTIONAL — only for narrative text in the report (<$10; not required)
```

## 2. Pretrained weights

| Model | Put weights in | Source |
|-------|----------------|--------|
| RAFT | — (auto via torchvision) | downloaded on first use |
| RIFE | `weights/rife/` | Practical-RIFE release (Google-Drive link in its README) |
| FILM | `weights/film/*.pt` | TorchScript export from `referred_clones/film` (dajes port) |
| Super-SloMo | `weights/superslomo/SuperSloMo.ckpt` | author Google-Drive checkpoint |
| UNetVFI (ours) | `weights/unet/` | produced by training in Step 6 |

## 3. Download data (official methods)

```bash
# GOES-19 ABI Ch13 (anonymous S3) — pick an event window
python data_setup.py --download goes --start 2025-10-01 --end 2025-10-03 --max-gb 30
# Himawari-9 B13 (anonymous S3)
python data_setup.py --download himawari --start 2025-10-01 --end 2025-10-02 --max-gb 25
# INSAT-3DR: FIRST order on MOSDAC web portal (Order Data → Archive → INSAT-3DR → Imager →
#   3RIMG_L1C_SGP → date range → All Band / HDF / Products Only → Add to cart). Then:
python data_setup.py --download insat --max-gb 25
```

INSAT bulk download on Linux can also use the MOSDAC-recommended `lftp` (equivalent to the paramiko
path in `data_setup.py`):

```bash
lftp -u "$MOSDAC_USERNAME","$MOSDAC_PASSWORD" sftp://download.mosdac.gov.in <<'EOF'
set ftp:ssl-force true
set ssl:verify-certificate no
mirror --use-pget-n=10 /Order data/insat
exit
EOF
```

## 4. Build triplet indices (+ keep storage in check)

```bash
python data_setup.py --build-index --source goes19   --step-min 10
python data_setup.py --build-index --source himawari9 --step-min 10
python data_setup.py --build-index --source insat3dr  --step-min 30   # also writes leave-one-out
# prune raw files you no longer need once indices/tiles are cached, to stay under 100 GB
```

## 5. Deterministic battery (GATE)

```bash
pytest tests/ -q          # must be green BEFORE any training/inference spend
```

## 6. Train (Steps 1–3 on GOES/Himawari)

```bash
python -m src.train.finetune --index data/index/goes19_triplets.json \
       --val-index data/index/himawari9_triplets.json \
       --steps 20000 --batch 8 --patch 256 --out weights/unet
```
Estimated: ~2–5 GPU-h on T4 (256² patches, batch 8 + grad-accum). Validation PSNR/SSIM logged; best
checkpoint at `weights/unet/best.pt`.

Fine-tune the EXISTING pretrained RIFE on the same triplets (satellite-adapted RIFE, for the
finetuned-vs-custom comparison):
```bash
python -m src.train.rife_finetune --index data/index/goes19_triplets.json \
       --weights weights/rife --out weights/rife_ft --steps 15000
```

## 7. Adapt to INSAT (Step 4, self-supervised)

```bash
python -m src.train.insat_selfsup --index data/index/insat3dr_triplets.json \
       --init weights/unet/best.pt --out weights/unet_insat --steps 5000
```
Trains on INSAT's own 30-min triplets (00:00,00:30,01:00 → predict 00:30). ~1–2 GPU-h.

## 8. Evaluate + report

```bash
python - <<'PY'
import json
from src.eval.report import run_eval
for src, idx in [("goes19","goes19"), ("himawari9","himawari9")]:
    d = json.load(open(f"data/index/{idx}_triplets.json"))
    run_eval(d["triplets"], src,
             ["classical","linear","raft","unet","rife","rife_ft","superslomo"],
             f"validation_report/{src}", max_triplets=20)        # -> committable validation_report/
# INSAT quantitative (leave-one-out): predict real 00:30 from 00:00 & 01:00
idx2 = json.load(open("data/index/insat3dr_triplets.json"))
run_eval(idx2["leave_one_out"], "insat3dr", ["classical","unet"], "validation_report/insat", max_triplets=20)
PY
git add validation_report/ && git commit -m "validation: GOES + Himawari + INSAT results"
```
Produces `validation_report/<src>/report.md` + PSNR/SSIM bar plots + `metrics_raw.json` +
`comparison_triplet0.png` + a side-by-side time-lapse GIF. **Commit `validation_report/`** so the
validation is visible in the dashboard's Validation Report tab without re-running the GPU job.
(`rife_ft` is the fine-tuned RIFE; `unet` the custom model.)

## 9. Produce INSAT 15-min product + animations

```bash
python - <<'PY'
from pathlib import Path
from src.infer.interpolate import interpolate_nc
from src.models.factory import get_model
m = get_model("unet")                      # or best model from Step 8
# 00:00 & 00:30 -> synthetic 00:15
interpolate_nc("data/insat/3RIMG_..._0000_...h5", "data/insat/3RIMG_..._0030_...h5",
               "insat3ds", m, "outputs/insat/3RIMG_0015_synth.nc", t=0.5)
PY
```

## 10. Dashboard

```bash
streamlit run src/viz/dashboard.py
```
Pick two frames, choose a model + factor (2×→15 min, 4×→7.5 min), view original-vs-interpolated
time-lapse, motion overlay, and metrics vs GT.

---

### GPU-hour budget summary (T4)
| Stage | Est. | Notes |
|-------|------|-------|
| Train UNetVFI (GOES+Himawari) | 2–5 h | 256² tiles, fine-tune-scale |
| INSAT self-sup adaptation | 1–2 h | warm-started |
| Pretrained RIFE/FILM/RAFT inference | minutes | no training |
| Full-disk inference + animations | <1 h | tiled |
Total well within a modest T4 allocation; a stronger GPU just shortens it. Closed-weight models are
not used as backbones (open-source only).
