#!/usr/bin/env python
"""connect.py — connect the PS12 pipeline to a cloud T4 GPU + persistent storage.

Interactive: choose a provider; it prompts for the creds/tokens that provider needs (or reads them from
.env.local), connects, ensures the dataset exists in persistent storage (running `data_setup.py` there
if it's missing), and can launch training. Credentials are read from `.env.local` (gitignored) or asked
for with getpass and saved there for reuse — never hardcoded or committed.

Providers
---------
  1) Lightning.ai  — Studio on a T4 with 100 GB persistent storage (fully scripted via lightning_sdk).
  2) Kaggle        — generate + push a GPU (T4) kernel that runs the pipeline; data as a Kaggle dataset.
  3) Google Colab  — generate a ready-to-run notebook that mounts Drive (persistent) and runs everything.

Usage
-----
  python connect.py                 # interactive menu
  python connect.py --provider lightning --train
  python connect.py --provider colab
"""
from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_LOCAL = ROOT / ".env.local"
sys.path.insert(0, str(ROOT))
from src.data.env import load_env  # noqa: E402

# readiness probes run on the remote persistent storage (a server that completed walkthrough.md).
# MANDATORY data: GOES-19 AND Himawari-9 present, plus at least ~1 day of INSAT-3DS/3DR
# (48 frames at 30-min cadence) so the INSAT self-supervised triplets + leave-one-out eval are real.
GOES_MIN, HIMA_MIN, INSAT_MIN = 3, 10, 48  # INSAT_MIN = 48 = one full day at 30-min cadence
DATA_PROBE = (
    f"[ $(ls data/goes19/*.nc 2>/dev/null | wc -l) -ge {GOES_MIN} ] && "
    f"[ $(ls data/himawari9/* 2>/dev/null | wc -l) -ge {HIMA_MIN} ] && "
    f"[ $(ls data/insat/*.h5 data/insat/*.hdf5 2>/dev/null | wc -l) -ge {INSAT_MIN} ]")
CKPT_PROBE = "ls weights/unet*/*.pt >/dev/null 2>&1 || ls weights/rife_ft/* >/dev/null 2>&1"
READY_PROBE = f"cd ps12 2>/dev/null && (( {DATA_PROBE} ) && ( {CKPT_PROBE} ) && echo PS12_READY || echo PS12_NOTREADY)"
DATA_SETUP_SAMPLE = ("python data_setup.py --download goes --sample && "
                     "python data_setup.py --download himawari --sample && "
                     "python data_setup.py --download insat --sample")
# Full bootstrap: GOES + Himawari (mandatory) + ≥1 day INSAT, then multi-granularity indices for all 3.
DATA_SETUP_FULL = ("python data_setup.py --download goes --start 2025-10-01 --end 2025-10-03 --max-gb 30 && "
                   "python data_setup.py --download himawari --start 2025-10-01 --end 2025-10-02 --max-gb 25 && "
                   "python data_setup.py --download insat --max-gb 25 && "
                   "python data_setup.py --build-index --source goes19 --step-min 10 --levels 3 && "
                   "python data_setup.py --build-index --source himawari9 --step-min 10 --levels 3 && "
                   "python data_setup.py --build-index --source insat3dr --step-min 30 --levels 2")


# ------------------------------------------------------------------ creds helpers
def set_env_local(key: str, value: str) -> None:
    """Upsert KEY=VALUE in .env.local (created if absent). Keeps the file gitignored."""
    lines = ENV_LOCAL.read_text(encoding="utf-8").splitlines() if ENV_LOCAL.exists() else []
    out, found = [], False
    for ln in lines:
        if ln.strip().startswith(f"{key}="):
            out.append(f"{key}={value}"); found = True
        else:
            out.append(ln)
    if not found:
        out.append(f"{key}={value}")
    ENV_LOCAL.write_text("\n".join(out) + "\n", encoding="utf-8")


def get_or_prompt(name: str, prompt: str, secret: bool = True, required: bool = True) -> str:
    val = os.environ.get(name, "").strip()
    if val:
        return val
    entry = getpass.getpass(f"{prompt} ({name}): ") if secret else input(f"{prompt} ({name}): ").strip()
    if not entry and required:
        print(f"  ! {name} is required."); sys.exit(2)
    if entry:
        os.environ[name] = entry
        if input(f"  save {name} to .env.local for reuse? [Y/n] ").strip().lower() in ("", "y", "yes"):
            set_env_local(name, entry)
    return entry


def ensure_pip(pkg: str, import_name: str | None = None) -> None:
    try:
        __import__(import_name or pkg.replace("-", "_"))
    except Exception:
        print(f"[connect] installing {pkg} …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-input", pkg])


def remote_env_exports() -> str:
    """Shell exports for creds the remote needs (MOSDAC/HF/OpenAI), pulled from the local env."""
    parts = []
    for k in ("MOSDAC_USERNAME", "MOSDAC_PASSWORD", "HF_TOKEN", "OPENAI_API_KEY"):
        v = os.environ.get(k)
        if v:
            parts.append(f"export {k}='{v}'")
    return " && ".join(parts)


# ------------------------------------------------------------------ Lightning.ai
def connect_lightning(do_train: bool, full_data: bool, do_serve: bool = False, bootstrap: bool = False) -> None:
    ensure_pip("lightning-sdk", "lightning_sdk")
    get_or_prompt("LIGHTNING_USER_ID", "Lightning user id")
    get_or_prompt("LIGHTNING_API_KEY", "Lightning API key")
    teamspace = get_or_prompt("LIGHTNING_TEAMSPACE", "Lightning teamspace (e.g. you/vision)",
                              secret=False, required=False) or None
    studio_name = os.environ.get("LIGHTNING_STUDIO", "ps12")
    repo_url = get_or_prompt("REPO_URL", "git URL of this repo (recommended)", secret=False, required=False)

    from lightning_sdk import Machine, Studio
    print(f"[lightning] starting Studio '{studio_name}' on a T4 (on-demand / non-interruptible) …")
    studio = Studio(name=studio_name, teamspace=teamspace, create_ok=True)
    try:
        studio.start(Machine.T4, interruptible=False)   # T4 = cheapest GPU; on-demand = non-interruptible
    except TypeError:
        studio.start(Machine.T4)

    if repo_url:
        studio.run(f"[ -d ps12 ] || git clone {repo_url} ps12")
    else:
        print("[lightning] no REPO_URL set — upload this folder via the Studio UI or set REPO_URL, then re-run.")

    if bootstrap:
        exports = remote_env_exports()
        setup_cmd = DATA_SETUP_FULL if full_data else DATA_SETUP_SAMPLE
        print("[lightning] --bootstrap: cloning models + downloading data (one-time setup) …")
        print(studio.run(f"cd ps12 && pip install -r requirements-local.txt && python data_setup.py --clone all && "
                         f"{exports + ' && ' if exports else ''}(( {DATA_PROBE} ) || ({setup_cmd}))"))

    # the server must already have data + trained checkpoints (i.e. walkthrough.md was completed)
    status = studio.run(READY_PROBE)
    if "PS12_READY" not in status:
        print("[lightning] ✗ Studio is NOT ready. MANDATORY data: GOES-19 + Himawari-9 both present,")
        print(f"            and ≥1 day of INSAT-3DS/3DR (≥{INSAT_MIN} frames at 30-min cadence) — plus a")
        print("            trained checkpoint. INSAT must be ordered on MOSDAC first (see walkthrough.md);")
        print("            then re-run with --bootstrap --full-data (--train) to set it up. Probe said:")
        print("           ", (status.strip().splitlines() or ["<no output>"])[-1])
        if not (do_train or bootstrap):
            studio.stop() if False else None
            return
    else:
        print("[lightning] ✓ Studio ready (data + checkpoints present in persistent storage).")

    if do_train:
        print("[lightning] launching training (UNetVFI on GOES) …")
        print(studio.run("cd ps12 && python -m src.train.finetune "
                         "--index data/index/goes19_triplets.json --steps 20000 --out weights/unet"))

    if do_serve:
        print("[lightning] TIP (simplest, no ngrok): in the Studio terminal run "
              "`streamlit run src/viz/dashboard.py` and click the Streamlit plugin's public-link icon.")
        token = get_or_prompt("NGROK_AUTHTOKEN", "ngrok authtoken (free: dashboard.ngrok.com; "
                              "Enter to skip and use the Lightning plugin instead)", required=False)
        if token:
            print("[lightning] serving the dashboard on the Studio + opening an ngrok tunnel …")
            serve = (f"cd ps12 && pip install -q streamlit pyngrok && "
                     f"NGROK_AUTHTOKEN='{token}' nohup python cloud/serve_dashboard.py > dash.log 2>&1 & "
                     f"sleep 14 && (grep -A1 'Open the dashboard' dash.log || tail -25 dash.log)")
            print(studio.run(serve))
            print("[lightning] ↑ open that URL in YOUR PC browser — the UI runs on the T4, you drive it locally.")
        else:
            print("[lightning] start it yourself on the Studio: `streamlit run src/viz/dashboard.py` "
                  "then use the Streamlit plugin public link.")
    print("[lightning] done. Studio keeps your data + weights in its 100 GB persistent home. "
          "Stop it from the UI or: studio.stop()")


# ------------------------------------------------------------------ Kaggle
def connect_kaggle(full_data: bool) -> None:
    user = get_or_prompt("KAGGLE_USERNAME", "Kaggle username", secret=False)
    key = get_or_prompt("KAGGLE_KEY", "Kaggle API key")
    # the kaggle CLI reads ~/.kaggle/kaggle.json
    kdir = Path.home() / ".kaggle"; kdir.mkdir(exist_ok=True)
    (kdir / "kaggle.json").write_text(f'{{"username":"{user}","key":"{key}"}}', encoding="utf-8")
    try:
        os.chmod(kdir / "kaggle.json", 0o600)
    except Exception:
        pass
    ensure_pip("kaggle")

    repo_url = get_or_prompt("REPO_URL", "git URL of this repo", secret=False)
    kernel_dir = ROOT / "cloud" / "kaggle"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{user}/ps12-frame-interpolation"
    setup_cmd = DATA_SETUP_FULL if full_data else DATA_SETUP_SAMPLE
    # kernel notebook source
    code = (f"!git clone {repo_url} ps12 && cd ps12 && pip install -q -r requirements-local.txt && "
            f"python data_setup.py --clone all && ({DATA_PROBE} || ({setup_cmd})) && "
            f"python -m src.train.finetune --index data/index/goes19_triplets.json --steps 5000 --out weights/unet")
    _write_kaggle_kernel(kernel_dir, slug, code)
    print(f"[kaggle] kernel scaffold written to {kernel_dir}")
    print("[kaggle] enable GPU(T4) in kernel-metadata.json (already set) and push:")
    print(f"    kaggle kernels push -p {kernel_dir}")
    if input("  push now? [y/N] ").strip().lower() in ("y", "yes"):
        subprocess.check_call(["kaggle", "kernels", "push", "-p", str(kernel_dir)])
        print(f"[kaggle] pushed -> https://www.kaggle.com/code/{slug}")


def _write_kaggle_kernel(kernel_dir: Path, slug: str, code: str) -> None:
    import json
    meta = {
        "id": slug,
        "title": "PS12 frame interpolation",
        "code_file": "ps12_kernel.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,          # T4
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": code.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "nbformat": 4, "nbformat_minor": 5}
    (kernel_dir / "ps12_kernel.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ Google Colab
def connect_colab() -> None:
    nb = ROOT / "cloud" / "colab" / "ps12_colab.ipynb"
    if not nb.exists():
        print(f"[colab] notebook not found at {nb} — it ships in the repo (cloud/colab/).")
        return
    repo_url = os.environ.get("REPO_URL", "<your repo URL>")
    print("[colab] Open this notebook in Google Colab (Runtime → Change runtime type → T4 GPU):")
    print(f"    {nb}")
    print("    or upload it to https://colab.research.google.com")
    print("[colab] It mounts Google Drive (persistent), clones the repo, runs data_setup.py only if the")
    print("        dataset is not already in Drive, then trains/evaluates and saves weights to Drive.")
    print("        The last cell serves the dashboard via an ngrok tunnel, so you drive the UI from your")
    print("        local PC browser while the T4 does the compute.")
    print(f"        Set REPO_URL in the first cell (currently: {repo_url}).")


# ------------------------------------------------------------------ menu
def main() -> None:
    load_env()
    ap = argparse.ArgumentParser(description="Connect PS12 to a cloud T4 GPU + persistent storage")
    ap.add_argument("--provider", choices=["lightning", "kaggle", "colab"])
    ap.add_argument("--train", action="store_true", help="also launch training after connecting")
    ap.add_argument("--serve", action="store_true", help="serve the dashboard on the remote (Lightning plugin or ngrok)")
    ap.add_argument("--bootstrap", action="store_true", help="(Lightning) one-time setup: clone models + download data")
    ap.add_argument("--full-data", action="store_true", help="download full event windows, not just samples")
    args = ap.parse_args()

    provider = args.provider
    if not provider:
        print("Connect PS12 to cloud GPU + persistent storage:\n"
              "  1) Lightning.ai (T4, 100 GB persistent)  [most automated]\n"
              "  2) Kaggle (T4 kernel)\n"
              "  3) Google Colab + Drive")
        choice = input("Choose [1/2/3]: ").strip()
        provider = {"1": "lightning", "2": "kaggle", "3": "colab"}.get(choice)
        if not provider:
            print("invalid choice"); sys.exit(2)

    if provider == "lightning":
        connect_lightning(args.train, args.full_data, args.serve, args.bootstrap)
    elif provider == "kaggle":
        connect_kaggle(args.full_data)
    else:
        connect_colab()


if __name__ == "__main__":
    main()
