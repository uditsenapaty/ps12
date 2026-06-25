# cloud/ — connect to a T4 GPU + persistent storage

`python connect.py` (repo root) is the single entry point. It prompts for the creds of the provider you
pick (or reads them from `.env.local`), connects to a box that has **already completed `walkthrough.md`**,
**verifies that data + trained checkpoints exist** in persistent storage, and serves the dashboard. If
the box isn't set up yet it **errors** (checkpoints are large and not committed — they live only in the
server's persistent storage). Use `--bootstrap` for a one-time setup (clone models + download data).

## The three commands you'll actually use
Creds live in **`.env.local`** (gitignored). To use a **different account**, just edit those values
(`LIGHTNING_USER_ID`, `LIGHTNING_API_KEY`, `MOSDAC_USERNAME/PASSWORD`, `REPO_URL`) before running.

```bash
# 1) RUN THE WEB APP (anywhere) — connects to the already-set-up Studio from the in-page "Connect" button
streamlit run src/viz/dashboard.py
#    sidebar → Compute = "Lightning.ai (T4)" → 🔌 Connect → "Use server files" → pick INSAT → Interpolate.

# 2) SET UP DATA on the Studio (one-time, or with different creds after editing .env.local)
python connect.py --provider lightning --bootstrap --full-data
#    clones model repos + downloads GOES/Himawari/INSAT into the Studio's 100 GB persistent storage.

# 3) RE-TRAIN with chosen args + re-commit new validation results to git (one command)
python scripts/cloud_retrain.py --steps 8000 --pinn --source goes19 --commit
#    trains on the T4 with your args → runs the validation report → fetches report.md →
#    git commit + push (validation_report/<source>[_pinn]/report.md). Omit --pinn / --commit as needed.
```
Tip (different creds inline, instead of editing the file): on PowerShell
`$env:LIGHTNING_API_KEY="…"; streamlit run src/viz/dashboard.py`.

| Provider | What it does | Persistent storage | Creds (.env.local or prompted) |
|----------|--------------|--------------------|--------------------------------|
| **Lightning.ai** | starts/reuses a **Studio on a T4** via `lightning_sdk`, syncs the repo, bootstraps data, trains | Studio home (**100 GB**, free T4 hrs) | `LIGHTNING_USER_ID`, `LIGHTNING_API_KEY`, `LIGHTNING_TEAMSPACE?`, `REPO_URL?` |
| **Kaggle** | generates + pushes a **GPU (T4) kernel** (`cloud/kaggle/`) that runs the pipeline | Kaggle Datasets | `KAGGLE_USERNAME`, `KAGGLE_KEY`, `REPO_URL` |
| **Google Colab** | opens `cloud/colab/ps12_colab.ipynb` which mounts **Drive** and runs everything | Google Drive (`MyDrive/ps12`) | prompted in the notebook (MOSDAC/HF) |

```bash
python connect.py                                   # interactive menu (verify + serve)
python connect.py --provider lightning --serve      # connect to a ready Studio, serve the dashboard
python connect.py --provider lightning --bootstrap --train   # one-time: set up data + train
python connect.py --provider colab                  # open the Drive-backed notebook
```

On **Lightning**, the simplest way to view the dashboard needs **no ngrok**: run
`streamlit run src/viz/dashboard.py` in the Studio and click the **Streamlit plugin's public link**
(or expose port 8501 via the ports plugin). ngrok is the universal fallback (and what Colab uses).

## Use the dashboard from your laptop while compute runs on the cloud
The UI runs on the remote GPU box; a tunnel exposes it to your local browser:
```bash
python connect.py --provider lightning --serve     # Lightning: serves + prints a public ngrok URL
# Colab: the notebook's last cell does the same (asks for your ngrok token)
# any box:  NGROK_AUTHTOKEN=... python cloud/serve_dashboard.py
```
Get a free ngrok token at https://dashboard.ngrok.com (put it in `.env.local` as `NGROK_AUTHTOKEN`).
Open the printed `https://….ngrok…` URL — you interact locally, the T4 does the work.

Notes
- Creds are read from `.env.local` (gitignored) or asked with `getpass` and saved there for reuse —
  never hardcoded or committed. See `.env.local.example` for the variable names.
- `REPO_URL` lets the remote `git clone` this repo. Push it to GitHub first, or (Lightning) upload via
  the Studio UI.
- The data check on the remote (`data/` or `samples/` present) decides whether to run `data_setup.py`,
  so you don't re-download into persistent storage every run.
- Lightning.ai docs: https://lightning.ai/docs/overview/studios/sdk · Kaggle API: https://www.kaggle.com/docs/api
