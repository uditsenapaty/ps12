# cloud/ — connect to a T4 GPU + persistent storage

`python connect.py` (repo root) is the single entry point. It prompts for the creds of the provider you
pick (or reads them from `.env.local`), connects to a box that has **already completed `walkthrough.md`**,
**verifies that data + trained checkpoints exist** in persistent storage, and serves the dashboard. If
the box isn't set up yet it **errors** (checkpoints are large and not committed — they live only in the
server's persistent storage). Use `--bootstrap` for a one-time setup (clone models + download data).

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
