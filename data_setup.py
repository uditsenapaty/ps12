#!/usr/bin/env python
"""data_setup.py — real data/environment bootstrap for the PS12 pipeline.

Runs on BOTH the local PC (small `--sample` pulls to learn structure) and the GPU server (bulk pulls
+ triplet index + weights). No GPU is touched here. Heavy execution (training, full-disk inference)
lives elsewhere; this script only fetches data, builds the index, and (optionally) installs deps.

Sources & official access methods
---------------------------------
  GOES-19   anonymous AWS S3  s3://noaa-goes19/ABI-L1b-RadF/<YYYY>/<DOY>/<HH>/  (channel C13)
  Himawari  anonymous AWS S3  s3://noaa-himawari9/AHI-L1b-FLDK/<YYYY>/<MM>/<DD>/<HHMM>/  (band B13)
  INSAT     MOSDAC SFTP       sftp://download.mosdac.gov.in:22  /Order/...  (3RIMG/3DIMG L1C_SGP .h5)
            Ordering is a manual web step (see walkthrough.md); this script downloads what was ordered.

Credentials (MOSDAC_USERNAME / MOSDAC_PASSWORD / HF_TOKEN) come from .env.local — never hardcoded.

Examples
--------
  python data_setup.py --download goes  --sample
  python data_setup.py --download insat --sample
  python data_setup.py --download goes  --start 2026-06-20 --end 2026-06-21 --max-gb 40
  python data_setup.py --build-index --source goes19 --step-min 10
  python data_setup.py --env            # install GPU deps (server only)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SAMPLES = ROOT / "samples"

GOES_BUCKET = "noaa-goes19"
HIMA_BUCKET = "noaa-himawari9"
MOSDAC_HOST = "download.mosdac.gov.in"
MOSDAC_PORT = 22
MOSDAC_REMOTE = "/Order"

CLONE_DIR = ROOT / "referred_clones"
# Deep backbones cloned for real inference/fine-tuning. RAFT is omitted — it ships in torchvision.
CLONE_REPOS = {
    "rife": "https://github.com/hzwer/Practical-RIFE",
    "superslomo": "https://github.com/avinashpaliwal/Super-SloMo",
    "film": "https://github.com/dajes/frame-interpolation-pytorch",  # Torch port of Google FILM
}


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def _human_gb(nbytes: int) -> float:
    return nbytes / (1024 ** 3)


def _dir_size_gb(path: Path) -> float:
    return _human_gb(sum(f.stat().st_size for f in path.rglob("*") if f.is_file())) if path.exists() else 0.0


def _s3_anon():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def _daterange(start: datetime, end: datetime):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


# --------------------------------------------------------------------------------------------------
# GOES-19 (anonymous S3)
# --------------------------------------------------------------------------------------------------
def download_goes(start: datetime, end: datetime, out: Path, channel: str = "C13",
                  max_gb: float = 50.0, sample_n: int | None = None) -> list[Path]:
    s3 = _s3_anon()
    out.mkdir(parents=True, exist_ok=True)
    got: list[Path] = []
    for day in _daterange(start, end):
        doy = day.timetuple().tm_yday
        for hour in range(24):
            prefix = f"ABI-L1b-RadF/{day.year}/{doy:03d}/{hour:02d}/"
            token = None
            while True:
                kw = {"Bucket": GOES_BUCKET, "Prefix": prefix, "MaxKeys": 1000}
                if token:
                    kw["ContinuationToken"] = token
                resp = s3.list_objects_v2(**kw)
                for obj in resp.get("Contents", []):
                    key = obj["Key"]
                    if f"-M6{channel}_" not in key and f"-M3{channel}_" not in key and f"{channel}_" not in key:
                        continue
                    if not key.endswith(".nc"):
                        continue
                    dest = out / Path(key).name
                    if not dest.exists():
                        if _dir_size_gb(out) > max_gb:
                            print(f"[goes] max-gb {max_gb} reached, stopping."); return got
                        print(f"[goes] {key}  ({_human_gb(obj['Size']):.2f} GB)")
                        s3.download_file(GOES_BUCKET, key, str(dest))
                    got.append(dest)
                    if sample_n and len(got) >= sample_n:
                        return got
                if resp.get("IsTruncated"):
                    token = resp["NextContinuationToken"]
                else:
                    break
    return got


# --------------------------------------------------------------------------------------------------
# Himawari-9 (anonymous S3, AHI L1b FLDK, HSD segments)
# --------------------------------------------------------------------------------------------------
def download_himawari(start: datetime, end: datetime, out: Path, band: str = "B13",
                      max_gb: float = 50.0, sample_n: int | None = None) -> list[Path]:
    s3 = _s3_anon()
    out.mkdir(parents=True, exist_ok=True)
    got: list[Path] = []
    for day in _daterange(start, end):
        prefix = f"AHI-L1b-FLDK/{day.year}/{day.month:02d}/{day.day:02d}/"
        token = None
        while True:
            kw = {"Bucket": HIMA_BUCKET, "Prefix": prefix, "MaxKeys": 1000}
            if token:
                kw["ContinuationToken"] = token
            resp = s3.list_objects_v2(**kw)
            for obj in resp.get("Contents", []):
                key = obj["Key"]
                if f"_{band}_" not in key:
                    continue
                dest = out / Path(key).name
                if not dest.exists():
                    if _dir_size_gb(out) > max_gb:
                        print(f"[himawari] max-gb {max_gb} reached, stopping."); return got
                    print(f"[himawari] {key}")
                    s3.download_file(HIMA_BUCKET, key, str(dest))
                got.append(dest)
                if sample_n and len(got) >= sample_n:
                    return got
            if resp.get("IsTruncated"):
                token = resp["NextContinuationToken"]
            else:
                break
    return got


# --------------------------------------------------------------------------------------------------
# INSAT (MOSDAC SFTP via paramiko — cross-platform; lftp recipe in walkthrough.md for the server)
# --------------------------------------------------------------------------------------------------
def download_insat(out: Path, max_gb: float = 50.0, sample_n: int | None = None,
                   remote_dir: str = MOSDAC_REMOTE) -> list[Path]:
    import paramiko  # noqa
    sys.path.insert(0, str(ROOT))
    from src.data.env import mosdac_credentials
    user, pwd = mosdac_credentials()
    out.mkdir(parents=True, exist_ok=True)
    got: list[Path] = []

    transport = paramiko.Transport((MOSDAC_HOST, MOSDAC_PORT))
    transport.connect(username=user, password=pwd)
    sftp = paramiko.SFTPClient.from_transport(transport)
    try:
        def walk(remote: str):
            for entry in sftp.listdir_attr(remote):
                rpath = f"{remote}/{entry.filename}"
                from stat import S_ISDIR
                if S_ISDIR(entry.st_mode):
                    yield from walk(rpath)
                elif entry.filename.lower().endswith((".h5", ".hdf", ".hdf5")):
                    yield rpath, entry.st_size

        for rpath, size in walk(remote_dir):
            dest = out / Path(rpath).name
            if not dest.exists():
                if _dir_size_gb(out) > max_gb:
                    print(f"[insat] max-gb {max_gb} reached, stopping."); break
                print(f"[insat] {rpath}  ({_human_gb(size):.2f} GB)")
                sftp.get(rpath, str(dest))
            got.append(dest)
            if sample_n and len(got) >= sample_n:
                break
    finally:
        sftp.close()
        transport.close()
    return got


# --------------------------------------------------------------------------------------------------
# triplet index
# --------------------------------------------------------------------------------------------------
def build_index(data_dir: Path, source: str, step_min: float, out_json: Path) -> dict:
    sys.path.insert(0, str(ROOT))
    from src.data.triplets import index_frames, build_triplets, build_leave_one_out
    files = [p for p in data_dir.rglob("*") if p.suffix.lower() in (".nc", ".h5", ".hdf", ".hdf5")]
    indexed = index_frames(files, source)
    trips = build_triplets(indexed, step_min)
    loo = build_leave_one_out(indexed, step_min)
    index = {
        "source": source,
        "step_min": step_min,
        "n_frames": len(indexed),
        "n_triplets": len(trips),
        "n_leave_one_out": len(loo),
        "triplets": [[str(a), str(b), str(c)] for a, b, c in trips],
        "leave_one_out": [[str(a), str(b), str(c)] for a, b, c in loo],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"[index] {source}: {len(indexed)} frames -> {len(trips)} triplets, "
          f"{len(loo)} leave-one-out -> {out_json}")
    return index


def install_gpu_env() -> None:
    import subprocess
    req = ROOT / "requirements-gpu.txt"
    print("[env] installing GPU requirements (ensure CUDA torch is installed first — see file header)")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])


def clone_models(names: list[str] | None = None) -> None:
    """Clone deep-model repos into referred_clones/ and strip their nested .git so the code commits
    to THIS repo (heavy artifacts are gitignored). Idempotent."""
    import shutil
    import subprocess
    CLONE_DIR.mkdir(parents=True, exist_ok=True)
    for name in (names or list(CLONE_REPOS)):
        url = CLONE_REPOS[name]
        dest = CLONE_DIR / name
        if dest.exists() and any(dest.rglob("*.py")):
            print(f"[clone] {name}: already present, skipping")
            continue
        print(f"[clone] {name} <- {url}")
        subprocess.check_call(["git", "clone", "--depth", "1", url, str(dest)])
        gitdir = dest / ".git"
        if gitdir.exists():
            shutil.rmtree(gitdir, ignore_errors=True)  # untrack: make files committable here
        print(f"[clone] {name}: ready ({dest})  weights -> weights/{name}/ (see walkthrough.md)")


# --------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="PS12 data/environment bootstrap")
    ap.add_argument("--download", choices=["goes", "himawari", "insat"], help="source to download")
    ap.add_argument("--sample", action="store_true", help="grab only a few frames to learn structure")
    ap.add_argument("--start", type=str, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", type=str, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--max-gb", type=float, default=50.0, help="stop once the target dir exceeds this")
    ap.add_argument("--build-index", action="store_true")
    ap.add_argument("--source", default="goes19", help="source tag for --build-index")
    ap.add_argument("--step-min", type=float, default=10.0, help="triplet spacing in minutes")
    ap.add_argument("--env", action="store_true", help="install GPU requirements (server)")
    ap.add_argument("--clone", nargs="?", const="all", help="clone deep-model repos into referred_clones/ "
                    "(all | rife | film | superslomo)")
    args = ap.parse_args()

    if args.env:
        install_gpu_env(); return

    if args.clone:
        clone_models(None if args.clone == "all" else [args.clone]); return

    if args.download:
        sample_n = 3 if args.sample else None
        base = SAMPLES if args.sample else DATA
        today = datetime.utcnow()
        start = datetime.strptime(args.start, "%Y-%m-%d") if args.start else today - timedelta(days=2)
        end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else start
        if args.download == "goes":
            files = download_goes(start, end, base / "goes19", max_gb=args.max_gb, sample_n=sample_n)
        elif args.download == "himawari":
            files = download_himawari(start, end, base / "himawari9", max_gb=args.max_gb, sample_n=sample_n)
        else:
            files = download_insat(base / "insat", max_gb=args.max_gb, sample_n=sample_n)
        print(f"[done] {len(files)} files in {base / args.download}")
        return

    if args.build_index:
        data_dir = (SAMPLES if (SAMPLES / args.source.replace('insat3dr', 'insat')).exists() else DATA)
        build_index(data_dir, args.source, args.step_min, DATA / "index" / f"{args.source}_triplets.json")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
