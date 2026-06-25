#!/usr/bin/env python
"""Run a shell command on the ps12 Lightning Studio (for setup/training/debugging).

Reads creds from .env.local. Reconnects to the named Studio (creating/starting it on a T4 if needed),
runs the command in the persistent home, and prints the output. Used to drive walkthrough.md remotely.

  python cloud/lightning_exec.py --start                       # just start the Studio
  python cloud/lightning_exec.py "nvidia-smi -L"               # run a command
  python cloud/lightning_exec.py --start "cd ps12 && pytest -q tests/"
  python cloud/lightning_exec.py --stop                        # stop the Studio (save hours)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.env import load_env  # noqa: E402

load_env()


def get_studio():
    from lightning_sdk import Studio
    name = os.environ.get("LIGHTNING_STUDIO", "ps12")
    teamspace = os.environ.get("LIGHTNING_TEAMSPACE") or None
    org = os.environ.get("LIGHTNING_ORG") or None
    user = os.environ.get("LIGHTNING_USER") or None
    # try the richest signature first, then fall back as the SDK allows
    for kwargs in (
        {"name": name, "teamspace": teamspace, "org": org},
        {"name": name, "teamspace": teamspace, "user": user},
        {"name": name, "teamspace": teamspace},
        {"name": name},
    ):
        try:
            from lightning_sdk import Studio
            return Studio(create_ok=True, **{k: v for k, v in kwargs.items() if v is not None})
        except Exception as e:
            last = e
    raise RuntimeError(f"could not open Studio '{name}': {last}")


def main():
    args = sys.argv[1:]
    do_start = "--start" in args
    do_stop = "--stop" in args
    cmd = " ".join(a for a in args if not a.startswith("--")).strip()

    studio = get_studio()
    if do_start:
        from lightning_sdk import Machine
        print(f"[lightning] starting Studio on T4 …")
        try:
            studio.start(Machine.T4)
        except Exception as e:
            print(f"[lightning] start note: {e} (may already be running)")
    if cmd:
        print(f"[lightning] $ {cmd}")
        print(studio.run(cmd))
    if do_stop:
        print("[lightning] stopping Studio …")
        studio.stop()


if __name__ == "__main__":
    main()
