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
    # NOTE: this account has no org (org_id=None); "2305745" is the USERNAME, not an org.
    os.environ.pop("LIGHTNING_ORG", None)
    user = os.environ.get("LIGHTNING_USER") or os.environ.get("LIGHTNING_ORG") or None
    if not user:  # auto-resolve the username from the authenticated account
        try:
            from lightning_sdk.lightning_cloud.rest_client import LightningClient
            user = getattr(LightningClient().auth_service_get_user(), "username", None)
        except Exception:
            pass

    attempts: list[str] = []
    for kwargs in (
        {"name": name, "teamspace": teamspace, "user": user},  # personal teamspace under the user
        {"name": name, "user": user},                          # default teamspace, user-owned
        {"name": name, "teamspace": teamspace},
        {"name": name},
    ):
        kw = {k: v for k, v in kwargs.items() if v}
        try:
            return Studio(create_ok=True, **kw)
        except Exception as e:
            attempts.append(f"{kw} -> {e}")
    raise RuntimeError("could not open Studio. Tried:\n  " + "\n  ".join(attempts))


def whoami():
    """Print this account's username + every teamspace it can see (name + owner) so we can set
    LIGHTNING_TEAMSPACE / LIGHTNING_USER / LIGHTNING_ORG correctly. No Studio boot."""
    from lightning_sdk.lightning_cloud.rest_client import LightningClient
    c = LightningClient()
    try:
        u = c.auth_service_get_user()
        print("USERNAME:", getattr(u, "username", None), "| user_id:", getattr(u, "id", None))
    except Exception as e:
        print("get_user err:", e)
    listed = False
    for meth in ("projects_service_list_memberships", "projects_service_list_projects"):
        try:
            res = getattr(c, meth)()
            items = getattr(res, "memberships", None) or getattr(res, "projects", None) or []
            for m in items:
                print("TEAMSPACE:", getattr(m, "name", None),
                      "| display:", getattr(m, "display_name", None),
                      "| owner_id:", getattr(m, "owner_id", None),
                      "| org_id:", getattr(m, "organization_id", None) or getattr(m, "org_id", None))
            listed = True
            break
        except Exception as e:
            print(f"{meth} err:", e)
    if not listed:
        print("[lightning] could not list teamspaces via known methods; tell me your Lightning username "
              "+ teamspace from the web URL: lightning.ai/<username-or-org>/<teamspace>")


def main():
    args = sys.argv[1:]
    if "--whoami" in args:
        whoami(); return
    if "--pull" in args:
        paths = [a for a in args if not a.startswith("--")]
        remote, local = paths[0], paths[1]
        from pathlib import Path as _P
        _P(local).mkdir(parents=True, exist_ok=True)
        studio = get_studio()
        for meth in ("download_folder", "download", "download_file"):
            fn = getattr(studio, meth, None)
            if fn:
                try:
                    fn(remote, local)
                    print(f"pulled {remote} -> {local} via {meth}()"); return
                except Exception as e:
                    print(f"{meth}() failed: {e}")
        print("no working download method on Studio"); return
    do_start = "--start" in args
    do_stop = "--stop" in args
    cmd = " ".join(a for a in args if not a.startswith("--")).strip()

    studio = get_studio()
    if do_start:
        from lightning_sdk import Machine
        # T4 = cheapest GPU on Lightning; on-demand (interruptible=False) = non-interruptible default.
        print("[lightning] starting Studio on T4 (on-demand / non-interruptible) …")
        try:
            studio.start(Machine.T4, interruptible=False)
        except TypeError:
            try:
                studio.start(Machine.T4)
            except Exception as e:
                print(f"[lightning] start note: {e} (may already be running)")
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
