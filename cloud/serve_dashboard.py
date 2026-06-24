#!/usr/bin/env python
"""Serve the Streamlit dashboard from a remote box (Colab / Lightning / any server) and expose it to
YOUR local browser through a public tunnel.

Why: the heavy compute (GPU, data) lives on the remote, but you want to click around the UI from your
PC. This starts Streamlit headless and opens an ngrok tunnel (preferred) or a localtunnel fallback,
then prints the public URL to paste into your browser.

  NGROK_AUTHTOKEN=...  python cloud/serve_dashboard.py            # ngrok (free token: dashboard.ngrok.com)
  python cloud/serve_dashboard.py --port 8501 --tunnel localtunnel
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "viz" / "dashboard.py"


def _start_streamlit(port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(APP),
         "--server.port", str(port), "--server.headless", "true", "--server.address", "0.0.0.0"],
        cwd=str(ROOT),
    )


def _ngrok(port: int) -> str | None:
    try:
        from pyngrok import conf, ngrok
    except Exception:
        subprocess.call([sys.executable, "-m", "pip", "install", "-q", "pyngrok"])
        try:
            from pyngrok import conf, ngrok
        except Exception as e:
            print(f"[serve] pyngrok unavailable: {e}")
            return None
    token = os.environ.get("NGROK_AUTHTOKEN")
    if token:
        conf.get_default().auth_token = token
    try:
        return str(ngrok.connect(port, "http").public_url)
    except Exception as e:
        print(f"[serve] ngrok failed ({e}); set NGROK_AUTHTOKEN or use --tunnel localtunnel.")
        return None


def serve(port: int = 8501, tunnel: str = "ngrok") -> None:
    proc = _start_streamlit(port)
    time.sleep(6)  # let Streamlit bind the port
    url = _ngrok(port) if tunnel == "ngrok" else None
    if url:
        print("\n" + "=" * 64 + f"\n🌐  Open the dashboard in your browser:\n    {url}\n" + "=" * 64 + "\n")
        try:
            proc.wait()
        except KeyboardInterrupt:
            pass
    else:
        print("[serve] starting localtunnel (npx localtunnel) — install Node if missing…")
        subprocess.call(["npx", "localtunnel", "--port", str(port)])
        proc.wait()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Serve the PS12 dashboard via a public tunnel")
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--tunnel", choices=["ngrok", "localtunnel"], default="ngrok")
    serve(**vars(ap.parse_args()))
