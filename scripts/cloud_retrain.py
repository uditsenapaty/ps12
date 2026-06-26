#!/usr/bin/env python
"""Re-train UNetVFI on the Lightning Studio with chosen args, validate, and re-commit the results.

Drives the (already set-up) Studio from your machine: syncs it to GitHub, trains with YOUR arguments,
runs the validation report there, then fetches the report text and git-commits + pushes it locally
(the heavy .nc/weights stay on the Studio). Creds come from .env.local — edit those to use a different
account.

  python scripts/cloud_retrain.py --steps 8000 --pinn --source goes19 --commit
  python scripts/cloud_retrain.py --steps 3000 --batch 8            # no PINN, no commit (dry look)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.env import load_env  # noqa: E402

load_env()


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-train on the Lightning Studio + commit validation results")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--pinn", action="store_true", help="add the physics-informed (advection) loss")
    ap.add_argument("--pinn-weight", type=float, default=0.1)
    ap.add_argument("--anytime", action="store_true",
                    help="arbitrary-time training on variable-(gap, t) samples (30→15→7.5 ready)")
    ap.add_argument("--multigap", action="store_true",
                    help="temporal multi-granularity: each target supervised from symmetric gaps (combined loss)")
    ap.add_argument("--source", default="goes19", help="goes19 | himawari9 | insat3dr")
    ap.add_argument("--out", default="weights/unet")
    ap.add_argument("--models", default="classical,raft,unet", help="models to compare in the report")
    ap.add_argument("--max-triplets", type=int, default=20)
    ap.add_argument("--commit", action="store_true", help="fetch the report and git commit + push it")
    a = ap.parse_args()

    from lightning_sdk import Machine
    from cloud.lightning_exec import get_studio
    studio = get_studio()
    print("[retrain] starting Studio (T4, on-demand) …")
    try:
        studio.start(Machine.T4, interruptible=False)
    except TypeError:
        try:
            studio.start(Machine.T4)
        except Exception as e:
            print("[retrain] start:", e)
    except Exception as e:
        print("[retrain] start:", e)

    pinn = f"--pinn --pinn-weight {a.pinn_weight}" if a.pinn else ""
    at = "--anytime" if a.anytime else ""
    mg = "--multigap" if a.multigap else ""
    exp = a.source + ("_pinn" if a.pinn else "") + ("_at" if a.anytime else "") + ("_mg" if a.multigap else "")
    models = "[" + ",".join(f"'{m.strip()}'" for m in a.models.split(",")) + "]"

    extras = " ".join(x for x in ("+PINN" if a.pinn else "", "+anytime" if a.anytime else "",
                                  "+multigap" if a.multigap else "") if x)
    print(f"[retrain] training {a.source}: {a.steps} steps {extras} on the T4 …")
    print(studio.run(
        "cd ~/ps12 && git fetch -q && git reset --hard origin/main -q && git clean -fd -q 2>/dev/null; "
        f"python -m src.train.finetune --index data/index/{a.source}_triplets.json "
        f"--steps {a.steps} --batch {a.batch} {pinn} {at} {mg} --out {a.out} --device cuda 2>&1 | tail -6"))

    print("[retrain] validating …")
    print(studio.run(
        "cd ~/ps12 && python - <<'PY'\n"
        "import json\nfrom src.eval.report import run_eval\n"
        f"idx = json.load(open('data/index/{a.source}_triplets.json'))['triplets']\n"
        f"run_eval(idx, '{a.source}', {models}, 'validation_report/{exp}', max_triplets={a.max_triplets})\n"
        "PY"))

    if a.commit:
        print("[retrain] fetching report.md + committing locally …")
        report = studio.run(f"cd ~/ps12 && cat validation_report/{exp}/report.md")
        local = ROOT / "validation_report" / exp
        local.mkdir(parents=True, exist_ok=True)
        (local / "report.md").write_text(report.split("[lightning]")[-1].strip() + "\n", encoding="utf-8")
        subprocess.run(["git", "add", f"validation_report/{exp}/report.md"], cwd=ROOT)
        subprocess.run(["git", "commit", "-m",
                        f"validation: {a.source} retrain ({a.steps} steps{', +PINN' if a.pinn else ''})"], cwd=ROOT)
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT)
        print(f"[retrain] committed + pushed validation_report/{exp}/report.md")
    else:
        print("[retrain] done (no --commit; report stayed on the Studio).")


if __name__ == "__main__":
    main()
