"""Self-supervised INSAT adaptation.

INSAT has no 15-min ground truth, but its own 30-min cadence gives supervised triplets
(00:00, 00:30, 01:00) -> predict the real 00:30. Fine-tuning on these adapts the GOES/Himawari-trained
UNetVFI to INSAT's 4 km resolution, geometry, and slightly different band (10.8 µm) — with NO external
labels. Reuses the standard training loop with the INSAT leave-one-out index and a warm start.

  python -m src.train.insat_selfsup --index data/index/insat3dr_triplets.json \
        --init weights/unet/best.pt --out weights/unet_insat --steps 5000
"""
from __future__ import annotations

import argparse

from .finetune import train


def main():
    ap = argparse.ArgumentParser(description="Self-supervised INSAT adaptation of UNetVFI")
    ap.add_argument("--index", required=True, help="INSAT triplet index (30-min spacing)")
    ap.add_argument("--init", required=True, help="GOES/Himawari-trained weights to warm-start from")
    ap.add_argument("--out", default="weights/unet_insat")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    train(a.index, a.out, a.steps, a.lr, a.batch, a.patch, device=a.device,
          workers=a.workers, init_weights=a.init)


if __name__ == "__main__":
    main()
