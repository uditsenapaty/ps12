"""Fine-tune the EXISTING pretrained RIFE on satellite triplets (real, server-only).

Drives the cloned RIFE repo's own training step (`Model.update`) with our SatTripletDataset, warm-started
from the pretrained checkpoint. Single-channel BT is replicated to RIFE's 3-channel input. Produces a
satellite-adapted RIFE alongside our custom UNetVFI, so the dashboard can offer *finetuned* RIFE vs the
custom model. Honestly gated: if the repo/weights/torch are missing it raises a clear error.

  python -m src.train.rife_finetune --index data/index/goes19_triplets.json \
        --weights weights/rife --out weights/rife_ft --steps 15000
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..models.vendor import add_to_path, has_vendor, has_weights, torch_available
from .dataset import SatTripletDataset


def _load_rife(weights_dir: str, device: str):
    add_to_path("rife")
    Model = None
    for imp in ("model.RIFE_HDv3", "train_log.RIFE_HDv3", "RIFE_HDv3", "model.RIFE_HD"):
        try:
            Model = __import__(imp, fromlist=["Model"]).Model
            break
        except Exception:
            continue
    if Model is None:
        raise RuntimeError("RIFE Model not found in referred_clones/rife (run: python data_setup.py --clone rife).")
    m = Model()
    if has_weights("rife") or Path(weights_dir).exists():
        try:
            m.load_model(weights_dir, -1)
        except TypeError:
            m.load_model(weights_dir)
    m.train()
    return m


def finetune(index: str, weights: str, out: str, steps: int = 15000, lr: float = 1e-5,
             batch: int = 8, patch: int = 256, device: str | None = None, workers: int = 0) -> Path:
    if not torch_available():
        raise RuntimeError("torch not available — run on the GPU server.")
    if not has_vendor("rife"):
        raise RuntimeError("RIFE repo missing — python data_setup.py --clone rife")
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_rife(weights, device)
    ds = SatTripletDataset(index_json=index, patch=patch)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers, drop_last=True)
    out_dir = Path(out); out_dir.mkdir(parents=True, exist_ok=True)

    def to3(x):  # (B,1,H,W) -> (B,3,H,W) on device
        return x.repeat(1, 3, 1, 1).to(device)

    step = 0
    print(f"[rife-ft] {len(ds)} triplets | device={device} | steps={steps}")
    while step < steps:
        for b in loader:
            i0, gt, i1 = to3(b["I0"]), to3(b["GT"]), to3(b["I1"])
            imgs = torch.cat((i0, i1), 1)
            cur_lr = lr * (1 - step / steps)
            try:
                pred, info = model.update(imgs, gt, cur_lr, training=True)
            except TypeError:
                pred, info = model.update(torch.cat((imgs, gt), 1), cur_lr, training=True)
            step += 1
            if step % 50 == 0:
                loss = info.get("loss_l1", info.get("loss_G", 0.0)) if isinstance(info, dict) else 0.0
                print(f"[rife-ft] step {step}/{steps}  loss {float(loss):.4f}")
            if step % 1000 == 0 or step == steps:
                try:
                    model.save_model(str(out_dir), -1)
                except TypeError:
                    model.save_model(str(out_dir))
                print(f"[rife-ft] checkpoint -> {out_dir}")
            if step >= steps:
                break
    print(f"[rife-ft] done -> {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="Fine-tune pretrained RIFE on satellite triplets")
    ap.add_argument("--index", required=True)
    ap.add_argument("--weights", default="weights/rife", help="pretrained RIFE weights dir (warm start)")
    ap.add_argument("--out", default="weights/rife_ft")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    finetune(a.index, a.weights, a.out, a.steps, a.lr, a.batch, a.patch, a.device, a.workers)


if __name__ == "__main__":
    main()
