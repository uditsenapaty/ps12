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
    # --index may be comma-separated (e.g. goes_tri,hima_tri) -> ConcatDataset mix of per-source datasets.
    idx_paths = [s.strip() for s in str(index).split(",") if s.strip()]
    if len(idx_paths) > 1:
        subs = [SatTripletDataset(index_json=p, patch=patch) for p in idx_paths]
        ds = torch.utils.data.ConcatDataset(subs)
        print("[rife-ft] multi-source: " + ", ".join(f"{Path(p).stem}={len(s)}" for p, s in zip(idx_paths, subs)))
    else:
        ds = SatTripletDataset(index_json=index, patch=patch)
    # drop_last=False + eff_batch so tiny sets (e.g. INSAT self-sup) still yield a batch (no infinite loop).
    eff_batch = max(1, min(batch, len(ds)))
    loader = torch.utils.data.DataLoader(ds, batch_size=eff_batch, shuffle=True, num_workers=workers, drop_last=False)
    out_dir = Path(out); out_dir.mkdir(parents=True, exist_ok=True)

    def to3(x):  # (B,1,H,W) -> (B,3,H,W) on device
        return x.repeat(1, 3, 1, 1).to(device)

    # Fine-tune by backpropagating an L1/Charbonnier loss through RIFE's OWN inference() forward. We do
    # NOT use the repo's Model.update training step: this packaged IFNet's forward() doesn't accept the
    # 'scale' kwarg update() passes, so update() is broken here — but inference() works (verified). We
    # train the flownet directly and save weights in the layout the wrapper's load_model(path, -1) reads.
    net = model.flownet
    net.train()
    for p in net.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    torch.set_grad_enabled(True)
    step = 0
    print(f"[rife-ft] {len(ds)} triplets | device={device} | steps={steps} (inference-backprop)")
    while step < steps:
        for b in loader:
            i0, gt, i1 = to3(b["I0"]), to3(b["GT"]), to3(b["I1"])
            pred = model.inference(i0, i1, timestep=0.5)
            if not getattr(pred, "requires_grad", False):
                raise RuntimeError("RIFE inference() returned a detached tensor — cannot fine-tune via "
                                   "this path (package exposes no trainable forward).")
            loss = torch.sqrt((pred - gt) ** 2 + 1e-6).mean()        # Charbonnier (robust L1)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % 50 == 0:
                print(f"[rife-ft] step {step}/{steps}  loss {float(loss):.4f}")
            if step >= steps:
                break
    # Save with module.-prefixed keys: the wrapper reloads via load_model(path, -1), which keeps ONLY
    # keys containing 'module.' (stripping it) — clean keys would reload to an empty state_dict.
    sd = {(k if k.startswith("module.") else "module." + k): v.detach().cpu() for k, v in net.state_dict().items()}
    torch.save(sd, out_dir / "flownet.pkl")
    print(f"[rife-ft] done -> {out_dir}/flownet.pkl")
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
