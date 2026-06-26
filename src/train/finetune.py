"""Train / fine-tune UNetVFI on satellite triplets (GOES-19 / Himawari).

Real end-to-end training loop: SatTripletDataset -> UNetVFI -> combined photometric+structure loss ->
AdamW, with periodic validation (PSNR/SSIM) and checkpointing. Runs on a single T4 in hours; a tiny
CPU `smoke_train` is exercised by the deterministic battery.

  python -m src.train.finetune --index data/index/goes19_triplets.json --steps 20000 --out weights/unet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..eval.metrics import psnr, ssim
from ..models.unet_vfi import build_net
from .dataset import MultiGapDataset, SatTripletDataset
from .losses import advection_physics_loss, combined_loss


def _loader(ds, batch_size, shuffle, workers=0):
    import torch
    # drop_last=False so small datasets still yield a batch (UNetVFI has no batchnorm) — avoids an
    # empty loader silently spinning the training loop forever.
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=workers, drop_last=False)


def validate(net, ds, device, n: int = 8) -> dict:
    import torch
    net.eval()
    ps, ss = [], []
    with torch.no_grad():
        for i in range(min(n, len(ds))):
            b = ds[i]
            gt = b["GT"].numpy()[0]
            if "LEFTS" in b:                      # symmetric multi-gap: score the tightest bracket (t=0.5)
                i0 = b["LEFTS"][0][None].to(device)
                i1 = b["RIGHTS"][0][None].to(device)
                t = 0.5
            else:
                i0 = b["I0"][None].to(device)
                i1 = b["I1"][None].to(device)
                t = float(b["t"])
            pred = net(i0, i1, t)[0, 0].cpu().numpy()
            ps.append(psnr(pred, gt))
            ss.append(ssim(pred, gt))
    net.train()
    return {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss))}


def _multigap_loss(net, gt, lefts, rights, mask, pinn: bool, pinn_weight: float):
    """Symmetric combined multi-gap loss: reconstruct the SAME target `gt` from each VALID symmetric
    bracket (lefts[:,L], rights[:,L]) at t=0.5 and average over valid brackets. lefts/rights=(B,M,1,H,W),
    mask=(B,M) flags which levels are real (boundary targets have fewer). Invalid views are skipped, so
    each sample contributes its own number of brackets."""
    import torch
    M = lefts.shape[1]
    total, count = 0.0, 0.0
    for L in range(M):
        idx = (mask[:, L] > 0.5).nonzero(as_tuple=True)[0]   # batch items that have level L
        if idx.numel() == 0:
            continue
        i0, i1, g = lefts[idx, L], rights[idx, L], gt[idx]
        if pinn:
            pred, f_t0, f_t1, _m, src = net(i0, i1, 0.5, return_aux=True)
            l = combined_loss(pred, g) + pinn_weight * advection_physics_loss(i0, i1, f_t0, f_t1, src)
        else:
            l = combined_loss(net(i0, i1, 0.5), g)
        total = total + l * idx.numel()                      # weight by count so the mean is exact
        count = count + idx.numel()
    return total / max(count, 1.0)


def train(index: str, out: str, steps: int = 20000, lr: float = 1e-4, batch: int = 8,
          patch: int = 256, base: int = 32, device: str | None = None, val_index: str | None = None,
          val_every: int = 500, workers: int = 0, init_weights: str | None = None,
          pinn: bool = False, pinn_weight: float = 0.1, anytime: bool = False,
          multigap: bool = False) -> Path:
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if multigap:
        ds = MultiGapDataset(index_json=index, patch=patch)
        val_ds = MultiGapDataset(index_json=val_index or index, patch=patch, augment=False)
        print(f"[train] temporal multi-granularity ON: {len(ds)} multi-gap groups "
              f"(same target supervised from several gaps, combined loss)")
    else:
        ds = SatTripletDataset(index_json=index, patch=patch, anytime=anytime)
        val_ds = SatTripletDataset(index_json=val_index or index, patch=patch, augment=False, anytime=anytime)
        if anytime:
            print(f"[train] arbitrary-time ON: {len(ds)} variable-(gap, t) samples")
    if len(ds) == 0:
        raise RuntimeError(f"No samples in {index}. Download more frames "
                           f"(e.g. `python data_setup.py --download goes --max-gb 5`) then rebuild the index.")
    eff_batch = max(1, min(batch, len(ds)))
    loader = _loader(ds, eff_batch, shuffle=True, workers=workers)

    net = build_net()(base).to(device).train()
    if init_weights and Path(init_weights).exists():
        net.load_state_dict(torch.load(init_weights, map_location=device)["model"], strict=False)
        print(f"[train] initialized from {init_weights}")
    if pinn:
        print(f"[train] PINN physics loss ON (weight {pinn_weight})")
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    out_dir = Path(out); out_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    step = 0
    print(f"[train] {len(ds)} triplets | device={device} | steps={steps}")
    while step < steps:
        for b in loader:
            gt = b["GT"].to(device)
            if multigap:                            # symmetric combined loss over gap levels -> same target
                loss = _multigap_loss(net, gt, b["LEFTS"].to(device), b["RIGHTS"].to(device),
                                      b["MASK"].to(device), pinn, pinn_weight)
            else:
                i0 = b["I0"].to(device)
                i1 = b["I1"].to(device)
                t = b["t"].to(device).float()       # per-sample time (0.5 for plain triplets)
                if pinn:
                    pred, f_t0, f_t1, _mask, source = net(i0, i1, t, return_aux=True)
                    loss = combined_loss(pred, gt) + pinn_weight * advection_physics_loss(i0, i1, f_t0, f_t1, source)
                else:
                    pred = net(i0, i1, t)
                    loss = combined_loss(pred, gt)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            step += 1
            if step % 50 == 0:
                print(f"[train] step {step}/{steps}  loss {loss.item():.4f}{'  (+PINN)' if pinn else ''}")
            if step % val_every == 0 or step == steps:
                m = validate(net, val_ds, device)
                print(f"[val] step {step}  psnr {m['psnr']:.3f}  ssim {m['ssim']:.4f}")
                ckpt = {"model": net.state_dict(), "step": step, "val": m, "base": base}
                torch.save(ckpt, out_dir / "last.pt")
                if m["ssim"] > best:
                    best = m["ssim"]
                    torch.save(ckpt, out_dir / "best.pt")
                    (out_dir / "best.json").write_text(json.dumps({"step": step, **m}, indent=2))
            if step >= steps:
                break
    print(f"[train] done. best ssim {best:.4f} -> {out_dir/'best.pt'}")
    return out_dir / "best.pt"


def smoke_train(steps: int = 2, size: int = 64) -> float:
    """Tiny real training step on synthetic data — CPU sanity used by the battery (returns final loss)."""
    import torch
    net = build_net()(base=8).train()
    opt = torch.optim.SGD(net.parameters(), lr=1e-2)
    rng = np.random.default_rng(0)
    last = 0.0
    for _ in range(steps):
        i0 = torch.from_numpy(rng.random((1, 1, size, size)).astype("float32"))
        i1 = torch.from_numpy(rng.random((1, 1, size, size)).astype("float32"))
        gt = (i0 + i1) / 2
        loss = combined_loss(net(i0, i1, 0.5), gt)
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.item())
    return last


def main():
    ap = argparse.ArgumentParser(description="Train/fine-tune UNetVFI on satellite triplets")
    ap.add_argument("--index", required=True)
    ap.add_argument("--val-index", default=None)
    ap.add_argument("--out", default="weights/unet")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--init", default=None, help="warm-start weights (.pt)")
    ap.add_argument("--pinn", action="store_true", help="add the physics-informed (advection) loss")
    ap.add_argument("--pinn-weight", type=float, default=0.1)
    ap.add_argument("--anytime", action="store_true",
                    help="arbitrary-time training on variable-(gap, t) samples (30→15→7.5 ready)")
    ap.add_argument("--multigap", action="store_true",
                    help="temporal multi-granularity: supervise each target from several gaps with a combined loss")
    a = ap.parse_args()
    train(a.index, a.out, a.steps, a.lr, a.batch, a.patch, a.base, a.device, a.val_index,
          val_every=a.val_every, workers=a.workers, init_weights=a.init,
          pinn=a.pinn, pinn_weight=a.pinn_weight, anytime=a.anytime, multigap=a.multigap)


if __name__ == "__main__":
    main()
