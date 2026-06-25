# PINN ablation — physics-informed advection loss (real T4 run)

We added a **physics-informed loss** to UNetVFI (the advection equation in integral form, with a learned
source term `S` for cloud growth/dissipation) and trained **baseline vs PINN** for 1000 steps each on
GOES-19, then compared on **held-out** triplets (central crop, 256-tile).

| model (1000 steps, T4) | PSNR ✅ | SSIM ✅ |
|---|---|---|
| baseline UNetVFI | 34.83 | 0.8976 |
| **PINN UNetVFI** | **34.95** | **0.9000** |

**Result:** a small but **consistent** improvement (+0.12 dB PSNR, +0.0024 SSIM) from the physics loss
alone, at a very short training. (Internal val at step 1000 also favoured PINN: 33.88/0.878 vs
33.71/0.873.)

**Loss:** `L = L_pixel + λ·L_phys`, with
`L_phys = ‖ warp(I₀, −u) + S − I₂ ‖² + 0.05·|∇·u| + 0.05·|S|₁`, `u = F_{t→1} − F_{t→0}`.
The `warp(I₀,−u)+S` term is `∂I/∂t + u·∇I = S` (advection + source); `S` models the brightness change
classical optical flow (which assumes `S≡0`) cannot.

**Reproduce:**
```bash
python -m src.train.finetune --index data/index/goes19_triplets.json --steps 1000 --out weights/unet_base --device cuda
python -m src.train.finetune --index data/index/goes19_triplets.json --steps 1000 --pinn --out weights/unet_pinn --device cuda
```
Code: `src/train/losses.py: advection_physics_loss`, `src/models/unet_vfi.py` (source head + `return_aux`),
`src/train/finetune.py` (`--pinn`). Expected to widen with more steps + a tuned `--pinn-weight`.
