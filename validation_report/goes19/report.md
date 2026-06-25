# Validation report — interpolated vs ground truth (GOES-19, real run)

Generated on a Lightning T4 Studio. Validated on 5 triplets: input frames **20 min apart**, predict
the held-out **real 10-min middle** frame, score against ground truth (eval on the central crop,
256-tile). Models: classical TV-L1/Farnebäck, RAFT (pretrained), UNetVFI (ours, 2000-step quick train).

| model | psnr | ssim | fsim | edge_ssim | mse | mae_kelvin | lpips |
|---|---|---|---|---|---|---|---|
| classical | 37.5780 | 0.9435 | 0.9949 | 0.9193 | 0.0002 | 1.0877 | 0.1566 |
| raft | 36.6981 | 0.9448 | 0.9940 | 0.8965 | 0.0002 | 1.0938 | 0.1495 |
| unet | 33.5012 | 0.8870 | 0.9870 | 0.7758 | 0.0004 | 1.6735 | 0.2165 |

Higher PSNR/SSIM/FSIM/edge_SSIM = better; lower MSE/MAE(K)/LPIPS = better. MAE is in Kelvin (physical).
edge_SSIM targets cloud-edge structure (motion fidelity).

**Training:** UNetVFI, 2000 steps on a T4, best val PSNR 41.7 / SSIM 0.95 (step 750); checkpoint at
`weights/unet/best.pt` (on the Studio's persistent storage, not committed — too large).

**Honest read:** the lightly-trained (2000-step) UNetVFI **trails the optical-flow baselines** here.
That's expected for a quick run — more steps + more/varied GOES days (and the census/edge losses doing
their work over longer training) are needed for the learned model to surpass classical/RAFT. The
pipeline, metrics, and comparison are real and reproducible via `walkthrough.md` Step 8.

> Plots (`psnr_by_model.png`, `ssim_by_model.png`), the qualitative panel (`comparison_triplet0.png`),
> and the time-lapse GIF are produced alongside this report on the Studio; re-run `run_eval(...)` there
> to regenerate them (the lightning-sdk file-download was unreliable on Windows, so only this table is
> committed for now).
