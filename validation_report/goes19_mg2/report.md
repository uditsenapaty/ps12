# Validation report — interpolated vs ground truth

Validated on 6 triplets: input frames 20 min apart, predict the held-out real 10-min middle, score against ground truth (INSAT: leave-one-out at 30 min).

Model: **UNetVFI trained with temporal multi-granularity (`--multigap`, level 2)** — each target supervised from symmetric gap brackets at t=0.5 with a combined loss. This run is a **pipeline verification only**: 150 steps on ~1.7 GB of GOES-19 (85 frames → 83 multi-gap groups) on a Lightning T4, so the model is intentionally undertrained and trails the baselines. It confirms the multigap path runs end-to-end and produces real, committable metrics.

| model | psnr | ssim | fsim | edge_ssim | mse | mae_kelvin | lpips |
|---|---|---|---|---|---|---|---|
| classical | 38.3454 | 0.9564 | 0.9959 | 0.9373 | 0.0001 | 1.0081 | 0.1357 |
| raft | 37.5136 | 0.9519 | 0.9954 | 0.9205 | 0.0002 | 1.0558 | 0.1400 |
| unet | 34.6991 | 0.9135 | 0.9926 | 0.8419 | 0.0003 | 1.4841 | 0.1995 |

Higher PSNR/SSIM/FSIM/edge_SSIM = better; lower MSE/MAE(K)/LPIPS = better.
MAE is in Kelvin (physical). edge_SSIM targets cloud-edge structure (motion fidelity).
