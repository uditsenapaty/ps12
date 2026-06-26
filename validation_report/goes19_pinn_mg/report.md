# Validation report — interpolated vs ground truth

Validated on 20 triplets: input frames 20 min apart, predict the held-out real 10-min middle, score against ground truth (INSAT: leave-one-out at 30 min).

| model | psnr | ssim | fsim | edge_ssim | mse | mae_kelvin | lpips |
|---|---|---|---|---|---|---|---|
| classical | 36.9731 | 0.9345 | 0.9938 | 0.9140 | 0.0002 | 1.2364 | 0.1682 |
| raft | 36.2591 | 0.9343 | 0.9929 | 0.8938 | 0.0002 | 1.2458 | 0.1626 |
| unet | 36.8444 | 0.9385 | 0.9939 | 0.9069 | 0.0002 | 1.1836 | 0.1647 |

Higher PSNR/SSIM/FSIM/edge_SSIM = better; lower MSE/MAE(K)/LPIPS = better.
MAE is in Kelvin (physical). edge_SSIM targets cloud-edge structure (motion fidelity).
