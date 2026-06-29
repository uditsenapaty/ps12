# Validation report — interpolated vs ground truth

Validated on 12 triplets: input frames 20 min apart, predict the held-out real 10-min middle, score against ground truth (INSAT: leave-one-out at 30 min).

| model | psnr | ssim | fsim | edge_ssim | mse | mae_kelvin | lpips |
|---|---|---|---|---|---|---|---|
| classical | 38.2697 | 0.9506 | 0.9938 | 0.9194 | 0.0001 | 0.9909 | 0.1502 |
| film | 38.8567 | 0.9619 | 0.9948 | 0.9270 | 0.0001 | 0.8557 | 0.0955 |
| raft | 36.9846 | 0.9445 | 0.9920 | 0.8922 | 0.0002 | 1.0668 | 0.1558 |
| superslomo | 35.5562 | 0.9459 | 0.9885 | 0.8880 | 0.0003 | 1.1104 | 0.1710 |
| unet | 38.3472 | 0.9541 | 0.9944 | 0.9176 | 0.0001 | 0.9514 | 0.1564 |

Higher PSNR/SSIM/FSIM/edge_SSIM = better; lower MSE/MAE(K)/LPIPS = better.
MAE is in Kelvin (physical). edge_SSIM targets cloud-edge structure (motion fidelity).