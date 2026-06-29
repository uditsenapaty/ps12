# Validation report — interpolated vs ground truth

Validated on 24 triplets: input frames 20 min apart, predict the held-out real 10-min middle, score against ground truth (INSAT: leave-one-out at 30 min).

| model | psnr | ssim | fsim | edge_ssim | mse | mae_kelvin | lpips |
|---|---|---|---|---|---|---|---|
| classical | 38.3029 | 0.9508 | 0.9936 | 0.9203 | 0.0001 | 0.9883 | 0.1501 |
| film | 38.9123 | 0.9615 | 0.9947 | 0.9280 | 0.0001 | 0.8582 | 0.0973 |
| raft | 37.0011 | 0.9440 | 0.9918 | 0.8927 | 0.0002 | 1.0715 | 0.1576 |
| superslomo | 35.4619 | 0.9457 | 0.9881 | 0.8891 | 0.0003 | 1.1125 | 0.1730 |
| unet | 38.7566 | 0.9574 | 0.9945 | 0.9256 | 0.0001 | 0.9075 | 0.1454 |

Higher PSNR/SSIM/FSIM/edge_SSIM = better; lower MSE/MAE(K)/LPIPS = better.
MAE is in Kelvin (physical). edge_SSIM targets cloud-edge structure (motion fidelity).