# Validation report — interpolated vs ground truth

Validated on 12 triplets: input frames 20 min apart, predict the held-out real 10-min middle, score against ground truth (INSAT: leave-one-out at 30 min).

| model | psnr | ssim | fsim | edge_ssim | mse | mae_kelvin | lpips |
|---|---|---|---|---|---|---|---|
| classical | 36.7359 | 0.9354 | 0.9919 | 0.9156 | 0.0002 | 1.2699 | 0.1605 |
| film | 37.0724 | 0.9476 | 0.9930 | 0.9167 | 0.0002 | 1.1139 | 0.1040 |
| raft | 35.8652 | 0.9295 | 0.9908 | 0.8927 | 0.0003 | 1.3277 | 0.1645 |
| superslomo | 34.5191 | 0.9280 | 0.9870 | 0.8806 | 0.0004 | 1.4134 | 0.1717 |
| unet | 33.5208 | 0.8777 | 0.9863 | 0.8013 | 0.0004 | 1.8174 | 0.1941 |

Higher PSNR/SSIM/FSIM/edge_SSIM = better; lower MSE/MAE(K)/LPIPS = better.
MAE is in Kelvin (physical). edge_SSIM targets cloud-edge structure (motion fidelity).