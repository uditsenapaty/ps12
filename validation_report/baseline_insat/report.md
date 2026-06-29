# Validation report — interpolated vs ground truth

Validated on 10 triplets: input frames 20 min apart, predict the held-out real 10-min middle, score against ground truth (INSAT: leave-one-out at 30 min).

| model | psnr | ssim | fsim | edge_ssim | mse | mae_kelvin | lpips |
|---|---|---|---|---|---|---|---|
| classical | 29.6238 | 0.8104 | 0.9563 | 0.6956 | 0.0011 | 2.8498 | 0.2789 |
| film | 29.4664 | 0.8199 | 0.9591 | 0.6935 | 0.0011 | 2.7633 | 0.2255 |
| raft | 29.4345 | 0.8120 | 0.9522 | 0.6791 | 0.0011 | 2.8655 | 0.2795 |
| superslomo | 28.5025 | 0.7930 | 0.9421 | 0.6278 | 0.0014 | 3.0685 | 0.2937 |
| unet | 27.7379 | 0.7463 | 0.9375 | 0.5285 | 0.0017 | 3.4693 | 0.2983 |

Higher PSNR/SSIM/FSIM/edge_SSIM = better; lower MSE/MAE(K)/LPIPS = better.
MAE is in Kelvin (physical). edge_SSIM targets cloud-edge structure (motion fidelity).