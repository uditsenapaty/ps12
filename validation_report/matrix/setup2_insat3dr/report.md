# SETUP2 → validate on INSAT3DR (60-min gap, leave-one-out)

Held-out chronological split: trained on the earlier frames, scored on 3 later INSAT3DR triplets it never saw (input frames 60-min apart, predict the real middle).

| model | psnr | ssim | fsim | edge_ssim | mse | mae_kelvin | lpips |
|---|---|---|---|---|---|---|---|
| rife_ft (finetuned · setup2) | 28.6882 | 0.8031 | 0.9359 | 0.7466 | 0.0014 | 3.2857 | 0.2550 |
| film (zero-shot) | 28.3845 | 0.8010 | 0.9372 | 0.7306 | 0.0015 | 3.3472 | 0.2235 |
| rife (zero-shot) | 28.5018 | 0.7918 | 0.9351 | 0.7285 | 0.0014 | 3.4024 | 0.2460 |
| classical (zero-shot) | 28.4561 | 0.7849 | 0.9319 | 0.7298 | 0.0014 | 3.4662 | 0.2726 |
| raft (zero-shot) | 28.2723 | 0.7856 | 0.9294 | 0.7209 | 0.0015 | 3.5142 | 0.2857 |
| unet (ours · Himawari-trained) | 27.8451 | 0.7699 | 0.9247 | 0.6895 | 0.0016 | 3.5946 | 0.2956 |
| superslomo (zero-shot) | 27.4410 | 0.7626 | 0.9198 | 0.6718 | 0.0018 | 3.7213 | 0.2935 |

**Rank (Borda avg across metrics, 1 = best):** rife_ft (finetuned · setup2) (1.43), film (zero-shot) (2.29), rife (zero-shot) (2.71), classical (zero-shot) (3.71), raft (zero-shot) (4.86), unet (ours · Himawari-trained) (6.14), superslomo (zero-shot) (6.86)

Higher PSNR/SSIM/FSIM/edge_SSIM = better; lower MSE/MAE(K)/LPIPS = better. MAE in Kelvin.

- Zero-shot = classical (algorithmic) + pretrained nets used frozen (no training on satellite IR). Runnable: classical, film, raft, rife, superslomo.
- Finetuned baseline = RIFE fine-tuned on satellite IR the SAME 3 ways as our model (GOES / Himawari / GOES+Himawari → INSAT self-sup). FILM/Super-SloMo have no fine-tune loop in this repo, so a 'finetuned FILM/SloMo' would be fabricated and is omitted.
