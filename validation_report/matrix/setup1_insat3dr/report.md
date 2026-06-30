# SETUP1 → validate on INSAT3DR (60-min gap, leave-one-out)

Held-out chronological split: trained on the earlier frames, scored on 3 later INSAT3DR triplets it never saw (input frames 60-min apart, predict the real middle).

| model | psnr | ssim | fsim | edge_ssim | mse | mae_kelvin | lpips |
|---|---|---|---|---|---|---|---|
| rife_ft (finetuned · setup1) | 28.6282 | 0.8013 | 0.9351 | 0.7418 | 0.0014 | 3.3030 | 0.2530 |
| film (zero-shot) | 28.3845 | 0.8010 | 0.9372 | 0.7306 | 0.0015 | 3.3472 | 0.2235 |
| rife (zero-shot) | 28.5018 | 0.7918 | 0.9351 | 0.7285 | 0.0014 | 3.4024 | 0.2460 |
| classical (zero-shot) | 28.4561 | 0.7849 | 0.9319 | 0.7298 | 0.0014 | 3.4662 | 0.2726 |
| raft (zero-shot) | 28.2723 | 0.7856 | 0.9294 | 0.7209 | 0.0015 | 3.5142 | 0.2857 |
| superslomo (zero-shot) | 27.4410 | 0.7626 | 0.9198 | 0.6718 | 0.0018 | 3.7213 | 0.2935 |
| unet (ours · GOES-trained) | 26.8765 | 0.7351 | 0.9115 | 0.6161 | 0.0021 | 3.9824 | 0.2971 |

**Rank (Borda avg across metrics, 1 = best):** rife_ft (finetuned · setup1) (1.43), film (zero-shot) (2.29), rife (zero-shot) (2.71), classical (zero-shot) (3.71), raft (zero-shot) (4.86), superslomo (zero-shot) (6.00), unet (ours · GOES-trained) (7.00)

Higher PSNR/SSIM/FSIM/edge_SSIM = better; lower MSE/MAE(K)/LPIPS = better. MAE in Kelvin.

- Zero-shot = classical (algorithmic) + pretrained nets used frozen (no training on satellite IR). Runnable: classical, film, raft, rife, superslomo.
- Finetuned baseline = RIFE fine-tuned on satellite IR the SAME 3 ways as our model (GOES / Himawari / GOES+Himawari → INSAT self-sup). FILM/Super-SloMo have no fine-tune loop in this repo, so a 'finetuned FILM/SloMo' would be fabricated and is omitted.
