# SETUP3 → validate on HIMAWARI9 (20-min gap, leave-one-out)

Held-out chronological split: trained on the earlier frames, scored on 10 later HIMAWARI9 triplets it never saw (input frames 20-min apart, predict the real middle).

| model | psnr | ssim | fsim | edge_ssim | mse | mae_kelvin | lpips |
|---|---|---|---|---|---|---|---|
| rife_ft (finetuned · setup3) | 37.5874 | 0.9483 | 0.9971 | 0.9304 | 0.0002 | 1.1099 | 0.1367 |
| film (zero-shot) | 37.3587 | 0.9480 | 0.9972 | 0.9226 | 0.0002 | 1.1103 | 0.1051 |
| rife (zero-shot) | 37.1175 | 0.9428 | 0.9970 | 0.9187 | 0.0002 | 1.1719 | 0.1180 |
| classical (zero-shot) | 36.7804 | 0.9329 | 0.9966 | 0.9176 | 0.0002 | 1.2787 | 0.1616 |
| raft (zero-shot) | 36.1742 | 0.9310 | 0.9965 | 0.9004 | 0.0002 | 1.3041 | 0.1630 |
| unet (ours · GOES+Hima → INSAT self-sup) | 36.3133 | 0.9275 | 0.9964 | 0.9059 | 0.0002 | 1.3386 | 0.2182 |
| superslomo (zero-shot) | 35.1081 | 0.9271 | 0.9944 | 0.8880 | 0.0003 | 1.3962 | 0.1797 |

**Rank (Borda avg across metrics, 1 = best):** rife_ft (finetuned · setup3) (1.43), film (zero-shot) (1.71), rife (zero-shot) (2.86), classical (zero-shot) (4.00), raft (zero-shot) (5.43), unet (ours · GOES+Hima → INSAT self-sup) (5.71), superslomo (zero-shot) (6.86)

Higher PSNR/SSIM/FSIM/edge_SSIM = better; lower MSE/MAE(K)/LPIPS = better. MAE in Kelvin.

- Zero-shot = classical (algorithmic) + pretrained nets used frozen (no training on satellite IR). Runnable: classical, film, raft, rife, superslomo.
- Finetuned baseline = RIFE fine-tuned on satellite IR the SAME 3 ways as our model (GOES / Himawari / GOES+Himawari → INSAT self-sup). FILM/Super-SloMo have no fine-tune loop in this repo, so a 'finetuned FILM/SloMo' would be fabricated and is omitted.
