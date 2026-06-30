# SETUP1 → validate on HIMAWARI9 (20-min gap, leave-one-out)

Held-out chronological split: trained on the earlier frames, scored on 10 later HIMAWARI9 triplets it never saw (input frames 20-min apart, predict the real middle).

| model | psnr | ssim | fsim | edge_ssim | mse | mae_kelvin | lpips |
|---|---|---|---|---|---|---|---|
| rife_ft (finetuned · setup1) | 37.6665 | 0.9498 | 0.9973 | 0.9311 | 0.0002 | 1.0904 | 0.1247 |
| film (zero-shot) | 37.3587 | 0.9480 | 0.9972 | 0.9226 | 0.0002 | 1.1103 | 0.1051 |
| rife (zero-shot) | 37.1175 | 0.9428 | 0.9970 | 0.9187 | 0.0002 | 1.1719 | 0.1180 |
| classical (zero-shot) | 36.7804 | 0.9329 | 0.9966 | 0.9176 | 0.0002 | 1.2787 | 0.1616 |
| raft (zero-shot) | 36.1742 | 0.9310 | 0.9965 | 0.9004 | 0.0002 | 1.3041 | 0.1630 |
| unet (ours · GOES-trained) | 35.8064 | 0.9284 | 0.9962 | 0.8935 | 0.0003 | 1.3093 | 0.1628 |
| superslomo (zero-shot) | 35.1081 | 0.9271 | 0.9944 | 0.8880 | 0.0003 | 1.3962 | 0.1797 |

**Rank (Borda avg across metrics, 1 = best):** rife_ft (finetuned · setup1) (1.29), film (zero-shot) (1.86), rife (zero-shot) (2.86), classical (zero-shot) (4.00), raft (zero-shot) (5.14), unet (ours · GOES-trained) (5.86), superslomo (zero-shot) (7.00)

Higher PSNR/SSIM/FSIM/edge_SSIM = better; lower MSE/MAE(K)/LPIPS = better. MAE in Kelvin.

- Zero-shot = classical (algorithmic) + pretrained nets used frozen (no training on satellite IR). Runnable: classical, film, raft, rife, superslomo.
- Finetuned baseline = RIFE fine-tuned on satellite IR the SAME 3 ways as our model (GOES / Himawari / GOES+Himawari → INSAT self-sup). FILM/Super-SloMo have no fine-tune loop in this repo, so a 'finetuned FILM/SloMo' would be fabricated and is omitted.
