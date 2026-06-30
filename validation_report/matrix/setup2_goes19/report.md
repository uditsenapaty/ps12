# SETUP2 → validate on GOES19 (20-min gap, leave-one-out)

Held-out chronological split: trained on the earlier frames, scored on 12 later GOES19 triplets it never saw (input frames 20-min apart, predict the real middle).

| model | psnr | ssim | fsim | edge_ssim | mse | mae_kelvin | lpips |
|---|---|---|---|---|---|---|---|
| rife_ft (finetuned · setup2) | 38.1608 | 0.9526 | 0.9938 | 0.9277 | 0.0002 | 1.0191 | 0.1287 |
| film (zero-shot) | 37.7581 | 0.9497 | 0.9935 | 0.9176 | 0.0002 | 1.0455 | 0.1076 |
| rife (zero-shot) | 37.5413 | 0.9451 | 0.9929 | 0.9136 | 0.0002 | 1.1047 | 0.1194 |
| unet (ours · Himawari-trained) | 37.6108 | 0.9443 | 0.9934 | 0.9174 | 0.0002 | 1.1135 | 0.1614 |
| classical (zero-shot) | 37.3732 | 0.9388 | 0.9924 | 0.9150 | 0.0002 | 1.1874 | 0.1606 |
| raft (zero-shot) | 36.4240 | 0.9333 | 0.9914 | 0.8917 | 0.0002 | 1.2403 | 0.1671 |
| superslomo (zero-shot) | 35.5372 | 0.9316 | 0.9890 | 0.8808 | 0.0003 | 1.3037 | 0.1763 |

**Rank (Borda avg across metrics, 1 = best):** rife_ft (finetuned · setup2) (1.29), film (zero-shot) (1.86), rife (zero-shot) (3.57), unet (ours · Himawari-trained) (3.57), classical (zero-shot) (4.71), raft (zero-shot) (6.00), superslomo (zero-shot) (7.00)

Higher PSNR/SSIM/FSIM/edge_SSIM = better; lower MSE/MAE(K)/LPIPS = better. MAE in Kelvin.

- Zero-shot = classical (algorithmic) + pretrained nets used frozen (no training on satellite IR). Runnable: classical, film, raft, rife, superslomo.
- Finetuned baseline = RIFE fine-tuned on satellite IR the SAME 3 ways as our model (GOES / Himawari / GOES+Himawari → INSAT self-sup). FILM/Super-SloMo have no fine-tune loop in this repo, so a 'finetuned FILM/SloMo' would be fabricated and is omitted.
