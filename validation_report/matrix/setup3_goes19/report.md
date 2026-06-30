# SETUP3 → validate on GOES19 (20-min gap, leave-one-out)

Held-out chronological split: trained on the earlier frames, scored on 12 later GOES19 triplets it never saw (input frames 20-min apart, predict the real middle).

| model | psnr | ssim | fsim | edge_ssim | mse | mae_kelvin | lpips |
|---|---|---|---|---|---|---|---|
| rife_ft (finetuned · setup3) | 38.0487 | 0.9508 | 0.9934 | 0.9264 | 0.0002 | 1.0401 | 0.1403 |
| film (zero-shot) | 37.7581 | 0.9497 | 0.9935 | 0.9176 | 0.0002 | 1.0455 | 0.1076 |
| rife (zero-shot) | 37.5413 | 0.9451 | 0.9929 | 0.9136 | 0.0002 | 1.1047 | 0.1194 |
| classical (zero-shot) | 37.3732 | 0.9388 | 0.9924 | 0.9150 | 0.0002 | 1.1874 | 0.1606 |
| unet (ours · GOES+Hima → INSAT self-sup) | 37.0197 | 0.9339 | 0.9919 | 0.9058 | 0.0002 | 1.2417 | 0.2216 |
| raft (zero-shot) | 36.4240 | 0.9333 | 0.9914 | 0.8917 | 0.0002 | 1.2403 | 0.1671 |
| superslomo (zero-shot) | 35.5372 | 0.9316 | 0.9890 | 0.8808 | 0.0003 | 1.3037 | 0.1763 |

**Rank (Borda avg across metrics, 1 = best):** rife_ft (finetuned · setup3) (1.43), film (zero-shot) (1.71), rife (zero-shot) (3.00), classical (zero-shot) (3.86), unet (ours · GOES+Hima → INSAT self-sup) (5.43), raft (zero-shot) (5.71), superslomo (zero-shot) (6.86)

Higher PSNR/SSIM/FSIM/edge_SSIM = better; lower MSE/MAE(K)/LPIPS = better. MAE in Kelvin.

- Zero-shot = classical (algorithmic) + pretrained nets used frozen (no training on satellite IR). Runnable: classical, film, raft, rife, superslomo.
- Finetuned baseline = RIFE fine-tuned on satellite IR the SAME 3 ways as our model (GOES / Himawari / GOES+Himawari → INSAT self-sup). FILM/Super-SloMo have no fine-tune loop in this repo, so a 'finetuned FILM/SloMo' would be fabricated and is omitted.
