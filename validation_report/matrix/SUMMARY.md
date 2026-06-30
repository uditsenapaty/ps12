# Validation matrix — 3 setups x {GOES, Himawari, INSAT}

Each cell trains OUR custom model one way, then scores every runnable model on a held-out chronological split of each satellite. Classical + pretrained rows are zero-shot (identical across setups). Finetuned baseline = RIFE fine-tuned on satellite IR the SAME 3 ways as our model (GOES / Himawari / GOES+Himawari → INSAT self-sup). FILM/Super-SloMo have no fine-tune loop in this repo, so a 'finetuned FILM/SloMo' would be fabricated and is omitted.

## setup1: GOES held-out — custom trained on GOES (20-min triplets)
- **GOES19** (20-min): best = `rife_ft (finetuned · setup1)`; ours rank = 3.43
- **HIMAWARI9** (20-min): best = `rife_ft (finetuned · setup1)`; ours rank = 5.86
- **INSAT3DR** (60-min): best = `rife_ft (finetuned · setup1)`; ours rank = 7.00

## setup2: HIMAWARI held-out — custom trained on Himawari (20-min triplets)
- **GOES19** (20-min): best = `rife_ft (finetuned · setup2)`; ours rank = 3.57
- **HIMAWARI9** (20-min): best = `rife_ft (finetuned · setup2)`; ours rank = 3.86
- **INSAT3DR** (60-min): best = `rife_ft (finetuned · setup2)`; ours rank = 6.14

## setup3: COMBINATION — GOES+Himawari multigap (20/40) then INSAT self-sup (60-min)
- **GOES19** (20-min): best = `rife_ft (finetuned · setup3)`; ours rank = 5.43
- **HIMAWARI9** (20-min): best = `rife_ft (finetuned · setup3)`; ours rank = 5.71
- **INSAT3DR** (60-min): best = `rife_ft (finetuned · setup3)`; ours rank = 3.43
