# PS-12 explained — from a flipbook to a custom neural network

A plain-English walkthrough of **what we're doing, how each method works step by step, how they compare,
and how to make it better.** No prior ML needed to follow the first half.

---

## 1. The problem, in one sentence

Weather satellites take a picture every so often — **INSAT every 30 minutes**, GOES/Himawari every
10 minutes. Storms, cyclones and fires change in **minutes**, so the important action happens *between*
two pictures. We want to **invent the missing in-between pictures** so the "video" of the sky is
smoother — **without launching a new satellite.**

> Think of a **flipbook**. If you only have every 3rd page, the motion looks jumpy. We draw the missing
> pages so it flows. The trick: each "page" is a real satellite image, and the missing page must look
> like a *real* satellite image of that exact moment.

---

## 2. What the satellite picture actually is (this matters)

It is **not** a normal colour photo. It is a **thermal-infrared** image at about **10 micrometres**
wavelength — the camera measures **how hot each spot is** (the "brightness temperature", in Kelvin).

- **Cold = high cloud tops** (storms), **warm = ground/sea**.
- It is a **single channel** (one number per pixel = temperature), like a grayscale image.
- A full image is huge (GOES is ~5424×5424 pixels).

So our job is really: *given the temperature map at 00:00 and 00:20, predict the temperature map at
00:10.* Everything below works on these temperature maps.

---

## 3. The common pipeline (the same 6 steps for every method)

Every method — traditional, fine-tuned, or custom — sits inside this pipeline. Only **Step 4** (the
"engine") changes.

| Step | Plain English | In the code |
|------|---------------|-------------|
| 1. **Read** | Open the satellite file (`.nc`/`.h5`) and pull out the temperature map. | `src/data/readers.py` |
| 2. **Calibrate** | Convert raw sensor numbers → real temperature in Kelvin (physics formula). | `radiance_to_bt_goes()` |
| 3. **Normalise** | Squash temperatures (≈180–330 K) into 0–1 so the maths is stable; remember how to undo it. | `src/data/normalize.py` |
| 4. **Interpolate (THE ENGINE)** | Look at two frames, figure out how clouds moved, and paint the in-between frame. | `src/models/…` |
| 5. **Stitch & un-normalise** | The image is processed in 256×256 tiles; blend them back seamlessly and convert 0–1 → Kelvin. | `src/data/tiling.py` |
| 6. **Save & show** | Write a new `.nc` file at the new time; measure quality; animate in the dashboard. | `src/infer`, `src/eval`, `src/viz` |

**Step 4 is the whole game.** Here are the three ways to do it.

---

## 4. Method 1 — The TRADITIONAL way (classical optical flow)

This is the decades-old method the problem statement says "fails with blur and artefacts."

### How it works, step by step
1. **Measure motion.** Compare frame A and frame B and compute, for every pixel, *which way and how far
   it moved* — a field of little arrows called **optical flow** (we use OpenCV's TV-L1 / Farnebäck).
   - Core assumption (the "brightness constancy" rule): *a point keeps the same brightness as it moves.*
   - Plain English: "this bright cloud edge here is the same cloud edge that's now over *there*."
2. **Move halfway.** To get the 00:10 frame, slide each pixel **half** of its arrow from A, and **half**
   back from B. (We use the Super-SloMo linear formula: the in-between motion is a simple blend of the
   forward and backward arrows.)
3. **Blend.** Average the two halfway-warped images into one in-between frame.

### Where it breaks
- Clouds **grow, shrink, and dissipate** — their brightness *changes*, breaking the "brightness
  constancy" rule → the arrows are wrong → **ghosting and blur**.
- The "move in a straight line" assumption fails for **swirling, accelerating** storm motion.
- It has **no idea what a cloud is** — it's pure geometry, no learning.

**Pros:** no training, no GPU, instant, fully explainable. **Cons:** blurry/ghosted on fast non-linear
cloud growth — exactly the events we care about.

---

## 5. Method 2 — ML with FINE-TUNED EXISTING models (RIFE / FILM / Super-SloMo / RAFT)

These are **neural networks built by other researchers for slow-motion video** (think turning 30 fps
phone video into 240 fps). They already "know" how to interpolate moving things. We **borrow them and
re-train (fine-tune) them on satellite images.**

### How they work, step by step
1. **They learned from millions of video frames.** Instead of *assuming* the brightness-constancy rule,
   they **learned** what realistic motion and in-between frames look like, from data.
2. **They estimate smarter motion.**
   - **RAFT** builds a "cost volume" — it compares every patch with every nearby patch to find the best
     match, giving very accurate arrows even for big jumps.
   - **RIFE** predicts the *intermediate* flow directly (the motion from the missing frame to each
     neighbour) — fast and tidy.
   - **FILM** is built for **large motion** (good for the big 30-min cloud jumps).
   - **Super-SloMo** also predicts a **visibility map** (what gets hidden/revealed as things move).
3. **They synthesise, not just average.** A small network *paints* the final pixels (fixing edges and
   occlusions) instead of blindly blending.
4. **We fine-tune on satellite IR.** We continue training them on GOES/Himawari triplets (frame A,
   real middle, frame B) so they adapt from "people and cars" to "clouds and storms"
   (`src/train/rife_finetune.py`). RAFT we use pretrained (it already gives excellent flow).

**Pros:** much better than classical on fast/non-linear motion; battle-tested; RAFT/RIFE give crisp
motion. **Cons:** big models, need a GPU; pretrained on *natural video* (RGB, everyday scenes), so they
need adapting to single-channel thermal-IR; less control over the design.

---

## 6. Method 3 — OUR CUSTOM model (UNetVFI), trained on satellite data

The problem statement specifically wants a model **trained on satellite data**. So we built our own
small network and trained it from scratch on thermal-IR — combining the best ideas from the models
above, tailored to one-channel satellite imagery.

### What it is
A compact **U-Net** (an encoder that zooms out to "understand the big picture", a decoder that zooms
back in to "draw the details", with shortcut "skip" links so fine detail isn't lost).

### How it works, step by step
1. **Look at both frames together.** Stack frame A and frame B and feed them in.
2. **Predict three things at once** (this is the novel combination):
   - **flow A→t** — how to drag frame A to the missing time *t* (RIFE-style intermediate flow),
   - **flow B→t** — how to drag frame B to time *t*,
   - **a visibility mask M** — *who to trust* at each pixel (Super-SloMo-style occlusion handling): if a
     cloud edge is covering something, trust the frame that can actually see it.
3. **Warp + blend.** Pull A and B to the middle time using their flows, then mix them using the mask:
   `prediction = M · warp(A) + (1−M) · warp(B)`.
4. **Train it on satellite IR.** Show it thousands of GOES/Himawari triplets; it adjusts itself to
   minimise the error vs the *real* middle frame (loss = robust pixel error + edge-sharpness + a
   texture/structure term tuned for smooth cloud gradients). `src/train/finetune.py`.
5. **Adapt to INSAT for free (self-supervised).** INSAT has no 15-min "answer key", but it *does* have
   its own 30-min frames — so we make triplets `(00:00, 00:30, 01:00)` and train it to predict the real
   `00:30`. No human labels needed. `src/train/insat_selfsup.py`.

### Why it's nice
- **Single-channel by design** (no RGB hacks), **small (~2–5M params)** → trains in *hours* on one free
  T4, runs fast.
- **Owns the whole design** — we can add satellite-specific features (Section 9).
- **Cross-satellite**: train where data is dense (GOES/Himawari), apply where it's sparse (INSAT).

**Honest caveat (measured):** with only a quick **2000-step** training it does **not yet beat** the
strong classical/RAFT baselines (numbers below). That's expected — a small net needs more training and
more varied data to overtake them. The pipeline and the path to get there are real.

---

## 7. Side-by-side comparison

| | **Traditional** (TV-L1) | **Fine-tuned existing** (RIFE/FILM/RAFT) | **Custom UNetVFI** (ours) |
|---|---|---|---|
| **Core idea** | geometry + "brightness stays constant" | *learned* motion + synthesis from video | *learned* flow + visibility, trained on IR |
| **Learns from data?** | ❌ no | ✅ yes (natural video → fine-tuned) | ✅ yes (satellite IR from scratch) |
| **Handles fast / non-linear cloud growth** | ✗ poor (blur, ghosts) | ✓ good | ✓ good (with enough training) |
| **Needs a GPU** | no | yes | yes (but small/cheap) |
| **Training data needed** | none | a lot (pretrained) + a little to fine-tune | satellite triplets (GOES/Himawari) |
| **Made for satellite IR?** | n/a | no (adapted) | **yes, by design** |
| **Control / extensibility** | low | low–medium | **high** (we can add bands, physics, etc.) |
| **Speed** | instant | fast | fast |
| **Explainability** | high | low | medium |
| **Our measured PSNR / SSIM** (5 GOES triplets) | **37.6 / 0.94** | **RAFT 36.7 / 0.94** | **33.5 / 0.89** (2000-step) |

### So which is "best"?
- **Right now, on our quick run:** classical and RAFT are the strongest, because RAFT's learned flow is
  excellent and classical does fine on the *gentle* motion in our small sample. **This is honest** —
  we're not hiding it.
- **Why the custom model still matters:** it is the only one **trained on satellite physics**, the only
  one we can **extend** (more bands, longer time context, physics priors), and the only one that
  **self-adapts to INSAT** without labels. With more training it is designed to pass the baselines —
  and it's tiny/cheap to run. The other models are the *yardstick* it must beat.
- **Best practical recipe:** use **RAFT/RIFE** as a strong, ready baseline today; develop **UNetVFI** as
  the satellite-native model that can be pushed further than any off-the-shelf net.

---

## 8. What "features" are we using now vs. the traditional method?

A **feature** = a piece of information the method gets to look at.

| | Traditional | Ours (now) |
|---|---|---|
| Spectral bands | **1** thermal-IR band (~10 µm) | **1** thermal-IR band (~10 µm) |
| Frames of context | 2 (before & after) | 2 (before & after) |
| Motion model | hand-coded, assumes constant brightness + straight lines | **learned** motion field |
| Occlusion / appearance change | **ignored** | **learned visibility mask** |
| Knowledge of "what a cloud is" | none | learned from data |

Key point: **we use the same raw inputs as the traditional method (one IR band, two frames)** — the
gain comes from *learning* the motion + the appearance changes instead of *assuming* them. That's the
honest, apples-to-apples improvement. The big future gains come from **adding more features** (next).

---

## 9. How can we make it better? (the roadmap)

### A. Use **more of the satellite's data** (the biggest lever)
Geostationary imagers see **many channels**, not just one. Each adds physical information:
- **Split-window IR** (e.g., 10.8 + 12.0 µm, INSAT TIR1+TIR2): the *difference* reveals thin cirrus and
  cloud microphysics → better cloud-edge tracking.
- **Water-vapour band (~6.7 µm):** shows mid-level moisture and **winds** even where there are no
  clouds → motion cues classical flow can't get.
- **Mid-IR (~3.9 µm):** great for **fire/hot-spot** detection and low cloud at night.
- **Visible band (daytime):** very high-resolution cloud texture for sharper daytime motion.
- → **Multi-band joint interpolation:** feed several channels together so the network estimates one
  consistent motion field from all of them (this is what NASA's Vandal & Nemani did for GOES-R).

### B. Use **more time** (more than two frames)
- Feed **3–5 past frames** (or a recurrent/transformer model) so the network learns **acceleration and
  rotation** (cyclone spin), not just straight-line motion between two frames.

### C. Add **physics**
- **Advection / continuity** constraints (mass is conserved as clouds move), or **NWP wind fields** as a
  prior, so the motion is physically plausible, not just visually plausible.

### D. Stronger models
- **Attention / transformer flow (EMA-VFI, VFIMamba)** for long-range motion.
- **Diffusion-based interpolation** to *generate* crisp, realistic clouds for the hardest non-linear
  cases (growth/dissipation) instead of blurring — at higher compute cost.

### E. Train harder & smarter
- More steps, **more event days** (cyclones, convective outbreaks), data augmentation.
- **Loss functions tuned for motion:** add the optical-flow endpoint error and a temporal-consistency
  term so consecutive interpolated frames don't flicker.
- **Uncertainty output:** also predict *how confident* each pixel is (vital for operational forecasting).

### F. Resolution & products
- **Spatial super-resolution** alongside temporal (sharper *and* faster).
- Output not just images but **derived products** (cloud-top height/temperature motion) that forecasters
  use directly.

### Quick priority list
1. **Train UNetVFI longer on more GOES/Himawari days** → it should pass the baselines. *(cheapest win)*
2. **Add the water-vapour + split-window bands** → biggest physical accuracy gain.
3. **Use 3+ frames of context** → captures rotation/acceleration.
4. **Add a flow-consistency + temporal loss** → smoother, flicker-free animations.
5. **(Stretch) diffusion refinement** → sharp clouds for extreme convective growth.

---

## 10. Every evaluation metric, explained

To judge an interpolated frame we compare it to the **real** frame that we hid (the "ground truth").
Each metric measures a *different kind* of "how close." None is complete on its own — that's why we use
several. (Code: `src/eval/metrics.py`.)

> Convention below: ✅ = we want it **high**, 🔻 = we want it **low**.

### Pixel-accuracy metrics (do the numbers match?)
1. **MSE — Mean Squared Error** 🔻
   - **What:** average of (predicted − true)² over all pixels. `mean((a−b)²)`.
   - **Lay:** "on average, how wrong is each pixel's brightness — with big mistakes punished extra hard"
     (because of the square).
   - **Blind to:** *structure*. A slightly **blurry** image can have a low MSE even though it looks wrong.
   - **Gotcha:** squaring means a few big errors dominate; it secretly **rewards blur** (averaging reduces
     squared error).
2. **MAE(K) — Mean Absolute Error in Kelvin** 🔻
   - **What:** average |predicted − true|, in **real temperature degrees**. `mean(|a−b|)`.
   - **Lay:** "on average we're off by *X degrees*." The only metric in physical, human units.
   - **Why we like it:** interpretable (MAE 1.1 K = good; 1.7 K = noticeably worse) and **robust** to a few
     outliers (no squaring).
3. **PSNR — Peak Signal-to-Noise Ratio** ✅ (decibels, dB)
   - **What:** `10·log₁₀(1 / MSE)` (data range 0–1). A log-rescaled MSE. 30 dB = okay, 40 dB = very good.
   - **Lay:** "signal vs. error, on a loudness-style scale." Higher = cleaner.
   - **Gotcha:** it is **just MSE in disguise** → same blur-reward problem. The most *reported* metric but
     one of the **weakest** for ranking interpolation quality (see §12).

### Structural & perceptual metrics (does it look right?)
4. **SSIM — Structural Similarity Index** ✅ (range −1…1, 1 = identical)
   - **What:** in small sliding windows it compares three things — **luminance** (mean brightness),
     **contrast** (variance), and **structure** (correlation of patterns):
     `SSIM = (2μₐμ_b + c₁)(2σ_ab + c₂) / ((μₐ² + μ_b² + c₁)(σₐ² + σ_b² + c₂))`.
   - **Lay:** "do the cloud **shapes and textures** line up?" — not just individual pixels.
   - **Why it matters:** much closer to how a person judges an image than PSNR. A blurry frame loses
     contrast/structure → SSIM drops even if PSNR is okay.
5. **FSIM — Feature Similarity Index** ✅ (0…1)
   - **What:** SSIM's cousin that **weights** the comparison by **phase congruency** (perceptually salient
     features like edges/corners) and **gradient magnitude**.
   - **Lay:** "do the **important features** (edges, structures) match?"
   - **In our data it's the least discriminative** — between two frames only 20 min apart the *gross*
     features barely move, so every method scores ~0.99. Useful as a sanity check, weak as a separator.
6. **LPIPS — Learned Perceptual Image Patch Similarity** 🔻 (0 = identical)
   - **What:** feed both images through a pretrained deep network (VGG) and measure the **distance in its
     feature space** (we replicate the 1-channel IR to 3 channels for VGG).
   - **Lay:** "does it look wrong **to a trained eye**?" — it captures perceptual errors (smearing, wrong
     texture) that pixel metrics miss.
   - **Why it matters:** LPIPS is the metric that best matches **human judgement of realism** — crucial
     because forecasters *look* at these animations.

### Cloud-motion-specific metrics (the ones that really matter here)
7. **edge-SSIM** ✅ — SSIM computed **only on high-gradient (cloud-edge) pixels**.
   - **Lay:** "are the cloud **boundaries** in the right place and still **sharp**?" Edges are where motion
     is visible, so this is a direct **motion-fidelity** score. A model that blurs edges is exposed here
     even if whole-image SSIM looks fine.
8. **flow-EPE — optical-flow End-Point Error** 🔻 — average distance between the **predicted motion field**
   and a reference (RAFT) motion field, in pixels.
   - **Lay:** "is the **motion itself** correct?" — not the pixels, the *velocity*. This is the closest
     thing to "did we get the physics of how the storm moved right."
9. **temporal warping error** 🔻 — take two consecutive output frames, push one onto the next using flow,
   and measure the leftover mismatch.
   - **Lay:** "does the **time-lapse flow smoothly**, or does it **flicker/jitter**?" Measures temporal
     consistency across the animation.

---

## 11. Reading our validation results — who won *where*, and *why*

Our run (5 GOES triplets: input frames 20 min apart, predict the held-out real 10-min middle):

| model | PSNR ✅ | SSIM ✅ | FSIM ✅ | edge-SSIM ✅ | MAE(K) 🔻 | LPIPS 🔻 |
|---|---|---|---|---|---|---|
| **classical** | **37.58** | 0.9435 | **0.9949** | **0.9193** | **1.09** | 0.1566 |
| **RAFT** | 36.70 | **0.9448** | 0.9940 | 0.8965 | **1.09** | **0.1495** |
| **UNetVFI (ours)** | 33.50 | 0.8870 | 0.9870 | 0.7758 | 1.67 | 0.2165 |

**The headline flips depending on the metric — read carefully:**
- **By PSNR/MAE (pixel accuracy): classical wins.** On this small sample the motion over a 10-minute gap
  is *gentle*, so classical's simple "warp halfway + average" lands pixels accurately — and the slight
  smoothing from averaging is exactly what PSNR/MSE **reward**. So classical tops PSNR (37.6).
- **By SSIM + LPIPS (structure + human perception): RAFT wins.** RAFT's learned, correlation-based flow
  places cloud structure most faithfully (SSIM 0.9448, the best) and looks the most realistic to a trained
  eye (LPIPS 0.1495, the lowest). So **the "better" metrics rank RAFT #1, even though PSNR ranks classical #1.**
- **By edge-SSIM (cloud-edge motion fidelity): classical ≳ RAFT ≫ UNetVFI.** Classical 0.919, RAFT 0.897,
  UNet 0.776. The **big gap at UNet** is the clearest signal: the under-trained UNet **blurs cloud edges**.
- **FSIM is flat (~0.99 for all):** the gross features are unchanged over 20 min, so FSIM can't separate
  the methods here. Don't rank on FSIM for short gaps.
- **UNetVFI is last on everything** (PSNR 33.5, SSIM 0.887, edge-SSIM 0.776, LPIPS 0.217, MAE 1.67 K).
  **Why:** it was trained for only **2000 steps on a single GOES day** → under-fit → it produces softer,
  less-accurate frames (low edge-SSIM + high LPIPS = "blurry to a person"). During training its *internal*
  validation hit PSNR ≈ 42 / SSIM ≈ 0.95, but that used training-set patches (optimistic); the table above
  is the **honest, held-out** number.

**The crucial caveat — the test was *kind to classical*.** The problem statement's claim that classical
optical flow "fails with blur and artefacts" shows up most at **long gaps (INSAT's 30 min)** and during
**rapid convective growth**, where brightness-constancy breaks. Our easy 10-min-gap GOES sample doesn't
stress those failure modes much. On **longer gaps / fast-growing storms**, the learned methods
(RAFT/RIFE/fine-tuned, and a properly-trained UNetVFI) are expected to pull clearly ahead — that's the
regime the PS cares about, and the next experiment to run.

---

## 12. Which metric is the best-ranked indicator of a "good" model?

**No single one — but they are not equal, and PSNR (the most-quoted) is among the weakest here.**

Why PSNR/MSE mislead for *this* task: they are minimised by **predicting the blurry average** of the two
frames. A model can win PSNR by being *cautiously blurry* — which is the **opposite** of the goal
(sharp clouds in the *right moved position*). Our own table proves it: PSNR crowns classical, but the
perceptual/structural metrics crown RAFT.

**A sensible ranking for satellite frame interpolation (best → supporting):**

| Rank | Metric | Why it's a strong indicator |
|------|--------|-----------------------------|
| 1 | **flow-EPE** | directly checks the **motion** (did the storm move correctly) — the actual physics. |
| 1 | **edge-SSIM** | checks cloud **boundaries** are sharp and correctly placed — where motion lives. |
| 2 | **LPIPS** | best match to **human/forecaster perception** of realism; catches blur/smear pixel metrics miss. |
| 3 | **SSIM** | solid structural score; robust, widely understood. |
| 4 | **temporal warping error** | the **animation** must flow without flicker (operational viewing). |
| 5 | **PSNR / MSE / MAE(K)** | necessary sanity checks + physical units, but **easy to game by blurring** → weak for ranking. |
| 6 | **FSIM** | fine as a check; too saturated to separate models at short gaps. |

**Practical rule:** judge models by a **small basket** — primarily **edge-SSIM + LPIPS + flow-EPE**, with
SSIM/PSNR as guards. For an operational "is the cyclone in the right place" question, weight **flow-EPE +
edge-SSIM**; for "does the time-lapse look real", weight **LPIPS + temporal consistency**. A single number
that captures most of it would be a **weighted blend**, e.g. `score = LPIPS + (1−edge_SSIM) + α·flowEPE`
(lower = better) — report it alongside, never instead of, the individual metrics.

---

## 13. Can we use PINNs (Physics-Informed Neural Networks)? Yes — and it's a great fit

### What a PINN is (plain English)
A normal network learns **only from examples** ("here's the right answer, match it"). A **PINN** also
forces the network to **obey the laws of physics**, by adding the physics equation as an **extra penalty
in the loss**. So it's trained to *both* fit the data *and* not violate physics. This is powerful when
data is limited (physics fills the gaps) and stops the model from producing **physically impossible**
clouds.

### The physics of moving clouds (the equations we can embed)
A thermal-IR brightness field `I(x, y, t)` is, to first order, **transported** by a wind/velocity field
`(u, v)`. That is the **advection equation**:

```
∂I/∂t  +  u · ∂I/∂x  +  v · ∂I/∂y  =  S
```

- The left side says "the brightness at a point changes because stuff **flows** past it."
- **`S` is a source/sink term** — it represents clouds **growing or dissipating** (brightness appearing
  or fading, *not* from motion).
- **The classical method secretly assumes `S = 0` and that `(u,v)` is simple** — that's *exactly* why it
  blurs when storms grow (`S ≠ 0`). A PINN can **predict `S` too**, modelling the very thing classical
  can't.

Two more useful physics priors:
- **Mass continuity** `∂ρ/∂t + ∇·(ρu) = 0` → penalise unrealistic **divergence** (clouds shouldn't appear
  from nowhere).
- **Flow smoothness / small-divergence** → motion fields should be locally smooth (real winds are).

### How we'd add it to *our* model (concrete)
Our `UNetVFI` **already predicts the motion fields** `F_{t→0}, F_{t→1}` and a mask `M`. To make it
physics-informed, add a **physics-residual loss** on top of the existing pixel loss:

1. From the predicted flows, get an instantaneous velocity `(u, v)`.
2. Estimate `∂I/∂t` from the two real frames (finite difference) and `∂I/∂x, ∂I/∂y` (spatial gradients).
3. Add a **new tiny head** that predicts the **source term `S(x,y)`** (cloud growth/decay).
4. **Penalise the advection residual:**
   `L_phys = ‖ ∂I/∂t + u·∂I/∂x + v·∂I/∂y − S ‖²`  + small `λ_div·‖∇·(u,v)‖²` + small `λ_S·‖S‖₁` (keep S sparse).
5. Train with `L_total = L_pixel + L_edge + λ_phys · L_phys`.

That single extra loss makes the learned motion **obey transport physics**, gives the model an explicit
way to represent **growth/dissipation** (the `S` head), and typically improves accuracy **with less data**
because physics constrains the answer. (Bonus: feed a weather model's **NWP wind field** as a soft prior
on `(u,v)` for even better grounding.)

### Two flavours
- **Physics-informed UNetVFI (recommended):** same architecture + the `L_phys` loss + an `S` head. Drops
  straight into `src/train/finetune.py` as an extra loss term. Operationally practical.
- **Pure PINN field (research-y):** a small network `MLP(x, y, t) → I` trained per scene to satisfy the PDE
  between the two frames; interpolate by querying any `t`. Elegant but slower and per-scene.

### Pros & cons of going PINN
**Pros:** physically plausible motion; models cloud **growth/dissipation** (the classical failure mode);
better with **scarce data**; better **extrapolation**; naturally extends to multi-band/3-frame inputs.
**Cons:** extra loss weighting to tune (`λ_phys`); needs differentiable finite-difference operators;
can be trickier/slower to train; the advection model is first-order (very explosive convection still
benefits from a generative/diffusion refinement on top).

### Measured (first quick test on the T4)
We implemented exactly the recommended version — the `S` source head + the advection loss — and trained
baseline vs PINN for 1000 steps each on GOES, then compared on held-out triplets:

| model (1000 steps) | PSNR | SSIM |
|---|---|---|
| baseline UNetVFI | 34.83 | 0.8976 |
| **PINN UNetVFI** | **34.95** | **0.9000** |

A small but **consistent improvement** (+0.12 dB PSNR, +0.0024 SSIM) from the physics loss alone, at a
very short training. It's wired in as `--pinn` in `src/train/finetune.py` (loss in
`src/train/losses.py: advection_physics_loss`). With more steps and a tuned `λ_phys` the gap should grow.

**Bottom line:** PINNs are arguably the **most principled upgrade** for this exact problem — they attack
the precise reason the traditional method fails, they bolt onto our existing UNetVFI cleanly, and the
first measured test already shows a consistent gain.

---

## 14. Honest status today
- ✅ **GOES end-to-end works** on a real Tesla T4: data → train UNetVFI → validate (classical 37.6,
  RAFT 36.7, unet 33.5 PSNR) → dashboard.
- ⏳ **INSAT inference** needs its calibrated `.h5` reader (`readers_insat.py`) written against a real
  INSAT file — download + indexing already work.
- ⏳ **UNetVFI** is currently under-trained (2000 steps) — Section 9.E is the path to make it the best.

Everything here is reproducible from `walkthrough.md`, and the code for all three methods lives under
`src/models/` (`classical.py`, `rife.py`/`film.py`/`superslomo.py`/`raft.py`, `unet_vfi.py`).

---

# Appendix A — how each method *actually* works (the mechanics & formulas)

Notation: `I₀, I₂` = the two real input frames (before & after); `I_t` = the frame to predict at
fraction `t ∈ (0,1)` (t = 0.5 is the exact middle); `F_{a→b}` = optical-flow field (per-pixel motion
vector) from frame *a* to frame *b*; `warp(I, F)(x) = I(x + F(x))` = move image `I` by flow `F`
(bilinear sampling).

## A.1 Traditional — classical optical flow (TV-L1 / Farnebäck) + linear interpolation
**Step 1 — estimate the flow `F_{0→2}` (and `F_{2→0}`).** TV-L1 finds the motion field `u` that minimises
an **energy**:

```
E(u) = ∫ ( |∇uₓ| + |∇u_y| )  +  λ ∫ | I₂(x + u(x)) − I₀(x) | dx
        └── smoothness (Total Variation) ──┘   └──── brightness-constancy data term (L1) ────┘
```
- The **data term** is the "a point keeps its brightness as it moves" assumption (`I₂(x+u) = I₀(x)`).
- The **TV term** keeps the flow smooth but allows sharp motion edges.
- Solved by linearising `I₂(x+u) ≈ I₂(x+u₀) + ∇I₂·(u−u₀)` and alternating a soft-threshold (data) with
  TV-denoising (a primal–dual loop). *(Farnebäck, our fallback, instead fits a local quadratic
  `f(x) ≈ xᵀAx + bᵀx + c` to each window; a shift `d` changes `b` predictably, so `d ≈ −½A⁻¹Δb`.)*

**Step 2 — place the middle frame (Super-SloMo linear approximation).** Approximate the flow from the
unknown middle time `t` to each end:
```
F_{t→0} = −(1−t)·t·F_{0→2} + t²·F_{2→0}
F_{t→1} = (1−t)²·F_{0→2} − t·(1−t)·F_{2→0}
```
**Step 3 — warp & blend:** `I_t = (1−t)·warp(I₀, F_{t→0}) + t·warp(I₂, F_{t→1})`.
**Why it blurs:** the data term *hard-codes* brightness-constancy, so when a cloud **grows** (brightness
genuinely changes) the flow is wrong → the warp smears. There is no term for "new cloud appeared."
*(Code: `src/models/classical.py`.)*

## A.2 Fine-tuned existing models — what each one really computes
**RAFT (the flow engine we use, pretrained in torchvision).**
1. CNN features at ¼–⅛ resolution: `g(I₀), g(I₂)`.
2. **All-pairs correlation volume** — similarity of *every* location with *every* location:
   `C(i,j,k,l) = Σ_c g(I₀)[c,i,j] · g(I₂)[c,k,l]`.
3. **Iterative refinement (a GRU)**: start `F=0`; each step looks up `C` around the current estimate and
   predicts an update `ΔF`, so `F ← F + ΔF`, repeated ~12–32×. This is why RAFT nails **large** motion.

**RIFE (our primary deep interpolator).** Its **IFNet** predicts the **intermediate flows directly**
(`F_{t→0}, F_{t→1}`) in a coarse-to-fine pyramid — skipping the "estimate forward/backward then
approximate" of A.1 — plus a fusion mask `M`. Output:
`I_t = M ⊙ warp(I₀, F_{t→0}) + (1−M) ⊙ warp(I₂, F_{t→1})`, refined by a small CNN. Trained end-to-end on
millions of video triplets to minimise reconstruction error; **we fine-tune it on satellite triplets**
(`src/train/rife_finetune.py`).

**FILM (large motion).** A **shared feature pyramid**; a *scale-agnostic* bidirectional motion estimator
gives flow at every scale (handles big jumps); both frames' feature pyramids are warped to `t` and a
**fusion U-Net** synthesises the frame. Trained with **L1 + perceptual (VGG) + style (Gram)** losses →
crisp results.

**Super-SloMo.** Two U-Nets: one estimates `F_{0→2}, F_{2→0}`; a second refines the intermediate flows
**and predicts visibility maps `V`** (occlusion). Final blend is visibility-weighted:
`I_t = ( (1−t)·V_{t←0}·warp(I₀,F_{t→0}) + t·V_{t←1}·warp(I₂,F_{t→1}) ) / ( (1−t)V_{t←0} + t·V_{t←1} )`.
→ This *visibility* idea is what we borrow for the mask `M`.

## A.3 Our custom UNetVFI — the exact operations
1. **Input:** concatenate `I₀, I₂` → a 2-channel tensor.
2. **U-Net:** encoder `32→64→128→256` (each block halves resolution), decoder back up with **skip
   connections**; final 1×1 conv **head** outputs **5 channels** → `raw_a (2), raw_b (2), mask_logit (1)`
   (+ a parallel **source head** `S (1)` for the PINN).
3. **Scale to time & activate:** `F_{t→0} = t·raw_a`, `F_{t→1} = (1−t)·raw_b`, `M = σ(mask_logit)`.
4. **Warp (bilinear `grid_sample`):** `w₀ = warp(I₀, F_{t→0})`, `w₂ = warp(I₂, F_{t→1})`.
5. **Blend:** `I_t = M ⊙ w₀ + (1−M) ⊙ w₂`, clamped to [0,1].
6. **Training loss:** `L = Charbonnier(I_t, GT) + 0.1·gradient-L1 + 0.1·soft-census`
   - **Charbonnier** `√((I_t−GT)² + ε²)` = robust L1 (less blur-prone than MSE),
   - **gradient-L1** sharpens edges, **census** matches local structure (robust to smooth IR gradients).
7. **PINN add-on (`--pinn`):** with `u = F_{t→1} − F_{t→0}` (the frame0→frame2 displacement),
   `L_phys = ‖ warp(I₀, −u) + S − I₂ ‖²  +  0.05·|∇·u|  +  0.05·|S|₁`, added as `λ·L_phys`. The `warp+S`
   term *is* the advection equation in integral form (`∂I/∂t + u·∇I = S`); `S` lets the model represent
   **growth/dissipation** the classical method can't. *(Code: `src/models/unet_vfi.py`,
   `src/train/losses.py`, `src/train/finetune.py`.)*

### One-line contrast
- **Traditional:** *assume* brightness-constancy + smoothness, solve an energy → one flow → linear blend.
- **Fine-tuned:** *learn* flow (RAFT correlation / RIFE intermediate-flow) **and** a synthesis net, from
  video, then adapt to IR.
- **Ours:** *learn* both flows + a visibility mask in one small net trained on IR, optionally constrained
  by the **advection physics** (PINN) and self-supervised on INSAT.
