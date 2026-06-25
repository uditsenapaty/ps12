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

## 10. Honest status today
- ✅ **GOES end-to-end works** on a real Tesla T4: data → train UNetVFI → validate (classical 37.6,
  RAFT 36.7, unet 33.5 PSNR) → dashboard.
- ⏳ **INSAT inference** needs its calibrated `.h5` reader (`readers_insat.py`) written against a real
  INSAT file — download + indexing already work.
- ⏳ **UNetVFI** is currently under-trained (2000 steps) — Section 9.E is the path to make it the best.

Everything here is reproducible from `walkthrough.md`, and the code for all three methods lives under
`src/models/` (`classical.py`, `rife.py`/`film.py`/`superslomo.py`/`raft.py`, `unet_vfi.py`).
