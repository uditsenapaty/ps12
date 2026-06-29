# Problem Statement 12

# Fill in the Frames Seamlessly - Enhancing Temporal Resolution of Satellite Imagery using AI/ML-based Optical Flow

## Description

Satellite images from geostationary satellites are captured at fixed intervals (e.g., every **30 minutes** for INSAT, every **10 minutes** for Himawari/GOES, or every few days for polar-orbiting satellites). This limited temporal resolution restricts near real-time monitoring of rapidly changing phenomena such as:

- Wildfires
- Cyclones
- Thunderstorms
- Floods
- Rapid land-use changes

Traditional optical flow-based temporal interpolation methods often produce blurred images and visual artifacts, especially when dealing with fast-moving or highly non-linear cloud dynamics.

This problem aims to develop an **AI/ML-based Optical Flow frame interpolation system** capable of generating realistic intermediate satellite frames between two consecutive observations, thereby increasing temporal resolution without requiring additional satellite resources.

---

# Objective

The developed software should accomplish the following:

1. Develop an optical flow-based model to estimate motion vectors between consecutive satellite frames.
2. Generate synthetic intermediate frames using deep learning-based frame interpolation methods.
3. Improve the temporal resolution of satellite imagery (e.g., from **30 minutes → 15 minutes → 7.5 minutes**, or equivalent).
4. Validate the generated frames against real higher temporal-resolution datasets (such as Himawari or GOES-19) using image similarity metrics including:
   - SSIM
   - MSE
   - PSNR
   - FSIM
   - Other suitable metrics

---

# Expected Outcomes

Develop a deep learning-based optical flow frame interpolation system capable of generating synthetic intermediate frames for **INSAT-3DS/3DR satellite imagery**.

---

# Dataset Required

Use **Thermal Infrared (TIR)** band data (around **10 μm**) from geostationary satellites.

### Input format

- `.nc`
- `.h5`

### Datasets

- **GOES-19 ABI Channel 13** data from the NOAA GOES-19 AWS bucket
- **INSAT-3DS / INSAT-3DR TIR1** channel data from MOSDAC
- **Himawari-8**

---

# Suggested Tools / Technologies

Possible deep learning video interpolation models include:

- Super SloMo
- RIFE
- Other suitable optical flow or frame interpolation models

Additional technologies:

- Web technologies for visualization dashboard

---

# Expected Solution

The solution should consist of the following modules.

## 1. Frame Interpolation

Generate intermediate frames between consecutive satellite images.

### Optical Flow Estimation

- Explore AI/ML-based optical flow models.
- Preferably train the models on satellite imagery collected in the dataset preparation stage.

### Frame Interpolation

- Use deep learning interpolation networks to synthesize intermediate frames.

### File Format

- Input: `.nc`
- Output: `.nc`

---

## 2. Visualization

Develop a dashboard for comparing original and interpolated satellite imagery.

The dashboard should include:

- Time-lapse animation of original satellite frames.
- Time-lapse animation of interpolated frames.
- Side-by-side comparison.

---

## 3. Evaluation Report

Generate a report comparing interpolated frames with ground truth.

The report should include:

- SSIM
- MSE
- FSIM
- PSNR
- Plots of evaluation metrics
- Any additional metrics suitable for measuring cloud motion

---

## 4. INSAT-3DS Application

Apply the best-performing model to INSAT-3DS / INSAT-3DR imagery.

Requirements:

- Generate intermediate frames at **15-minute temporal resolution**.
- Produce animations using the interpolated INSAT imagery.

For model development (Steps 1–3), high temporal frequency datasets such as **GOES-19** (or another suitable geostationary satellite) may be used.

Example:

Input frames:

- 00:00
- 00:20

Expected output:

- 00:10

After training, apply the same model to **INSAT-3DS** imagery.

---

# Evaluation Parameters

## Frame Interpolation

Evaluation will be based on image quality metrics such as:

- MSE
- PSNR
- SSIM
- FSIM
- Other suitable image similarity metrics

The generated frame should closely match the corresponding ground truth frame.

---

## Visualization

Evaluation of:

- Web GUI design
- User experience
- Animation quality

---

## INSAT-3DS Results

Final evaluation will also consider the visual quality of the interpolated INSAT-3DS imagery.