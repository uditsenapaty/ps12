#!/usr/bin/env python
"""Fill the ISRO BAH 2026 idea-submission template with the PS-12 idea.

Beginner->advanced, concise, with small math snippets and colourful research-paper-style diagrams
(process flow on slide 5, dashboard wireframe on 6, custom UNetVFI architecture on 7).
Run:  python scripts/build_ppt.py   ->  writes "ISRO_BAH_2026_PS12 - FILLED.pptx"
"""
import glob
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ---- palette ------------------------------------------------------------------
NAVY = RGBColor(0x0E, 0x24, 0x44); BLUE = RGBColor(0x2E, 0x86, 0xC1); TEAL = RGBColor(0x11, 0x9D, 0x8B)
PURPLE = RGBColor(0x7D, 0x3C, 0x98); GREEN = RGBColor(0x1F, 0x9D, 0x55); ORANGE = RGBColor(0xE5, 0x7E, 0x1E)
RED = RGBColor(0xC0, 0x39, 0x2B); GOLD = RGBColor(0xD4, 0xA0, 0x17)
DARK = RGBColor(0x1B, 0x26, 0x38); GREY = RGBColor(0x6B, 0x7A, 0x8C); WHITE = RGBColor(0xFF, 0xFF, 0xFF)
CARD = RGBColor(0xFF, 0xFF, 0xFF); LIGHT = RGBColor(0xEF, 0xF3, 0xF7); INK = RGBColor(0x23, 0x2F, 0x3E)

SRC = glob.glob(r"D:\Udit\gitclones\ps12\*.pptx")
SRC = [s for s in SRC if "FILLED" not in s][0]
prs = Presentation(SRC)
S = list(prs.slides)


# ---- helpers ------------------------------------------------------------------
def _noshadow(sh):
    sh.shadow.inherit = False
    return sh


def box(sl, l, t, w, h, fill, line=None, rounded=True):
    sh = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
                             Inches(l), Inches(t), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line; sh.line.width = Pt(1)
    return _noshadow(sh)


def arrow(sl, l, t, w, h, color=GREY, shape=MSO_SHAPE.RIGHT_ARROW):
    a = sl.shapes.add_shape(shape, Inches(l), Inches(t), Inches(w), Inches(h))
    a.fill.solid(); a.fill.fore_color.rgb = color; a.line.fill.background()
    return _noshadow(a)


def settext(sh, runs, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, wrap=True):
    tf = sh.text_frame; tf.word_wrap = wrap; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Pt(4); tf.margin_top = tf.margin_bottom = Pt(2)
    for i, r in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = r.get("align", align); p.space_after = Pt(r.get("sa", 2)); p.space_before = Pt(0)
        run = p.add_run(); run.text = r["t"]
        run.font.size = Pt(r.get("s", 12)); run.font.bold = r.get("b", False)
        run.font.italic = r.get("i", False); run.font.color.rgb = r.get("c", WHITE)
    return sh


def textbox(sl, l, t, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = sl.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    settext(tb, runs, align=align, anchor=anchor)
    return tb


def title_bar(sl, text, sub=None):
    b = box(sl, 0.35, 0.6, 9.3, 0.62, NAVY)
    runs = [{"t": text, "s": 20, "b": True, "c": WHITE, "align": PP_ALIGN.LEFT}]
    settext(b, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE)
    if sub:
        textbox(sl, 0.4, 1.18, 9.2, 0.3, [{"t": sub, "s": 11, "i": True, "c": GREY}])


def card(sl, l, t, w, h, runs, fill=CARD, bar=None):
    c = box(sl, l, t, w, h, fill, line=RGBColor(0xD5, 0xDC, 0xE3))
    if bar:
        box(sl, l, t, 0.08, h, bar)
    settext(c, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)
    return c


def clear_prompts(sl):
    for sh in sl.shapes:
        if sh.has_text_frame and sh.shape_type == 17:  # TEXT_BOX
            sh.text_frame.clear()


def bullet(t, s=12, c=INK, b=False, sa=4, align=PP_ALIGN.LEFT):
    return {"t": t, "s": s, "c": c, "b": b, "sa": sa, "align": align}


# ===============================================================================
# SLIDE 1 — cover fields
for sh in S[0].shapes:
    if sh.has_text_frame:
        tx = sh.text_frame.text
        if tx.startswith("Problem Statement"):
            settext(sh, [{"t": "Problem Statement : PS-12 — Fill in the Frames Seamlessly: Enhancing "
                          "Temporal Resolution of Satellite Imagery using AI/ML based on Optical Flow",
                          "s": 13, "b": True, "c": WHITE, "align": PP_ALIGN.LEFT}],
                    align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE)
        elif tx.startswith("Team Name"):
            settext(sh, [{"t": "Team Name : [your team name]", "s": 13, "c": WHITE, "align": PP_ALIGN.LEFT}],
                    align=PP_ALIGN.LEFT)
        elif tx.startswith("Team Leader"):
            settext(sh, [{"t": "Team Leader Name : [your name]", "s": 13, "c": WHITE, "align": PP_ALIGN.LEFT}],
                    align=PP_ALIGN.LEFT)

# ===============================================================================
# SLIDE 3 — Opportunity & USP
sl = S[2]; clear_prompts(sl)
title_bar(sl, "The Opportunity & Our Edge")
card(sl, 0.35, 1.45, 4.55, 1.75, [
    bullet("The gap", 13, BLUE, True),
    bullet("INSAT-3DR/3DS image every 30 min. Cyclones, thunderstorms, fire fronts and floods change "
           "in minutes — so we miss them between frames.", 11),
    bullet("The idea (simple): teach an AI to draw the missing in-between frames from two real frames "
           "→ 30 → 15 → 7.5 min, with no new satellite.", 11),
], bar=BLUE)
card(sl, 0.35, 3.3, 4.55, 1.95, [
    bullet("How it's different", 13, PURPLE, True),
    bullet("Classical optical flow (TV-L1) assumes straight, constant-brightness motion → it blurs & "
           "ghosts on fast, non-linear cloud growth.", 11),
    bullet("We LEARN the motion field + how to fuse it, directly from satellite thermal-IR.", 11),
    bullet("Predict I(t₁) from I(t₀), I(t₂) at t = (t₁−t₀)/(t₂−t₀);  30→15 min ⇒ t = 0.5.", 11, GREY, False),
], bar=PURPLE)
card(sl, 5.05, 1.45, 4.6, 3.8, [
    bullet("Unique selling points (USP)", 13, GREEN, True),
    bullet("• Trained on satellite IR (brightness temperature) — not natural video.", 11),
    bullet("• Cross-satellite transfer: train on dense GOES-19 / Himawari (10-min) → apply to INSAT.", 11),
    bullet("• Self-supervised INSAT adaptation — needs NO labels (uses INSAT's own 30-min frames).", 11),
    bullet("• Custom UNetVFI = intermediate-flow (RIFE-style) ⊕ visibility/occlusion blend "
           "(Super-SloMo-style).", 11),
    bullet("• Full product: .nc → .nc, a web dashboard, and a metric-validated report.", 11),
    bullet("• Already real: trained on a Tesla T4, validated on real GOES-19 (val PSNR≈42, SSIM≈0.95).",
           11, GREEN, True),
], bar=GREEN)

# ===============================================================================
# SLIDE 4 — Features
sl = S[3]; clear_prompts(sl)
title_bar(sl, "What the Solution Does")
feats = [
    ("5 interpolation engines", "Custom UNetVFI + RIFE + FILM + Super-SloMo + RAFT, with a classical "
     "TV-L1 baseline for honest comparison.", BLUE),
    ("Temporal upscaling", "2× (30→15) and 4× (30→7.5 min) by recursively inserting AI frames.", TEAL),
    ("Standards-compliant I/O", "Reads .nc/.h5 (GOES/Himawari/INSAT), writes CF NetCDF brightness "
     "temperature (the PS contract).", PURPLE),
    ("Validation vs ground truth", "PSNR, SSIM, FSIM, MSE, MAE(K), LPIPS + cloud-motion metrics "
     "(flow-EPE, edge-SSIM, temporal warping).", ORANGE),
    ("Web dashboard (3 tabs)", "Interpolate · Temporal Upscaling · Validation Report — animations, "
     "motion-vector overlay, live metrics.", GREEN),
    ("Cloud-ready", "One command connects to a free T4 (Lightning.ai / Colab / Kaggle) with 100 GB "
     "persistent storage.", RED),
]
xs = [0.35, 3.45, 6.55]; ys = [1.45, 3.05]
for k, (h, b, col) in enumerate(feats):
    x = xs[k % 3]; y = ys[k // 3]
    card(sl, x, y, 2.95, 1.45, [bullet(h, 12, col, True), bullet(b, 10.5)], bar=col)
textbox(sl, 0.35, 4.7, 9.3, 0.6, [
    {"t": "Math: PSNR = 10·log₁₀(1 / MSE) → a perfect frame (MSE 0) ⇒ PSNR ∞.  "
          "Our UNetVFI reached val PSNR ≈ 42 dB, SSIM ≈ 0.95 on held-out GOES.", "s": 11, "i": True, "c": INK}])

# ===============================================================================
# SLIDE 5 — Process flow  (colourful pipeline)
sl = S[4]; clear_prompts(sl)
title_bar(sl, "Process Flow")
stages = [
    (BLUE,  "1 · INPUT", ["Two real frames", "I(t₀), I(t₂)", ".nc / .h5  ·  TIR ~10 µm"]),
    (TEAL,  "2 · PREP", ["Calibrate → BT (K)", "Normalize [180–330K]→[0,1]", "Tile 256²"]),
    (PURPLE, "3 · AI OPTICAL FLOW", ["Estimate motion +", "synthesize I(t)", "t = ½, ¼, ¾"]),
    (GREEN, "4 · REBUILD", ["Untile (feather-blend)", "Denormalize → BT", "Write .nc @ new time"]),
    (ORANGE, "5 · VALIDATE & SHOW", ["PSNR/SSIM/FSIM…", "Web dashboard", "time-lapse animation"]),
]
w, gap, y, h = 1.66, 0.18, 1.7, 2.0
x = 0.32
for i, (col, head, lines) in enumerate(stages):
    b = box(sl, x, y, w, h, col)
    runs = [{"t": head, "s": 11.5, "b": True, "c": WHITE, "sa": 4}]
    runs += [{"t": ln, "s": 10, "c": WHITE, "sa": 1} for ln in lines]
    settext(b, runs, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    if i < len(stages) - 1:
        arrow(sl, x + w + 0.005, y + h / 2 - 0.16, gap - 0.01, 0.32, GREY)
    x += w + gap
box(sl, 0.32, 4.0, 9.36, 0.95, LIGHT, line=RGBColor(0xCF, 0xD8, 0xE0))
textbox(sl, 0.5, 4.08, 9.0, 0.8, [
    {"t": "Synthesis equation (per pixel x):", "s": 11, "b": True, "c": NAVY},
    {"t": "I_t(x) ≈ M(x)·I₀(x + F_{t→0}(x))  +  (1 − M(x))·I₂(x + F_{t→1}(x))", "s": 13, "b": True, "c": PURPLE},
    {"t": "F = learned optical flow (motion vectors),  M = learned visibility/occlusion mask.", "s": 10.5,
     "i": True, "c": GREY}], align=PP_ALIGN.LEFT)

# ===============================================================================
# SLIDE 6 — Dashboard wireframe
sl = S[5]; clear_prompts(sl)
title_bar(sl, "Dashboard (Web GUI)")
# browser frame
box(sl, 0.5, 1.5, 9.0, 3.7, RGBColor(0xF7, 0xF9, 0xFB), line=RGBColor(0xC2, 0xCD, 0xD6))
box(sl, 0.5, 1.5, 9.0, 0.4, NAVY)
settext(box(sl, 0.5, 1.5, 9.0, 0.4, NAVY), [{"t": "🛰  PS-12  ·  Satellite Frame Interpolation", "s": 11,
        "b": True, "c": WHITE, "align": PP_ALIGN.LEFT}], align=PP_ALIGN.LEFT)
# sidebar
sb = box(sl, 0.65, 2.05, 2.0, 3.0, LIGHT, line=RGBColor(0xC2, 0xCD, 0xD6))
settext(sb, [bullet("Configuration", 11, NAVY, True),
             bullet("Source ▾", 10, INK), bullet("Model ▾  (UNetVFI…)", 10, INK),
             bullet("Factor: 2× / 4×", 10, INK), bullet("☑ motion overlay", 10, INK),
             bullet("● runnable: unet, raft,", 9.5, GREEN), bullet("  classical", 9.5, GREEN)],
        align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)
# tabs
for i, (lab, c) in enumerate([("▶ Interpolate", BLUE), ("🔼 Temporal Upscaling", TEAL),
                              ("📊 Validation Report", ORANGE)]):
    settext(box(sl, 2.8 + i * 2.2, 2.05, 2.1, 0.35, c), [{"t": lab, "s": 9.5, "b": True, "c": WHITE}])
# frame strip
labels = [("t₀ (input)", GREY), ("t=0.5 (AI)", PURPLE), ("t₂ (input)", GREY)]
for i, (lab, c) in enumerate(labels):
    box(sl, 2.85 + i * 1.7, 2.6, 1.5, 1.35, RGBColor(0x33, 0x3A, 0x44))
    settext(box(sl, 2.85 + i * 1.7, 3.98, 1.5, 0.28, c), [{"t": lab, "s": 9.5, "b": True, "c": WHITE}])
# metrics strip
for i, (lab, val) in enumerate([("PSNR", "33.5"), ("SSIM", "0.89"), ("FSIM", "0.99"), ("MAE(K)", "1.7")]):
    box(sl, 2.85 + i * 1.55, 4.45, 1.4, 0.55, RGBColor(0xEA, 0xF1, 0xF6), line=RGBColor(0xC2, 0xCD, 0xD6))
    settext(sl.shapes[-1], [{"t": val, "s": 13, "b": True, "c": NAVY}, {"t": lab, "s": 8.5, "c": GREY}])
textbox(sl, 0.5, 5.18, 9.0, 0.35, [{"t": "Served from the cloud T4 → opened in your browser (Lightning "
        "Streamlit plugin / ngrok). Validation tab shows committed comparison results.", "s": 10, "i": True,
        "c": GREY}])

# ===============================================================================
# SLIDE 7 — Architecture (custom UNetVFI)
sl = S[6]; clear_prompts(sl)
title_bar(sl, "Architecture — Custom UNetVFI")
# inputs
box(sl, 0.3, 1.95, 0.95, 0.55, BLUE); settext(sl.shapes[-1], [{"t": "I₀", "s": 13, "b": True, "c": WHITE}])
box(sl, 0.3, 2.75, 0.95, 0.55, BLUE); settext(sl.shapes[-1], [{"t": "I₂", "s": 13, "b": True, "c": WHITE}])
box(sl, 1.4, 2.2, 0.8, 0.9, NAVY); settext(sl.shapes[-1], [{"t": "concat", "s": 10, "b": True, "c": WHITE},
                                                          {"t": "2 ch", "s": 9, "c": WHITE}])
arrow(sl, 1.25, 2.4, 0.13, 0.5, GREY); arrow(sl, 2.2, 2.4, 0.13, 0.5, GREY)
# encoder (down) and decoder (up) columns
enc = [("32", TEAL), ("64", TEAL), ("128", TEAL), ("256", PURPLE)]
dec = [("128", GREEN), ("64", GREEN), ("32", GREEN)]
ex = 2.45
for i, (lab, c) in enumerate(enc):
    yy = 1.7 + i * 0.62
    box(sl, ex, yy, 0.95, 0.5, c); settext(sl.shapes[-1], [{"t": "enc " + lab, "s": 10, "b": True, "c": WHITE}])
# bottleneck arrow to decoder
dx = 3.95
for i, (lab, c) in enumerate(dec):
    yy = 2.0 + (len(dec) - 1 - i) * 0.62
    box(sl, dx, yy, 0.95, 0.5, c); settext(sl.shapes[-1], [{"t": "dec " + lab, "s": 10, "b": True, "c": WHITE}])
    # skip-connection arrow enc->dec (dashed look via thin grey arrow)
    arrow(sl, ex + 0.97, 1.7 + i * 0.62 + 0.12, dx - (ex + 0.97), 0.18, RGBColor(0xB7, 0xC2, 0xCC))
textbox(sl, 2.4, 4.18, 2.6, 0.3, [{"t": "U-Net encoder ↘ / decoder ↗  + skip connections", "s": 9.5,
        "i": True, "c": GREY}])
# head -> flow + mask
arrow(sl, 4.92, 2.4, 0.18, 0.5, GREY)
hx = 5.2
for i, (lab, c, sub) in enumerate([("flow_{t→0}", ORANGE, "2 ch"), ("flow_{t→1}", ORANGE, "2 ch"),
                                   ("mask M", GOLD, "1 ch, σ")]):
    box(sl, hx, 1.75 + i * 0.78, 1.55, 0.62, c)
    settext(sl.shapes[-1], [{"t": lab, "s": 10.5, "b": True, "c": WHITE}, {"t": sub, "s": 8.5, "c": WHITE}])
# warp + blend
arrow(sl, 6.78, 2.4, 0.18, 0.5, GREY)
box(sl, 7.0, 1.95, 1.45, 1.4, NAVY)
settext(sl.shapes[-1], [{"t": "backward-warp", "s": 10, "b": True, "c": WHITE},
                        {"t": "I₀, I₂ by F", "s": 9.5, "c": WHITE},
                        {"t": "blend by M", "s": 9.5, "c": WHITE}])
arrow(sl, 8.45, 2.5, 0.2, 0.3, GREEN)
box(sl, 8.7, 1.95, 1.0, 1.4, GREEN)
settext(sl.shapes[-1], [{"t": "I(t)", "s": 14, "b": True, "c": WHITE}, {"t": "synthetic", "s": 9, "c": WHITE},
                        {"t": "frame", "s": 9, "c": WHITE}])
# caption + system strip
box(sl, 0.3, 4.55, 9.4, 0.95, LIGHT, line=RGBColor(0xCF, 0xD8, 0xE0))
textbox(sl, 0.45, 4.6, 9.1, 0.9, [
    {"t": "RIFE-style intermediate flow ⊕ Super-SloMo visibility blending · ~2–5 M params · trains "
          "from scratch on GOES/Himawari in hours on a T4 · self-supervised on INSAT.", "s": 10.5, "c": INK},
    {"t": "System: Local dev  →  GitHub  →  Lightning.ai T4 (train + infer, 100 GB persist)  →  "
          "Streamlit dashboard (browser).", "s": 10.5, "b": True, "c": NAVY}], align=PP_ALIGN.LEFT)

# ===============================================================================
# SLIDE 8 — Technologies
sl = S[7]; clear_prompts(sl)
title_bar(sl, "Technology Stack")
groups = [
    ("Deep learning", BLUE, "PyTorch · RAFT (torchvision) · RIFE · FILM · Super-SloMo · custom UNetVFI"),
    ("Satellite I/O", TEAL, "xarray · netCDF4 · h5py · satpy · boto3 (GOES S3) · paramiko/lftp (MOSDAC SFTP)"),
    ("Classical + metrics", PURPLE, "OpenCV TV-L1/Farnebäck · scikit-image · piq (FSIM/LPIPS)"),
    ("Web + serving", ORANGE, "Streamlit (3-tab dashboard) · ngrok / Lightning Streamlit plugin"),
    ("Compute + DevOps", GREEN, "Lightning.ai / Colab / Kaggle free T4 + 100 GB persist · Git / GitHub"),
    ("Datasets", RED, "GOES-19 ABI Ch13 (10.3µm) · Himawari AHI B13 (10.4µm) · INSAT-3DR/3DS TIR1 (10.8µm)"),
]
y = 1.5
for h, c, b in groups:
    card(sl, 0.4, y, 9.2, 0.58, [{"t": h + ":  ", "s": 12.5, "b": True, "c": c},
                                 {"t": b, "s": 11.5, "c": INK}], bar=c)
    # put header+body on one wrapped line
    sl.shapes[-1].text_frame.paragraphs[0].runs  # noop
    y += 0.62

# ===============================================================================
# SLIDE 9 — Cost
sl = S[8]; clear_prompts(sl)
title_bar(sl, "Estimated Cost")
card(sl, 0.4, 1.5, 5.6, 3.4, [
    bullet("Software", 13, BLUE, True), bullet("₹0 — fully open-source.", 11),
    bullet("Data", 13, TEAL, True),
    bullet("₹0 — NOAA GOES/Himawari on AWS (free); INSAT via MOSDAC (free account).", 11),
    bullet("Compute", 13, GREEN, True),
    bullet("Free tiers cover it — Lightning.ai (~35 T4-hrs/mo), Colab/Kaggle free T4.", 11),
    bullet("If paid: training ≈ 2–5 T4-GPU-hours.", 11),
    bullet("Optional OpenAI (report narrative): < $10 (not required).", 11),
], bar=GREEN)
card(sl, 6.2, 1.5, 3.4, 3.4, [
    bullet("Bottom line", 13, ORANGE, True),
    bullet("₹0 on free tiers.", 14, GREEN, True),
    bullet("Worst case ≤ $10.", 12, INK),
    bullet("Cost model:", 11, NAVY, True),
    bullet("cost ≈ GPU-hrs × $/hr", 11, INK),
    bullet("≈ 5 × $0.35 ≈ $1.75 per", 11, INK),
    bullet("full training run.", 11, INK),
], bar=ORANGE)

# ===============================================================================
# SLIDE 10 — closing
sl = S[9]
box(sl, 1.2, 1.5, 7.6, 2.6, NAVY)
settext(sl.shapes[-1], [
    {"t": "Thank You", "s": 30, "b": True, "c": WHITE, "sa": 8},
    {"t": "Enhancing temporal resolution — no new satellite required.", "s": 14, "c": RGBColor(0xCF, 0xE0, 0xF0)},
    {"t": "Live & reproducible: trained on a real Tesla T4, validated on real GOES-19.", "s": 12,
     "c": RGBColor(0xBE, 0xD3, 0xE6)},
])
textbox(sl, 1.2, 4.3, 7.6, 0.5, [{"t": "Repository:  github.com/uditsenapaty/ps12", "s": 14, "b": True,
        "c": ORANGE, "align": PP_ALIGN.CENTER}], align=PP_ALIGN.CENTER)

OUT = r"D:\Udit\gitclones\ps12\ISRO_BAH_2026_PS12 - FILLED.pptx"
prs.save(OUT)
print("SAVED:", OUT, "| slides:", len(S))
