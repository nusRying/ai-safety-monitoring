"""Build 1600x1200 (4:3) Upwork portfolio slides for the Safety Monitoring project.

    python portfolio/make_portfolio.py                          # uses portfolio/dashboard_screenshot.webp
    python portfolio/make_portfolio.py path/to/new_screenshot.png

Upwork: landscape cover, 1000-4000 px wide; 4:3 fills the thumbnail frame without cropping.
Slide 2 is built from outputs/demo_site_track.mp4 and outputs/demo_scene_pipeline.mp4.
"""
import sys
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "portfolio"
OUT.mkdir(exist_ok=True)
SHOT = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "dashboard_screenshot.webp"

W, H = 1600, 1200
BG, PANEL, LINE = (14, 17, 23), (24, 28, 36), (48, 54, 66)
FG, MUTED, RED, CYAN = (240, 242, 246), (150, 158, 172), (255, 75, 75), (0, 212, 255)
FONTS = r"C:\Windows\Fonts"


def font(size, weight="regular"):
    name = {"regular": "segoeui.ttf", "bold": "segoeuib.ttf", "semibold": "seguisb.ttf",
            "light": "segoeuil.ttf"}[weight]
    path = Path(FONTS) / name
    if not path.exists():
        path = Path(FONTS) / "segoeuib.ttf" if weight != "regular" else Path(FONTS) / "segoeui.ttf"
    return ImageFont.truetype(str(path), size)


def canvas():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # Subtle vertical gradient for depth.
    for y in range(H):
        t = y / H
        c = tuple(int(BG[i] * (1 - t) + (10, 12, 17)[i] * t) for i in range(3))
        d.line([(0, y), (W, y)], fill=c)
    return img, ImageDraw.Draw(img)


def header(d, kicker, title, subtitle, y=56):
    d.text((70, y), kicker.upper(), font=font(22, "semibold"), fill=RED)
    d.text((68, y + 34), title, font=font(60, "bold"), fill=FG)
    d.text((70, y + 116), subtitle, font=font(27), fill=MUTED)
    return y + 170


def paste_rounded(img, tile, xy, radius=18, border=LINE, shadow=True):
    w, h = tile.size
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius, fill=255)
    if shadow:
        sh = Image.new("RGBA", (w + 60, h + 60), (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle([30, 36, w + 30, h + 36], radius, fill=(0, 0, 0, 150))
        sh = sh.filter(ImageFilter.GaussianBlur(18))
        img.paste(sh, (xy[0] - 30, xy[1] - 30), sh)
    img.paste(tile, xy, mask)
    ImageDraw.Draw(img).rounded_rectangle([xy[0], xy[1], xy[0] + w - 1, xy[1] + h - 1], radius,
                                          outline=border, width=2)


def chips(d, items, x, y, max_x=W - 70, size=22):
    f = font(size, "semibold")
    for text, color in items:
        tw = d.textlength(text, font=f)
        if x + tw + 36 > max_x:
            x, y = 70, y + size + 34
        d.rounded_rectangle([x, y, x + tw + 32, y + size + 20], 999, fill=PANEL, outline=color, width=2)
        d.text((x + 16, y + 8), text, font=f, fill=FG)
        x += tw + 46
    return y + size + 20


def frame_at(video, t_s):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000)
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read {video} at {t_s}s")
    return Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))


def fit(tile, max_w, max_h):
    s = min(max_w / tile.width, max_h / tile.height)
    return tile.resize((round(tile.width * s), round(tile.height * s)), Image.LANCZOS)


# ------------------------------------------------------------------ 1. cover
img, d = canvas()
y = header(d, "Computer vision  ·  Real-time AI",
           "AI Safety Monitoring System",
           "Live PPE compliance, danger zones and fall detection from any camera")
shot = fit(Image.open(SHOT).convert("RGB"), W - 140, 720)
sx = (W - shot.width) // 2
paste_rounded(img, shot, (sx, y + 24))
y = y + 24 + shot.height + 48
y = chips(d, [("YOLO26 fine-tuned", RED), ("Pose estimation", CYAN), ("BoT-SORT tracking", CYAN),
              ("TensorRT  3× faster", RED), ("Vision LLM reports", CYAN), ("Streamlit dashboard", CYAN),
              ("Python · PyTorch · OpenCV", LINE)], 70, y)
d.text((70, y + 34), "21+ FPS live on a laptop GPU (RTX 3050)  ·  webcam, CCTV or video files",
       font=font(26), fill=MUTED)
img.save(OUT / "01_cover_dashboard.png", optimize=True)

# ---------------------------------------------------------- 2. how it works
img, d = canvas()
y = header(d, "What it detects", "Rules on top of deep learning",
           "Detection + tracking + pose keypoints, turned into safety events")
left = frame_at(ROOT / "outputs" / "demo_site_track.mp4", 1.2)
right = frame_at(ROOT / "outputs" / "demo_scene_pipeline.mp4", 8.0)
tile_w = (W - 140 - 40) // 2
lt, rt = fit(left, tile_w, 620), fit(right, tile_w, 620)
th = max(lt.height, rt.height)
paste_rounded(img, lt, (70, y + 30))                       # top-aligned
paste_rounded(img, rt, (70 + tile_w + 40, y + 30))
cy = y + 30 + th + 34
cap_f, body_f = font(28, "bold"), font(22)
d.text((70, cy), "PPE compliance in a crowd", font=cap_f, fill=FG)
d.text((70, cy + 42), "Tracks every worker; flags missing hard hats only\nwhen the head is actually visible.",
       font=body_f, fill=MUTED, spacing=8)
d.text((70 + tile_w + 40, cy), "Danger zones + fall detection", font=cap_f, fill=FG)
d.text((70 + tile_w + 40, cy + 42), "Zone dwell time and occupancy; falls caught even when\n"
       "the detector loses the person lying down.", font=body_f, fill=MUTED, spacing=8)
chips(d, [("Missing helmet / vest", RED), ("Restricted zone entry", RED), ("Dwell time & capacity", RED),
          ("Fall + person down", RED), ("VLM incident reports", CYAN)], 70, cy + 140)
img.save(OUT / "02_detections.png", optimize=True)

# ---------------------------------------------------------- 3. results
img, d = canvas()
y = header(d, "Measured results", "Engineered, tested, measured",
           "Every change checked against labelled test video and a held-out test set")
cards = [
    ("0.84", "mAP50 on PPE test set", "Fine-tuned YOLO26 vs 0.66 for the best\nzero-shot model (Grounding DINO)"),
    ("3×", "faster with TensorRT", "28.5 → 9.8 ms per frame, same accuracy\n(mAP50 0.543 → 0.552)"),
    ("8 → 1", "false alarms", "Head-visibility rule with pose keypoints;\nall real violations still caught"),
    ("21 FPS", "live on a laptop GPU", "Two models + tracking + rules on an\nRTX 3050 laptop at 720p"),
    ("$0.0004", "per incident report", "Vision LLM writes a structured report and\nsecond-opinions each alert"),
    ("0.85", "confidence on fallen people", "Rotated search re-finds people the\ndetector loses once they're lying down"),
]
cw, ch, gap = (W - 140 - 2 * 32) // 3, 300, 32
for i, (big, label, note) in enumerate(cards):
    x = 70 + (i % 3) * (cw + gap)
    yy = y + 40 + (i // 3) * (ch + gap)
    d.rounded_rectangle([x, yy, x + cw, yy + ch], 22, fill=PANEL, outline=LINE, width=2)
    d.text((x + 32, yy + 26), big, font=font(72, "bold"), fill=RED if i % 2 == 0 else CYAN)
    d.text((x + 34, yy + 122), label, font=font(27, "semibold"), fill=FG)
    d.text((x + 34, yy + 172), note, font=font(21), fill=MUTED, spacing=8)
sy = y + 40 + 2 * ch + gap + 56
d.text((70, sy), "TECH STACK", font=font(22, "semibold"), fill=RED)
chips(d, [("Python", LINE), ("PyTorch", LINE), ("Ultralytics YOLO26", LINE), ("SAM 2.1", LINE),
          ("Grounding DINO", LINE), ("TensorRT", LINE), ("OpenCV", LINE), ("Streamlit", LINE),
          ("OpenRouter / Claude API", LINE)], 70, sy + 44, size=22)
img.save(OUT / "03_results.png", optimize=True)

for p in sorted(OUT.glob("*.png")):
    im = Image.open(p)
    print(p.name, im.size, f"{p.stat().st_size / 1e6:.1f} MB")
