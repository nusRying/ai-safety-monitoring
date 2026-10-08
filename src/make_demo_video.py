"""Make reproducible test videos from still images.

Real CCTV footage is the best test, but these clips are enough to exercise the
pipeline when you don't have any.

Modes:
    pan     A virtual camera pans across still images placed side by side.
            People move across the frame, which exercises tracking (Stage 4).
    scene   A fixed camera. Two people are cut out of photos with a segmentation
            model and animated over an empty background: one walks into the
            middle of the yard and lingers (Stage 5 zones), the other falls over
            (Stage 6 fall detection).

Usage:
    python src/make_demo_video.py                     # pan  -> data/samples/demo_site.mp4
    python src/make_demo_video.py --mode scene        # scene -> data/samples/demo_scene.mp4
    python src/make_demo_video.py a.jpg b.jpg --seconds 12 --out data/samples/my.mp4
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TEST_IMGS = DATA / "construction-ppe" / "images" / "test"
DEFAULT_IMAGES = [TEST_IMGS / "image536.jpg", TEST_IMGS / "image1145.jpg"]

# Scene ingredients: an empty yard (cropped to remove a worker at the right edge),
# a worker wearing a helmet, and a man without one.
SCENE_BG = (DATA / "construction-ppe" / "images" / "train" / "image738.jpg", (0, 200, 555, 640))
SCENE_FALLER = TEST_IMGS / "image536.jpg"
SCENE_WALKER = DATA / "samples" / "bus.jpg"


# ------------------------------------------------------------------- pan mode

def make_pan(args):
    frames = [cv2.imread(str(i)) for i in args.images]
    if any(f is None for f in frames):
        raise SystemExit(f"Could not read one of: {args.images}")
    h = max(f.shape[0] for f in frames)
    canvas = np.hstack([cv2.resize(f, (round(f.shape[1] * h / f.shape[0]), h)) for f in frames])

    out_w, out_h = args.size
    n = int(args.seconds * args.fps)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (out_w, out_h))
    for i in range(n):
        t = i / (n - 1)
        ease = 0.5 - 0.5 * np.cos(np.pi * t)              # slow start and stop
        zoom = 1.0 + 0.15 * np.sin(np.pi * t)              # zoom in mid-pan
        win_h = int(h / zoom)
        win_w = min(int(win_h * out_w / out_h), canvas.shape[1])
        x = int(ease * (canvas.shape[1] - win_w))
        y = (h - win_h) // 2
        writer.write(cv2.resize(canvas[y:y + win_h, x:x + win_w], (out_w, out_h)))
    writer.release()
    return n


# ----------------------------------------------------------------- scene mode

def cut_person(seg_model, img_path, pick=0):
    """Return a tight BGRA sprite of one person, using an instance-segmentation mask.

    pick: 0 = largest person by mask area, 1 = second largest, ...
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"Could not read {img_path}")
    r = seg_model.predict(img, classes=[0], conf=0.4, verbose=False)[0]
    if r.masks is None:
        raise SystemExit(f"No person found in {img_path}")
    polys = sorted(r.masks.xy, key=lambda p: -cv2.contourArea(p.astype(np.float32)))
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [polys[pick].astype(np.int32)], 255)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)                      # soften the cut edge
    x, y, w, h = cv2.boundingRect(polys[pick].astype(np.int32))
    return np.dstack([img, mask])[y:y + h, x:x + w]


def scale_to_height(sprite, height):
    return cv2.resize(sprite, (round(sprite.shape[1] * height / sprite.shape[0]), height))


def paste(dst, sprite, feet_xy, angle=0.0):
    """Alpha-blend sprite onto dst with its bottom-centre (feet) at feet_xy, rotated by angle degrees."""
    h, w = sprite.shape[:2]
    size = 2 * max(h, w) + 2
    canvas = np.zeros((size, size, 4), np.uint8)
    c = size // 2
    canvas[c - h:c, c - w // 2:c - w // 2 + w] = sprite              # feet at canvas centre
    if angle:
        rot = cv2.getRotationMatrix2D((c, c), angle, 1.0)          # pivot around the feet
        canvas = cv2.warpAffine(canvas, rot, (size, size), flags=cv2.INTER_LINEAR)

    x0, y0 = int(feet_xy[0]) - c, int(feet_xy[1]) - c
    H, W = dst.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(W, x0 + size), min(H, y0 + size)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    patch = canvas[sy0:sy0 + dy1 - dy0, sx0:sx0 + dx1 - dx0]
    alpha = patch[..., 3:4].astype(np.float32) / 255
    roi = dst[dy0:dy1, dx0:dx1]
    roi[:] = (alpha * patch[..., :3] + (1 - alpha) * roi).astype(np.uint8)


def make_scene(args):
    from ultralytics import YOLO

    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    weights = models / "yolo26n-seg.pt"
    seg = YOLO(str(weights) if weights.exists() else "yolo26n-seg.pt")
    if not weights.exists() and Path("yolo26n-seg.pt").exists():
        Path("yolo26n-seg.pt").replace(weights)

    bg_path, (x0, y0, x1, y1) = SCENE_BG
    bg = cv2.imread(str(bg_path))[y0:y1, x0:x1]
    out_w = args.size[0]
    out_h = round(bg.shape[0] * out_w / bg.shape[1])
    bg = cv2.resize(bg, (out_w, out_h))

    person_h = int(0.48 * out_h)
    walker = scale_to_height(cut_person(seg, SCENE_WALKER), person_h)
    faller = scale_to_height(cut_person(seg, SCENE_FALLER), person_h)
    floor_y = int(0.93 * out_h)

    n = int(args.seconds * args.fps)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (out_w, out_h))
    for i in range(n):
        t = i / args.fps
        frame = bg.copy()

        # Walker: enters from the left, stops mid-yard from 4 s to 9 s, walks back out.
        stop_x, start_x = 0.37 * out_w, -0.1 * out_w
        if t < 4:
            wx = start_x + (stop_x - start_x) * t / 4
        elif t < 9:
            wx = stop_x
        else:
            wx = stop_x + (start_x - stop_x) * min((t - 9) / 3, 1)
        walking = t < 4 or t >= 9
        bob = 4 * abs(np.sin(t * 2 * np.pi * 1.6)) if walking else 0
        paste(frame, walker, (wx, floor_y - bob))

        # Faller: stands at the right, tips over to the left at 6 s and stays down.
        fall_start, fall_dur = 6.0, 0.6
        p = np.clip((t - fall_start) / fall_dur, 0, 1)
        angle = 85 * p ** 2                                      # accelerates like a real fall
        sway = 0 if t >= fall_start else 1.5 * np.sin(t * 2.0)
        paste(frame, faller, (0.86 * out_w, floor_y), angle=angle + sway)

        writer.write(frame)
    writer.release()
    return n


def main():
    p = argparse.ArgumentParser(description="Make test videos from still images")
    p.add_argument("images", nargs="*", type=Path, default=DEFAULT_IMAGES, help="pan mode: images to pan across")
    p.add_argument("--mode", choices=["pan", "scene"], default="pan")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--seconds", type=float, default=None, help="default 10 (pan) or 12 (scene)")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--size", type=int, nargs=2, default=(960, 540), metavar=("W", "H"),
                   help="output size; scene mode keeps W and derives H from the background")
    args = p.parse_args()

    args.seconds = args.seconds or (10 if args.mode == "pan" else 12)
    args.out = args.out or DATA / "samples" / f"demo_{'site' if args.mode == 'pan' else 'scene'}.mp4"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n = make_pan(args) if args.mode == "pan" else make_scene(args)
    print(f"Wrote {args.out} ({n} frames, {args.seconds:.0f}s @ {args.fps} fps)")


if __name__ == "__main__":
    main()
