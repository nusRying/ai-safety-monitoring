"""Stage 1: real-time object detection with a pretrained YOLO model.

Runs a COCO-pretrained YOLO model on a webcam, video file or image and draws
boxes, labels and an FPS / person-count overlay.

The pretrained model knows 80 COCO classes ("person", "car", "chair", ...) but
not "helmet" or "vest" yet -- that comes in Stage 2 when we fine-tune on a PPE
dataset. Here the goal is to understand what a detector outputs and how the
confidence and IoU thresholds change it.

Examples:
    python src/stage1_detect.py                          # webcam 0
    python src/stage1_detect.py --source data/samples/street.mp4
    python src/stage1_detect.py --source img.jpg --save
    python src/stage1_detect.py --classes person         # only people
    python src/stage1_detect.py --model yolo11n.pt       # compare models

Keys while the window is open:
    q / Esc   quit
    + / -     raise / lower confidence threshold by 0.05
    s         save the current frame to outputs/
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 - YOLO detection")
    p.add_argument("--source", default="0", help="webcam index, video path or image path")
    p.add_argument("--model", default="yolo26n.pt", help="YOLO weights (downloaded to models/ on first use)")
    p.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS (ignored by NMS-free models like YOLO26)")
    p.add_argument("--imgsz", type=int, default=640, help="inference image size")
    p.add_argument("--classes", nargs="*", default=None, help="class names to keep, e.g. --classes person car")
    p.add_argument("--save", action="store_true", help="save annotated output to outputs/")
    p.add_argument("--no-show", action="store_true", help="don't open a window (useful for batch runs)")
    return p.parse_args()


def load_model(name: str) -> YOLO:
    MODELS_DIR.mkdir(exist_ok=True)
    local = MODELS_DIR / name
    model = YOLO(str(local) if local.exists() else name)
    # Ultralytics downloads bare names into the CWD; move them into models/.
    downloaded = Path(name)
    if not local.exists() and downloaded.exists():
        downloaded.replace(local)
    return model


def class_ids_from_names(model: YOLO, names):
    if not names:
        return None
    lookup = {v.lower(): k for k, v in model.names.items()}
    unknown = [n for n in names if n.lower() not in lookup]
    if unknown:
        raise SystemExit(f"Unknown classes {unknown}. Available: {sorted(lookup)}")
    return [lookup[n.lower()] for n in names]


class Annotator:
    # PPE models (Stage 2) name violations "no_helmet", "no_gloves", ...; draw those in red.
    RED = sv.Color.from_hex("#e5343a")

    def __init__(self):
        self.box = sv.BoxAnnotator(thickness=2)
        self.label = sv.LabelAnnotator(text_scale=0.5, text_padding=4)
        self.bad_box = sv.BoxAnnotator(color=self.RED, thickness=3)
        self.bad_label = sv.LabelAnnotator(color=self.RED, text_scale=0.5, text_padding=4)

    def __call__(self, frame, detections: sv.Detections, names, fps, conf):
        is_bad = np.array([names[c].lower().startswith("no_") for c in detections.class_id], dtype=bool)
        out = frame.copy()
        for dets, box, label in ((detections[~is_bad], self.box, self.label),
                                 (detections[is_bad], self.bad_box, self.bad_label)):
            labels = [f"{names[c]} {p:.2f}" for c, p in zip(dets.class_id, dets.confidence)]
            out = box.annotate(out, dets)
            out = label.annotate(out, dets, labels)

        people = int(np.sum([names[c].lower() == "person" for c in detections.class_id]))
        hud = (f"FPS {fps:5.1f} | objects {len(detections)} | people {people} | "
               f"violations {int(is_bad.sum())} | conf>={conf:.2f}")
        cv2.rectangle(out, (0, 0), (len(hud) * 9 + 12, 28), (0, 0, 0), -1)
        cv2.putText(out, hud, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return out


def detect(model, frame, args, class_ids, conf):
    result = model.predict(
        frame, conf=conf, iou=args.iou, imgsz=args.imgsz,
        classes=class_ids, verbose=False,
    )[0]
    return sv.Detections.from_ultralytics(result)


def run_image(model, args, class_ids, annotate):
    frame = cv2.imread(args.source)
    if frame is None:
        raise SystemExit(f"Could not read image: {args.source}")
    t0 = time.perf_counter()
    dets = detect(model, frame, args, class_ids, args.conf)
    fps = 1.0 / (time.perf_counter() - t0)
    out = annotate(frame, dets, model.names, fps, args.conf)

    for c, p, xyxy in zip(dets.class_id, dets.confidence, dets.xyxy):
        print(f"{model.names[c]:>12s}  conf={p:.2f}  box={xyxy.round().astype(int).tolist()}")

    if args.save:
        dst = OUTPUTS_DIR / f"{Path(args.source).stem}_det.jpg"
        cv2.imwrite(str(dst), out)
        print(f"Saved {dst}")
    if not args.no_show:
        cv2.imshow("Stage 1 - detection", out)
        cv2.waitKey(0)


def run_stream(model, args, class_ids, annotate):
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    writer = None
    if args.save:
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        name = "webcam" if isinstance(source, int) else Path(args.source).stem
        dst = OUTPUTS_DIR / f"{name}_det.mp4"
        writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
        print(f"Recording to {dst}")

    conf, fps = args.conf, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        dets = detect(model, frame, args, class_ids, conf)
        # Exponential moving average keeps the FPS readout stable.
        fps = 0.9 * fps + 0.1 * (1.0 / (time.perf_counter() - t0)) if fps else 1.0 / (time.perf_counter() - t0)
        out = annotate(frame, dets, model.names, fps, conf)

        if writer:
            writer.write(out)
        if args.no_show:
            continue

        cv2.imshow("Stage 1 - detection (q to quit)", out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key in (ord("+"), ord("=")):
            conf = min(conf + 0.05, 0.95)
        elif key == ord("-"):
            conf = max(conf - 0.05, 0.05)
        elif key == ord("s"):
            dst = OUTPUTS_DIR / f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(str(dst), out)
            print(f"Saved {dst}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    OUTPUTS_DIR.mkdir(exist_ok=True)

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    model = load_model(args.model)
    print(f"Model: {args.model} | device: {device} | classes: {len(model.names)}")
    class_ids = class_ids_from_names(model, args.classes)
    annotate = Annotator()

    # The first GPU call pays for CUDA init and kernel selection; do it before timing.
    model.predict(np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8), imgsz=args.imgsz, verbose=False)

    if Path(args.source).suffix.lower() in IMAGE_EXTS:
        run_image(model, args, class_ids, annotate)
    else:
        run_stream(model, args, class_ids, annotate)


if __name__ == "__main__":
    main()
