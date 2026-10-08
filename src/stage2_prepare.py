"""Stage 2a: download and inspect the Construction-PPE dataset.

Downloads the dataset into data/construction-ppe, writes data/construction-ppe.yaml
pointing at it, prints how many boxes each class has per split, and saves a few
images with their ground-truth boxes drawn so you can see what the labels look like.

YOLO label format (one .txt per image, one line per box):
    class_id  x_center  y_center  width  height      (all normalised to 0..1)

Usage:
    python src/stage2_prepare.py
    python src/stage2_prepare.py --samples 8
"""

import argparse
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import yaml
from ultralytics.utils import ROOT as ULTRA_ROOT
from ultralytics.utils.downloads import download

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATASET_DIR = DATA_DIR / "construction-ppe"
DATA_YAML = DATA_DIR / "construction-ppe.yaml"
OUTPUTS_DIR = ROOT / "outputs" / "stage2_samples"
SPLITS = ("train", "val", "test")


def fetch():
    src_cfg = yaml.safe_load((ULTRA_ROOT / "cfg/datasets/construction-ppe.yaml").read_text(encoding="utf-8"))
    if not (DATASET_DIR / "images" / "train").exists():
        DATA_DIR.mkdir(exist_ok=True)
        download(src_cfg["download"], dir=DATA_DIR, unzip=True, delete=True)
    if not (DATASET_DIR / "images" / "train").exists():
        raise SystemExit(f"Download finished but {DATASET_DIR / 'images/train'} was not found.")

    cfg = {
        "path": str(DATASET_DIR),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": src_cfg["names"],
    }
    DATA_YAML.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"Wrote {DATA_YAML}")
    return cfg["names"]


def read_labels(label_file: Path):
    if not label_file.exists():
        return np.zeros((0, 5))
    rows = [l.split() for l in label_file.read_text().splitlines() if l.strip()]
    return np.array(rows, dtype=float).reshape(-1, 5)


def label_path(img: Path) -> Path:
    return DATASET_DIR / "labels" / img.parent.name / f"{img.stem}.txt"


def class_stats(names):
    print(f"\n{'class':>12s} " + " ".join(f"{s:>7s}" for s in SPLITS))
    counts = {s: Counter() for s in SPLITS}
    n_images = {}
    for split in SPLITS:
        imgs = list((DATASET_DIR / "images" / split).glob("*.*"))
        n_images[split] = len(imgs)
        for img in imgs:
            for row in read_labels(label_path(img)):
                counts[split][int(row[0])] += 1
    for cid, name in names.items():
        print(f"{name:>12s} " + " ".join(f"{counts[s][cid]:>7d}" for s in SPLITS))
    print(f"{'images':>12s} " + " ".join(f"{n_images[s]:>7d}" for s in SPLITS))


def save_samples(names, n):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    imgs = sorted((DATASET_DIR / "images" / "train").glob("*.*"))
    random.seed(0)
    box = sv.BoxAnnotator(thickness=2)
    label = sv.LabelAnnotator(text_scale=0.5, text_padding=3)
    for img_path in random.sample(imgs, min(n, len(imgs))):
        frame = cv2.imread(str(img_path))
        h, w = frame.shape[:2]
        rows = read_labels(label_path(img_path))
        if len(rows) == 0:
            continue
        # Convert normalised (cx, cy, w, h) to pixel (x1, y1, x2, y2).
        cx, cy, bw, bh = rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        dets = sv.Detections(xyxy=xyxy, class_id=rows[:, 0].astype(int))
        out = box.annotate(frame.copy(), dets)
        out = label.annotate(out, dets, [names[c] for c in dets.class_id])
        cv2.imwrite(str(OUTPUTS_DIR / f"{img_path.stem}_gt.jpg"), out)
    print(f"\nSaved ground-truth samples to {OUTPUTS_DIR}")


def main():
    p = argparse.ArgumentParser(description="Stage 2a - prepare PPE dataset")
    p.add_argument("--samples", type=int, default=6, help="number of labelled samples to save")
    args = p.parse_args()

    names = fetch()
    class_stats(names)
    save_samples(names, args.samples)


if __name__ == "__main__":
    main()
