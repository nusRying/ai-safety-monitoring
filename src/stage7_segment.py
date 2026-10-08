"""Stage 7: segmentation with SAM 2.1.

Boxes say roughly where an object is; masks say exactly which pixels belong to it.
SAM (Segment Anything) doesn't know classes. You give it a prompt (a box or a
point) and it returns the mask of whatever is there. So it pairs naturally with a
detector: the detector finds and names objects, SAM outlines them.

Two commands:

    compare     Detector boxes -> SAM masks, side by side with YOLO26-seg masks.
                Prints mask IoU between the two and the time each takes.

    autolabel   Text prompt -> YOLOE boxes -> SAM masks -> a YOLO segmentation
                dataset (images + polygon labels + data.yaml). This is the
                "Grounded-SAM" idea: label a new class without drawing anything.
                On the PPE test images it also scores the auto-labels against the
                human labels, so you can see how much review they'd need.

Examples:
    python src/stage7_segment.py compare --source data/samples/bus.jpg
    python src/stage7_segment.py compare --source data/construction-ppe/images/test/image536.jpg --sam sam2.1_s.pt
    python src/stage7_segment.py autolabel --prompts "safety vest" "hard hat" --limit 40
    python src/stage7_segment.py autolabel --images path/to/folder --prompts ladder --name ladders
"""

import argparse
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import yaml
from ultralytics import SAM

from common import OUTPUTS_DIR, ROOT, load_yolo

DATASET_DIR = ROOT / "data" / "construction-ppe"
# Auto-label prompt -> class name in the PPE dataset, used to score auto-labels.
PPE_CLASS_FOR_PROMPT = {"hard hat": "helmet", "helmet": "helmet", "safety vest": "vest",
                        "vest": "vest", "person": "Person", "gloves": "gloves", "boots": "boots"}


def sam_masks(sam, image, boxes):
    """Box-prompted SAM. Returns an (n, H, W) bool array, one mask per box, in box order."""
    if len(boxes) == 0:
        return np.zeros((0, *image.shape[:2]), bool)
    r = sam(image, bboxes=boxes.tolist(), verbose=False)[0]
    return r.masks.data.cpu().numpy().astype(bool)


def mask_iou(a, b):
    union = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / union if union else 0.0


def timed(fn, *args, repeat=3):
    fn(*args)                                   # warm-up (CUDA init, kernel selection)
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = fn(*args)
    return out, (time.perf_counter() - t0) / repeat * 1000


# ------------------------------------------------------------------ compare

def cmd_compare(args):
    image = cv2.imread(args.source)
    if image is None:
        raise SystemExit(f"Could not read {args.source}")
    seg = load_yolo(args.seg)
    sam = load_yolo(args.sam, SAM)

    def run_seg(img):
        return sv.Detections.from_ultralytics(seg.predict(img, conf=args.conf, classes=[0], verbose=False)[0])

    yolo_dets, t_yolo = timed(run_seg, image)
    if len(yolo_dets) == 0:
        raise SystemExit("No people found.")
    # Same boxes go to SAM, so any difference is down to the mask quality.
    masks, t_sam = timed(sam_masks, sam, image, yolo_dets.xyxy)
    sam_dets = sv.Detections(xyxy=yolo_dets.xyxy, mask=masks, class_id=yolo_dets.class_id,
                             confidence=yolo_dets.confidence)

    print(f"{len(yolo_dets)} people | {args.seg}: {t_yolo:.0f} ms (detect + masks) | "
          f"{args.sam}: {t_sam:.0f} ms (masks only, given boxes)")
    print(f"\n{'#':>3s} {'box':>24s} {'mask IoU':>9s} {'YOLO px':>9s} {'SAM px':>9s}")
    for i, (b, ym, sm) in enumerate(zip(yolo_dets.xyxy.astype(int), yolo_dets.mask, masks)):
        print(f"{i:3d} {str(b.tolist()):>24s} {mask_iou(ym, sm):9.3f} {ym.sum():9d} {sm.sum():9d}")

    ann = sv.MaskAnnotator(opacity=0.5, color_lookup=sv.ColorLookup.INDEX)
    edge = sv.PolygonAnnotator(thickness=2, color_lookup=sv.ColorLookup.INDEX)
    panels = []
    for title, dets in ((f"YOLO26-seg  {t_yolo:.0f} ms", yolo_dets), (f"SAM 2.1  {t_sam:.0f} ms", sam_dets)):
        p = edge.annotate(ann.annotate(image.copy(), dets), dets)
        cv2.rectangle(p, (0, 0), (len(title) * 14 + 16, 36), (0, 0, 0), -1)
        cv2.putText(p, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(p)
    out = np.hstack(panels)

    # Close-up of the largest person: mask edges are easier to judge zoomed in.
    i = int(np.argmax([m.sum() for m in masks]))
    x1, y1, x2, y2 = yolo_dets.xyxy[i].astype(int)
    crops = [cv2.resize(p[y1:y2, x1:x2], None, fx=out.shape[0] / (y2 - y1), fy=out.shape[0] / (y2 - y1))
             for p in panels]
    out = np.hstack([out, *crops])

    OUTPUTS_DIR.mkdir(exist_ok=True)
    dst = OUTPUTS_DIR / f"{Path(args.source).stem}_sam_compare.jpg"
    cv2.imwrite(str(dst), out)
    print(f"\nSaved {dst}")


# ---------------------------------------------------------------- autolabel

def mask_to_polygon(mask, min_area=50):
    """Largest outer contour of a mask, simplified, as an (n, 2) array. None if too small.

    YOLO segmentation labels hold one polygon per object, so for masks in several
    pieces (e.g. an arm in front of a vest) we keep the biggest piece.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    c = cv2.approxPolyDP(c, 0.002 * cv2.arcLength(c, True), True)   # fewer points, same shape
    return c.reshape(-1, 2) if len(c) >= 3 else None


def gt_boxes(img_path, class_id):
    lbl = DATASET_DIR / "labels" / img_path.parent.name / f"{img_path.stem}.txt"
    if not lbl.exists() or lbl.stat().st_size == 0:
        return np.zeros((0, 4))
    rows = np.loadtxt(lbl, ndmin=2)
    rows = rows[rows[:, 0].astype(int) == class_id]
    h, w = cv2.imread(str(img_path)).shape[:2]
    cx, cy, bw, bh = rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1).reshape(-1, 4)


def cmd_autolabel(args):
    from stage3_zeroshot import YoloeDetector

    src = Path(args.images)
    images = sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})[:args.limit]
    if not images:
        raise SystemExit(f"No images in {src}")

    detector = YoloeDetector(args.prompts)
    sam = load_yolo(args.sam, SAM)
    out_dir = ROOT / "data" / "autolabel" / args.name
    preview_dir = OUTPUTS_DIR / "stage7_autolabel" / args.name
    for d in (out_dir / "images", out_dir / "labels", preview_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Score against human labels when the images come from the PPE dataset.
    ppe_names = yaml.safe_load((ROOT / "data" / "construction-ppe.yaml").read_text())["names"] \
        if (ROOT / "data" / "construction-ppe.yaml").exists() else {}
    ppe_ids = {v: k for k, v in ppe_names.items()}
    scorable = {i: ppe_ids[PPE_CLASS_FOR_PROMPT[p]] for i, p in enumerate(args.prompts)
                if PPE_CLASS_FOR_PROMPT.get(p) in ppe_ids and DATASET_DIR in src.parents}
    stats = {i: [0, 0, 0] for i in scorable}            # true positives, auto-labels, human labels

    mask_ann = sv.MaskAnnotator(opacity=0.45)
    label_ann = sv.LabelAnnotator(text_scale=0.45, text_padding=3)
    n_objects, t0 = 0, time.perf_counter()
    for img_path in images:
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]
        dets = detector(image, args.conf)
        masks = sam_masks(sam, image, dets.xyxy)

        lines, keep = [], []
        for k, (cid, mask) in enumerate(zip(dets.class_id, masks)):
            poly = mask_to_polygon(mask)
            if poly is None:
                continue
            keep.append(k)
            coords = " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in poly)
            lines.append(f"{cid} {coords}")
        n_objects += len(lines)

        shutil.copy(img_path, out_dir / "images" / img_path.name)
        (out_dir / "labels" / f"{img_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")

        kept = dets[np.array(keep, dtype=int)]
        kept.mask = masks[keep] if keep else None
        if len(kept):
            preview = mask_ann.annotate(image.copy(), kept)
            preview = label_ann.annotate(preview, kept, [args.prompts[c] for c in kept.class_id])
        else:
            preview = image
        cv2.imwrite(str(preview_dir / img_path.name), preview)

        for pid, ds_id in scorable.items():
            pred = kept.xyxy[kept.class_id == pid] if len(kept) else np.zeros((0, 4))
            gt = gt_boxes(img_path, ds_id)
            stats[pid][1] += len(pred)
            stats[pid][2] += len(gt)
            if len(pred) and len(gt):
                iou = sv.box_iou_batch(pred, gt)
                # Greedy one-to-one matching at IoU >= 0.5.
                matched = 0
                while iou.size and iou.max() >= 0.5:
                    r, c = np.unravel_index(iou.argmax(), iou.shape)
                    matched += 1
                    iou[r, :], iou[:, c] = -1, -1
                stats[pid][0] += matched

    names = dict(enumerate(args.prompts))
    (out_dir / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(out_dir), "train": "images", "val": "images", "names": names}, sort_keys=False), encoding="utf-8")
    dt = time.perf_counter() - t0
    print(f"\nAuto-labelled {len(images)} images, {n_objects} objects in {dt:.0f}s "
          f"({dt / len(images):.2f}s per image)")
    print(f"Dataset: {out_dir}  (YOLO-seg format, train with model=yolo26n-seg.pt data={out_dir / 'data.yaml'})")
    print(f"Previews: {preview_dir}")

    if stats:
        print(f"\nAgainst the human labels (box IoU >= 0.5):")
        print(f"{'prompt':>14s} {'precision':>10s} {'recall':>8s} {'auto':>6s} {'human':>6s}")
        for pid, (tp, n_pred, n_gt) in stats.items():
            prec = tp / n_pred if n_pred else float("nan")
            rec = tp / n_gt if n_gt else float("nan")
            print(f"{args.prompts[pid]:>14s} {prec:10.2f} {rec:8.2f} {n_pred:6d} {n_gt:6d}")


def main():
    p = argparse.ArgumentParser(description="Stage 7 - SAM segmentation")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compare", help="SAM vs YOLO26-seg masks on one image")
    c.add_argument("--source", default=str(ROOT / "data" / "samples" / "bus.jpg"))
    c.add_argument("--seg", default="yolo26n-seg.pt")
    c.add_argument("--sam", default="sam2.1_t.pt", help="sam2.1_t / _s / _b / _l: bigger = better edges, slower")
    c.add_argument("--conf", type=float, default=0.4)

    a = sub.add_parser("autolabel", help="text prompts -> YOLOE boxes -> SAM masks -> YOLO-seg dataset")
    a.add_argument("--images", default=str(DATASET_DIR / "images" / "test"))
    a.add_argument("--prompts", nargs="+", default=["safety vest", "hard hat"])
    a.add_argument("--name", default="ppe_auto")
    a.add_argument("--limit", type=int, default=40)
    a.add_argument("--conf", type=float, default=0.3)
    a.add_argument("--sam", default="sam2.1_t.pt")

    args = p.parse_args()
    cmd_compare(args) if args.cmd == "compare" else cmd_autolabel(args)


if __name__ == "__main__":
    main()
