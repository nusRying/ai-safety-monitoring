"""Stage 3: zero-shot (open-vocabulary) detection from text prompts.

No training: you type what you want to find ("hard hat", "ladder", "forklift")
and the model finds it. Two backends:

    yoloe   YOLOE-26 (Ultralytics). Real-time; text prompts are encoded once by a
            small CLIP text encoder and turned into class weights.
    gdino   Grounding DINO tiny (Hugging Face). A transformer that reads the image
            and the text together. Slower but often more accurate on odd prompts.

Examples:
    python src/stage3_zeroshot.py --source data/samples/bus.jpg --prompts person bus "traffic sign"
    python src/stage3_zeroshot.py --backend gdino --source img.jpg --save
    python src/stage3_zeroshot.py                       # webcam, default safety prompts
    python src/stage3_zeroshot.py --eval                # zero-shot vs fine-tuned on PPE test set
    python src/stage3_zeroshot.py --eval --backend gdino
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from supervision.metrics import MeanAveragePrecision
from tqdm import tqdm
from ultralytics import YOLO, YOLOE

from stage1_detect import IMAGE_EXTS, MODELS_DIR, OUTPUTS_DIR, Annotator

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows without Developer Mode

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "construction-ppe"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_PROMPTS = ["person", "hard hat", "safety vest", "ladder", "forklift"]

# Prompt used for zero-shot -> class name in the PPE dataset, for --eval.
EVAL_PROMPTS = {"person": "Person", "hard hat": "helmet", "safety vest": "vest", "gloves": "gloves"}


class YoloeDetector:
    def __init__(self, prompts, weights="yoloe-26s-seg.pt"):
        MODELS_DIR.mkdir(exist_ok=True)
        local = MODELS_DIR / weights
        self.model = YOLOE(str(local) if local.exists() else weights)
        if not local.exists() and Path(weights).exists():
            Path(weights).replace(local)
        # Encode the prompts once; they become the classifier weights.
        self.model.set_classes(prompts, self.model.get_text_pe(prompts))
        self.names = dict(enumerate(prompts))

    def __call__(self, frame, conf):
        result = self.model.predict(frame, conf=conf, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)


class GroundingDinoDetector:
    def __init__(self, prompts, model_id="IDEA-Research/grounding-dino-tiny"):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE).eval()
        self.prompts = list(prompts)
        self.names = dict(enumerate(prompts))

    def _prompt_index(self, phrase):
        # Grounding DINO returns the text span it matched, which can be a fragment
        # ("hard") or a merge ("person hard hat"). Map it back to one prompt.
        phrase = phrase.strip().lower()
        if phrase in self.prompts:
            return self.prompts.index(phrase)
        for i, p in enumerate(self.prompts):
            if phrase and (phrase in p or p in phrase):
                return i
        return -1

    @torch.inference_mode()
    def __call__(self, frame, conf):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, text=[self.prompts], return_tensors="pt").to(DEVICE)
        outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=conf, text_threshold=conf,
            target_sizes=[frame.shape[:2]],
        )[0]
        if len(res["boxes"]) == 0:
            return sv.Detections.empty()
        labels = res["text_labels"] if "text_labels" in res else res["labels"]
        ids = np.array([self._prompt_index(l) for l in labels], dtype=int)
        keep = ids >= 0
        return sv.Detections(
            xyxy=res["boxes"].cpu().numpy().reshape(-1, 4)[keep],
            confidence=res["scores"].cpu().numpy()[keep],
            class_id=ids[keep],
        )


def build(backend, prompts):
    return YoloeDetector(prompts) if backend == "yoloe" else GroundingDinoDetector(prompts)


def run_source(det, args, annotate):
    is_image = Path(args.source).suffix.lower() in IMAGE_EXTS
    source = int(args.source) if args.source.isdigit() else args.source
    cap = None if is_image else cv2.VideoCapture(source)
    if cap is not None and not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    fps = 0.0
    while True:
        frame = cv2.imread(args.source) if is_image else cap.read()[1]
        if frame is None:
            break
        t0 = time.perf_counter()
        dets = det(frame, args.conf)
        dt = time.perf_counter() - t0
        fps = 0.9 * fps + 0.1 / dt if fps else 1.0 / dt
        out = annotate(frame, dets, det.names, fps, args.conf)

        if is_image:
            for c, p in zip(dets.class_id, dets.confidence):
                print(f"{det.names[c]:>14s}  conf={p:.2f}")
            if args.save:
                dst = OUTPUTS_DIR / f"{Path(args.source).stem}_{args.backend}.jpg"
                cv2.imwrite(str(dst), out)
                print(f"Saved {dst}")
            if not args.no_show:
                cv2.imshow(f"Stage 3 - {args.backend}", out)
                cv2.waitKey(0)
            break

        if args.no_show:
            continue
        cv2.imshow(f"Stage 3 - {args.backend} (q to quit)", out)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------- evaluation

def load_targets(img_path, w, h, class_map):
    """Read a YOLO label file and keep only the classes in class_map (dataset id -> eval id)."""
    lbl = DATASET_DIR / "labels" / "test" / f"{img_path.stem}.txt"
    rows = np.loadtxt(lbl, ndmin=2) if lbl.exists() and lbl.stat().st_size else np.zeros((0, 5))
    rows = rows[np.isin(rows[:, 0].astype(int), list(class_map))]
    cx, cy, bw, bh = rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
    return sv.Detections(
        xyxy=np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1).reshape(-1, 4),
        class_id=np.array([class_map[int(c)] for c in rows[:, 0]], dtype=int),
    )


def evaluate(args):
    imgs = sorted((DATASET_DIR / "images" / "test").glob("*.*"))
    if not imgs:
        raise SystemExit("PPE dataset not found. Run: python src/stage2_prepare.py")

    prompts = list(EVAL_PROMPTS)
    finetuned = YOLO(str(MODELS_DIR / args.finetuned))
    ds_ids = {v: k for k, v in finetuned.names.items()}
    # dataset class id -> index into prompts (the shared label space for this comparison)
    class_map = {ds_ids[EVAL_PROMPTS[p]]: i for i, p in enumerate(prompts)}

    def finetuned_det(frame, conf):
        r = finetuned.predict(frame, conf=conf, classes=list(class_map), verbose=False)[0]
        d = sv.Detections.from_ultralytics(r)
        d.class_id = np.array([class_map[int(c)] for c in d.class_id], dtype=int)
        return d

    contenders = {f"zero-shot {args.backend}": build(args.backend, prompts),
                  f"fine-tuned {args.finetuned}": finetuned_det}

    results = {}
    for name, det in contenders.items():
        metric, times = MeanAveragePrecision(), []
        det(np.zeros((640, 640, 3), np.uint8), 0.5)  # warm-up
        for img_path in tqdm(imgs, desc=name):
            frame = cv2.imread(str(img_path))
            h, w = frame.shape[:2]
            t0 = time.perf_counter()
            # Low threshold: mAP sweeps over confidence itself, so keep low-score boxes.
            preds = det(frame, args.eval_conf)
            times.append(time.perf_counter() - t0)
            metric.update(preds, load_targets(img_path, w, h, class_map))
        results[name] = (metric.compute(), 1.0 / np.mean(times[1:]))

    print(f"\nPPE test set ({len(imgs)} images), prompts -> dataset classes: {EVAL_PROMPTS}\n")
    header = f"{'model':>28s} {'mAP50':>7s} {'mAP50-95':>9s} {'FPS':>6s}   " + \
             " ".join(f"{p[:10]:>10s}" for p in prompts)
    print(header + "\n" + " " * 63 + "(per-class mAP50)")
    for name, (m, fps) in results.items():
        per_cls = dict(zip(m.matched_classes, m.ap_per_class[:, 0]))
        cells = " ".join(f"{per_cls.get(i, float('nan')):10.3f}" for i in range(len(prompts)))
        print(f"{name:>28s} {m.map50:7.3f} {m.map50_95:9.3f} {fps:6.1f}   {cells}")


def main():
    p = argparse.ArgumentParser(description="Stage 3 - zero-shot detection")
    p.add_argument("--backend", choices=["yoloe", "gdino"], default="yoloe")
    p.add_argument("--source", default="0", help="webcam index, video or image path")
    p.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS, help="what to look for")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--save", action="store_true")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--eval", action="store_true", help="compare against the fine-tuned model on the PPE test set")
    p.add_argument("--eval-conf", type=float, default=0.05)
    p.add_argument("--finetuned", default="ppe_yolo26n.pt", help="fine-tuned weights in models/")
    args = p.parse_args()

    OUTPUTS_DIR.mkdir(exist_ok=True)
    print(f"Backend: {args.backend} | device: {DEVICE}")
    if args.eval:
        evaluate(args)
        return
    det = build(args.backend, args.prompts)
    run_source(det, args, Annotator())


if __name__ == "__main__":
    main()
