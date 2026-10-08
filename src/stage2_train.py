"""Stage 2b: fine-tune YOLO on Construction-PPE and evaluate it.

Starts from COCO-pretrained weights (transfer learning) and trains on the PPE
dataset prepared by stage2_prepare.py. When training finishes it evaluates the
best checkpoint on the held-out test split and copies it to models/.

Defaults are tuned for a 4 GB laptop GPU (RTX 3050).

Usage:
    python src/stage2_train.py                      # 50 epochs, yolo26n
    python src/stage2_train.py --epochs 1           # quick smoke test
    python src/stage2_train.py --model yolo26s.pt --batch 8
    python src/stage2_train.py --eval-only models/ppe_yolo26n.pt

Training curves, confusion matrix and PR curves are written to runs/ppe/<name>/.
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data" / "construction-ppe.yaml"
MODELS_DIR = ROOT / "models"
RUNS_DIR = ROOT / "runs" / "ppe"


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2b - fine-tune YOLO on PPE")
    p.add_argument("--model", default="yolo26n.pt", help="pretrained weights to start from")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16, help="lower this if you run out of GPU memory")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workers", type=int, default=2, help="dataloader processes (keep low on Windows)")
    p.add_argument("--patience", type=int, default=15, help="stop early after N epochs without improvement")
    p.add_argument("--name", default=None, help="run name, defaults to the model name")
    p.add_argument("--eval-only", default=None, metavar="WEIGHTS", help="skip training and evaluate these weights")
    return p.parse_args()


def evaluate(weights: Path, imgsz: int, batch: int):
    model = YOLO(str(weights))
    m = model.val(data=str(DATA_YAML), split="test", imgsz=imgsz, batch=batch,
                  project=str(RUNS_DIR), name=f"{weights.stem}_test", exist_ok=True, plots=True)

    print(f"\nTest set results for {weights.name}")
    print(f"  mAP50-95 {m.box.map:.3f}   mAP50 {m.box.map50:.3f}   "
          f"precision {m.box.mp:.3f}   recall {m.box.mr:.3f}")
    print(f"\n{'class':>12s} {'mAP50':>7s} {'mAP50-95':>9s}")
    for i, cid in enumerate(m.box.ap_class_index):
        print(f"{model.names[int(cid)]:>12s} {m.box.ap50[i]:7.3f} {m.box.ap[i]:9.3f}")


def main():
    args = parse_args()
    if not DATA_YAML.exists():
        raise SystemExit("Dataset not prepared. Run: python src/stage2_prepare.py")

    if args.eval_only:
        evaluate(Path(args.eval_only), args.imgsz, args.batch)
        return

    MODELS_DIR.mkdir(exist_ok=True)
    local = MODELS_DIR / args.model
    model = YOLO(str(local) if local.exists() else args.model)
    name = args.name or Path(args.model).stem

    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        patience=args.patience,
        project=str(RUNS_DIR),
        name=name,
        exist_ok=True,
        amp=True,       # mixed precision: less VRAM, faster on RTX cards
        cache=False,    # set "ram" if you have 16 GB+ RAM to speed up epochs
        plots=True,
    )

    best = RUNS_DIR / name / "weights" / "best.pt"
    dst = MODELS_DIR / f"ppe_{name}.pt"
    shutil.copy(best, dst)
    print(f"\nCopied best checkpoint to {dst}")
    evaluate(dst, args.imgsz, args.batch)


if __name__ == "__main__":
    main()
