"""Export models to TensorRT engines and check they're faster and still accurate.

TensorRT compiles a network for one specific GPU: it fuses layers, picks the
fastest kernels for your card and, with FP16, halves the precision of the maths.
The result (.engine) only runs on this GPU model with this TensorRT version, so
it's built locally rather than downloaded, and rebuilt after a driver/TensorRT upgrade.

Commands:
    export      .pt -> .onnx -> .engine (FP16, 640x640) for each model, into models/
    benchmark   PyTorch vs TensorRT on the demo video: ms per frame, and how many
                detections agree (same class, box IoU >= 0.5)
    val         PPE test-set mAP for both, to confirm FP16 didn't cost accuracy

Examples:
    python src/export_trt.py export                       # PPE + pose models (~3 min each)
    python src/export_trt.py export --models yolo26n.pt --force
    python src/export_trt.py benchmark
    python src/export_trt.py val

Then use the engines with --trt on Stages 4-6 and the pipeline, or the dashboard toggle.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from common import MODELS_DIR, ROOT

DEFAULT_MODELS = ["ppe_yolo26n.pt", "yolo26n-pose.pt"]
DEMO_VIDEO = ROOT / "data" / "samples" / "demo_site.mp4"


def engine_path(pt_name: str) -> Path:
    return MODELS_DIR / f"{Path(pt_name).stem}.engine"


def cmd_export(args):
    for name in args.models:
        pt = MODELS_DIR / name
        if not pt.exists():
            print(f"skip {name}: not in models/ (run the stage that downloads or trains it first)")
            continue
        eng = engine_path(name)
        if eng.exists() and eng.stat().st_mtime > pt.stat().st_mtime and not args.force:
            print(f"skip {name}: {eng.name} is up to date (use --force to rebuild)")
            continue
        print(f"exporting {name} -> {eng.name} (FP16, {args.imgsz}x{args.imgsz}); this takes a few minutes...")
        t0 = time.perf_counter()
        # quantize="fp16" (replaces the deprecated half=True). With TensorRT 11, Ultralytics
        # converts the ONNX graph to FP16 with NVIDIA modelopt before building the engine.
        out = YOLO(str(pt)).export(format="engine", quantize="fp16", imgsz=args.imgsz, device=0, verbose=False)
        print(f"  done in {time.perf_counter() - t0:.0f}s: {out}")


def read_frames(n):
    cap = cv2.VideoCapture(str(DEMO_VIDEO))
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit(f"Could not read {DEMO_VIDEO}. Run: python src/make_demo_video.py")
    return frames


def run(model, frames):
    """Predict every frame; return (detections per frame, mean ms per stage)."""
    model.predict(frames[0], verbose=False)                # warm-up
    dets, speeds = [], []
    for f in frames:
        r = model.predict(f, conf=0.25, verbose=False)[0]
        dets.append(sv.Detections.from_ultralytics(r))
        speeds.append([r.speed["preprocess"], r.speed["inference"], r.speed["postprocess"]])
    return dets, np.mean(speeds, axis=0)


def agreement(a_list, b_list):
    """Share of detections in a with a same-class match in b at IoU >= 0.5 (and vice versa)."""
    matched_a = matched_b = total_a = total_b = 0
    for a, b in zip(a_list, b_list):
        total_a, total_b = total_a + len(a), total_b + len(b)
        if len(a) == 0 or len(b) == 0:
            continue
        iou = sv.box_iou_batch(a.xyxy, b.xyxy)
        iou[a.class_id[:, None] != b.class_id[None, :]] = 0
        matched_a += int((iou.max(axis=1) >= 0.5).sum())
        matched_b += int((iou.max(axis=0) >= 0.5).sum())
    return matched_a / max(total_a, 1), matched_b / max(total_b, 1), total_a, total_b


def cmd_benchmark(args):
    frames = read_frames(args.frames)
    print(f"{len(frames)} frames of {DEMO_VIDEO.name}, conf 0.25\n")
    print(f"{'model':>18s} {'backend':>9s} {'pre':>6s} {'infer':>7s} {'post':>6s} {'total ms':>9s} {'FPS':>6s}")
    for name in args.models:
        eng = engine_path(name)
        if not eng.exists():
            print(f"{name}: no engine yet, run: python src/export_trt.py export")
            continue
        results = {}
        for backend, path in (("pytorch", MODELS_DIR / name), ("tensorrt", eng)):
            dets, (pre, inf, post) = run(YOLO(str(path)), frames)
            total = pre + inf + post
            results[backend] = (dets, total)
            print(f"{Path(name).stem:>18s} {backend:>9s} {pre:6.1f} {inf:7.1f} {post:6.1f} {total:9.1f} {1000 / total:6.1f}")
        pt_dets, pt_ms = results["pytorch"]
        trt_dets, trt_ms = results["tensorrt"]
        a, b, na, nb = agreement(pt_dets, trt_dets)
        print(f"{'':>18s} speed-up x{pt_ms / trt_ms:.2f} | detections: pytorch {na}, tensorrt {nb} | "
              f"{a:.1%} of pytorch's found by tensorrt, {b:.1%} of tensorrt's found by pytorch\n")


def cmd_val(args):
    data = ROOT / "data" / "construction-ppe.yaml"
    if not data.exists():
        raise SystemExit("PPE dataset not prepared. Run: python src/stage2_prepare.py")
    rows = []
    for backend, path in (("pytorch", MODELS_DIR / "ppe_yolo26n.pt"), ("tensorrt", engine_path("ppe_yolo26n.pt"))):
        if not path.exists():
            raise SystemExit(f"Missing {path.name}")
        # Engines are built for batch 1 and a fixed 640x640 input.
        m = YOLO(str(path)).val(data=str(data), split="test", imgsz=640, batch=1, verbose=False, plots=False,
                                project=str(ROOT / "runs" / "trt_val"), name=backend, exist_ok=True)
        rows.append((backend, m.box.map50, m.box.map, m.box.mp, m.box.mr))
    print(f"\n{'backend':>9s} {'mAP50':>7s} {'mAP50-95':>9s} {'precision':>10s} {'recall':>7s}")
    for b, m50, m, p, r in rows:
        print(f"{b:>9s} {m50:7.3f} {m:9.3f} {p:10.3f} {r:7.3f}")


def main():
    p = argparse.ArgumentParser(description="TensorRT export and benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    e.add_argument("--imgsz", type=int, default=640)
    e.add_argument("--force", action="store_true", help="rebuild even if the engine is up to date")
    b = sub.add_parser("benchmark")
    b.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    b.add_argument("--frames", type=int, default=150)
    sub.add_parser("val")
    args = p.parse_args()
    {"export": cmd_export, "benchmark": cmd_benchmark, "val": cmd_val}[args.cmd](args)


if __name__ == "__main__":
    main()
