"""Stage 6: fall detection from body pose.

A pose model finds 17 body keypoints per person (COCO order: nose, eyes, ears,
shoulders, elbows, wrists, hips, knees, ankles). From them we compute the torso
angle: the line from mid-hip to mid-shoulder, measured from vertical.

    upright   torso angle < 35 deg
    lying     torso angle > 60 deg

Lying down is not a fall: someone working under a vehicle lies down on purpose.
So a fall needs two things:
    1. The person went from upright to lying FAST (within --fall-window seconds).
    2. They stay down for --confirm-s seconds (a stumble they recover from is ignored).
Anyone lying still for --down-s seconds raises a second alert, however they got there.

If the shoulders or hips aren't visible, the box shape is used as a fallback:
a box much wider than it is tall usually means lying.

Detectors trained on COCO rarely see people lying down, so they often lose a
person the moment they fall. When a tracked person vanishes mid-frame (not at an
edge), we crop the area where they were, rotate it 90 degrees both ways, and run
the pose model again. A lying person becomes an upright person in the rotated
crop, which the model finds easily. Keypoints are then rotated back.

Examples:
    python src/stage6_fall.py --source data/samples/demo_scene.mp4 --save
    python src/stage6_fall.py                 # webcam: try sitting, then lying down slowly vs quickly
"""

import argparse
from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv
import torch

from common import EventLog, FpsMeter, TrackStitcher, VideoIO, draw_hud, head_point, load_yolo, track, trt_weights

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12
COLORS = {  # BGR
    "upright": (68, 179, 47),
    "bending": (200, 200, 200),
    "unknown": (150, 150, 150),
    "lying": (0, 165, 255),
    "FALLEN": (58, 52, 229),
}


def parse_args():
    p = argparse.ArgumentParser(description="Stage 6 - fall detection")
    p.add_argument("--source", default="0")
    p.add_argument("--model", default="yolo26n-pose.pt")
    p.add_argument("--track-conf", type=float, default=0.1)
    p.add_argument("--tracker", default="botsort.yaml", choices=["botsort.yaml", "bytetrack.yaml"])
    p.add_argument("--min-size", type=int, default=60, help="ignore people whose longest box side is below this")
    p.add_argument("--kp-conf", type=float, default=0.3, help="keypoint confidence needed to use it")
    p.add_argument("--fall-window", type=float, default=1.0, help="max seconds from upright to lying for a fall")
    p.add_argument("--confirm-s", type=float, default=0.5, help="seconds lying before a fall is confirmed")
    p.add_argument("--down-s", type=float, default=5.0, help="seconds lying before a 'person down' alert")
    p.add_argument("--search-s", type=float, default=30.0,
                   help="keep looking for a vanished person (rotated search) for this many seconds")
    p.add_argument("--no-recovery", action="store_true", help="disable the rotated search, to compare")
    p.add_argument("--no-stitch", action="store_true", help="disable track stitching, to compare")
    p.add_argument("--trt", action="store_true", help="use the TensorRT engine (build with src/export_trt.py)")
    p.add_argument("--save", action="store_true")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def torso_angle(kp, kc, min_conf):
    """Degrees between the hip->shoulder line and vertical. None if not visible.

    0 = upright, 90 = horizontal, >90 = upside down. Image y grows downwards,
    hence the minus sign.
    """
    sh = kp[[L_SHOULDER, R_SHOULDER]][kc[[L_SHOULDER, R_SHOULDER]] >= min_conf]
    hp = kp[[L_HIP, R_HIP]][kc[[L_HIP, R_HIP]] >= min_conf]
    if len(sh) == 0 or len(hp) == 0:
        return None
    dx, dy = sh.mean(axis=0) - hp.mean(axis=0)
    if dx == 0 and dy == 0:
        return None
    return float(np.degrees(np.arctan2(abs(dx), -dy)))


@dataclass
class PersonState:
    state: str = "upright"
    last_upright_t: float | None = None
    lying_since: float | None = None
    fast_drop: bool = False
    angle: float | None = None
    last_seen: float | None = None
    last_lying_t: float | None = None    # last frame that actually showed the person lying


def box_cut_by_frame(xyxy, frame_wh, margin=0.02):
    """True if the box reaches the left, right or bottom frame edge (within `margin` x frame
    size): part of the body is probably outside. Boxes jitter, so a box a few pixels short of
    the edge still counts (a fixed 4 px let one frame through and caused a false alarm)."""
    x1, _, x2, y2 = xyxy
    w, h = frame_wh
    return x1 <= margin * w or x2 >= w * (1 - margin) or y2 >= h * (1 - margin)


class FallDetector:
    # "Down for 5 s" must mean 5 s of being seen lying. If the person hasn't been seen for
    # longer than this, or no frame has shown them lying for longer than this (frames with
    # no posture verdict don't count), the lying timer starts again.
    MAX_GAP_S = 1.0

    def __init__(self, args):
        self.args = args
        self.people: dict[int, PersonState] = {}

    def update(self, tid, t, angle, aspect, box_cut=False):
        """box_cut: the person's box is cut off by the frame edge (see box_cut_by_frame)."""
        a = self.args
        s = self.people.setdefault(tid, PersonState())
        unseen_too_long = s.last_seen is not None and t - s.last_seen > self.MAX_GAP_S
        no_recent_lying = s.lying_since is not None and t - (s.last_lying_t or s.lying_since) > self.MAX_GAP_S
        if unseen_too_long or no_recent_lying:
            s.lying_since, s.fast_drop = None, False
            if s.state in ("FALLEN", "lying"):
                s.state = "unknown"
        s.last_seen = t
        s.angle = angle
        if angle is not None:
            upright, lying = angle < 35, angle > 60
        elif not box_cut:                       # keypoints hidden: fall back to box shape
            upright, lying = aspect < 0.8, aspect > 1.3
        else:
            # Hips/shoulders not visible and the box is cut by the frame edge (e.g. someone
            # at a desk seen from the chest up: a wide box, but not lying). No posture verdict.
            upright = lying = False
            if s.state not in ("FALLEN", "lying"):
                s.state = "unknown"

        if upright:
            s.last_upright_t, s.lying_since, s.fast_drop = t, None, False
            s.state = "upright"
        elif lying:
            s.last_lying_t = t
            if s.lying_since is None:
                s.lying_since = t
                s.fast_drop = s.last_upright_t is not None and t - s.last_upright_t <= a.fall_window
            down_for = t - s.lying_since
            s.state = "FALLEN" if s.fast_drop and down_for >= a.confirm_s else "lying"
        elif angle is not None or not box_cut:
            # In between (sitting, bending, mid-fall): keep a confirmed fall, otherwise "bending".
            if s.state not in ("FALLEN", "lying"):
                s.state = "bending"

        reasons = []
        if s.state == "FALLEN":
            reasons.append("fall detected")
        if s.lying_since is not None and t - s.lying_since >= a.down_s:
            reasons.append(f"person down > {a.down_s:g}s")
        return s, reasons


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class RotationRecovery:
    """Re-find people the tracker lost because they are lying down."""

    # cv2 rotation -> maps (x, y) in the rotated crop back to the unrotated crop (w, h = crop size)
    ROTATIONS = {
        cv2.ROTATE_90_CLOCKWISE: lambda x, y, w, h: (y, h - 1 - x),
        cv2.ROTATE_90_COUNTERCLOCKWISE: lambda x, y, w, h: (w - 1 - y, x),
    }

    def __init__(self, model, args, frame_wh):
        self.model, self.args = model, args
        self.W, self.H = frame_wh
        self.last = {}                          # tid -> (last box, time last seen)

    def remember(self, tid, xyxy, t):
        self.last[tid] = (np.asarray(xyxy, dtype=float), t)

    def search(self, frame, seen_ids, tracked_boxes, t):
        """Return {tid: (xyxy, kp_xy, kp_conf)} for vanished people found by rotated search."""
        found = {}
        for tid, (box, t_seen) in list(self.last.items()):
            if tid in seen_ids:
                continue
            m = 8
            left_frame = box[0] <= m or box[2] >= self.W - m or box[1] <= m
            if t - t_seen > self.args.search_s or left_frame:
                del self.last[tid]
                continue
            hit = self._search_one(frame, box)
            if hit is not None and not any(box_iou(hit[0], tb) > 0.5 for tb in tracked_boxes):
                found[tid] = hit
                self.remember(tid, hit[0], t)
        return found

    def _search_one(self, frame, box):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        side = 1.5 * max(box[2] - box[0], box[3] - box[1])     # a lying person spans their standing height
        x0, y0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
        x1, y1 = int(min(cx + side / 2, self.W)), int(min(cy + side / 2, self.H))
        crop = frame[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        best = None
        for code, unrotate in self.ROTATIONS.items():
            r = self.model.predict(cv2.rotate(crop, code), conf=0.4, verbose=False)[0]
            if r.keypoints is None or len(r.boxes) == 0:
                continue
            i = int(r.boxes.conf.argmax())
            conf = float(r.boxes.conf[i])
            if best is not None and conf <= best[3]:
                continue
            kp = r.keypoints.xy[i].cpu().numpy()
            kc = r.keypoints.conf[i].cpu().numpy()
            kp = np.array([unrotate(x, y, cw, ch) for x, y in kp]) + [x0, y0]
            bx1, by1, bx2, by2 = r.boxes.xyxy[i].cpu().numpy()
            corners = np.array([unrotate(x, y, cw, ch) for x, y in ((bx1, by1), (bx2, by2))]) + [x0, y0]
            xyxy = np.concatenate([corners.min(axis=0), corners.max(axis=0)])
            best = (xyxy, kp, kc, conf)
        return None if best is None else best[:3]


def main():
    args = parse_args()
    args.model = trt_weights(args.model, args.trt)
    model = load_yolo(args.model)
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Model: {args.model} | device: {device}")

    video = VideoIO(args.source, "fall" if args.save else None, not args.no_show, "Stage 6 - fall detection")
    events = EventLog(video.name, "stage6")
    detector, meter = FallDetector(args), FpsMeter()
    # A separate model instance: predicting on crops with the tracking model would
    # overwrite the tracker's previous frame and break BoT-SORT's motion compensation.
    recovery = None if args.no_recovery else RotationRecovery(load_yolo(args.model), args, video.size)
    stitcher = None if args.no_stitch else TrackStitcher()
    edges = sv.EdgeAnnotator(color=sv.Color.from_hex("#00d4ff"), thickness=2)
    vertices = sv.VertexAnnotator(color=sv.Color.from_hex("#ffffff"), radius=3)

    for frame in video.frames():
        meter.start()
        now = video.now()
        dets, result = track(model, frame, args.track_conf, args.tracker)
        out = frame.copy()

        # people: (tid, xyxy, keypoints xy, keypoint conf, found by rotated search?)
        people = []
        if len(dets) and result.keypoints is not None:
            kps = sv.KeyPoints.from_ultralytics(result)
            w = dets.xyxy[:, 2] - dets.xyxy[:, 0]
            h = dets.xyxy[:, 3] - dets.xyxy[:, 1]
            keep = np.maximum(w, h) >= args.min_size
            dets, kps = dets[keep], kps[keep]
            out = edges.annotate(out, kps)
            out = vertices.annotate(out, kps)
            people = [(int(t), b, kps.xy[i], kps.keypoint_confidence[i], False)
                      for i, (b, t) in enumerate(zip(dets.xyxy, dets.tracker_id))]

        if recovery:
            for tid, b, *_ in people:
                recovery.remember(tid, b, now)
            found = recovery.search(frame, {p[0] for p in people}, [p[1] for p in people], now)
            for tid, (b, kp, kc) in found.items():
                people.append((tid, b, kp, kc, True))
                for (x, y), c in zip(kp.astype(int), kc):
                    if c >= args.kp_conf:
                        cv2.circle(out, (int(x), int(y)), 4, (255, 0, 255), -1)

        n_fallen = 0
        canon = {p[0]: p[0] for p in people}
        if stitcher and people:
            canon = stitcher.update(frame, [p[0] for p in people], [p[1] for p in people], now,
                                    [head_point(p[2], p[3]) for p in people])
        for tid, xyxy, kp, kc, recovered in people:
            angle = torso_angle(kp, kc, args.kp_conf)
            aspect = (xyxy[2] - xyxy[0]) / max(xyxy[3] - xyxy[1], 1)
            s, reasons = detector.update(tid, now, angle, aspect, box_cut_by_frame(xyxy, video.size))
            for r in reasons:
                events.log(frame, now, video.idx, canon[tid], r, xyxy)
            n_fallen += s.state == "FALLEN"

            color = COLORS[s.state]
            x1, y1, x2, y2 = np.asarray(xyxy).astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3 if s.state == "FALLEN" else 2)
            ang = f"{s.angle:.0f}deg" if s.angle is not None else "box"
            label = f"#{tid} {s.state} {ang}"
            if s.lying_since is not None:
                label += f" {now - s.lying_since:.1f}s"
            if recovered:
                label += " (rotated)"
            cv2.putText(out, label, (x1, max(y1 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        fps = meter.stop()
        hud = f"FPS {fps:5.1f} | t {now:5.1f}s | people {len(people)} | fallen {n_fallen} | events {events.count}"
        if not video.emit(draw_hud(out, hud)):
            break

    video.close()
    events.close()
    if stitcher:
        stitcher.report()
    print(f"\n{video.idx + 1} frames | {events.count} events | {events.csv_path}")


if __name__ == "__main__":
    main()
