"""Stage 4: track people and flag missing PPE with a rule.

Stage 2 showed the model is good at finding helmets and people but poor at
finding "no_helmet" directly. So instead of trusting a single "no_helmet" box,
we combine reliable detections with a rule:

    1. Detect people and PPE items with the fine-tuned model.
    2. Track people (BoT-SORT) so each one keeps the same ID across frames.
       People smaller than --min-height are skipped: too few pixels to judge PPE.
    3. For each person, look for each required PPE item inside the body region
       where it should be (helmet: top of the box; vest: torso).
    4. Smooth over time: a person is only flagged if the item has been missing
       in most of their last N frames. One missed detection doesn't raise an alarm.
    5. Head cut-off rule: head items (helmet, goggles) are only judged in frames
       where the head is in view. A pose model runs alongside the PPE model; its
       detections are paired with the tracked people by box overlap, and a person's
       head counts as visible when enough head keypoints (nose, eyes, ears) are
       confidently inside the frame. People the pose model misses fall back to a
       box test (box not touching the top/left/right edge). Frames that fail are
       skipped, not counted as missing.
    6. Track stitching: when the tracker loses someone and gives them a new ID,
       common.TrackStitcher re-links the new ID to the old person (nearby, similar
       size, similar colours), so the same person doesn't raise the same alarm twice.
    7. When someone becomes a violator, write a row to events.csv and save a crop.
       Stage 8 will send these crops to a vision LLM for a written report.

Examples:
    python src/make_demo_video.py                        # builds data/samples/demo_site.mp4
    python src/stage4_track.py --source data/samples/demo_site.mp4 --save
    python src/stage4_track.py                           # webcam
    python src/stage4_track.py --require helmet vest     # check vests too
    python src/stage4_track.py --head-check box          # compare: box-only head rule (or off)
"""

import argparse
from collections import defaultdict, deque

import numpy as np
import supervision as sv
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from common import (HEAD_KEYPOINTS, MODELS_DIR, EventLog, FpsMeter, TrackStitcher, VideoIO, draw_hud, head_point,
                    load_yolo, track, trt_weights)

# Where each PPE item should appear, as (top, bottom) fractions of the person box height.
PPE_REGIONS = {
    "helmet": (0.0, 0.30),
    "goggles": (0.0, 0.30),
    "vest": (0.15, 0.70),
    "gloves": (0.25, 0.90),
    "boots": (0.75, 1.05),
}

HEAD_ITEMS = {"helmet", "goggles"}       # items that need the head in view to judge

GREEN, RED, GREY = (sv.Color.from_hex(c) for c in ("#2fb344", "#e5343a", "#9aa0a6"))


def parse_args():
    p = argparse.ArgumentParser(description="Stage 4 - tracking + PPE rules")
    p.add_argument("--source", default="0", help="webcam index or video path")
    p.add_argument("--model", default="ppe_yolo26n.pt", help="fine-tuned weights in models/")
    p.add_argument("--conf", type=float, default=0.3, help="confidence needed for a PPE item to count")
    p.add_argument("--track-conf", type=float, default=0.1,
                   help="lowest detection score passed to the tracker; ByteTrack uses weak boxes to keep tracks alive")
    p.add_argument("--tracker", default="botsort.yaml", choices=["botsort.yaml", "bytetrack.yaml"],
                   help="BoT-SORT adds camera-motion compensation on top of ByteTrack")
    p.add_argument("--require", nargs="+", default=["helmet"], choices=sorted(PPE_REGIONS))
    p.add_argument("--window", type=int, default=15, help="frames of history per person")
    p.add_argument("--min-frames", type=int, default=15,
                   help="frames before a person gets a verdict; higher = fewer false alarms from ID switches, slower alerts")
    p.add_argument("--ratio", type=float, default=0.6, help="flag if item missing in >= this share of the window")
    p.add_argument("--min-height", type=int, default=80,
                   help="ignore people shorter than this in pixels; too small to judge PPE reliably")
    p.add_argument("--edge-margin", type=int, default=4,
                   help="a box within this many px of the top/left/right edge may have its head cut off")
    p.add_argument("--head-check", default="pose", choices=["pose", "box", "off"],
                   help="pose: head keypoints (box test for people the pose model misses); "
                        "box: box-edge test only; off: always judge helmets")
    p.add_argument("--pose-model", default="yolo26n-pose.pt")
    p.add_argument("--trt", action="store_true", help="use TensorRT engines (build with src/export_trt.py)")
    p.add_argument("--no-stitch", action="store_true",
                   help="disable track stitching (re-linking IDs after an ID switch), to compare")
    p.add_argument("--save", action="store_true", help="save annotated video to outputs/")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def has_item(person_xyxy, item_boxes, region):
    """True if any item box centre lies inside the given vertical band of the person box."""
    if len(item_boxes) == 0:
        return False
    x1, y1, x2, y2 = person_xyxy
    top, bottom = y1 + region[0] * (y2 - y1), y1 + region[1] * (y2 - y1)
    cx = (item_boxes[:, 0] + item_boxes[:, 2]) / 2
    cy = (item_boxes[:, 1] + item_boxes[:, 3]) / 2
    return bool(np.any((cx >= x1) & (cx <= x2) & (cy >= top) & (cy <= bottom)))


def head_in_frame_box(xyxy, frame_wh, margin=4):
    """Box-only head check: False if the box touches the top, left or right frame edge.

    A box cut by the frame edge often means part of the person, possibly the head,
    is outside the picture. Coarse: someone at the side edge with their head in view
    is skipped too, until they move further in.
    """
    x1, y1, x2, _ = xyxy
    w, _ = frame_wh
    return y1 > margin and x1 > margin and x2 < w - margin


def head_in_frame_keypoints(xyxy, kp, kc, frame_wh, margin=4, min_conf=0.5, min_points=2):
    """Keypoint head check: at least `min_points` head keypoints confidently inside the frame,
    and the box top clear of the frame top (a helmet sits above the eyes, so it can be cut
    off even when the face is visible)."""
    w, h = frame_wh
    pts = kp[HEAD_KEYPOINTS][kc[HEAD_KEYPOINTS] >= min_conf]
    inside = (pts[:, 0] > margin) & (pts[:, 0] < w - margin) & (pts[:, 1] > margin) & (pts[:, 1] < h - margin)
    return int(inside.sum()) >= min_points and xyxy[1] > margin


def match_boxes(a, b, min_iou=0.3):
    """Pair rows of a with rows of b one-to-one, maximising total IoU (Hungarian algorithm).

    Returns {index in a: index in b} for pairs with IoU >= min_iou. Used to find which
    pose detection belongs to which tracked person: two models, two sets of boxes for
    the same people, slightly different in size.
    """
    if len(a) == 0 or len(b) == 0:
        return {}
    iou = sv.box_iou_batch(a, b)
    rows, cols = linear_sum_assignment(-iou)
    return {int(r): int(c) for r, c in zip(rows, cols) if iou[r, c] >= min_iou}


class HeadChecker:
    """Decides, per tracked person and frame, whether their head is in view (head cut-off rule).

    mode "pose": run a pose model, pair its people with the tracked people (match_boxes)
                 and use head_in_frame_keypoints; unmatched people fall back to the box test.
    mode "box":  head_in_frame_box only.
    mode "off":  no check (returns None, so every frame is judged).

    Used by Stage 4 and Stage 5, which track people with the PPE model (no keypoints).
    """

    def __init__(self, mode="pose", pose_model="yolo26n-pose.pt", edge_margin=4):
        self.mode, self.margin = mode, edge_margin
        self.pose = load_yolo(pose_model) if mode == "pose" else None
        self.counts = {"pose": 0, "box fallback": 0}   # how each person-frame's head was judged
        self.frame_pose = None                         # (pose boxes, sv.KeyPoints) from the last call

    def __call__(self, frame, persons: sv.Detections, frame_wh):
        """Return (head_ok bool array or None, sv.KeyPoints for drawing or None)."""
        self.frame_pose = None
        if self.mode == "off":
            return None, None
        head_ok = np.array([head_in_frame_box(b, frame_wh, self.margin) for b in persons.xyxy], dtype=bool)
        if self.pose is None or len(persons) == 0:
            return head_ok, None
        r = self.pose.predict(frame, conf=0.25, verbose=False)[0]
        if r.keypoints is None or len(r.boxes) == 0:
            self.counts["box fallback"] += len(persons)
            return head_ok, None
        keypoints = sv.KeyPoints.from_ultralytics(r)
        self.frame_pose = (r.boxes.xyxy.cpu().numpy(), keypoints)
        pairs = match_boxes(persons.xyxy, self.frame_pose[0])
        for i, j in pairs.items():
            head_ok[i] = head_in_frame_keypoints(persons.xyxy[i], keypoints.xy[j],
                                                 keypoints.keypoint_confidence[j], frame_wh, self.margin)
        self.counts["pose"] += len(pairs)
        self.counts["box fallback"] += len(persons) - len(pairs)
        return head_ok, keypoints

    def head_points(self, boxes):
        """Head position per tracked box, from the pose detection it overlaps most (IoU >= 0.3).

        Many-to-one on purpose: two duplicate boxes on one person both get that person's
        head, while two people standing close get two different heads. Used by the track
        stitcher to tell the two cases apart. None per box where no head was seen; None
        overall only when no pose model is running.
        """
        if self.pose is None:
            return None
        if self.frame_pose is None or len(boxes) == 0:
            return [None] * len(boxes)
        pose_boxes, kps = self.frame_pose
        iou = sv.box_iou_batch(np.asarray(boxes).reshape(-1, 4), pose_boxes)
        return [head_point(kps.xy[j], kps.keypoint_confidence[j]) if iou[i, j] >= 0.3 else None
                for i, j in enumerate(iou.argmax(axis=1))]

    def report(self):
        if self.pose is not None:
            total = sum(self.counts.values()) or 1
            print(f"Head judged by pose keypoints for {self.counts['pose'] / total:.0%} of person-frames, "
                  f"box fallback for {self.counts['box fallback'] / total:.0%}")


class PpeMonitor:
    def __init__(self, names, args):
        self.args = args
        lookup = {v.lower(): k for k, v in names.items()}
        self.person_id = lookup["person"]
        self.item_ids = {item: lookup[item] for item in args.require}
        # Explicit "no_x" detections count as evidence too, when the model finds them.
        self.no_ids = {item: lookup.get(f"no_{item}") for item in args.require}
        self.history = defaultdict(lambda: {i: deque(maxlen=args.window) for i in args.require})
        self.flagged = set()        # track ids that have ever been a violator

    def update(self, persons: sv.Detections, items: sv.Detections, head_ok=None):
        """Return a status per tracked person plus the items they're missing.

        head_ok: optional bool per person; where False, head items are not judged this
        frame (their history is left unchanged rather than counted as missing).

        Statuses: 'violation', 'ok', 'head hidden' (a head item can't be judged yet
        because the head isn't in view), or 'checking' (not enough frames yet).
        """
        if head_ok is None:
            head_ok = np.ones(len(persons), dtype=bool)
        statuses = []
        for xyxy, tid, head_visible in zip(persons.xyxy, persons.tracker_id, head_ok):
            hist = self.history[tid]
            for item in self.args.require:
                if item in HEAD_ITEMS and not head_visible:
                    continue
                present = has_item(xyxy, items.xyxy[items.class_id == self.item_ids[item]], PPE_REGIONS[item])
                no_id = self.no_ids[item]
                explicit_no = no_id is not None and has_item(
                    xyxy, items.xyxy[items.class_id == no_id], PPE_REGIONS[item])
                hist[item].append(explicit_no or not present)

            judged = {i: len(hist[i]) >= self.args.min_frames for i in self.args.require}
            missing = [i for i in self.args.require if judged[i] and np.mean(hist[i]) >= self.args.ratio]
            if missing:
                statuses.append(("violation", missing))
            elif all(judged.values()):
                statuses.append(("ok", []))
            elif not head_visible and any(not judged[i] for i in self.args.require if i in HEAD_ITEMS):
                statuses.append(("head hidden", []))
            else:
                statuses.append(("checking", []))
        return statuses


class Drawer:
    def __init__(self):
        colors = (("ok", GREEN), ("violation", RED), ("checking", GREY), ("head hidden", GREY))
        self.boxes = {s: sv.BoxAnnotator(color=c, thickness=3) for s, c in colors}
        self.labels = {s: sv.LabelAnnotator(color=c, text_scale=0.55, text_padding=4) for s, c in colors}
        self.items = sv.BoxAnnotator(thickness=1)
        self.item_labels = sv.LabelAnnotator(text_scale=0.4, text_padding=2, text_position=sv.Position.TOP_RIGHT)
        self.trace = sv.TraceAnnotator(thickness=2, trace_length=40)
        self.skeleton = sv.EdgeAnnotator(color=sv.Color.from_hex("#00d4ff"), thickness=1)

    def __call__(self, frame, persons, statuses, items, names, hud, keypoints=None, canon=None):
        out = frame.copy()
        if keypoints is not None and len(keypoints):
            out = self.skeleton.annotate(out, keypoints)
        out = self.items.annotate(out, items)
        out = self.item_labels.annotate(out, items, [names[c] for c in items.class_id])
        if len(persons):
            out = self.trace.annotate(out, persons)
        for state in ("checking", "head hidden", "ok", "violation"):
            mask = np.array([s == state for s, _ in statuses], dtype=bool)
            if not mask.any():
                continue
            sub = persons[mask]
            canon = canon or {}
            text = [f"#{tid}" + (f"={canon[int(tid)]}" if canon.get(int(tid), tid) != tid else "") + " "
                    + ("NO " + "/".join(m).upper() if state == "violation" else state.upper())
                    for tid, (_, m) in zip(sub.tracker_id, np.array(statuses, dtype=object)[mask])]
            out = self.boxes[state].annotate(out, sub)
            out = self.labels[state].annotate(out, sub, text)
        return draw_hud(out, hud)


def main():
    args = parse_args()
    args.model, args.pose_model = trt_weights(args.model, args.trt), trt_weights(args.pose_model, args.trt)
    model = YOLO(str(MODELS_DIR / args.model))
    heads = HeadChecker(args.head_check, args.pose_model, args.edge_margin)
    monitor = PpeMonitor(model.names, args)
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Model: {args.model} | device: {device} | tracker: {args.tracker} | "
          f"head check: {args.head_check} | require: {', '.join(args.require)}")

    video = VideoIO(args.source, "track" if args.save else None, not args.no_show, "Stage 4 - PPE tracking")
    events = EventLog(video.name, "stage4")
    draw, meter = Drawer(), FpsMeter()
    stitcher = None if args.no_stitch else TrackStitcher()

    for frame in video.frames():
        meter.start()
        dets, _ = track(model, frame, args.track_conf, args.tracker)
        is_person = dets.class_id == monitor.person_id
        tall = (dets.xyxy[:, 3] - dets.xyxy[:, 1]) >= args.min_height
        persons = dets[is_person & tall]
        items = dets[~is_person & (dets.confidence >= args.conf)]

        head_ok, keypoints = heads(frame, persons, video.size)
        statuses = monitor.update(persons, items, head_ok)
        # Canonical identity per track: an ID the tracker re-assigned keeps the old person's identity.
        canon = stitcher.update(frame, persons.tracker_id, persons.xyxy, video.now(),
                                heads.head_points(persons.xyxy)) if stitcher else \
            {int(t): int(t) for t in persons.tracker_id}

        for (state, missing), tid, xyxy in zip(statuses, persons.tracker_id, persons.xyxy):
            if state == "violation":
                monitor.flagged.add(canon[int(tid)])
                events.log(frame, video.now(), video.idx, canon[int(tid)], "missing " + "+".join(missing), xyxy)

        fps = meter.stop()
        now_bad = sum(s == "violation" for s, _ in statuses)
        hud = (f"FPS {fps:5.1f} | people {len(persons)} | violating now {now_bad} | "
               f"unique violators {len(monitor.flagged)}")
        if not video.emit(draw(frame, persons, statuses, items, model.names, hud, keypoints, canon)):
            break

    video.close()
    events.close()
    print(f"\n{video.idx + 1} frames | {len(monitor.flagged)} unique violators | events: {events.csv_path}")
    heads.report()
    if stitcher:
        stitcher.report()


if __name__ == "__main__":
    main()
