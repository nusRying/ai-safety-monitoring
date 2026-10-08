"""Stage 5: danger zones, dwell time and occupancy.

Zones are polygons drawn on a fixed camera view and stored in configs/zones_<name>.json
as fractions of the frame size (so they work at any resolution). Each zone can have rules:

    restricted       true  -> anyone entering raises an event
    max_dwell_s      N     -> event when someone stays longer than N seconds
    max_occupancy    N     -> event when more than N people are inside at once
    require          ["helmet", ...] -> PPE required inside this zone only (Stage 4 rule,
                                        including its pose-based head cut-off check)

A person is "in" a zone when their feet (bottom-centre of the box) are inside the
polygon. Using the box centre would put a person standing just behind a zone
"inside" it, because their torso overlaps it in the image.

Examples:
    python src/stage5_zones.py --source data/samples/demo_scene.mp4 --save
    python src/stage5_zones.py --draw --source 0 --zones configs/zones_webcam.json
    python src/stage5_zones.py --source 0 --zones configs/zones_webcam.json

Zone editor (--draw): left-click to add points, right-click or Enter to close the
polygon, u = undo point, s = save and quit, q = quit without saving.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

from common import CONFIGS_DIR, MODELS_DIR, EventLog, FpsMeter, TrackStitcher, VideoIO, draw_hud, track, trt_weights
from stage4_track import HeadChecker, PpeMonitor

ZONE_COLOR = (0, 200, 255)      # BGR amber
ALERT_COLOR = (58, 52, 229)     # BGR red
OK_COLOR = (68, 179, 47)        # BGR green


def parse_args():
    p = argparse.ArgumentParser(description="Stage 5 - zones")
    p.add_argument("--source", default="data/samples/demo_scene.mp4")
    p.add_argument("--zones", type=Path, default=CONFIGS_DIR / "zones_demo_scene.json")
    p.add_argument("--model", default="ppe_yolo26n.pt")
    p.add_argument("--conf", type=float, default=0.3, help="confidence needed for a PPE item to count")
    p.add_argument("--track-conf", type=float, default=0.1)
    p.add_argument("--tracker", default="botsort.yaml", choices=["botsort.yaml", "bytetrack.yaml"])
    p.add_argument("--min-height", type=int, default=80)
    p.add_argument("--grace-s", type=float, default=0.5,
                   help="seconds inside a restricted zone before it counts; ignores boxes jittering on the border")
    p.add_argument("--head-check", default="pose", choices=["pose", "box", "off"],
                   help="head cut-off rule for zone helmet requirements (see stage4_track.py)")
    p.add_argument("--pose-model", default="yolo26n-pose.pt")
    p.add_argument("--trt", action="store_true", help="use TensorRT engines (build with src/export_trt.py)")
    p.add_argument("--no-stitch", action="store_true", help="disable track stitching, to compare")
    p.add_argument("--edge-margin", type=int, default=4)
    p.add_argument("--draw", action="store_true", help="draw zones on the first frame and save them")
    p.add_argument("--save", action="store_true")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


# ------------------------------------------------------------------ editor

def draw_zones_interactive(source, path: Path):
    video = VideoIO(source, show=True)
    frame = next(video.frames())
    video.close()
    h, w = frame.shape[:2]
    zones, points = [], []
    win = "Draw zones: L-click add, R-click/Enter close, u undo, s save, q quit"

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            close_polygon()

    def close_polygon():
        if len(points) >= 3:
            zones.append(list(points))
            print(f"zone{len(zones)}: {len(points)} points")
        points.clear()

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        view = frame.copy()
        for poly in zones:
            cv2.polylines(view, [np.array(poly)], True, ZONE_COLOR, 2)
        if points:
            cv2.polylines(view, [np.array(points)], False, (255, 255, 255), 1)
            for pt in points:
                cv2.circle(view, pt, 4, (255, 255, 255), -1)
        cv2.imshow(win, view)
        key = cv2.waitKey(20) & 0xFF
        if key == 13:
            close_polygon()
        elif key == ord("u") and points:
            points.pop()
        elif key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("Quit without saving.")
        elif key == ord("s"):
            close_polygon()
            break
    cv2.destroyAllWindows()

    cfg = {"zones": [{
        "name": f"zone{i + 1}",
        "polygon": [[round(x / w, 4), round(y / h, 4)] for x, y in poly],
        "restricted": True,
        "max_dwell_s": None,
        "max_occupancy": None,
        "require": [],
    } for i, poly in enumerate(zones)]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Saved {len(zones)} zone(s) to {path}. Edit names and rules there.")


# ------------------------------------------------------------------- zones

class Zone:
    def __init__(self, cfg, frame_wh):
        w, h = frame_wh
        self.name = cfg["name"]
        self.restricted = cfg.get("restricted", False)
        self.max_dwell = cfg.get("max_dwell_s")
        self.max_occ = cfg.get("max_occupancy")
        self.require = cfg.get("require", [])
        self.polygon = (np.array(cfg["polygon"]) * [w, h]).astype(int)
        self.sv_zone = sv.PolygonZone(self.polygon, triggering_anchors=(sv.Position.BOTTOM_CENTER,))
        self.entered_at = {}            # track id -> time they entered (current visit)
        self.visits = 0
        self.inside_ids = []
        self.alert = False

    def update(self, persons: sv.Detections, now: float):
        inside = self.sv_zone.trigger(persons) if len(persons) else np.zeros(0, bool)
        ids = set(persons.tracker_id[inside].tolist())
        for tid in ids - set(self.entered_at):
            self.entered_at[tid] = now
            self.visits += 1
        for tid in set(self.entered_at) - ids:
            del self.entered_at[tid]    # left the zone (or lost track): reset their timer
        self.inside_ids = sorted(ids)
        return inside

    def dwell(self, tid, now):
        return now - self.entered_at[tid] if tid in self.entered_at else 0.0


def check_rules(zones, persons, statuses, now, grace_s):
    """Return {track_id: [reasons]} for people breaking a rule, plus zone-level reasons."""
    person_reasons = defaultdict(list)
    zone_reasons = []
    status_by_id = dict(zip(persons.tracker_id.tolist(), statuses)) if statuses else {}
    for z in zones:
        z.alert = False
        for tid in z.inside_ids:
            d = z.dwell(tid, now)
            if z.restricted and d >= grace_s:
                person_reasons[tid].append(f"entered {z.name}")
            if z.max_dwell is not None and d > z.max_dwell:
                person_reasons[tid].append(f"in {z.name} > {z.max_dwell:g}s")
            state, missing = status_by_id.get(tid, ("ok", []))
            lacking = [m for m in missing if m in z.require]
            if state == "violation" and lacking:
                person_reasons[tid].append(f"no {'+'.join(lacking)} in {z.name}")
            if any(z.name in r for r in person_reasons[tid]):
                z.alert = True
        if z.max_occ is not None and len(z.inside_ids) > z.max_occ:
            zone_reasons.append((z, f"{z.name} over capacity ({len(z.inside_ids)}>{z.max_occ})"))
            z.alert = True
    return person_reasons, zone_reasons


SKELETON = sv.EdgeAnnotator(color=sv.Color.from_hex("#00d4ff"), thickness=1)


def draw(frame, zones, persons, person_reasons, now, hud, keypoints=None):
    overlay = frame.copy()
    for z in zones:
        cv2.fillPoly(overlay, [z.polygon], ALERT_COLOR if z.alert else ZONE_COLOR)
    out = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
    if keypoints is not None and len(keypoints):
        out = SKELETON.annotate(out, keypoints)
    for z in zones:
        color = ALERT_COLOR if z.alert else ZONE_COLOR
        cv2.polylines(out, [z.polygon], True, color, 2)
        x, y = z.polygon.min(axis=0)
        label = f"{z.name}: {len(z.inside_ids)} inside, {z.visits} visits"
        cv2.putText(out, label, (int(x) + 4, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    for xyxy, tid in zip(persons.xyxy.astype(int), persons.tracker_id):
        reasons = person_reasons.get(tid, [])
        in_zone = [z for z in zones if tid in z.inside_ids]
        color = ALERT_COLOR if reasons else (ZONE_COLOR if in_zone else OK_COLOR)
        x1, y1, x2, y2 = xyxy
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.circle(out, ((x1 + x2) // 2, y2), 5, color, -1)            # the anchor point used for zones
        text = f"#{tid}"
        if in_zone:
            text += f" {in_zone[0].dwell(tid, now):.1f}s"
        cv2.putText(out, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        for i, r in enumerate(reasons):
            cv2.putText(out, r.upper(), (x1, y2 + 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return draw_hud(out, hud)


def main():
    args = parse_args()
    if args.draw:
        draw_zones_interactive(args.source, args.zones)
        return
    if not args.zones.exists():
        raise SystemExit(f"No zone file at {args.zones}. Create one with --draw.")

    args.model, args.pose_model = trt_weights(args.model, args.trt), trt_weights(args.pose_model, args.trt)
    model = YOLO(str(MODELS_DIR / args.model))
    person_id = next(k for k, v in model.names.items() if v.lower() == "person")
    video = VideoIO(args.source, "zones" if args.save else None, not args.no_show, "Stage 5 - zones")
    zones = [Zone(c, video.size) for c in json.loads(args.zones.read_text(encoding="utf-8"))["zones"]]

    required = sorted({item for z in zones for item in z.require})
    monitor, heads = None, None
    if required:
        monitor = PpeMonitor(model.names, SimpleNamespace(require=required, window=15, min_frames=15, ratio=0.6))
        # The pose model is only loaded when a zone requires PPE; plain zones don't need it.
        heads = HeadChecker(args.head_check, args.pose_model, args.edge_margin)

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Model: {args.model} | device: {device} | zones: {', '.join(z.name for z in zones)}"
          + (f" | head check: {args.head_check}" if heads else ""))
    events = EventLog(video.name, "stage5")
    meter = FpsMeter()
    stitcher = None if args.no_stitch else TrackStitcher()

    for frame in video.frames():
        meter.start()
        now = video.now()
        dets, _ = track(model, frame, args.track_conf, args.tracker)
        tall = (dets.xyxy[:, 3] - dets.xyxy[:, 1]) >= args.min_height
        persons = dets[(dets.class_id == person_id) & tall]
        items = dets[(dets.class_id != person_id) & (dets.confidence >= args.conf)]
        statuses, keypoints = [], None
        if monitor:
            head_ok, keypoints = heads(frame, persons, video.size)
            statuses = monitor.update(persons, items, head_ok)

        # From here on, people are identified by canonical ID: a duplicate box or an ID switch
        # must not count as an extra person in a zone, restart someone's dwell timer, or alarm twice.
        if stitcher and len(persons):
            canon = stitcher.update(frame, persons.tracker_id, persons.xyxy, now,
                                    heads.head_points(persons.xyxy) if heads else None)
            persons.tracker_id = np.array([canon[int(t)] for t in persons.tracker_id], dtype=int)

        for z in zones:
            z.update(persons, now)
        person_reasons, zone_reasons = check_rules(zones, persons, statuses, now, args.grace_s)

        for tid, xyxy in zip(persons.tracker_id, persons.xyxy):
            for reason in person_reasons.get(tid, []):
                events.log(frame, now, video.idx, int(tid), reason, xyxy)
        for z, reason in zone_reasons:
            x1, y1 = z.polygon.min(axis=0)
            x2, y2 = z.polygon.max(axis=0)
            # Zone-level events are logged once per visit count, so a new crowding episode logs again.
            events.log(frame, now, video.idx, f"zone-{z.name}-{z.visits}", reason, (x1, y1, x2, y2))

        fps = meter.stop()
        inside = sum(len(z.inside_ids) for z in zones)
        n_people = len(set(persons.tracker_id.tolist()))
        hud = f"FPS {fps:5.1f} | t {now:5.1f}s | people {n_people} | in zones {inside} | events {events.count}"
        if not video.emit(draw(frame, zones, persons, person_reasons, now, hud, keypoints)):
            break

    video.close()
    events.close()
    print(f"\n{video.idx + 1} frames | {events.count} events | {events.csv_path}")
    if heads:
        heads.report()
    if stitcher:
        stitcher.report()


if __name__ == "__main__":
    main()
