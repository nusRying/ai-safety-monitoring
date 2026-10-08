"""Combined safety pipeline: PPE rules + zones + fall detection on one video.

Stages 4-6 each run their own detector and tracker. Run side by side, the same
person would get a different ID in each, so here one model owns the people:

    pose model (tracked)   -> every person, one ID each, plus keypoints for falls
                              (+ the rotated search for people lying down)
    PPE model (detect)     -> helmets, vests, ... ; matched to people by the Stage 4 rule
    zones                  -> Stage 5 rules on the same tracked people
    fall detector          -> Stage 6 state machine on the same tracked people

Used by the Streamlit dashboard (Stage 9); also runnable on its own:

    python src/pipeline.py --source data/samples/demo_scene.mp4 --zones configs/zones_demo_scene.json --save
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import supervision as sv

from common import (CONFIGS_DIR, STATIC_TRACKER, EventLog, FpsMeter, TrackStitcher, VideoIO, draw_hud, head_point,
                    load_yolo, track, trt_weights)
from stage4_track import PpeMonitor, head_in_frame_keypoints
from stage5_zones import Zone, check_rules
from stage6_fall import FallDetector, RotationRecovery, box_cut_by_frame, torso_angle

GREEN, AMBER, ORANGE, RED, GREY = (68, 179, 47), (0, 200, 255), (0, 140, 255), (58, 52, 229), (170, 170, 170)


@dataclass
class PipelineConfig:
    ppe: bool = True
    require: list = field(default_factory=lambda: ["helmet"])
    zones_file: Path | None = None
    falls: bool = True
    ppe_model: str = "ppe_yolo26n.pt"
    pose_model: str = "yolo26n-pose.pt"
    item_conf: float = 0.3
    min_size: int = 60             # longest box side, px, to consider a person at all
    min_ppe_height: int = 80       # box height, px, to judge PPE (too few pixels below this)
    grace_s: float = 0.5
    head_check: bool = True        # only judge helmets when head keypoints are in view
    edge_margin: int = 4
    tracker: str = "botsort.yaml"
    trt: bool = False              # use TensorRT engines (src/export_trt.py)
    stitch: bool = True            # one canonical ID per person (common.TrackStitcher)


@dataclass
class FrameResult:
    image: np.ndarray
    people: int
    ppe_violations: int
    in_zones: int
    fallen: int
    new_events: list            # [(track_id, reason)]
    fps: float


class SafetyPipeline:
    def __init__(self, cfg: PipelineConfig, frame_wh, source_name: str):
        self.cfg = cfg
        pose_w, ppe_w = trt_weights(cfg.pose_model, cfg.trt), trt_weights(cfg.ppe_model, cfg.trt)
        self.pose = load_yolo(pose_w)
        self.ppe_model = load_yolo(ppe_w) if cfg.ppe or cfg.zones_file else None

        self.frame_wh = frame_wh
        self.zones = []
        if cfg.zones_file:
            spec = json.loads(Path(cfg.zones_file).read_text(encoding="utf-8"))["zones"]
            self.zones = [Zone(z, frame_wh) for z in spec]

        # PPE is checked for --require everywhere (if enabled) plus whatever a zone requires.
        required = set(cfg.require if cfg.ppe else []) | {i for z in self.zones for i in z.require}
        self.monitor = None
        if required:
            self.monitor = PpeMonitor(self.ppe_model.names, SimpleNamespace(
                require=sorted(required), window=15, min_frames=15, ratio=0.6))
            self.item_ids = [k for k, v in self.ppe_model.names.items() if v.lower() != "person"]

        fall_args = SimpleNamespace(fall_window=1.0, confirm_s=0.5, down_s=5.0, search_s=30.0)
        self.falls = FallDetector(fall_args) if cfg.falls else None
        self.recovery = RotationRecovery(load_yolo(pose_w), fall_args, frame_wh) if cfg.falls else None

        self.stitcher = TrackStitcher() if cfg.stitch else None
        self.events = EventLog(source_name, "dashboard")
        self.meter = FpsMeter()
        self.edges = sv.EdgeAnnotator(color=sv.Color.from_hex("#00d4ff"), thickness=2)
        self.vertices = sv.VertexAnnotator(color=sv.Color.from_hex("#ffffff"), radius=3)
        self._warm_up(frame_wh)

    def _warm_up(self, frame_wh):
        """Run every model once on a blank frame. The first inference sets up CUDA / TensorRT and
        can take ~10 s; doing it here keeps it out of the first video frame (a frozen-looking
        start on a webcam). predict, not track, so the tracker's state stays untouched."""
        blank = np.zeros((frame_wh[1], frame_wh[0], 3), np.uint8)
        models = [self.pose, self.ppe_model, self.recovery.model if self.recovery else None]
        for m in models:
            if m is not None:
                m.predict(blank, verbose=False)

    def process(self, frame, now: float, frame_idx: int) -> FrameResult:
        cfg = self.cfg
        self.meter.start()
        out = frame.copy()

        # 1. People + keypoints (tracked)
        dets, result = track(self.pose, frame, 0.1, cfg.tracker)
        people = []                 # (tid, xyxy, kp_xy, kp_conf, recovered)
        if len(dets) and result.keypoints is not None:
            kps = sv.KeyPoints.from_ultralytics(result)
            size = np.maximum(dets.xyxy[:, 2] - dets.xyxy[:, 0], dets.xyxy[:, 3] - dets.xyxy[:, 1])
            keep = size >= cfg.min_size
            dets, kps = dets[keep], kps[keep]
            out = self.edges.annotate(out, kps)
            out = self.vertices.annotate(out, kps)
            people = [(int(t), b, kps.xy[i], kps.keypoint_confidence[i], False)
                      for i, (b, t) in enumerate(zip(dets.xyxy, dets.tracker_id))]
        if self.recovery:
            for tid, b, *_ in people:
                self.recovery.remember(tid, b, now)
            found = self.recovery.search(frame, {p[0] for p in people}, [p[1] for p in people], now)
            people += [(tid, b, kp, kc, True) for tid, (b, kp, kc) in found.items()]

        persons = sv.Detections(
            xyxy=np.array([p[1] for p in people], dtype=float).reshape(-1, 4),
            tracker_id=np.array([p[0] for p in people], dtype=int),
            class_id=np.zeros(len(people), dtype=int),
        )
        # One canonical ID per person (merges duplicate tracks and re-links ID switches).
        canon = self.stitcher.update(frame, persons.tracker_id, persons.xyxy, now,
                                     [head_point(p[2], p[3]) for p in people]) if self.stitcher else \
            {p[0]: p[0] for p in people}

        # 2. PPE items and the Stage 4 rule (only for people big enough to judge)
        statuses = [("checking", [])] * len(people)
        if self.monitor and len(people):
            r = self.ppe_model.predict(frame, conf=cfg.item_conf, classes=self.item_ids, verbose=False)[0]
            items = sv.Detections.from_ultralytics(r)
            tall = (persons.xyxy[:, 3] - persons.xyxy[:, 1]) >= cfg.min_ppe_height
            upright_like = np.array([not p[4] for p in people], dtype=bool)   # skip rotated finds
            judge = tall & upright_like
            head_ok = np.array([not cfg.head_check or head_in_frame_keypoints(
                p[1], p[2], p[3], self.frame_wh, cfg.edge_margin) for p in people], dtype=bool)
            judged = iter(self.monitor.update(persons[judge], items, head_ok[judge])) if judge.any() else iter(())
            statuses = [next(judged) if j else ("checking", []) for j in judge]

        # 3. Zones, on canonical IDs: a duplicate box must not count as a second person inside.
        zone_persons = sv.Detections(xyxy=persons.xyxy, class_id=persons.class_id,
                                     tracker_id=np.array([canon[p[0]] for p in people], dtype=int))
        for z in self.zones:
            z.update(zone_persons, now)
        zone_reasons_by_person, zone_level = check_rules(self.zones, zone_persons, statuses, now, cfg.grace_s) \
            if self.zones else ({}, [])

        # 4. Falls + collect reasons per person
        new_events, n_ppe, n_fallen = [], 0, 0
        for (tid, xyxy, kp, kc, recovered), (state, missing) in zip(people, statuses):
            cid = canon[tid]
            reasons = list(zone_reasons_by_person.get(cid, []))
            in_zone_rule = any("no " in r and " in " in r for r in reasons)
            if cfg.ppe and state == "violation" and not in_zone_rule:
                reasons.append("missing " + "+".join(m for m in missing if m in cfg.require))
            n_ppe += state == "violation"

            fall_state = None
            if self.falls:
                angle = torso_angle(kp, kc, 0.3)
                aspect = (xyxy[2] - xyxy[0]) / max(xyxy[3] - xyxy[1], 1)
                fs, fall_reasons = self.falls.update(tid, now, angle, aspect, box_cut_by_frame(xyxy, self.frame_wh))
                fall_state = fs.state
                reasons += fall_reasons
                n_fallen += fs.state == "FALLEN"

            reasons = list(dict.fromkeys(r for r in reasons if r != "missing "))
            for r in reasons:
                if (cid, r) not in self.events.logged:
                    new_events.append((cid, r))
                self.events.log(frame, now, frame_idx, cid, r, xyxy)
            self._draw_person(out, tid, xyxy, reasons, state, fall_state, recovered, now, cid)

        for z, reason in zone_level:
            x1, y1 = z.polygon.min(axis=0)
            x2, y2 = z.polygon.max(axis=0)
            key = f"zone-{z.name}-{z.visits}"
            if (key, reason) not in self.events.logged:
                new_events.append((key, reason))
            self.events.log(frame, now, frame_idx, key, reason, (x1, y1, x2, y2))

        out = self._draw_zones(out)
        fps = self.meter.stop()
        in_zones = sum(len(z.inside_ids) for z in self.zones)
        n_people = len(set(canon.values()))         # a duplicate box doesn't count as a second person
        hud = f"FPS {fps:4.1f} | t {now:5.1f}s | people {n_people} | events {self.events.count}"
        return FrameResult(draw_hud(out, hud), n_people, n_ppe, in_zones, n_fallen, new_events, fps)

    def _draw_person(self, out, tid, xyxy, reasons, ppe_state, fall_state, recovered, now, cid=None):
        cid = tid if cid is None else cid             # zones track canonical IDs
        if reasons or fall_state == "FALLEN":
            color = RED
        elif fall_state == "lying":
            color = ORANGE
        elif any(cid in z.inside_ids for z in self.zones):
            color = AMBER
        elif ppe_state in ("checking", "head hidden") and self.monitor:
            color = GREY
        else:
            color = GREEN
        x1, y1, x2, y2 = np.asarray(xyxy).astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3 if color == RED else 2)
        cv2.circle(out, ((x1 + x2) // 2, y2), 4, color, -1)
        label = (f"#{tid}" + (f"={cid}" if cid != tid else "") + (" (rotated)" if recovered else "")
                 + (" head hidden" if ppe_state == "head hidden" else ""))
        zone = next((z for z in self.zones if cid in z.inside_ids), None)
        if zone:
            label += f" {zone.dwell(cid, now):.1f}s"
        # Reasons go under the box, or above the ID label if they'd run off the bottom of the frame.
        below = y2 + 20 + 18 * len(reasons) <= out.shape[0]
        label_y = max(y1 - 8 - (0 if below else 18 * len(reasons)), 12)
        cv2.putText(out, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        for i, r in enumerate(reasons):
            y = y2 + 20 + 18 * i if below else label_y + 18 * (i + 1)
            cv2.putText(out, r.upper(), (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    def _draw_zones(self, out):
        if not self.zones:
            return out
        overlay = out.copy()
        for z in self.zones:
            cv2.fillPoly(overlay, [z.polygon], RED if z.alert else AMBER)
        out = cv2.addWeighted(overlay, 0.2, out, 0.8, 0)
        for z in self.zones:
            c = RED if z.alert else AMBER
            cv2.polylines(out, [z.polygon], True, c, 2)
            x, y = z.polygon.min(axis=0)
            cv2.putText(out, f"{z.name}: {len(z.inside_ids)} inside", (int(x) + 4, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
        return out

    def close(self):
        self.events.close()


def main():
    p = argparse.ArgumentParser(description="Combined safety pipeline")
    p.add_argument("--source", default="data/samples/demo_scene.mp4")
    p.add_argument("--zones", type=Path, default=None, help=f"e.g. {CONFIGS_DIR / 'zones_demo_scene.json'}")
    p.add_argument("--require", nargs="+", default=["helmet"])
    p.add_argument("--no-ppe", action="store_true")
    p.add_argument("--no-falls", action="store_true")
    p.add_argument("--no-head-check", action="store_true", help="disable the head cut-off rule, to compare")
    p.add_argument("--trt", action="store_true", help="use TensorRT engines (build with src/export_trt.py)")
    p.add_argument("--no-stitch", action="store_true", help="disable track stitching, to compare")
    p.add_argument("--save", action="store_true")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    video = VideoIO(args.source, "pipeline" if args.save else None, not args.no_show, "Safety pipeline")
    tracker = STATIC_TRACKER if video.is_cam else "botsort.yaml"   # webcams don't move: skip motion compensation
    cfg = PipelineConfig(tracker=tracker,ppe=not args.no_ppe, require=args.require, zones_file=args.zones, falls=not args.no_falls,
                         head_check=not args.no_head_check, trt=args.trt, stitch=not args.no_stitch)
    pipe = SafetyPipeline(cfg, video.size, video.name)
    for frame in video.frames():
        res = pipe.process(frame, video.now(), video.idx)    # EventLog prints each new event
        if not video.emit(res.image):
            break
    video.close()
    pipe.close()
    print(f"\n{video.idx + 1} frames | {pipe.events.count} events | {pipe.events.csv_path}")
    if pipe.stitcher:
        pipe.stitcher.report()


if __name__ == "__main__":
    main()
