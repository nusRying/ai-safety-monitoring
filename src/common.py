"""Shared helpers for the video stages (4, 5, 6): video input/output, tracking, HUD and event log."""

import csv
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
CONFIGS_DIR = ROOT / "configs"


def load_env(path: Path = ROOT / ".env"):
    """Load KEY=value lines from the project's .env into os.environ (without overriding).

    Keeps API keys out of the code. .env is git-ignored; never commit it.
    """
    import os

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


STATIC_TRACKER = str(CONFIGS_DIR / "botsort_static.yaml")   # BoT-SORT without camera-motion compensation


class VideoIO:
    """Read frames from a webcam or file, optionally save and show annotated frames.

    `now()` returns the current time in seconds: video time for files (so results
    don't depend on how fast your GPU is), wall-clock time for a webcam.

    Webcams are read on a background thread that keeps only the newest frame. Reading
    a camera blocks until the next frame arrives (30-70 ms), so without the thread every
    loop pays read time + processing time; with it the two overlap, and if processing is
    slower than the camera, stale frames are skipped instead of piling up as lag.
    """

    def __init__(self, source: str, save_suffix: str | None = None, show: bool = True, title: str = "",
                 cam_size=(1280, 720)):
        self.is_cam = source.isdigit()
        self.cap = cv2.VideoCapture(int(source) if self.is_cam else source)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open source: {source}")
        if self.is_cam and cam_size:
            # Webcams often start at 640x480; ask for more. The camera picks the nearest it supports.
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_size[1])
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1 else 30   # some webcam drivers report 0 or -1
        self.name = "webcam" if self.is_cam else Path(source).stem
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.show_window, self.title = show, f"{title} (q to quit)"
        self.idx, self.t_start = -1, time.time()
        self.writer, self.save_path = None, None
        if save_suffix:
            OUTPUTS_DIR.mkdir(exist_ok=True)
            self.save_path = OUTPUTS_DIR / f"{self.name}_{save_suffix}.mp4"
            self.writer = cv2.VideoWriter(str(self.save_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
        if self.is_cam:
            self._cond = threading.Condition()
            self._latest, self._seq, self._running = None, 0, True
            self._thread = threading.Thread(target=self._read_camera, daemon=True)
            self._thread.start()

    def _read_camera(self):
        while self._running:
            ok, frame = self.cap.read()
            with self._cond:
                if not ok:
                    self._running = False
                else:
                    self._latest, self._seq = frame, self._seq + 1
                self._cond.notify_all()

    def frames(self):
        if self.is_cam:
            seen = 0
            while True:
                with self._cond:
                    # Wait for a frame newer than the last one handed out.
                    self._cond.wait_for(lambda: self._seq != seen or not self._running, timeout=5)
                    if self._seq == seen:
                        return                        # camera stopped or stalled
                    frame, seen = self._latest, self._seq
                self.idx += 1
                yield frame
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return
            self.idx += 1
            yield frame

    def now(self) -> float:
        return time.time() - self.t_start if self.is_cam else self.idx / self.fps

    def emit(self, out) -> bool:
        """Write/show one annotated frame. Returns False when the user asks to quit."""
        if self.writer:
            self.writer.write(out)
        if not self.show_window:
            return True
        cv2.imshow(self.title, out)
        return cv2.waitKey(1) & 0xFF not in (ord("q"), 27)

    def close(self):
        if self.is_cam:
            with self._cond:
                self._running = False
            self._thread.join(timeout=2)
        self.cap.release()
        if self.writer:
            self.writer.release()
            print(f"Saved {self.save_path}")
        cv2.destroyAllWindows()


def trt_weights(name: str, use_trt: bool) -> str:
    """Return the TensorRT engine name for a .pt model in models/ when use_trt is set.

    Engines are built by src/export_trt.py and only run on the GPU they were built on.
    """
    if not use_trt:
        return name
    engine = Path(name).with_suffix(".engine").name
    if not (MODELS_DIR / engine).exists():
        raise SystemExit(f"No TensorRT engine for {name}. Build it with: python src/export_trt.py export")
    return engine


def load_yolo(name: str, cls=None):
    """Load Ultralytics weights from models/, downloading them there on first use.

    cls: the model class (YOLO by default; pass SAM, YOLOE, ... for others).
    """
    if cls is None:
        from ultralytics import YOLO as cls

    MODELS_DIR.mkdir(exist_ok=True)
    local = MODELS_DIR / name
    model = cls(str(local) if local.exists() else name)
    if not local.exists() and Path(name).exists():   # Ultralytics downloads into the CWD
        Path(name).replace(local)
    return model


def track(model, frame, conf=0.1, tracker="botsort.yaml"):
    """Run detection + tracking on one frame. Returns (sv.Detections with tracker_id, raw result)."""
    result = model.track(frame, conf=conf, persist=True, tracker=tracker, verbose=False)[0]
    dets = sv.Detections.from_ultralytics(result)
    if dets.tracker_id is None:  # nothing confirmed yet
        dets = sv.Detections.empty()
        dets.tracker_id = np.array([], dtype=int)
    return dets, result


class FpsMeter:
    def __init__(self):
        self.fps, self.t0 = 0.0, None

    def start(self):
        self.t0 = time.perf_counter()

    def stop(self):
        dt = time.perf_counter() - self.t0
        self.fps = 0.9 * self.fps + 0.1 / dt if self.fps else 1.0 / dt
        return self.fps


def draw_hud(frame, text):
    cv2.rectangle(frame, (0, 0), (len(text) * 9 + 12, 28), (0, 0, 0), -1)
    cv2.putText(frame, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


HEAD_KEYPOINTS = [0, 1, 2, 3, 4]         # COCO pose: nose, left/right eye, left/right ear


def head_point(kp, kc, min_conf=0.5):
    """Mean position of the confident head keypoints, or None if there are none."""
    pts = np.asarray(kp)[HEAD_KEYPOINTS][np.asarray(kc)[HEAD_KEYPOINTS] >= min_conf]
    return tuple(pts.mean(axis=0)) if len(pts) else None


class TrackStitcher:
    """Give each person one canonical ID, however many track IDs the tracker uses for them.

    Two ways one person ends up with several track IDs:

    1. Duplicate track: the detector puts two overlapping boxes on the same person and
       the tracker follows both. YOLO26 is NMS-free, so an occasional duplicate box isn't
       removed. A new track counts as a duplicate of an existing one (at once), and two
       existing tracks are merged (after `dup_frames` frames in a row), when their boxes
       overlap with IoU >= `dup_iou`, they look alike, and their heads (pose keypoints)
       are both visible and in the same place (within `head_tol` x box height). Overlap
       and colour alone can't tell a duplicate from two people standing close together
       (one bending in front of another overlaps just as much); a second head can. If
       either head isn't visible, the tracks are kept apart. Only when no pose model runs
       at all (heads=None) do overlap and colour decide on their own, and then never for
       merging two existing tracks.
    2. ID switch: the tracker loses someone in a crowd and gives them a new ID when they
       reappear. A new track is re-linked to a track that disappeared within `max_gap_s`
       seconds, was last seen close by (centre distance <= `max_dist` x person height),
       at a similar size (height ratio within `max_scale`), and looked alike.

    "Looked alike" = hue/saturation histogram correlation >= `min_appearance`: a crude but
    cheap re-identification feature, and what stops two different people standing in the
    same spot being merged. If several candidates fit, the best-overlapping (duplicates)
    or closest (switches) wins.

    The event log uses the canonical ID, so one person raises each alarm once, and people
    counts use it, so a duplicate box doesn't count twice.
    """

    # head_tol: duplicates measured 0-2 px apart; 0.15 x height (~one head width) merged two
    # people standing shoulder to shoulder in a test once the overlap requirement was lowered.
    def __init__(self, max_gap_s=2.0, max_dist=0.75, max_scale=1.6, min_appearance=0.5, dup_iou=0.6, head_tol=0.06,
                 dup_frames=5, dup_iou_same_head=0.3):
        self.max_gap_s, self.max_dist, self.max_scale, self.min_app = max_gap_s, max_dist, max_scale, min_appearance
        self.dup_iou, self.head_tol, self.dup_frames = dup_iou, head_tol, dup_frames
        # With both heads seen in the same place, less box overlap is needed: two people can't
        # share a head, but two boxes on one person can differ a lot (one includes the arms).
        # On a laptop camera, duplicate boxes on one person overlapped only ~0.49.
        self.dup_iou_same_head = dup_iou_same_head
        self.last = {}          # tid -> (time, box, histogram) when last seen
        self.parent = {}        # tid -> earlier tid it was stitched to
        self.overlap = {}       # (older root, younger root) -> consecutive frames overlapping as duplicates
        self.links = []         # (time, kind, new tid, canonical tid) for reporting

    def root(self, tid):
        while tid in self.parent:
            tid = self.parent[tid]
        return tid

    @staticmethod
    def _hist(frame, box):
        x1, y1, x2, y2 = np.asarray(box).astype(int)
        h, w = frame.shape[:2]
        # Central part of the body: avoids most of the background around the person.
        cx1, cx2 = x1 + (x2 - x1) // 4, x2 - (x2 - x1) // 4
        cy1, cy2 = y1 + (y2 - y1) // 8, y1 + (y2 - y1) * 3 // 4
        patch = frame[max(cy1, 0):min(cy2, h), max(cx1, 0):min(cx2, w)]
        if patch.size == 0:
            return None
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def update(self, frame, tids, boxes, now, heads=None) -> dict:
        """Call once per frame with the tracked people. Returns {tid: canonical id}.

        heads: (x, y) head position per track this frame, None where the pose model saw no
        head. Pass heads=None (not a list) only when no pose model is running.
        """
        tids = [int(t) for t in tids]
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        pose_available = heads is not None
        heads = list(heads) if pose_available else [None] * len(tids)

        def same_head(i, j):
            """Duplicates need both heads seen in the same place. Without a pose model at all
            (heads=None), overlap and colour have to decide on their own."""
            if not pose_available:
                return True
            if heads[i] is None or heads[j] is None:
                return False
            h = max(boxes[i][3] - boxes[i][1], boxes[j][3] - boxes[j][1], 1)
            return np.linalg.norm(np.subtract(heads[i], heads[j])) <= self.head_tol * h

        def dup_overlap(iou_ij, i, j):
            """Enough evidence that tracks i and j are one person: matching heads plus some box
            overlap, or (no pose model at all) a large overlap on its own."""
            if not pose_available:
                return iou_ij >= self.dup_iou
            return iou_ij >= self.dup_iou_same_head and same_head(i, j)
        established = [i for i, t in enumerate(tids) if t in self.last]     # on screen before this frame
        active_roots = {self.root(tids[i]) for i in established}
        lost = {t: v for t, v in self.last.items()
                if t not in tids and now - v[0] <= self.max_gap_s and self.root(t) not in active_roots}
        newborn = [i for i, t in enumerate(tids) if t not in self.last and t not in self.parent]
        hists = {i: self._hist(frame, boxes[i]) for i in range(len(tids))}

        def alike(h1, h2):
            return h1 is not None and h2 is not None and cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL) >= self.min_app

        # 1. Duplicates of a track that is on screen right now.
        if newborn and established:
            iou = sv.box_iou_batch(boxes[newborn], boxes[established])
            for r, i in enumerate(newborn):
                c = int(iou[r].argmax())
                old = tids[established[c]]
                if dup_overlap(iou[r, c], i, established[c]) and alike(hists[i], self.last[old][2]):
                    self.parent[tids[i]] = self.root(old)
                    self.links.append((now, "duplicate", tids[i], self.root(old)))
        newborn = [i for i in newborn if tids[i] not in self.parent]

        # 2. ID switches: re-link to a track that disappeared a moment ago.
        candidates = []
        for i in newborn:
            tid, box, hist = tids[i], boxes[i], hists[i]
            bh = max(box[3] - box[1], 1)
            bc = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            for old, (_, obox, ohist) in lost.items():
                oh = max(obox[3] - obox[1], 1)
                if max(bh / oh, oh / bh) > self.max_scale:
                    continue
                oc = np.array([(obox[0] + obox[2]) / 2, (obox[1] + obox[3]) / 2])
                dist = np.linalg.norm(bc - oc) / max(bh, oh)
                if dist > self.max_dist or not alike(hist, ohist):
                    continue
                candidates.append((dist, tid, old))

        used_new, used_old = set(), set()
        for dist, tid, old in sorted(candidates):
            if tid in used_new or self.root(old) in used_old:
                continue
            self.parent[tid] = self.root(old)
            used_new.add(tid)
            used_old.add(self.root(old))
            self.links.append((now, "ID switch", tid, self.root(old)))

        # 3. Duplicates that only become obvious later: two existing tracks that keep
        #    overlapping with the same head and look (e.g. a person walking in from the frame
        #    edge, first seen as two slivers). Merged after `dup_frames` frames in a row, so
        #    two people who just cross paths aren't merged. Merging two established
        #    identities needs positive evidence: both heads known and in the same place.
        #    Overlap and colour alone merged two neighbouring workers in identical vests.
        overlapping = set()
        if len(established) > 1:
            iou = sv.box_iou_batch(boxes[established], boxes[established])
            for a in range(len(established)):
                for b in range(a + 1, len(established)):
                    i, j = established[a], established[b]
                    ri, rj = self.root(tids[i]), self.root(tids[j])
                    if ri != rj and pose_available and dup_overlap(iou[a, b], i, j) and alike(hists[i], hists[j]):
                        overlapping.add((min(ri, rj), max(ri, rj)))
        self.overlap = {k: self.overlap.get(k, 0) + 1 for k in overlapping}
        for (older, younger), n in self.overlap.items():
            if n >= self.dup_frames and self.root(younger) != self.root(older):
                self.parent[younger] = self.root(older)
                self.links.append((now, "duplicate", younger, self.root(older)))

        for i, tid in enumerate(tids):
            self.last[tid] = (now, boxes[i], hists[i])
        # Forget tracks lost long ago; they can no longer be stitched.
        for t in [t for t, v in self.last.items() if now - v[0] > 10 * self.max_gap_s]:
            del self.last[t]
        return {t: self.root(t) for t in tids}

    def counts(self):
        """{"duplicate": n, "ID switch": n}: how many track IDs have been merged so far."""
        return {kind: sum(1 for _, k, _, _ in self.links if k == kind) for kind in ("duplicate", "ID switch")}

    def summary(self):
        c = self.counts()
        return (f"Track stitching merged {c['duplicate']} duplicate track{'s' * (c['duplicate'] != 1)} "
                f"and {c['ID switch']} ID switch{'es' * (c['ID switch'] != 1)}")

    def report(self):
        for kind, label in (("duplicate", "duplicate tracks"), ("ID switch", "ID switches")):
            links = [(t, new, old) for t, k, new, old in self.links if k == kind]
            print(f"Track stitching, {label} re-linked: {len(links)}" +
                  (": " + ", ".join(f"#{new}->#{old} at {t:.1f}s" for t, new, old in links) if links else ""))


class EventLog:
    """Write one CSV row and one image crop per new event.

    Each (track_id, reason) pair is logged once, so a person standing in a zone
    for a minute produces one event, not 1,500.
    Stage 8 reads events.csv and the crops to write incident reports.
    """

    def __init__(self, source_name: str, stage: str):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = OUTPUTS_DIR / "events" / f"{stage}_{source_name}_{stamp}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.dir / "events.csv"
        self.file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["time", "video_s", "frame", "track_id", "reason", "x1", "y1", "x2", "y2", "crop", "scene"])
        self.logged = set()
        self.count = 0

    def log(self, frame, video_s, frame_idx, tid, reason, xyxy):
        if (tid, reason) in self.logged:
            return
        self.logged.add((tid, reason))
        self.count += 1
        x1, y1, x2, y2 = np.asarray(xyxy).astype(int)
        pad = int(0.15 * max(y2 - y1, x2 - x1))
        h, w = frame.shape[:2]
        crop = frame[max(y1 - pad, 0):min(y2 + pad, h), max(x1 - pad, 0):min(x2 + pad, w)]
        stem = f"event{self.count:03d}_track{tid}_frame{frame_idx}"
        cv2.imwrite(str(self.dir / f"{stem}_crop.jpg"), crop)
        # Full frame with the box drawn: gives the VLM in Stage 8 context around the person.
        scene = frame.copy()
        cv2.rectangle(scene, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.imwrite(str(self.dir / f"{stem}_scene.jpg"), scene)
        self.writer.writerow([datetime.now().isoformat(timespec="seconds"), f"{video_s:.2f}", frame_idx, tid,
                              reason, x1, y1, x2, y2, f"{stem}_crop.jpg", f"{stem}_scene.jpg"])
        self.file.flush()
        print(f"[event] {video_s:6.2f}s  #{tid}: {reason}")

    def close(self):
        self.file.close()
