"""Stage 9: Streamlit dashboard.

    streamlit run src/dashboard.py

Live tab: pick a source and which checks to run, then switch Run on. The
annotated video, live counters and an event feed update as frames are processed.
Incidents tab: browse every events folder (from this dashboard or Stages 4-6),
with the crops, and write incident reports with Claude (Stage 8).
"""

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONFIGS_DIR, MODELS_DIR, OUTPUTS_DIR, ROOT, STATIC_TRACKER, VideoIO  # noqa: E402
from pipeline import PipelineConfig, SafetyPipeline  # noqa: E402

EVENTS_DIR = OUTPUTS_DIR / "events"
UPLOADS_DIR = ROOT / "data" / "uploads"
SOURCES = {
    "Demo: fixed camera (zones, falls)": ROOT / "data" / "samples" / "demo_scene.mp4",
    "Demo: panning camera (PPE)": ROOT / "data" / "samples" / "demo_site.mp4",
    "Webcam": None,
    "Upload a video": None,
}

st.set_page_config(page_title="Safety Monitor", page_icon="🦺", layout="wide")


# ------------------------------------------------------------------ sidebar

def sidebar():
    sb = st.sidebar
    sb.title("Safety Monitor")
    kind = sb.radio("Source", list(SOURCES))
    source = None
    if kind == "Webcam":
        source = str(sb.number_input("Camera index", 0, 9, 0))
    elif kind == "Upload a video":
        up = sb.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])
        if up:
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            dst = UPLOADS_DIR / up.name
            if not dst.exists() or dst.stat().st_size != up.size:
                dst.write_bytes(up.getbuffer())
            source = str(dst)
    else:
        source = str(SOURCES[kind])

    sb.subheader("Checks")
    ppe = sb.toggle("PPE check", value=True)
    require = sb.multiselect("Required PPE", ["helmet", "vest", "gloves", "goggles", "boots"],
                             default=["helmet"], disabled=not ppe)
    zone_files = sorted(CONFIGS_DIR.glob("zones_*.json"))
    zone_names = ["None"] + [f.name for f in zone_files]
    default_zone = "zones_demo_scene.json" if kind.startswith("Demo: fixed") else "None"
    zone_choice = sb.selectbox("Zones", zone_names,
                               index=zone_names.index(default_zone) if default_zone in zone_names else 0,
                               help="Zones only make sense for the camera they were drawn on. "
                                    "Draw your own: python src/stage5_zones.py --draw")
    falls = sb.toggle("Fall detection", value=True)
    with sb.expander("Advanced"):
        item_conf = st.slider("PPE item confidence", 0.1, 0.9, 0.3, 0.05)
        min_ppe_height = st.slider("Min person height for PPE (px)", 40, 200, 80, 10)
        head_check = st.toggle("Head cut-off rule", value=True,
                               help="Only judge helmets while the head keypoints are visible inside the frame.")
        engines_ready = all((MODELS_DIR / f"{m}.engine").exists() for m in ("ppe_yolo26n", "yolo26n-pose"))
        trt = st.toggle("TensorRT engines", value=engines_ready, disabled=not engines_ready,
                        help="About 3x faster inference. Build the engines first: python src/export_trt.py export")
        stitch = st.toggle("Track stitching", value=True,
                           help="One ID per person: merges duplicate tracks (two boxes on one person) and "
                                "re-links ID switches, so one person raises each alarm once and counts once. "
                                "Re-linked people are labelled #new=old in the video.")

    cfg = PipelineConfig(ppe=ppe and bool(require), require=require or ["helmet"],
                         zones_file=CONFIGS_DIR / zone_choice if zone_choice != "None" else None,
                         falls=falls, item_conf=item_conf, min_ppe_height=min_ppe_height, head_check=head_check,
                         trt=trt, stitch=stitch)
    return source, cfg


# --------------------------------------------------------------------- live

def live_tab(source, cfg):
    left, right = st.columns([3, 1.2], gap="medium")
    with right:
        run = st.toggle("Run", key="run", disabled=source is None,
                        help="Switch off to stop. Each run writes a new events folder.")
        c1, c2 = st.columns(2)
        m_people, m_ppe = c1.empty(), c2.empty()
        m_zone, m_fall = c1.empty(), c2.empty()
        m_fps, m_events = c1.empty(), c2.empty()
        st.markdown("**Event feed**")
        feed = st.empty()
        stitch_line = st.empty()
    with left:
        frame_slot = st.empty()
        status = st.empty()

    def show_metrics(r=None, n_events=0):
        m_people.metric("People", r.people if r else "-")
        m_ppe.metric("PPE violations", r.ppe_violations if r else "-")
        m_zone.metric("In zones", r.in_zones if r else "-")
        m_fall.metric("Fallen", r.fallen if r else "-")
        m_fps.metric("FPS", f"{r.fps:.1f}" if r else "-")
        m_events.metric("Events", n_events)

    last = st.session_state.get("last_run")
    if not run:
        show_metrics()
        if source is None:
            frame_slot.info("Upload a video in the sidebar to start.")
        elif last:
            frame_slot.image(last["frame"], channels="BGR", width="stretch")
            status.caption(f"Last run: {last['frames']} frames, {len(last['feed'])} events. "
                           f"See the Incidents tab to review and write reports.")
            feed.dataframe(pd.DataFrame(last["feed"]), hide_index=True, width="stretch")
            if last.get("stitch"):
                stitch_line.caption(last["stitch"])
        else:
            frame_slot.info("Choose a source and checks in the sidebar, then switch Run on.")
        return

    try:
        video = VideoIO(source, show=False)
    except SystemExit as e:
        st.error(str(e))
        return
    if video.is_cam:
        cfg.tracker = STATIC_TRACKER        # a webcam doesn't move: skip camera-motion compensation
    with st.spinner("Loading models..."):
        pipe = SafetyPipeline(cfg, video.size, video.name)
    events_feed = []
    st.session_state["last_events_dir"] = pipe.events.dir.name
    status.caption(f"Recording events to `{pipe.events.dir.relative_to(ROOT)}`")

    result, frame, stitch_counts = None, None, None
    stitch_text = None if pipe.stitcher else "Track stitching is off: one person can raise an alarm more than once."
    if stitch_text:
        stitch_line.caption(stitch_text)
    try:
        for frame in video.frames():
            result = pipe.process(frame, video.now(), video.idx)
            frame_slot.image(result.image, channels="BGR", width="stretch")
            show_metrics(result, pipe.events.count)
            if result.new_events:
                for tid, reason in result.new_events:
                    events_feed.insert(0, {"time (s)": round(video.now(), 1), "who": f"#{tid}", "event": reason})
                feed.dataframe(pd.DataFrame(events_feed), hide_index=True, width="stretch")
            # Only redraw the stitching line when something new was merged.
            if pipe.stitcher and pipe.stitcher.counts() != stitch_counts:
                stitch_counts = pipe.stitcher.counts()
                stitch_text = pipe.stitcher.summary()
                stitch_line.caption(stitch_text)
    finally:
        video.close()
        pipe.close()
        if result is not None:
            st.session_state["last_run"] = {"frame": result.image, "frames": video.idx + 1, "feed": events_feed,
                                            "stitch": stitch_text}
    status.success(f"Finished: {video.idx + 1} frames, {pipe.events.count} events. "
                   "Switch Run off, then open the Incidents tab.")


# ---------------------------------------------------------------- incidents

def load_folder(d: Path):
    rows = list(csv.DictReader(open(d / "events.csv", encoding="utf-8")))
    reports = {}
    if (d / "reports.jsonl").exists():
        for line in (d / "reports.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                reports[rec["event"]["crop"]] = rec["report"]
    return rows, reports


PROVIDER_MODELS = {
    "openrouter": ("MiniMax via OpenRouter", ["minimax/minimax-m3"]),
    "anthropic": ("Claude (Anthropic)", ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]),
}


def write_reports(d: Path, provider: str, model: str):
    import stage8_report

    reporter = stage8_report.make_reporter(provider, model, "low")
    args = SimpleNamespace(max_side=1024, limit=None, dry_run=False)
    return stage8_report.process(d, reporter, args)


def incidents_tab():
    folders = sorted((d for d in EVENTS_DIR.iterdir() if (d / "events.csv").exists()),
                     key=lambda d: d.stat().st_mtime, reverse=True) if EVENTS_DIR.exists() else []
    if not folders:
        st.info("No events yet. Run the Live tab or Stages 4-6.")
        return

    names = [d.name for d in folders]
    default = st.session_state.get("last_events_dir")
    d = folders[names.index(default) if default in names else 0]
    d = EVENTS_DIR / st.selectbox("Events folder", names, index=names.index(d.name))
    rows, reports = load_folder(d)

    top = st.columns([2, 1, 1, 1])
    confirmed = sum(1 for r in reports.values() if r["confirmed"])
    top[0].markdown(f"**{len(rows)} events** in `{d.name}`")
    top[1].metric("Reported", f"{len(reports)}/{len(rows)}")
    top[2].metric("Confirmed", confirmed if reports else "-")
    top[3].metric("False alarms", len(reports) - confirmed if reports else "-")

    with st.expander("Write incident reports with a vision model", expanded=not reports and bool(rows)):
        from stage8_report import available_providers

        providers = available_providers()          # also loads keys from .env
        if not providers:
            st.warning("No API key found. Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY in the environment or in "
                       "a .env file in the project folder, then restart the dashboard. See README, Stage 8.")
        provider = st.selectbox("Provider", providers or ["openrouter"],
                                format_func=lambda p: PROVIDER_MODELS[p][0], disabled=not providers)
        model = st.selectbox("Model", PROVIDER_MODELS[provider][1], disabled=not providers)
        st.caption(f"Sends {len(rows)} events (2 images each) to {PROVIDER_MODELS[provider][0]}. "
                   "The images show people; check your organisation's rules before sending real footage.")
        if st.button("Write reports", type="primary", disabled=not providers or not rows):
            with st.spinner(f"Asking {model} about {len(rows)} events..."):
                try:
                    cost = write_reports(d, provider, model)
                    st.success(f"Done. Cost ${cost:.4f}.")
                except SystemExit as e:
                    st.error(str(e))
            st.rerun()

    if not rows:
        return
    table = []
    for r in rows:
        rep = reports.get(r["crop"])
        table.append({"time (s)": float(r["video_s"]), "who": r["track_id"], "rule": r["reason"],
                      "verdict": ("confirmed" if rep["confirmed"] else "false alarm") if rep else "",
                      "severity": rep["severity"] if rep else "", "title": rep["title"] if rep else ""})
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")

    st.markdown("**Event images**")
    cols = st.columns(4)
    for i, r in enumerate(rows):
        rep = reports.get(r["crop"])
        with cols[i % 4]:
            st.image(str(d / r["crop"]), width="stretch")
            st.caption(f"{r['video_s']} s · #{r['track_id']} · {r['reason']}")
            if rep:
                badge = "✅ confirmed" if rep["confirmed"] else "⚪ false alarm"
                with st.popover(f"{badge} · {rep['severity']}", width="stretch"):
                    st.markdown(f"**{rep['title']}**\n\n{rep['description']}")
                    if not rep["confirmed"]:
                        st.markdown(f"*Why not:* {rep['false_alarm_reason']}")
                    for label, key in (("PPE seen", "ppe_observed"), ("PPE missing", "ppe_missing"),
                                       ("Hazards", "hazards"), ("Actions", "recommended_actions")):
                        if rep[key]:
                            st.markdown(f"**{label}:** " + ", ".join(rep[key]))
                    st.image(str(d / r["scene"]), width="stretch")


source, cfg = sidebar()
tab_live, tab_inc = st.tabs(["Live", "Incidents"])
with tab_live:
    live_tab(source, cfg)
with tab_inc:
    incidents_tab()
