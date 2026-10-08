"""Stage 8: incident reports from a vision-language model (Claude or MiniMax).

Stages 4-6 write events with a rule-based reason ("missing helmet", "fall
detected") plus two images: a crop of the person and the full frame. This
stage sends each event to a vision-language model and gets back a structured report:

    - confirmed:  do the images actually support the automated alert?
                  (the VLM acts as a second opinion that filters false alarms)
    - severity, title, description
    - PPE seen / missing, hazards, recommended actions

The fast models run on every frame; the slow, expensive VLM runs only on the
handful of frames where something happened. That split is how real systems
keep cost and latency down.

Providers (--provider, default: whichever key is set, Anthropic first):
    anthropic    Claude via the Anthropic SDK (ANTHROPIC_API_KEY), default claude-opus-5-5
    openrouter   any OpenRouter vision model (OPENROUTER_API_KEY), default minimax/minimax-m3

Keys are read from the environment or from a .env file in the project root
(git-ignored). Both providers return the same IncidentReport schema.

Output, written into the events folder:
    reports.jsonl     one JSON report per event
    report.html       a readable incident report with the images

Examples:
    python src/stage8_report.py --dry-run                        # newest events folder, no API calls
    python src/stage8_report.py                                  # newest events folder
    python src/stage8_report.py --provider openrouter --limit 2
    python src/stage8_report.py --events outputs/events/stage4_demo_site_20261009_005021
    python src/stage8_report.py --all                            # every events folder
"""

import argparse
import base64
import csv
import html
import json
import os
import re
import time
from pathlib import Path
from typing import Literal

import anthropic
import cv2
import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from common import OUTPUTS_DIR, load_env

EVENTS_DIR = OUTPUTS_DIR / "events"
DEFAULT_MODELS = {"anthropic": "claude-opus-5-5", "openrouter": "minimax/minimax-m3"}
CLAUDE_PRICE_PER_MTOK = {"claude-opus-5-5": (4.00, 20.00), "claude-sonnet-5-5": (2.00, 10.00),
                         "claude-haiku-5-5": (0.10, 0.50)}       # (input, output) USD per million tokens
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """You review alerts from an automated workplace-safety camera system.

The system uses object detectors, a tracker and simple rules. It is fast but makes
mistakes: missed helmets at the edge of the frame, people partly hidden, odd poses.
For each alert you get the rule that fired and two images: a close-up crop of the
flagged person and the full camera frame with that person boxed in red.

Judge only the person in the red box, and only from what is visible. "confirmed" means
the rule's claim is true for that person: for "missing helmet", confirmed is true when
the boxed person's head is visible and has no helmet or hard hat on it. Other people
and objects in the frame don't change that. Frames may come from a moving camera or be
stitched from several photos, so ignore odd backgrounds.

If the claim can't be checked (head turned away or out of frame, person hidden), set
confirmed to false, say why in false_alarm_reason and use low confidence rather than
guess. Describe first, decide last: the verdict must agree with your description.
Keep the description factual and short, as a safety officer would write it. Do not
try to identify people."""


class IncidentReport(BaseModel):
    # Field order matters: models write JSON in schema order, so the description and the
    # evidence come first and the verdict last ("describe first, decide last"). With the
    # verdict first, a model can commit to it before looking, then contradict it.
    model_config = ConfigDict(extra="forbid")      # strict JSON schema: no extra fields

    description: str = Field(description="2-4 factual sentences about what is visible")
    ppe_observed: list[str] = Field(description="PPE items visibly worn")
    ppe_missing: list[str] = Field(description="PPE items that appear to be missing")
    hazards: list[str] = Field(description="Hazards visible in the scene, if any")
    confirmed: bool = Field(description="Is the rule's claim true for the boxed person?")
    false_alarm_reason: str = Field(description="Why the alert is wrong or can't be checked; empty if confirmed")
    confidence: Literal["low", "medium", "high"]
    severity: Literal["none", "low", "medium", "high", "critical"]
    title: str = Field(description="Short headline, under 10 words")
    recommended_actions: list[str]


def parse_args():
    p = argparse.ArgumentParser(description="Stage 8 - VLM incident reports")
    p.add_argument("--events", type=Path, default=None, help="events folder (default: newest)")
    p.add_argument("--all", action="store_true", help="process every events folder")
    p.add_argument("--provider", choices=["auto", "anthropic", "openrouter"], default="auto")
    p.add_argument("--model", default=None, help="default: claude-opus-5-5 / minimax/minimax-m3")
    p.add_argument("--effort", default="low", choices=["low", "medium", "high"],
                   help="reasoning effort; low is enough for describing an image and keeps cost down")
    p.add_argument("--max-side", type=int, default=1024, help="downscale images so the longest side is at most this")
    p.add_argument("--limit", type=int, default=None, help="process at most N events per folder")
    p.add_argument("--dry-run", action="store_true", help="build the requests but don't call the API")
    return p.parse_args()


def available_providers():
    """Providers whose API key is set (environment or .env), Anthropic first."""
    load_env()
    return [p for p, key in (("anthropic", "ANTHROPIC_API_KEY"), ("openrouter", "OPENROUTER_API_KEY"))
            if os.environ.get(key)]


def pick_provider(choice="auto"):
    if choice != "auto":
        load_env()
        return choice
    found = available_providers()
    return found[0] if found else None


def encode_image(path: Path, max_side: int):
    """Load, downscale and base64-encode a JPEG. Smaller images = fewer input tokens."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    scale = max_side / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.standard_b64encode(buf.tobytes()).decode("ascii"), img.shape[1], img.shape[0]


def build_content(event_dir: Path, row: dict, max_side: int):
    """Provider-neutral message parts: [("text", str) | ("image", base64 jpeg), ...]."""
    stage = event_dir.name.split("_")[0]
    crop, cw, ch = encode_image(event_dir / row["crop"], max_side)
    scene, sw, sh = encode_image(event_dir / row["scene"], max_side)
    text = (f"Alert from {stage}: \"{row['reason']}\"\n"
            f"Video time {row['video_s']} s, track #{row['track_id']}.\n"
            "Image 1: close-up of the flagged person. Image 2: full frame, person boxed in red.")
    # Images before the question tends to work best for vision prompts.
    parts = [("text", "Image 1:"), ("image", crop), ("text", "Image 2:"), ("image", scene), ("text", text)]
    return parts, (cw, ch, sw, sh)


class ReportError(RuntimeError):
    """One event failed; the run continues with the next."""


class ClaudeReporter:
    provider = "anthropic"

    def __init__(self, model, effort):
        self.model, self.effort = model, effort
        self.client = anthropic.Anthropic()

    def report(self, parts):
        content = [{"type": "text", "text": v} if k == "text" else
                   {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": v}}
                   for k, v in parts]
        try:
            response = self.client.beta.messages.parse(
                model=self.model,
                max_tokens=4000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
                output_format=IncidentReport,
                output_config={"effort": self.effort},
                # If a safety classifier declines, the API retries on a suitable model in the same call.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError:
            raise SystemExit("Authentication failed. Check ANTHROPIC_API_KEY (see README, Stage 8).")
        except TypeError as e:
            # The SDK raises TypeError at request time when no credentials are configured at all.
            if "authentication" not in str(e).lower():
                raise
            raise SystemExit("No Anthropic credentials found. Set ANTHROPIC_API_KEY (see README, Stage 8).")
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APIStatusError) as e:
            raise ReportError(str(e))
        if response.stop_reason == "refusal":
            category = response.stop_details.category if response.stop_details else None
            raise ReportError(f"model declined (category: {category})")
        u = response.usage
        p_in, p_out = CLAUDE_PRICE_PER_MTOK.get(response.model, CLAUDE_PRICE_PER_MTOK["claude-opus-5-5"])
        cost = (u.input_tokens * p_in + u.output_tokens * p_out) / 1e6
        return response.parsed_output, u.input_tokens, u.output_tokens, cost, response.model


class OpenRouterReporter:
    """Any vision model on OpenRouter (OpenAI-style chat API), called with plain HTTP.

    Asks for strict JSON matching IncidentReport, then validates it in Python anyway:
    "strict" support varies by model and by the upstream provider OpenRouter routes to.
    """
    provider = "openrouter"

    def __init__(self, model, effort):
        self.model, self.effort = model, effort
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("OPENROUTER_API_KEY is not set (environment or .env). See README, Stage 8.")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {key}", "X-Title": "Safety Monitoring"})

    def _post(self, body):
        err = "no response"
        for attempt in range(3):
            try:
                r = self.session.post(OPENROUTER_URL, json=body, timeout=180)
            except requests.RequestException as e:
                err = f"connection error: {e}"
            else:
                if r.status_code == 200:
                    return r.json()
                msg = r.text[:300]
                if r.status_code == 401:
                    raise SystemExit("OpenRouter rejected the key (401). Check OPENROUTER_API_KEY.")
                if r.status_code == 402:
                    raise SystemExit("OpenRouter: not enough credits on this key (402).")
                if r.status_code not in (408, 429, 500, 502, 503, 504):
                    raise ReportError(f"HTTP {r.status_code}: {msg}")
                err = f"HTTP {r.status_code}: {msg}"
            time.sleep(2 * (attempt + 1))             # back off, then retry
        raise ReportError(err)

    def report(self, parts):
        content = [{"type": "text", "text": v} if k == "text" else
                   {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{v}"}} for k, v in parts]
        body = {
            "model": self.model,
            # Reasoning models spend hidden reasoning tokens from this budget before the JSON;
            # 4000 cut some replies off mid-JSON. You pay only for tokens actually used.
            "max_tokens": 16000,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "incident_report", "strict": True, "schema": IncidentReport.model_json_schema()}},
            "reasoning": {"effort": self.effort},
            "usage": {"include": True},                # ask OpenRouter to report the cost
        }
        # OpenRouter spreads a model over several hosting providers, and now and then a reply
        # comes back cut short or malformed. Those are retried; every attempt is paid for.
        cost, tok_in, tok_out, problem = 0.0, 0, 0, ""
        for _ in range(3):
            data = self._post(body)
            if "error" in data:
                raise ReportError(str(data["error"])[:300])
            u = data.get("usage", {})
            cost += float(u.get("cost") or 0)
            tok_in += u.get("prompt_tokens", 0)
            tok_out += u.get("completion_tokens", 0)
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
            # Some models wrap JSON in <think> blocks or ``` fences despite the schema.
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            match = re.search(r"\{.*\}", text, flags=re.S)
            try:
                report = IncidentReport.model_validate_json(match.group(0) if match else text)
                return report, tok_in, tok_out, cost, data.get("model", self.model)
            except ValidationError as e:
                problem = (f"reply didn't match the schema ({choice.get('finish_reason')}, "
                           f"via {data.get('provider')}): {e.errors()[0]['msg']} | {text[:120]!r}")
        raise ReportError(problem + " (3 attempts)")


def make_reporter(provider, model=None, effort="low"):
    model = model or DEFAULT_MODELS[provider]
    return ClaudeReporter(model, effort) if provider == "anthropic" else OpenRouterReporter(model, effort)


def write_html(event_dir: Path, results):
    sev_color = {"none": "#6b7280", "low": "#2563eb", "medium": "#d97706", "high": "#dc2626", "critical": "#7f1d1d"}
    cards = []
    for row, rep, err in results:
        esc = html.escape
        if err:
            body = f"<p class='err'>Not reported: {esc(err)}</p>"
            badge = "<span class='badge' style='background:#6b7280'>error</span>"
        else:
            verdict = "Confirmed" if rep.confirmed else "False alarm"
            badge = (f"<span class='badge' style='background:{sev_color[rep.severity]}'>{rep.severity}</span>"
                     f"<span class='badge {'ok' if rep.confirmed else 'fa'}'>{verdict} ({rep.confidence})</span>")
            lists = "".join(
                f"<h4>{t}</h4><ul>{''.join(f'<li>{esc(i)}</li>' for i in items)}</ul>"
                for t, items in (("PPE seen", rep.ppe_observed), ("PPE missing", rep.ppe_missing),
                                 ("Hazards", rep.hazards), ("Recommended actions", rep.recommended_actions)) if items)
            fa = f"<p><b>Why not:</b> {esc(rep.false_alarm_reason)}</p>" if not rep.confirmed else ""
            body = f"<h3>{esc(rep.title)}</h3><p>{esc(rep.description)}</p>{fa}{lists}"
        cards.append(f"""<article>
  <div class="imgs"><img src="{esc(row['crop'])}" alt="crop"><img src="{esc(row['scene'])}" alt="scene"></div>
  <div class="txt"><div class="meta">{badge}<span>{esc(row['video_s'])} s &middot; track #{esc(row['track_id'])}
  &middot; rule: <code>{esc(row['reason'])}</code></span></div>{body}</div>
</article>""")

    n_ok = sum(1 for _, r, e in results if r and r.confirmed)
    n_fa = sum(1 for _, r, e in results if r and not r.confirmed)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Incident report</title>
<style>
:root{{--bg:#f7f7f5;--card:#fff;--fg:#1f2328;--muted:#6b7280;--line:#e5e7eb}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16181c;--card:#1f2228;--fg:#e6e6e6;--muted:#9aa0a6;--line:#30343b}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:24px 16px}}
h1{{margin:0 0 4px}} .sub{{color:var(--muted);margin-bottom:20px}}
article{{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:16px;background:var(--card);
  border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:14px}}
@media (max-width:760px){{article{{grid-template-columns:1fr}}}}
.imgs{{display:grid;grid-template-columns:1fr 2fr;gap:8px;align-items:start}}
.imgs img{{width:100%;border-radius:6px;border:1px solid var(--line)}}
.meta{{display:flex;flex-wrap:wrap;gap:6px;align-items:center;color:var(--muted);font-size:13px}}
.badge{{color:#fff;border-radius:999px;padding:1px 9px;font-size:12px;text-transform:uppercase}}
.badge.ok{{background:#15803d}} .badge.fa{{background:#6b7280}}
h3{{margin:8px 0 4px}} h4{{margin:10px 0 2px;font-size:13px;color:var(--muted)}} ul{{margin:0;padding-left:18px}}
.err{{color:#dc2626}} code{{font-size:12px}}
</style></head><body><main>
<h1>Incident report</h1>
<div class="sub">{html.escape(event_dir.name)} &middot; {len(results)} alerts &middot;
{n_ok} confirmed &middot; {n_fa} judged false alarms</div>
{''.join(cards)}
</main></body></html>"""
    (event_dir / "report.html").write_text(page, encoding="utf-8")


def process(event_dir: Path, reporter, args):
    rows = list(csv.DictReader(open(event_dir / "events.csv", encoding="utf-8")))[:args.limit]
    if not rows:
        print(f"{event_dir.name}: no events")
        return 0.0
    print(f"\n{event_dir.name}: {len(rows)} events")

    results, total = [], 0.0
    out = None if args.dry_run else open(event_dir / "reports.jsonl", "w", encoding="utf-8")
    try:
        for row in rows:
            parts, (cw, ch, sw, sh) = build_content(event_dir, row, args.max_side)
            label = f"  #{row['track_id']:>4} {row['reason'][:32]:32s}"
            if args.dry_run:
                kb = sum(len(v) for k, v in parts if k == "image") * 3 / 4 / 1024
                print(f"{label} would send crop {cw}x{ch} + scene {sw}x{sh} ({kb:.0f} KB)")
                continue
            try:
                t0 = time.perf_counter()
                rep, tok_in, tok_out, c, served_by = reporter.report(parts)
                total += c
                verdict = "CONFIRMED  " if rep.confirmed else "FALSE ALARM"
                print(f"{label} {verdict} {rep.severity:8s} {rep.title}  "
                      f"[{tok_in}+{tok_out} tok, ${c:.4f}, {time.perf_counter() - t0:.1f}s]")
                out.write(json.dumps({"event": row, "provider": reporter.provider, "model": served_by,
                                      "report": rep.model_dump()}) + "\n")
                out.flush()
                results.append((row, rep, None))
            except ReportError as e:
                print(f"{label} ERROR {e}")
                results.append((row, None, str(e)))
    finally:
        if out:
            out.close()

    if not args.dry_run:
        write_html(event_dir, results)
        print(f"  -> {event_dir / 'report.html'}  (cost ${total:.4f})")
    return total


def main():
    args = parse_args()
    if args.all:
        dirs = sorted(d for d in EVENTS_DIR.iterdir() if (d / "events.csv").exists())
    elif args.events:
        dirs = [args.events]
    else:
        candidates = [d for d in EVENTS_DIR.iterdir() if (d / "events.csv").exists()] if EVENTS_DIR.exists() else []
        if not candidates:
            raise SystemExit("No events found. Run stage 4, 5 or 6 first.")
        dirs = [max(candidates, key=lambda d: d.stat().st_mtime)]

    provider = pick_provider(args.provider)
    reporter = None
    if not args.dry_run:
        if provider is None:
            raise SystemExit("No API key found. Set ANTHROPIC_API_KEY or OPENROUTER_API_KEY "
                             "(environment or .env; see README, Stage 8), or use --dry-run.")
        reporter = make_reporter(provider, args.model, args.effort)

    model = args.model or DEFAULT_MODELS.get(provider or "anthropic")
    print(f"Provider: {provider or '-'} | model: {model} | effort: {args.effort}"
          f"{' | DRY RUN' if args.dry_run else ''}")
    total = sum(process(d, reporter, args) for d in dirs)
    if not args.dry_run:
        print(f"\nTotal cost: ${total:.4f}")


if __name__ == "__main__":
    main()
