import json
import re
import urllib.request
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi

TRANSCRIPTS_DIR = Path("transcripts")
MAX_TOKENS = 2048
LITELLM_PRICES_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"


def fetch_model_pricing(model: str) -> tuple[float, float] | None:
    try:
        req = urllib.request.Request(LITELLM_PRICES_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
        entry = data.get(model)
        if entry is None:
            return None
        return (entry["input_cost_per_token"], entry["output_cost_per_token"])
    except Exception:
        return None


def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be",):
        return parsed.path.lstrip("/")
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        params = parse_qs(parsed.query)
        return params["v"][0]
    raise ValueError(f"Unrecognized YouTube URL format: {url}")


def fetch_video_title(video_id: str) -> str:
    url = f"https://www.youtube.com/watch?v={video_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            html = response.read().decode("utf-8", errors="replace")
        match = re.search(r'<meta property="og:title" content="([^"]+)"', html)
        if match:
            return match.group(1)
        match = re.search(r'<title>([^<]+)</title>', html)
        if match:
            return match.group(1).replace(" - YouTube", "").strip()
    except Exception:
        pass
    return video_id


def _pick_transcript_auto(available: list) -> tuple[object, str]:
    """
    Pick the best transcript for automated use, translating to English via
    YouTube's translation endpoint when no English transcript exists.

    Returns (transcript, description) where description is a short human-readable
    string like "manual EN" or "auto-generated PT → EN".

    Priority:
    1. Manual English
    2. Auto-generated English
    3. Manual (any language) — translated to English
    4. Auto-generated (any language) — translated to English
    5. Manual (any language) — untranslated fallback
    6. Auto-generated (any language) — untranslated fallback
    """
    manual = [t for t in available if not t.is_generated]
    generated = [t for t in available if t.is_generated]

    manual_en = [t for t in manual if t.language_code.startswith("en")]
    if manual_en:
        return manual_en[0], "manual EN"

    generated_en = [t for t in generated if t.language_code.startswith("en")]
    if generated_en:
        return generated_en[0], "auto-generated EN"

    manual_translatable = [t for t in manual if t.is_translatable]
    if manual_translatable:
        t = manual_translatable[0]
        return t.translate("en"), f"manual {t.language_code} → EN"

    generated_translatable = [t for t in generated if t.is_translatable]
    if generated_translatable:
        t = generated_translatable[0]
        return t.translate("en"), f"auto-generated {t.language_code} → EN"

    fallback = (manual + generated)[0]
    kind = "auto-generated" if fallback.is_generated else "manual"
    return fallback, f"{kind} {fallback.language_code}"


def pick_transcript(available: list, auto: bool = False) -> object:
    if len(available) == 1:
        return available[0]

    if auto:
        transcript, _ = _pick_transcript_auto(available)
        return transcript

    print("\nMultiple transcripts available:")
    for i, t in enumerate(available):
        kind = "manual" if not t.is_generated else "auto-generated"
        print(f"  [{i + 1}] {t.language} ({t.language_code}) — {kind}")

    while True:
        choice = input(f"Choose [1-{len(available)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(available):
            return available[int(choice) - 1]


def _seconds_to_timestamp(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 60}:{total % 60:02d}"


def fetch_and_save_transcript(video_id: str, auto: bool = False) -> tuple[str, str | None]:
    """Return (transcript_text, source_description).

    source_description is a short human-readable string such as "manual EN" or
    "auto-generated PT → EN" when auto=True, or None for interactive selection.
    """
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)
    available = list(transcript_list)

    if auto:
        chosen, description = _pick_transcript_auto(available)
    else:
        chosen = pick_transcript(available, auto=False)
        description = None

    fetched = chosen.fetch()

    TRANSCRIPTS_DIR.mkdir(exist_ok=True)

    # Save structured segments so ytsearch can cite timestamps later.
    segments = [{"start": s.start, "text": s.text} for s in fetched]
    segments_path = TRANSCRIPTS_DIR / f"{video_id}.json"
    segments_path.write_text(json.dumps(segments, ensure_ascii=False), encoding="utf-8")

    transcript = " ".join(s.text for s in fetched)
    transcript_path = TRANSCRIPTS_DIR / f"{video_id}.txt"
    transcript_path.write_text(transcript, encoding="utf-8")

    return transcript, description


def load_transcript_segments(video_id: str) -> list[dict] | None:
    """Return [{start, text}, …] from the JSON sidecar, or None if unavailable."""
    segments_path = TRANSCRIPTS_DIR / f"{video_id}.json"
    if not segments_path.exists():
        return None
    try:
        return json.loads(segments_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def format_transcript_with_timestamps(segments: list[dict]) -> str:
    return "\n".join(
        f"[{_seconds_to_timestamp(seg['start'])}] {seg['text']}" for seg in segments
    )


def load_transcript(video_id: str, auto: bool = False) -> tuple[str, bool, str | None]:
    """Return (transcript_text, from_cache, source_description).

    source_description is None for cache hits and a human-readable string like
    "auto-generated EN" for fresh fetches.
    """
    transcript_path = TRANSCRIPTS_DIR / f"{video_id}.txt"
    if transcript_path.exists():
        return transcript_path.read_text(encoding="utf-8"), True, None

    text, description = fetch_and_save_transcript(video_id, auto=auto)
    return text, False, description
