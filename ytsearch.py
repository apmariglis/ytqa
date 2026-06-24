from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import yt_dlp
from youtube_transcript_api import CouldNotRetrieveTranscript
from dotenv import load_dotenv
import anthropic
from textual.app import App, ComposeResult
from textual.widgets import Header, Input, Markdown, Static
from textual.containers import VerticalScroll
from textual import on, work
from textual.markup import escape

from ytlib import (
    load_transcript,
    load_transcript_segments,
    MAX_TOKENS,
)
from ytlog import SessionLog, serialize_content

load_dotenv()

PLANNING_SYSTEM_PROMPT = """\
You are a YouTube research assistant. Given a user's query, plan a research session \
by calling the plan_search tool with:
1. A list of YouTube search queries (2–5 queries) to find relevant videos — vary the phrasing \
and include synonyms so the searches complement each other.
2. A list of keywords and short phrases (5–10) that would appear in the transcript text of a \
video that genuinely answers the query — include technical terms, synonyms, and related concepts.\
"""

SYNTHESIS_SYSTEM_PROMPT = """\
You are a YouTube research assistant. Using the relevant transcript excerpts provided, write a \
comprehensive answer to the user's query in Markdown.

Excerpts include [M:SS] timestamps indicating where in the video the content appears. Cite \
specific claims with the source video and the nearest timestamp using this format: \
([Video Title](URL) ~M:SS)

For example: "The GIL prevents true parallelism ([Understanding the GIL](https://...) ~2:15)"

If an excerpt has no timestamp, cite with just the title and URL.

Base your entire answer on the provided excerpts. Do not suggest the user seek \
"additional sources" or "further research" — give the most complete answer the available \
excerpts allow, and state any gaps plainly within the answer itself.\
"""

SYNTHESIS_MAX_TOKENS = 4096
SEARCH_MAX_RESULTS = 10
MAX_POOL_SIZE = 20          # max unique videos to fetch transcripts for
MAX_CONCURRENT_FETCHES = 5  # parallel transcript fetches
DEFAULT_WINDOW_SIZE = 3

# Maps status kind → Rich markup tag used in the TUI.
_STATUS_MARKUP: dict[str, str] = {
    "info": "dim",
    "found": "dim",
    "success": "green",
    "skip": "yellow",
    "error": "red",
    "phase": "bold",
}

CSS = """
.user-query {
    border-left: thick cyan;
    padding: 1 2;
    height: auto;
}
.status-message {
    color: $text-muted;
    padding: 0 2;
    height: auto;
}
.answer {
    border-left: thick green;
    padding: 1 2;
    height: auto;
}
.answer > * {
    margin-top: 0;
    margin-bottom: 0;
}
.answer MarkdownH1,
.answer MarkdownH2,
.answer MarkdownH3 {
    margin-top: 1;
}
.answer MarkdownParagraph {
    margin: 0;
}
.answer MarkdownBulletList,
.answer MarkdownOrderedList {
    margin: 0;
    padding-left: 2;
}
.answer MarkdownTableOfContents {
    display: none;
}
#chat-log {
    height: 1fr;
    overflow-y: scroll;
}
Input {
    dock: bottom;
}
"""


_RETRYABLE_STATUS_CODES = {429, 529}
_MAX_API_RETRIES = 5


def _create_with_retry(client, status_callback=None, **kwargs):
    for attempt in range(_MAX_API_RETRIES):
        try:
            return client.messages.create(**kwargs)
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            is_retryable = isinstance(status_code, int) and status_code in _RETRYABLE_STATUS_CODES
            has_attempts_left = attempt < _MAX_API_RETRIES - 1
            if is_retryable and has_attempts_left:
                wait = 2 ** attempt
                if status_callback:
                    status_callback(f"API overloaded — retrying in {wait}s…", "info")
                time.sleep(wait)
            else:
                raise


class SearchError(Exception):
    pass


@dataclass
class VideoResult:
    video_id: str
    title: str
    url: str
    description: str | None
    view_count: int | None
    duration_seconds: int | None


def search_youtube(query: str, max_results: int = SEARCH_MAX_RESULTS) -> list[VideoResult]:
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
    except yt_dlp.utils.DownloadError as exc:
        raise SearchError(str(exc)) from exc

    entries = info.get("entries") if info else None
    if not entries:
        return []

    results = []
    for entry in entries[:max_results]:
        video_id = entry.get("id", "")
        results.append(
            VideoResult(
                video_id=video_id,
                title=entry.get("title", ""),
                url=f"https://www.youtube.com/watch?v={video_id}",
                description=entry.get("description"),
                view_count=entry.get("view_count"),
                duration_seconds=entry.get("duration"),
            )
        )

    return results


def build_plan_tool_definition() -> dict:
    return {
        "name": "plan_search",
        "description": (
            "Plan a YouTube research session by specifying search queries and "
            "keywords to locate in video transcripts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "search_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "YouTube search queries to find relevant videos.",
                },
                "transcript_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Keywords and short phrases to search for in video transcripts. "
                        "Include technical terms, synonyms, and related concepts."
                    ),
                },
            },
            "required": ["search_queries", "transcript_keywords"],
        },
    }


def fetch_transcripts_for_videos(video_ids: list[str]) -> dict[str, str]:
    if not video_ids:
        return {}

    transcripts: dict[str, str] = {}
    for video_id in video_ids:
        try:
            text, _ = load_transcript(video_id)
            transcripts[video_id] = text
        except Exception:
            pass

    return transcripts


def extract_keyword_windows(
    segments: list[dict],
    keywords: list[str],
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> list[dict]:
    """
    Find segments matching any keyword and return merged windows of surrounding context.
    Each returned window is {"start": float, "text": str}.
    Matching is case-insensitive substring search.
    """
    if not segments or not keywords:
        return []

    keywords_lower = [kw.lower() for kw in keywords]

    matched_indices = set()
    for i, seg in enumerate(segments):
        text_lower = seg["text"].lower()
        if any(kw in text_lower for kw in keywords_lower):
            matched_indices.add(i)

    if not matched_indices:
        return []

    # Expand each match to a window range
    ranges = []
    for idx in sorted(matched_indices):
        start = max(0, idx - window_size)
        end = min(len(segments) - 1, idx + window_size)
        ranges.append([start, end])

    # Merge overlapping ranges (not merely adjacent — adjacent windows have no
    # shared context so they should stay separate).
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    result = []
    for start, end in merged:
        window_segs = segments[start:end + 1]
        text = " ".join(seg["text"] for seg in window_segs)
        result.append({"start": window_segs[0]["start"], "text": text})

    return result


def build_video_excerpts(
    all_video_results: dict[str, VideoResult],
    segments_by_id: dict[str, list[dict]],
    keywords: list[str],
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> dict[str, dict]:
    """
    Build {video_id → {title, url, excerpts}} for every video where at least
    one keyword matches a transcript segment. Videos with no matches are omitted.
    """
    result = {}
    for video_id, segments in segments_by_id.items():
        windows = extract_keyword_windows(segments, keywords, window_size)
        if not windows:
            continue
        video = all_video_results.get(video_id)
        result[video_id] = {
            "title": video.title if video else video_id,
            "url": video.url if video else f"https://www.youtube.com/watch?v={video_id}",
            "excerpts": windows,
        }
    return result


def _format_excerpt_window(window: dict) -> str:
    total = int(window["start"])
    ts = f"{total // 60}:{total % 60:02d}"
    return f"[{ts}] {window['text']}"


def run_agent_loop(
    user_query: str,
    client,
    model: str,
    status_callback=None,
    session_log: SessionLog | None = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> str:
    # --- Phase 1: Planning ---
    if status_callback:
        status_callback("Planning", "phase")

    plan_response = _create_with_retry(
        client,
        status_callback,
        model=model,
        max_tokens=MAX_TOKENS,
        system=PLANNING_SYSTEM_PROMPT,
        tools=[build_plan_tool_definition()],
        messages=[{"role": "user", "content": user_query}],
    )

    search_queries: list[str] = []
    transcript_keywords: list[str] = []
    for block in plan_response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "plan_search":
            search_queries = block.input.get("search_queries", [])
            transcript_keywords = block.input.get("transcript_keywords", [])

    if session_log:
        session_log.record(
            "claude_planning_response",
            search_queries=search_queries,
            transcript_keywords=transcript_keywords,
        )

    if status_callback:
        for q in search_queries:
            status_callback(f'Search query: "{q}"', "info")
        status_callback(f"Keywords: {', '.join(transcript_keywords)}", "info")

    # --- Phase 2: Searching YouTube ---
    if status_callback:
        status_callback(f"Searching YouTube ({len(search_queries)} quer{'y' if len(search_queries) == 1 else 'ies'})", "phase")

    all_video_results: dict[str, VideoResult] = {}

    for query in search_queries:
        if status_callback:
            status_callback(f'Searching: "{query}"…', "info")
        try:
            results = search_youtube(query, SEARCH_MAX_RESULTS)
            for r in results:
                is_new = r.video_id not in all_video_results
                all_video_results[r.video_id] = r
                if is_new and status_callback:
                    status_callback(f'Found: "{r.title}" — {r.url}', "found")

            if session_log:
                session_log.record(
                    "youtube_search",
                    query=query,
                    results=[
                        {"video_id": r.video_id, "title": r.title, "url": r.url}
                        for r in results
                    ],
                )
        except SearchError as exc:
            if session_log:
                session_log.record("youtube_search_error", query=query, error=str(exc))
            if status_callback:
                status_callback(f'Search failed: "{query}"', "error")

    # Cap pool size to avoid fetching hundreds of transcripts.
    if len(all_video_results) > MAX_POOL_SIZE:
        all_video_results = dict(list(all_video_results.items())[:MAX_POOL_SIZE])

    # --- Phase 3: Fetching transcripts ---
    if status_callback:
        status_callback(f"Fetching transcripts ({len(all_video_results)} video{'s' if len(all_video_results) != 1 else ''})", "phase")

    segments_by_id: dict[str, list[dict]] = {}
    _segments_lock = threading.Lock()

    def _fetch_one(video_id: str, title: str) -> None:
        if status_callback:
            status_callback(f"Fetching transcript: {title}…", "info")
        try:
            text, _ = load_transcript(video_id, auto=True)
            segs = load_transcript_segments(video_id)
            if segs:
                with _segments_lock:
                    segments_by_id[video_id] = segs
            if status_callback:
                status_callback(f"Loaded: {title}", "success")
            if session_log:
                session_log.record(
                    "transcript_fetch",
                    video_id=video_id,
                    title=title,
                    status="success",
                )
        except CouldNotRetrieveTranscript:
            if status_callback:
                status_callback(f"No captions: {title}", "skip")
            if session_log:
                session_log.record(
                    "transcript_fetch",
                    video_id=video_id,
                    title=title,
                    status="no_captions",
                )
        except Exception as exc:
            if status_callback:
                status_callback(f"Failed: {title}", "error")
            if session_log:
                session_log.record(
                    "transcript_fetch",
                    video_id=video_id,
                    title=title,
                    status="failed",
                    error=str(exc),
                )

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FETCHES) as executor:
        futures = {
            executor.submit(_fetch_one, vid_id, vid.title): vid_id
            for vid_id, vid in all_video_results.items()
        }
        for future in as_completed(futures):
            future.result()  # exceptions are handled inside _fetch_one

    # --- Phase 4: Keyword matching ---
    if status_callback:
        status_callback("Matching keywords in transcripts", "phase")

    video_excerpts = build_video_excerpts(
        all_video_results, segments_by_id, transcript_keywords, window_size
    )

    if status_callback:
        for video_id, video in all_video_results.items():
            if video_id in video_excerpts:
                n = len(video_excerpts[video_id]["excerpts"])
                status_callback(f"Matched: {video.title} ({n} excerpt{'s' if n != 1 else ''})", "success")
            elif video_id in segments_by_id:
                status_callback(f"No matches: {video.title}", "skip")

    # --- Phase 5: Synthesising ---
    if status_callback:
        status_callback("Synthesising answer", "phase")
    transcript_sections = []
    for video_id, data in video_excerpts.items():
        excerpts_text = "\n".join(_format_excerpt_window(w) for w in data["excerpts"])
        transcript_sections.append(
            f"## {data['title']}\nURL: {data['url']}\n\nRelevant excerpts:\n{excerpts_text}"
        )

    transcript_context = "\n\n---\n\n".join(transcript_sections)

    if session_log:
        session_log.record(
            "synthesis_input",
            videos_with_excerpts=[
                {"video_id": vid, "title": data["title"]}
                for vid, data in video_excerpts.items()
            ],
            transcript_context=transcript_context,
        )

    synthesis_messages = [
        {
            "role": "user",
            "content": (
                f"User query: {user_query}\n\n"
                f"Here are relevant excerpts from video transcripts:\n\n{transcript_context}\n\n"
                "Please provide a comprehensive answer with citations."
            ),
        }
    ]

    synthesis_response = _create_with_retry(
        client,
        status_callback,
        model=model,
        max_tokens=SYNTHESIS_MAX_TOKENS,
        system=SYNTHESIS_SYSTEM_PROMPT,
        messages=synthesis_messages,
    )

    answer = synthesis_response.content[0].text

    if session_log:
        session_log.record("synthesis_output", answer=answer)

    return answer


class YtsearchApp(App):
    CSS = CSS
    BINDINGS = [("ctrl+q", "quit", "Quit")]

    def __init__(self, client, model: str, initial_query: str | None = None) -> None:
        super().__init__()
        self._client = client
        self._model = model
        self._initial_query = initial_query

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield VerticalScroll(id="chat-log")
        yield Input(placeholder="What do you want to learn from YouTube?", id="input-bar")

    def on_mount(self) -> None:
        self.title = "ytsearch"
        input_bar = self.query_one("#input-bar", Input)
        if self._initial_query:
            input_bar.value = self._initial_query
        input_bar.focus()

    @on(Input.Submitted, "#input-bar")
    def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return

        input_bar = self.query_one("#input-bar", Input)
        input_bar.value = ""
        input_bar.disabled = True

        chat_log = self.query_one("#chat-log", VerticalScroll)
        chat_log.mount(Static(f"[bold cyan]You:[/bold cyan] {escape(query)}", classes="user-query"))
        self._scroll_to_bottom()

        self._run_search(query)

    @work(thread=True)
    def _run_search(self, query: str) -> None:
        log = SessionLog(query=query, model=self._model)

        def status_callback(message: str, kind: str = "info") -> None:
            self.call_from_thread(self._append_status, message, kind)

        try:
            answer = run_agent_loop(query, self._client, self._model, status_callback, session_log=log)
            log_path = log.save()
            self.call_from_thread(self._show_answer, answer, log_path)
        except Exception as exc:
            log_path = log.save()
            self.call_from_thread(self._append_status, f"Error: {exc}", "error")
            self.call_from_thread(self._append_status, f"Session logged → {log_path}", "info")
            self.call_from_thread(self._reenable_input)

    def _append_status(self, message: str, kind: str = "info") -> None:
        markup = _STATUS_MARKUP.get(kind, "dim")
        chat_log = self.query_one("#chat-log", VerticalScroll)
        chat_log.mount(Static(f"[{markup}]{escape(message)}[/{markup}]", classes="status-message"))
        self._scroll_to_bottom()

    def _show_answer(self, answer: str, log_path) -> None:
        chat_log = self.query_one("#chat-log", VerticalScroll)
        chat_log.mount(Markdown(answer, classes="answer"))
        chat_log.mount(Static(f"[dim]Session logged → {log_path}[/dim]", classes="status-message"))
        self._reenable_input()
        self._scroll_to_bottom()

    def _reenable_input(self) -> None:
        input_bar = self.query_one("#input-bar", Input)
        input_bar.disabled = False
        input_bar.focus()

    def _scroll_to_bottom(self) -> None:
        self.query_one("#chat-log", VerticalScroll).scroll_end(animate=False)


def main() -> None:
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("What do you want to learn from YouTube? ").strip()

    model = os.environ["YTQA_MODEL"]
    client = anthropic.Anthropic()

    app = YtsearchApp(client=client, model=model, initial_query=query)
    app.run()


if __name__ == "__main__":
    main()
