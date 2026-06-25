from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import yt_dlp
from youtube_transcript_api import CouldNotRetrieveTranscript, IpBlocked, RequestBlocked
from dotenv import load_dotenv
import anthropic
from textual.app import App, ComposeResult
from textual.widgets import Header, Input, Markdown, Static
from textual.containers import VerticalScroll
from textual import on, work
from textual.markup import escape

from ytlib import (
    fetch_model_pricing,
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
FOLLOWUP_MAX_TOKENS = 4096
SEARCH_MAX_RESULTS = 10
MAX_POOL_SIZE = 20          # max unique videos to fetch transcripts for
MAX_CONCURRENT_FETCHES = 5  # parallel transcript fetches
DEFAULT_WINDOW_SIZE = 3

FOLLOWUP_SYSTEM_PROMPT = """\
You are a YouTube research assistant continuing a conversation.

The transcript excerpts below are from videos already fetched during this session. \
Use them to answer follow-up questions directly when possible. If the existing excerpts \
do not contain enough information, use the youtube_search tool to find additional videos \
— their transcripts will be fetched automatically and added to the context above.

Cite specific claims with the source video and nearest timestamp: ([Title](URL) ~M:SS)

--- Transcript excerpts ---

{context}
"""

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


@dataclass
class SearchSession:
    """Accumulated state from one research session, enabling follow-up conversation."""
    answer: str
    segments_by_id: dict  # video_id -> list of segment dicts
    all_video_results: dict  # video_id -> VideoResult
    transcript_context: str  # formatted excerpts used for synthesis
    transcript_keywords: list  # keywords planned in the initial search phase


@dataclass
class _CostTracker:
    input_cost_per_token: float
    output_cost_per_token: float
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @property
    def total_cost(self) -> float:
        return (
            self.total_input_tokens * self.input_cost_per_token
            + self.total_output_tokens * self.output_cost_per_token
        )

    def add(self, usage) -> None:
        self.total_input_tokens += getattr(usage, "input_tokens", 0)
        self.total_output_tokens += getattr(usage, "output_tokens", 0)


def _create_with_retry(client, status_callback=None, cost_tracker: _CostTracker | None = None, **kwargs):
    for attempt in range(_MAX_API_RETRIES):
        try:
            response = client.messages.create(**kwargs)
            if cost_tracker is not None and hasattr(response, "usage"):
                cost_tracker.add(response.usage)
                if status_callback:
                    in_tok = response.usage.input_tokens
                    out_tok = response.usage.output_tokens
                    status_callback(
                        f"[dim]↳ {in_tok:,} in + {out_tok:,} out — ${cost_tracker.total_cost:.4f} total[/dim]",
                        "markup",
                    )
            return response
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


def build_youtube_search_tool_definition() -> dict:
    return {
        "name": "youtube_search",
        "description": (
            "Search YouTube for videos relevant to the user's question. "
            "Use this when the existing transcript excerpts don't cover the topic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The YouTube search query.",
                },
            },
            "required": ["query"],
        },
    }


def fetch_transcripts_for_videos(video_ids: list[str]) -> dict[str, str]:
    if not video_ids:
        return {}

    transcripts: dict[str, str] = {}
    for video_id in video_ids:
        try:
            text, _, _source = load_transcript(video_id)
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


def _description_to_segments(description: str, chunk_words: int = 50) -> list[dict]:
    """Split a video description into fixed-word chunks for keyword matching."""
    words = description.split()
    return [
        {"start": None, "text": " ".join(words[i : i + chunk_words])}
        for i in range(0, len(words), chunk_words)
        if words[i : i + chunk_words]
    ]


def _format_excerpt_window(window: dict) -> str:
    if window["start"] is None:
        return f"[description] {window['text']}"
    total = int(window["start"])
    ts = f"{total // 60}:{total % 60:02d}"
    return f"[{ts}] {window['text']}"


def _keywords_matched_in_segments(segments: list[dict], keywords: list[str]) -> list[str]:
    """Return the subset of keywords found in at least one segment (case-insensitive)."""
    kw_pairs = [(kw, kw.lower()) for kw in keywords]
    return [kw for kw, kw_l in kw_pairs if any(kw_l in seg["text"].lower() for seg in segments)]


def _no_transcript_reason(exc: CouldNotRetrieveTranscript) -> str:
    """Extract a short human-readable reason from a CouldNotRetrieveTranscript exception."""
    msg = str(exc)
    marker = "This is most likely caused by:"
    if marker in msg:
        after = msg.split(marker, 1)[1].strip()
        reason = after.split("\n")[0].strip()
        if reason:
            return reason
    return "no captions available"


def run_agent_loop(
    user_query: str,
    client,
    model: str,
    status_callback=None,
    session_log: SessionLog | None = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> str:
    pricing = fetch_model_pricing(model)
    cost_tracker = _CostTracker(*pricing) if pricing else None

    # --- Phase 1: Planning ---
    if status_callback:
        status_callback("Planning", "phase")

    plan_response = _create_with_retry(
        client,
        status_callback,
        cost_tracker,
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
    _ip_blocked = threading.Event()

    def _fetch_one(video_id: str, title: str) -> None:
        if status_callback:
            status_callback(f"Fetching: {title}…", "info", widget_key=video_id)
        try:
            text, _, source = load_transcript(video_id, auto=True)
            segs = load_transcript_segments(video_id)
            if segs:
                with _segments_lock:
                    segments_by_id[video_id] = segs
            if status_callback:
                suffix = f" [dim]({source})[/dim]" if source else ""
                status_callback(f"[green]Loaded: {escape(title)}{suffix}[/green]", "markup", widget_key=video_id)
            if session_log:
                session_log.record(
                    "transcript_fetch",
                    video_id=video_id,
                    title=title,
                    status="success",
                )
        except CouldNotRetrieveTranscript as exc:
            exc_msg = str(exc).lower()
            is_ip_issue = isinstance(exc, (IpBlocked, RequestBlocked)) or (
                "ip" in exc_msg and "block" in exc_msg
            )
            if is_ip_issue:
                _ip_blocked.set()
                if status_callback:
                    status_callback(f"[red]IP blocked: {escape(title)}[/red]", "markup", widget_key=video_id)
                if session_log:
                    session_log.record(
                        "transcript_fetch",
                        video_id=video_id,
                        title=title,
                        status="ip_blocked",
                    )
            else:
                reason = _no_transcript_reason(exc)
                if status_callback:
                    status_callback(
                        f"[yellow]No captions:[/yellow] {escape(title)} [dim]— {escape(reason)}[/dim]",
                        "markup",
                        widget_key=video_id,
                    )
                if session_log:
                    session_log.record(
                        "transcript_fetch",
                        video_id=video_id,
                        title=title,
                        status="no_captions",
                        reason=reason,
                    )
        except Exception as exc:
            if status_callback:
                status_callback(f"Failed: {title}", "error", widget_key=video_id)
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

    if _ip_blocked.is_set() and status_callback:
        status_callback(
            "[red]YouTube blocked transcript requests from your IP.[/red] "
            "[dim]Try disabling your VPN or running from a different network.[/dim]",
            "markup",
        )

    # Append description segments so keyword matching also covers video descriptions.
    for video_id, video in all_video_results.items():
        if video.description:
            desc_segs = _description_to_segments(video.description)
            if desc_segs:
                segments_by_id[video_id] = segments_by_id.get(video_id, []) + desc_segs

    # --- Phase 4: Keyword matching ---
    if status_callback:
        status_callback("Matching keywords in transcripts", "phase")

    video_excerpts = build_video_excerpts(
        all_video_results, segments_by_id, transcript_keywords, window_size
    )

    if status_callback:
        all_matched_keywords: set[str] = set()
        for video_id, video in all_video_results.items():
            segs = segments_by_id.get(video_id, [])
            matched_kws = set(_keywords_matched_in_segments(segs, transcript_keywords)) if segs else set()
            all_matched_keywords.update(matched_kws)
            if video_id in video_excerpts:
                n = len(video_excerpts[video_id]["excerpts"])
                excerpt_label = f"excerpt{'s' if n != 1 else ''}"
                status_callback(
                    f"[green]Matched: {escape(video.title)} ({n} {excerpt_label})[/green]",
                    "markup",
                )
            elif segs:
                status_callback(f"No matches: {video.title}", "skip")
            else:
                continue
            if transcript_keywords and video_id in video_excerpts:
                parts = []
                for kw in transcript_keywords:
                    if kw in matched_kws:
                        parts.append(f"[cyan bold]✓ {escape(kw)}[/cyan bold]")
                    else:
                        parts.append(f"[dim]✗ {escape(kw)}[/dim]")
                status_callback(f"  {'  '.join(parts)}", "markup")
        unmatched_kws = [kw for kw in transcript_keywords if kw not in all_matched_keywords]
        if all_matched_keywords:
            kws = escape(", ".join(sorted(all_matched_keywords)))
            status_callback(f"[green]Keywords found:[/green] [cyan bold]{kws}[/cyan bold]", "markup")
        if unmatched_kws:
            kws = escape(", ".join(unmatched_kws))
            status_callback(f"[yellow]Not found in any video:[/yellow] [red]{kws}[/red]", "markup")

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
        cost_tracker,
        model=model,
        max_tokens=SYNTHESIS_MAX_TOKENS,
        system=SYNTHESIS_SYSTEM_PROMPT,
        messages=synthesis_messages,
    )

    answer = synthesis_response.content[0].text

    if session_log:
        session_log.record("synthesis_output", answer=answer)

    return SearchSession(
        answer=answer,
        segments_by_id=segments_by_id,
        all_video_results=all_video_results,
        transcript_context=transcript_context,
        transcript_keywords=transcript_keywords,
    )


def run_followup(
    question: str,
    session: SearchSession,
    conversation_history: list[dict],
    client,
    model: str,
    status_callback=None,
    session_log: SessionLog | None = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> tuple[str, SearchSession]:
    """Answer a follow-up question using accumulated context, searching YouTube if needed."""
    pricing = fetch_model_pricing(model)
    cost_tracker = _CostTracker(*pricing) if pricing else None

    # Working copies — do not mutate the caller's session.
    segments_by_id = dict(session.segments_by_id)
    all_video_results = dict(session.all_video_results)
    transcript_context = session.transcript_context

    messages = list(conversation_history) + [{"role": "user", "content": question}]

    if status_callback:
        status_callback("Answering follow-up", "phase")

    while True:
        system = FOLLOWUP_SYSTEM_PROMPT.format(context=transcript_context)
        response = _create_with_retry(
            client,
            status_callback,
            cost_tracker,
            model=model,
            max_tokens=FOLLOWUP_MAX_TOKENS,
            system=system,
            tools=[build_youtube_search_tool_definition()],
            messages=messages,
        )

        # Find tool_use block, if any.
        tool_block = next(
            (b for b in response.content if getattr(b, "type", None) == "tool_use"),
            None,
        )

        if tool_block is None:
            # Claude answered directly — done.
            answer = next(
                (b.text for b in response.content if getattr(b, "type", None) == "text"),
                "",
            )
            break

        # Claude wants to search YouTube.
        query = tool_block.input.get("query", "")
        if status_callback:
            status_callback(f'Searching YouTube: "{query}"', "info")

        try:
            results = search_youtube(query, SEARCH_MAX_RESULTS)
        except SearchError:
            results = []

        # Append assistant tool_use message.
        assistant_content = [
            {"type": "tool_use", "id": tool_block.id, "name": tool_block.name, "input": tool_block.input}
        ]
        messages.append({"role": "assistant", "content": assistant_content})

        # Fetch transcripts for newly discovered videos and extend context.
        new_sections: list[str] = []
        for r in results:
            if r.video_id in all_video_results:
                continue
            all_video_results[r.video_id] = r
            if status_callback:
                status_callback(f"Fetching: {r.title}...", "info", widget_key=f"fu_{r.video_id}")
            try:
                _text, _, source = load_transcript(r.video_id, auto=True)
                segs = load_transcript_segments(r.video_id)
                if segs:
                    segments_by_id[r.video_id] = segs
                    excerpts_data = build_video_excerpts(
                        {r.video_id: r},
                        {r.video_id: segs},
                        keywords=session.transcript_keywords,
                        window_size=window_size,
                    )
                    if r.video_id in excerpts_data:
                        excerpts_text = "\n".join(
                            _format_excerpt_window(w)
                            for w in excerpts_data[r.video_id]["excerpts"]
                        )
                        new_sections.append(
                            f"## {r.title}\nURL: {r.url}\n\nRelevant excerpts:\n{excerpts_text}"
                        )
                suffix = f" ({source})" if source else ""
                if status_callback:
                    status_callback(
                        f"[green]Loaded: {escape(r.title)}{suffix}[/green]", "markup",
                        widget_key=f"fu_{r.video_id}",
                    )
            except CouldNotRetrieveTranscript:
                if status_callback:
                    status_callback(f"No captions: {r.title}", "skip", widget_key=f"fu_{r.video_id}")
            except Exception:
                if status_callback:
                    status_callback(f"Failed: {r.title}", "error", widget_key=f"fu_{r.video_id}")

        if new_sections:
            transcript_context = transcript_context + "\n\n---\n\n" + "\n\n---\n\n".join(new_sections)

        found_titles = ", ".join(r.title for r in results[:3]) or "none"
        tool_result = f"Found {len(results)} video(s): {found_titles}. Relevant excerpts added to context."
        messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_block.id, "content": tool_result}],
        })

    updated_session = SearchSession(
        answer=answer,
        segments_by_id=segments_by_id,
        all_video_results=all_video_results,
        transcript_context=transcript_context,
        transcript_keywords=session.transcript_keywords,
    )

    if session_log:
        session_log.record("followup_output", question=question, answer=answer)

    return answer, updated_session


class YtsearchApp(App):
    CSS = CSS
    BINDINGS = [("ctrl+q", "quit", "Quit"), ("ctrl+r", "new_search", "New search")]

    def __init__(self, client, model: str, initial_query: str | None = None) -> None:
        super().__init__()
        self._client = client
        self._model = model
        self._initial_query = initial_query
        self._keyed_widgets: dict[str, Static] = {}
        self._session: SearchSession | None = None
        self._conversation: list[dict] = []

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

        if self._session is not None:
            self._run_followup(query)
        else:
            self._run_search(query)

    def action_new_search(self) -> None:
        self._session = None
        self._conversation = []
        input_bar = self.query_one("#input-bar", Input)
        input_bar.placeholder = "What do you want to learn from YouTube?"

    @work(thread=True)
    def _run_search(self, query: str) -> None:
        log = SessionLog(query=query, model=self._model)

        self._keyed_widgets.clear()

        def status_callback(message: str, kind: str = "info", widget_key: str | None = None, **_kw) -> None:
            self.call_from_thread(self._append_status, message, kind, widget_key)

        try:
            session = run_agent_loop(query, self._client, self._model, status_callback, session_log=log)
            log_path = log.save()
            self.call_from_thread(self._on_search_complete, query, session, log_path)
        except Exception as exc:
            log_path = log.save()
            self.call_from_thread(self._append_status, f"Error: {exc}", "error")
            self.call_from_thread(self._append_status, f"Session logged -> {log_path}", "info")
            self.call_from_thread(self._reenable_input)

    def _on_search_complete(self, query: str, session: SearchSession, log_path) -> None:
        self._session = session
        self._conversation = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": session.answer},
        ]
        self._show_answer(session.answer, log_path)
        input_bar = self.query_one("#input-bar", Input)
        input_bar.placeholder = "Ask a follow-up... (Ctrl+R for new search)"

    @work(thread=True)
    def _run_followup(self, question: str) -> None:
        log = SessionLog(query=question, model=self._model)

        self._keyed_widgets.clear()

        def status_callback(message: str, kind: str = "info", widget_key: str | None = None, **_kw) -> None:
            self.call_from_thread(self._append_status, message, kind, widget_key)

        try:
            answer, updated_session = run_followup(
                question, self._session, self._conversation,
                self._client, self._model, status_callback, session_log=log,
            )
            log_path = log.save()
            self.call_from_thread(self._on_followup_complete, question, answer, updated_session, log_path)
        except Exception as exc:
            log_path = log.save()
            self.call_from_thread(self._append_status, f"Error: {exc}", "error")
            self.call_from_thread(self._reenable_input)

    def _on_followup_complete(self, question: str, answer: str, session: SearchSession, log_path) -> None:
        self._session = session
        self._conversation.append({"role": "user", "content": question})
        self._conversation.append({"role": "assistant", "content": answer})
        self._show_answer(answer, log_path)

    def _append_status(self, message: str, kind: str = "info", widget_key: str | None = None) -> None:
        if kind == "markup":
            content = message
        else:
            tag = _STATUS_MARKUP.get(kind, "dim")
            content = f"[{tag}]{escape(message)}[/{tag}]"
        if widget_key and widget_key in self._keyed_widgets:
            self._keyed_widgets[widget_key].update(content)
        else:
            widget = Static(content, classes="status-message")
            chat_log = self.query_one("#chat-log", VerticalScroll)
            chat_log.mount(widget)
            if widget_key:
                self._keyed_widgets[widget_key] = widget
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
