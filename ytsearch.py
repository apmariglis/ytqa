from __future__ import annotations

import json
import os
import re
import sys
import time
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
    format_transcript_with_timestamps,
    MAX_TOKENS,
)
from ytlog import SessionLog, serialize_content

load_dotenv()

SEARCH_SYSTEM_PROMPT = """\
You are a YouTube research assistant. Use the youtube_search tool to find videos relevant \
to the user's query. You may search multiple times with different or refined queries to \
find the best results.

When you have identified the 1–3 most promising videos, finish your response with this \
exact block (no extra text after it):
<selected_videos>["video_id_1", "video_id_2"]</selected_videos>\
"""

SYNTHESIS_SYSTEM_PROMPT = """\
You are a YouTube research assistant. Using the transcript excerpts provided, write a \
comprehensive answer to the user's query in Markdown.

Transcripts include [M:SS] timestamps. Cite specific claims with the source video and the \
nearest timestamp using this format: ([Video Title](URL) ~M:SS)

For example: "The GIL prevents true parallelism ([Understanding the GIL](https://...) ~2:15)"

If a transcript has no timestamps, cite with just the title and URL.

Base your entire answer on the provided transcripts. Do not suggest the user seek \
"additional sources" or "further research" — give the most complete answer the available \
transcripts allow, and state any gaps plainly within the answer itself.\
"""

SYNTHESIS_MAX_TOKENS = 4096

# Maps status kind → Rich markup tag used in the TUI.
_STATUS_MARKUP: dict[str, str] = {
    "info": "dim",
    "found": "dim",
    "success": "green",
    "skip": "yellow",
    "error": "red",
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


def search_youtube(query: str, max_results: int = 5) -> list[VideoResult]:
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


def build_search_tool_definition() -> dict:
    return {
        "name": "youtube_search",
        "description": (
            "Search YouTube for videos matching a query. "
            "Returns video IDs, titles, URLs, and descriptions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant YouTube videos.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5).",
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
            text, _ = load_transcript(video_id)
            transcripts[video_id] = text
        except Exception:
            pass

    return transcripts


def _parse_selected_video_ids(text: str) -> list[str]:
    match = re.search(r"<selected_videos>(\[.*?\])</selected_videos>", text, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return []


def run_agent_loop(
    user_query: str,
    client,
    model: str,
    status_callback=None,
    session_log: SessionLog | None = None,
) -> str:
    messages = [{"role": "user", "content": user_query}]
    tools = [build_search_tool_definition()]
    all_video_results: dict[str, VideoResult] = {}

    if status_callback:
        status_callback(f'Searching YouTube for "{user_query}"…', "info")

    # --- Phase 1: search loop ---
    search_end_text = ""
    while True:
        response = _create_with_retry(
            client,
            status_callback,
            model=model,
            max_tokens=MAX_TOKENS,
            system=SEARCH_SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if session_log:
            session_log.record(
                "claude_search_response",
                stop_reason=response.stop_reason,
                content=serialize_content(response.content),
            )

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    search_end_text = block.text
            break

        # Process tool_use blocks and build tool_result reply
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            query = block.input.get("query", "")
            max_results = block.input.get("max_results", 5)

            if status_callback:
                status_callback(f'Searching YouTube for "{query}"…', "info")

            try:
                results = search_youtube(query, max_results)
                for r in results:
                    all_video_results[r.video_id] = r

                if status_callback:
                    for r in results:
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

                results_data = [
                    {
                        "video_id": r.video_id,
                        "title": r.title,
                        "url": r.url,
                        "description": r.description,
                        "view_count": r.view_count,
                    }
                    for r in results
                ]
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(results_data),
                    }
                )
            except SearchError as exc:
                if session_log:
                    session_log.record("youtube_search_error", query=query, error=str(exc))
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Search failed: {exc}",
                        "is_error": True,
                    }
                )

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    # --- Phase 2: fetch transcripts and synthesize ---
    selected_ids = _parse_selected_video_ids(search_end_text)

    # Fetch each transcript individually so we can report progress per video.
    transcripts: dict[str, str] = {}

    def _fetch_one(video_id: str, title: str, is_fallback: bool = False) -> None:
        if status_callback:
            label = f"Trying alternative: {title}…" if is_fallback else f"Fetching transcript: {title}…"
            status_callback(label, "info")
        try:
            text, _ = load_transcript(video_id)
            transcripts[video_id] = text
            if status_callback:
                status_callback(f"Loaded: {title}", "success")
            if session_log:
                session_log.record(
                    "transcript_fetch",
                    video_id=video_id,
                    title=title,
                    status="fallback" if is_fallback else "success",
                )
        except CouldNotRetrieveTranscript:
            if status_callback and not is_fallback:
                status_callback(f"No captions: {title}", "skip")
            if session_log:
                session_log.record(
                    "transcript_fetch",
                    video_id=video_id,
                    title=title,
                    status="no_captions",
                )
        except Exception as exc:
            if status_callback and not is_fallback:
                status_callback(f"Failed: {title}", "error")
            if session_log:
                session_log.record(
                    "transcript_fetch",
                    video_id=video_id,
                    title=title,
                    status="failed",
                    error=str(exc),
                )

    for video_id in selected_ids:
        video = all_video_results.get(video_id)
        _fetch_one(video_id, video.title if video else video_id)

    # If fewer than 3 transcripts loaded, try other discovered videos as fallbacks.
    TARGET_TRANSCRIPT_COUNT = 3
    if len(transcripts) < TARGET_TRANSCRIPT_COUNT:
        fallback_ids = [
            vid_id for vid_id in all_video_results
            if vid_id not in selected_ids and vid_id not in transcripts
        ]
        for video_id in fallback_ids:
            if len(transcripts) >= TARGET_TRANSCRIPT_COUNT:
                break
            video = all_video_results[video_id]
            _fetch_one(video_id, video.title, is_fallback=True)

    transcript_sections = []
    for video_id, text in transcripts.items():
        video = all_video_results.get(video_id)

        # Use timestamped segments when available so Claude can cite timestamps.
        segments = load_transcript_segments(video_id)
        if segments:
            transcript_content = format_transcript_with_timestamps(segments)[:12000]
        else:
            transcript_content = text[:8000]

        if video:
            transcript_sections.append(
                f"## {video.title}\nURL: {video.url}\n\nTranscript:\n{transcript_content}"
            )
        else:
            transcript_sections.append(
                f"## Video {video_id}\n\nTranscript:\n{transcript_content}"
            )

    transcript_context = "\n\n---\n\n".join(transcript_sections)

    if session_log:
        session_log.record(
            "synthesis_input",
            selected_ids=selected_ids,
            transcripts_loaded=[
                {"video_id": vid, "title": all_video_results[vid].title if vid in all_video_results else vid}
                for vid in transcripts
            ],
            transcript_context=transcript_context,
        )

    synthesis_messages = [
        {
            "role": "user",
            "content": (
                f"User query: {user_query}\n\n"
                f"Here are the relevant video transcripts:\n\n{transcript_context}\n\n"
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
            self.call_from_thread(self._append_status, f"Error: {exc}", "error")
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
