# ytqa + ytsearch — YouTube Transcript Tools

Two Claude-powered TUI tools for learning from YouTube:

- **`ytqa`** — Q&A a single video you already know
- **`ytsearch`** — Describe what you want to learn; an agent searches YouTube, reads transcripts, and synthesises a cited answer

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- An Anthropic API key

## Setup

```bash
uv sync
```

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_api_key_here
YTQA_MODEL=claude-sonnet-4-6
```

## Usage

### ytqa — single-video Q&A

```bash
uv run python ytqa.py https://www.youtube.com/watch?v=VIDEO_ID
# or
uv run python ytqa.py https://youtu.be/VIDEO_ID
```

Launches an interactive chat TUI. A summary of the video is shown on load, then you can ask follow-up questions. Press `Ctrl+Q` to exit.

### ytsearch — agentic search

```bash
uv run python ytsearch.py "how does Python's GIL work"
```

Or run without arguments to be prompted:

```bash
uv run python ytsearch.py
```

The agent runs in five phases:

1. **Planning** — Claude receives your query and returns a set of YouTube search queries and transcript keywords via a structured tool call.
2. **Searching YouTube** — Each query runs against YouTube (up to 10 results each). Duplicate videos across queries are deduplicated.
3. **Fetching transcripts** — Each video's transcript is fetched in parallel. Manual English transcripts are preferred; auto-generated English is used as a fallback, then translated transcripts for non-English videos. The TUI shows each video updating in place: the transcript source used (e.g. `auto-generated EN`), `No captions: title — reason` when a video has no subtitles at all, or `IP blocked: title` when YouTube is rate-limiting the host. If any video is IP-blocked a summary message explains the cause.
4. **Keyword matching** — Transcript segments (and video descriptions) matching any planned keyword are extracted with surrounding context as `[M:SS]` timestamped excerpts. The TUI shows which keywords hit or missed per video and a summary of which keywords appeared in any video. Videos with no matching segments are skipped.
5. **Synthesising** — Claude synthesises the matching excerpts into a cited Markdown answer. Each citation includes the video title, URL, and nearest timestamp.

The TUI also shows the running LLM cost (input + output tokens and cumulative USD) after each API call. Answers include `[M:SS]` timestamp citations so you can jump to the relevant moment in each video. Press `Ctrl+Q` to exit.

## Session logs

Every `ytsearch` query writes a structured JSON log to `logs/`:

```
logs/2026-06-23T16-18-17_how-does-python-s-gil-work.json
```

Each log captures the full agent session in order:

| Event | What it records |
|-------|----------------|
| `claude_planning_response` | Search queries and transcript keywords Claude chose |
| `youtube_search` | Query used and every video result returned |
| `youtube_search_error` | Query and error when yt-dlp fails |
| `transcript_fetch` | Per-video outcome: `success`, `no_captions` (with `reason`), `ip_blocked`, or `failed` |
| `synthesis_input` | Which videos had keyword matches and the full excerpt context given to Claude |
| `synthesis_output` | The final answer verbatim |

Every event has an ISO timestamp. Log files are self-contained for offline agent evaluation. The TUI shows the log path after each completed query.

## Notes

- `ytsearch` works best when videos have captions. It tries manual captions first, then auto-generated, then translated — so many non-English videos are covered too. Videos with subtitles entirely disabled are skipped with the reason shown.
- If YouTube rate-limits the host (e.g. on a VPN or cloud IP), the TUI shows which videos were blocked and suggests disabling the VPN.
- If a video has multiple transcript tracks, you'll be prompted to choose one before the TUI launches (ytqa only).
- `ytqa` responses are in the same language as the transcript.
- Transcripts are cached in `transcripts/<video_id>.txt` and `transcripts/<video_id>.json` (with timestamps) — subsequent runs on the same or similar topics reuse cached files, making keyword matching fast.
- `ytsearch` shows running LLM cost after each API call. `ytqa` shows it in the header.
- `logs/` and `transcripts/` are gitignored.

## Running tests

```bash
uv run pytest tests/ -v
```
