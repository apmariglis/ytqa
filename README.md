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

The agent runs in three phases:

1. **Planning** — Claude receives your query and returns a set of YouTube search queries and transcript keywords via a structured tool call.
2. **Search + fetch** — Each search query runs against YouTube (up to 10 results each). All discovered transcripts are fetched and cached locally.
3. **Keyword matching + synthesis** — Transcript segments matching any keyword are extracted with surrounding context (`[M:SS]` timestamped excerpts). Videos with no matching segments are skipped. Claude synthesises the excerpts into a cited Markdown answer.

Progress is shown in the TUI as it works — found videos, loaded transcripts (green), skipped videos with no captions (yellow). Answers include `[M:SS]` timestamp citations so you can jump to the relevant moment in each video. Press `Ctrl+Q` to exit.

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
| `transcript_fetch` | Per-video outcome: `success`, `no_captions`, or `failed` |
| `synthesis_input` | Which videos had keyword matches and the full excerpt context given to Claude |
| `synthesis_output` | The final answer verbatim |

Every event has an ISO timestamp. Log files are self-contained for offline agent evaluation. The TUI shows the log path after each completed query.

## Notes

- Videos must have captions/subtitles available.
- If a video has multiple transcript tracks, you'll be prompted to choose one before the TUI launches (ytqa only).
- `ytqa` responses are in the same language as the transcript.
- Transcripts are cached in `transcripts/<video_id>.txt` and `transcripts/<video_id>.json` (with timestamps) — subsequent runs on the same or similar topics reuse cached files, making keyword matching fast.
- `ytqa` shows running API cost in the header, updated after each call.
- `logs/` and `transcripts/` are gitignored.

## Running tests

```bash
uv run pytest tests/ -v
```
