# ytqa — YouTube Transcript Q&A

Ask Claude questions about any YouTube video using its transcript.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- An Anthropic API key

## Setup

```bash
uv venv && uv add youtube-transcript-api anthropic python-dotenv textual
```

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_api_key_here
YTQA_MODEL=claude-sonnet-4-6
```

## Usage

```bash
uv run python ytqa.py https://www.youtube.com/watch?v=VIDEO_ID
```

Or with a short URL:

```bash
uv run python ytqa.py https://youtu.be/VIDEO_ID
```

You'll be dropped into an interactive TUI. Press `Ctrl+Q` to exit.

## Notes

- The video must have captions/subtitles available.
- If a video has multiple transcript tracks, you'll be prompted to choose one before the TUI launches.
- Responses are in the same language as the transcript.
- Transcripts are cached in `transcripts/<video_id>.txt` — re-running the same URL skips the fetch.
- Running cost is shown in the header subtitle, updated after each API call.
