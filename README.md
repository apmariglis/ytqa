# ytqa — YouTube Transcript Q&A

Ask Claude questions about any YouTube video using its transcript.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- An Anthropic API key

## Setup

```bash
uv venv && uv add youtube-transcript-api anthropic python-dotenv
```

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_api_key_here
```

## Usage

```bash
uv run python ytqa.py https://www.youtube.com/watch?v=VIDEO_ID
```

Or with a short URL:

```bash
uv run python ytqa.py https://youtu.be/VIDEO_ID
```

You'll be dropped into an interactive Q&A session. Type `q` or `quit` to exit.

## Notes

- The video must have captions/subtitles available.
- Transcripts are cached in `transcripts/<video_id>.txt` — re-running the same URL skips the fetch.
