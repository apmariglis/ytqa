"""
Session logging for ytsearch agent runs.

Each query produces one JSON log file capturing the full sequence of Claude
API calls, YouTube searches, transcript fetches, and the final synthesis —
enough detail to evaluate agent quality offline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path("logs")


@dataclass
class SessionLog:
    query: str
    model: str
    started_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    events: list[dict] = field(default_factory=list)

    def record(self, event_type: str, **data) -> None:
        """Append a timestamped event to the log."""
        self.events.append(
            {
                "type": event_type,
                "t": datetime.now().isoformat(timespec="seconds"),
                **data,
            }
        )

    def save(self, logs_dir: Path = LOGS_DIR) -> Path:
        """Write the log to <logs_dir>/<timestamp>_<slug>.json and return the path."""
        logs_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^\w]+", "-", self.query[:50]).strip("-").lower()
        ts = self.started_at.replace(":", "-")
        path = logs_dir / f"{ts}_{slug}.json"
        path.write_text(
            json.dumps(
                {
                    "query": self.query,
                    "model": self.model,
                    "started_at": self.started_at,
                    "events": self.events,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return path


def serialize_content(content: list) -> list[dict]:
    """Convert a list of SDK content blocks to plain dicts for JSON logging."""
    result = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            result.append({"type": "text", "text": getattr(block, "text", "")})
        elif block_type == "tool_use":
            result.append(
                {
                    "type": "tool_use",
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                    "input": getattr(block, "input", {}),
                }
            )
        elif block_type is not None:
            result.append({"type": block_type})
    return result
