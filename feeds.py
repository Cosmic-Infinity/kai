"""Simple file-backed stubs for MQTT feeds used across the system.

Each feed is represented by a text file inside the `feeds` folder. Messages are
appended one per line and consumed atomically when read. This allows the rest of
the application to follow the MQTT-style contract while keeping the runtime
self-contained for local testing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

FEED_ROOT = Path("feeds")
FEED_ROOT.mkdir(exist_ok=True)


def _feed_path(feed_name: str) -> Path:
    """Return the on-disk path backing a feed name."""
    sanitized = feed_name.strip()
    if not sanitized:
        raise ValueError("feed_name must be non-empty")
    return FEED_ROOT / f"{sanitized}.feed"


def append_message(feed_name: str, message: str) -> None:
    """Append a single message to the specified feed."""
    path = _feed_path(feed_name)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip("\n") + "\n")


def append_messages(feed_name: str, messages: Iterable[str]) -> None:
    """Append multiple messages to the specified feed."""
    path = _feed_path(feed_name)
    with path.open("a", encoding="utf-8") as handle:
        for message in messages:
            handle.write(str(message).rstrip("\n") + "\n")


def consume_messages(feed_name: str) -> List[str]:
    """Return all pending messages for the feed and clear it."""
    path = _feed_path(feed_name)
    if not path.exists():
        return []
    with path.open("r+", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.readlines() if line.strip()]
        handle.seek(0)
        handle.truncate()
    return lines


def peek_messages(feed_name: str) -> List[str]:
    """Return all pending messages for the feed without clearing it."""
    path = _feed_path(feed_name)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle.readlines() if line.strip()]


def clear_feed(feed_name: str) -> None:
    """Remove all pending messages from the feed."""
    path = _feed_path(feed_name)
    if path.exists():
        path.write_text("", encoding="utf-8")
