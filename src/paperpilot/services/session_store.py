"""Persistence of session metadata.

The checkpointer already holds each conversation. What it does not hold is the
list of conversations — their ids, display names and creation times — so that
lives in a small JSON file next to it. Keeping the two separate means the
sidebar can be rendered without loading a single message.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from paperpilot.config import get_settings
from paperpilot.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SESSION_NAME = "New Session"


@dataclass(slots=True)
class SessionMeta:
    """Everything the sidebar needs to know about a conversation."""

    id: str
    name: str = DEFAULT_SESSION_NAME
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    is_named: bool = False

    @classmethod
    def new(cls) -> SessionMeta:
        """Create metadata for a fresh session."""
        return cls(id=str(uuid.uuid4()))


class SessionStore:
    """Reads and writes the session index."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path if path is not None else get_settings().sessions_file

    def load(self) -> dict[str, SessionMeta]:
        """Return every known session, keyed by id.

        A missing or unreadable index is treated as an empty one: losing the
        list of past conversations must not stop someone starting a new one.
        """
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, OSError):
            logger.warning("Session index at %s is unreadable; starting empty", self._path)
            return {}

        sessions: dict[str, SessionMeta] = {}
        for session_id, raw in payload.items():
            sessions[session_id] = SessionMeta(
                id=raw.get("id", session_id),
                name=raw.get("name", DEFAULT_SESSION_NAME),
                created_at=raw.get("created_at", datetime.now().isoformat()),
                is_named=raw.get("is_named", False),
            )
        return sessions

    def save(self, sessions: dict[str, SessionMeta]) -> None:
        """Write the session index, creating its directory if needed."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {session_id: asdict(meta) for session_id, meta in sessions.items()}
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def most_recent(sessions: dict[str, SessionMeta]) -> SessionMeta | None:
        """Return the newest session, or None when there are none."""
        if not sessions:
            return None
        return max(sessions.values(), key=lambda meta: meta.created_at)

    @staticmethod
    def sorted_by_recency(sessions: dict[str, SessionMeta]) -> list[SessionMeta]:
        """Return sessions newest first, the order the sidebar lists them in."""
        return sorted(sessions.values(), key=lambda meta: meta.created_at, reverse=True)
