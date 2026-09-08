"""Automatic session titles.

A sidebar of entries all reading "New Session" is unusable, so the first
message is condensed into a short title. Naming is best-effort: a failure
leaves the default name in place rather than blocking the answer the user is
actually waiting for.
"""

from __future__ import annotations

from paperpilot.core.logging import get_logger
from paperpilot.llm import get_chat_model

logger = get_logger(__name__)

MAX_MESSAGE_CHARACTERS = 500
NAMING_SYSTEM = (
    "Generate a concise 3-5 word title for a research chat session based on the "
    "user's first message. Return only the title, with no trailing punctuation "
    "and no quotes."
)
FALLBACK_NAME = "New Session"


def generate_session_name(first_message: str) -> str:
    """Return a short title for a conversation, or a safe default."""
    try:
        response = get_chat_model().invoke(
            [
                {"role": "system", "content": NAMING_SYSTEM},
                {"role": "user", "content": first_message[:MAX_MESSAGE_CHARACTERS]},
            ]
        )
        return response.content.strip() or FALLBACK_NAME
    except Exception:
        logger.exception("Session naming failed; keeping the default name")
        return FALLBACK_NAME
