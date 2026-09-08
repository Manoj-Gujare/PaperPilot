"""Logging setup.

Libraries should never configure the root logger on import, so configuration is
an explicit call made once by an entry point (the Streamlit app, the evaluation
CLI). Every module obtains its logger through :func:`get_logger`.
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_configured = False


def configure_logging(level: str = "INFO", *, force: bool = False) -> None:
    """Attach a stream handler to the ``paperpilot`` logger exactly once.

    Args:
        level: Minimum level to emit, e.g. ``"DEBUG"``.
        force: Reconfigure even if logging was already set up. Useful in tests.
    """
    global _configured
    if _configured and not force:
        return

    logger = logging.getLogger("paperpilot")
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    # Handled by our own handler; do not duplicate through the root logger.
    logger.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child of the ``paperpilot`` logger."""
    suffix = name.removeprefix("paperpilot.")
    return logging.getLogger(f"paperpilot.{suffix}" if suffix else "paperpilot")
