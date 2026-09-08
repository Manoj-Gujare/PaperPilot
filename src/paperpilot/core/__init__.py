"""Cross-cutting building blocks: errors, logging and shared schemas."""

from paperpilot.core.exceptions import (
    ConfigurationError,
    DocumentLoadError,
    PaperPilotError,
    UnsupportedSourceError,
    VectorStoreError,
    WebSearchError,
)
from paperpilot.core.logging import configure_logging, get_logger

__all__ = [
    "ConfigurationError",
    "DocumentLoadError",
    "PaperPilotError",
    "UnsupportedSourceError",
    "VectorStoreError",
    "WebSearchError",
    "configure_logging",
    "get_logger",
]
