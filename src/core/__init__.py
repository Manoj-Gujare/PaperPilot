"""Cross-cutting building blocks: errors, logging and shared schemas."""

from src.core.exceptions import (
    ConfigurationError,
    DocumentLoadError,
    PaperPilotError,
    UnsupportedSourceError,
    VectorStoreError,
    WebSearchError,
)
from src.core.logging import configure_logging, get_logger

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
