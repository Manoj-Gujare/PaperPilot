"""Exception hierarchy for PaperPilot.

A single root exception lets callers — most importantly the Streamlit layer —
distinguish failures this application understands and can explain to a user
from genuine programming errors, which should keep propagating.
"""

from __future__ import annotations


class PaperPilotError(Exception):
    """Base class for every error raised deliberately by PaperPilot."""


class ConfigurationError(PaperPilotError):
    """Raised when required configuration is missing or invalid."""


class DocumentLoadError(PaperPilotError):
    """Raised when a paper cannot be fetched, parsed or chunked."""


class UnsupportedSourceError(DocumentLoadError):
    """Raised when a source has no loader — an unknown file extension, say."""


class VectorStoreError(PaperPilotError):
    """Raised when the vector store cannot be reached or updated."""


class WebSearchError(PaperPilotError):
    """Raised when the web search provider fails or returns nothing usable."""
