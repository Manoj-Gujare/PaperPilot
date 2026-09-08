"""Construction of the chat and embedding models.

Model clients are built here and nowhere else. Centralising them means a model
name or provider change is a one-line edit to configuration, and it lets tests
patch a single seam instead of every call site. Instances are cached because
constructing a client opens connection pools that are safe to share.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

from paperpilot.config import get_settings
from paperpilot.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    """Return the shared chat model configured for this deployment."""
    settings = get_settings()
    logger.debug("Creating chat model %s", settings.chat_model)
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key.get_secret_value(),
    )


def get_structured_model[SchemaT: BaseModel](schema: type[SchemaT]) -> Runnable:
    """Return the chat model constrained to emit ``schema``."""
    return get_chat_model().with_structured_output(schema)


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return embeddings backed by an on-disk cache.

    Identical text is embedded once and reused across sessions and restarts,
    which removes the dominant cost of re-ingesting a paper someone has already
    uploaded. The cache is keyed by model name, so switching models cannot
    return stale vectors of the wrong dimensionality.
    """
    settings = get_settings()
    settings.ensure_directories()
    base = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key.get_secret_value(),
    )
    return CacheBackedEmbeddings.from_bytes_store(
        base,
        LocalFileStore(str(settings.embedding_cache_dir)),
        namespace=settings.embedding_model,
        query_embedding_cache=True,
        key_encoder="blake2b",
    )


def reset_provider_cache() -> None:
    """Drop cached clients. Used by tests and after a configuration change."""
    get_chat_model.cache_clear()
    get_embeddings.cache_clear()
