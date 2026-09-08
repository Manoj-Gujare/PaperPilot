"""Session-scoped vector storage backed by Qdrant.

Each chat session owns its own Qdrant collection. Isolation is the point:
papers uploaded in one session must never surface as context in another, and a
per-session collection makes that a property of the storage layer rather than
something every query has to remember to filter for.

The repository is a class so tests can drive it with a fake client, and
:func:`get_repository` provides the shared instance the application uses.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from paperpilot.config import Settings, get_settings
from paperpilot.core.exceptions import VectorStoreError
from paperpilot.core.logging import get_logger
from paperpilot.llm import get_embeddings

logger = get_logger(__name__)

SCROLL_PAGE_SIZE = 100


def collection_name_for(session_id: str, prefix: str | None = None) -> str:
    """Return the Qdrant collection name for a session.

    Hyphens in the session UUID are replaced because they are awkward in the
    identifiers Qdrant surfaces in its dashboard and API paths.
    """
    resolved_prefix = prefix if prefix is not None else get_settings().collection_prefix
    return f"{resolved_prefix}_{session_id.replace('-', '_')}"


class PaperRepository:
    """Stores and retrieves paper chunks, one collection per session."""

    def __init__(self, client: QdrantClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    # ── Collection management ────────────────────────────────────────────────

    def collection_name(self, session_id: str) -> str:
        """Return the collection backing ``session_id``."""
        return collection_name_for(session_id, self._settings.collection_prefix)

    def _vector_store(self, session_id: str) -> QdrantVectorStore:
        """Return the vector store for a session, creating its collection once."""
        name = self.collection_name(session_id)
        try:
            if not self._client.collection_exists(name):
                logger.info("Creating collection %s", name)
                self._client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=self._settings.embedding_dimension,
                        distance=Distance.COSINE,
                    ),
                )
            return QdrantVectorStore(
                client=self._client,
                collection_name=name,
                embedding=get_embeddings(),
            )
        except VectorStoreError:
            raise
        except Exception as exc:  # client raises transport-level errors
            raise VectorStoreError(f"Could not open collection '{name}': {exc}") from exc

    # ── Writes ───────────────────────────────────────────────────────────────

    def add_documents(self, documents: list[Document], session_id: str) -> int:
        """Embed and store chunks for a session. Returns the number stored."""
        if not documents:
            return 0
        try:
            self._vector_store(session_id).add_documents(documents)
        except VectorStoreError:
            raise
        except Exception as exc:  # client raises transport-level errors
            raise VectorStoreError(f"Could not store documents: {exc}") from exc
        logger.info("Stored %d chunk(s) for session %s", len(documents), session_id)
        return len(documents)

    def delete_session(self, session_id: str) -> None:
        """Remove a session's collection and everything in it."""
        name = self.collection_name(session_id)
        try:
            if self._client.collection_exists(name):
                self._client.delete_collection(name)
                logger.info("Deleted collection %s", name)
        except Exception as exc:  # client raises transport-level errors
            raise VectorStoreError(f"Could not delete collection '{name}': {exc}") from exc

    # ── Reads ────────────────────────────────────────────────────────────────

    def list_titles(self, session_id: str) -> list[str]:
        """Return the distinct paper titles loaded into a session, in load order.

        Qdrant paginates scroll results, so this walks every page: a session
        with more chunks than one page would otherwise appear to have lost the
        papers ingested last.
        """
        name = self.collection_name(session_id)
        try:
            if not self._client.collection_exists(name):
                return []
            seen: set[str] = set()
            titles: list[str] = []
            offset = None
            while True:
                points, offset = self._client.scroll(
                    collection_name=name,
                    with_payload=True,
                    limit=SCROLL_PAGE_SIZE,
                    offset=offset,
                )
                for point in points:
                    title = (point.payload or {}).get("metadata", {}).get("title")
                    if title and title not in seen:
                        seen.add(title)
                        titles.append(title)
                if offset is None:
                    return titles
        except Exception as exc:  # client raises transport-level errors
            raise VectorStoreError(f"Could not list papers for session: {exc}") from exc

    def search(self, query: str, session_id: str, k: int | None = None) -> list[Document]:
        """Return the ``k`` chunks most similar to ``query`` within a session."""
        top_k = k if k is not None else self._settings.retrieval_top_k
        try:
            results = self._vector_store(session_id).similarity_search(query, k=top_k)
        except VectorStoreError:
            raise
        except Exception as exc:  # client raises transport-level errors
            raise VectorStoreError(f"Search failed: {exc}") from exc
        logger.debug("Retrieved %d chunk(s) for query %r", len(results), query[:80])
        return results


@lru_cache(maxsize=1)
def get_repository() -> PaperRepository:
    """Return the process-wide repository, sharing one Qdrant connection pool."""
    settings = get_settings()
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key.get_secret_value(),
        timeout=settings.qdrant_timeout,
    )
    return PaperRepository(client=client, settings=settings)


def reset_repository_cache() -> None:
    """Drop the cached repository. Used by tests and after a settings change."""
    get_repository.cache_clear()
