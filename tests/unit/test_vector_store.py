"""Tests for the session-scoped paper repository."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from paperpilot.core.exceptions import VectorStoreError
from paperpilot.retrieval import vector_store
from paperpilot.retrieval.vector_store import PaperRepository, collection_name_for

SESSION = "3f2b1a44-5c6d-7e8f-9012-3456789abcde"


class FakeClient:
    """Minimal stand-in for QdrantClient covering the calls we make."""

    def __init__(self, pages=None, exists=True):
        self.pages = pages or []
        self.exists = exists
        self.created: list[str] = []
        self.deleted: list[str] = []

    def collection_exists(self, name):
        return self.exists

    def create_collection(self, collection_name, vectors_config):
        self.created.append(collection_name)
        self.exists = True

    def delete_collection(self, name):
        self.deleted.append(name)

    def scroll(self, collection_name, with_payload, limit, offset):
        index = offset or 0
        if index >= len(self.pages):
            return [], None
        next_offset = index + 1 if index + 1 < len(self.pages) else None
        return self.pages[index], next_offset


def point(title):
    return type("Point", (), {"payload": {"metadata": {"title": title}}})()


@pytest.fixture
def repository(settings):
    return PaperRepository(client=FakeClient(), settings=settings)


def test_collection_name_is_namespaced_and_hyphen_free(settings):
    name = collection_name_for(SESSION)
    assert name.startswith(f"{settings.collection_prefix}_")
    assert "-" not in name


def test_sessions_get_distinct_collections():
    assert collection_name_for("session-a") != collection_name_for("session-b")


def test_adding_no_documents_is_a_no_op(repository):
    assert repository.add_documents([], SESSION) == 0


def test_add_documents_reports_how_many_were_stored(repository, monkeypatch):
    stored = []
    monkeypatch.setattr(
        repository,
        "_vector_store",
        lambda session_id: type(
            "VS", (), {"add_documents": lambda self, docs: stored.extend(docs)}
        )(),
    )
    count = repository.add_documents(
        [Document(page_content="a"), Document(page_content="b")], SESSION
    )
    assert count == 2
    assert len(stored) == 2


def test_collection_is_created_only_when_missing(settings):
    client = FakeClient(exists=False)
    repository = PaperRepository(client=client, settings=settings)
    repository.list_titles(SESSION)
    assert client.created == []


def test_list_titles_walks_every_page(settings):
    client = FakeClient(pages=[[point("Paper A")], [point("Paper B")]])
    repository = PaperRepository(client=client, settings=settings)
    assert repository.list_titles(SESSION) == ["Paper A", "Paper B"]


def test_list_titles_deduplicates_and_preserves_order(settings):
    client = FakeClient(pages=[[point("B"), point("A"), point("B")]])
    repository = PaperRepository(client=client, settings=settings)
    assert repository.list_titles(SESSION) == ["B", "A"]


def test_list_titles_for_an_unknown_session_is_empty(settings):
    repository = PaperRepository(client=FakeClient(exists=False), settings=settings)
    assert repository.list_titles(SESSION) == []


def test_transport_failures_surface_as_vector_store_errors(settings):
    class BrokenClient(FakeClient):
        def collection_exists(self, name):
            raise ConnectionError("qdrant unreachable")

    repository = PaperRepository(client=BrokenClient(), settings=settings)
    with pytest.raises(VectorStoreError, match="Could not list papers"):
        repository.list_titles(SESSION)


def test_search_defaults_to_the_configured_top_k(repository, settings, monkeypatch):
    captured = {}

    class FakeStore:
        def similarity_search(self, query, k):
            captured["k"] = k
            return []

    monkeypatch.setattr(repository, "_vector_store", lambda session_id: FakeStore())
    repository.search("query", SESSION)
    assert captured["k"] == settings.retrieval_top_k


def test_search_honours_an_explicit_k(repository, monkeypatch):
    captured = {}

    class FakeStore:
        def similarity_search(self, query, k):
            captured["k"] = k
            return []

    monkeypatch.setattr(repository, "_vector_store", lambda session_id: FakeStore())
    repository.search("query", SESSION, k=9)
    assert captured["k"] == 9


def test_delete_session_removes_the_collection(settings):
    client = FakeClient()
    repository = PaperRepository(client=client, settings=settings)
    repository.delete_session(SESSION)
    assert client.deleted == [collection_name_for(SESSION)]


def test_repository_is_cached(monkeypatch, settings):
    vector_store.reset_repository_cache()
    monkeypatch.setattr(vector_store, "QdrantClient", lambda **kwargs: FakeClient())
    first = vector_store.get_repository()
    assert vector_store.get_repository() is first
    vector_store.reset_repository_cache()
