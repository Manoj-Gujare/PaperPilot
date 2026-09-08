"""Sidebar controls for getting papers into a session."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from paperpilot.core.exceptions import PaperPilotError
from paperpilot.core.logging import get_logger
from paperpilot.ingestion import load_arxiv, load_document, load_webpage
from paperpilot.ingestion.chunking import stamp_title
from paperpilot.retrieval import get_repository
from paperpilot.ui.state import AppState

logger = get_logger(__name__)

UPLOAD_TYPES = ["pdf", "txt", "md", "markdown"]
URL_PLACEHOLDER = "https://example.com/paper"
ARXIV_PLACEHOLDER = "1706.03762  or  Attention Is All You Need"


def _ingest_upload(uploaded_file, session_id: str) -> None:
    """Write an upload to a temporary file, ingest it, then remove the file.

    Streamlit hands over an in-memory buffer, but the PDF and text loaders read
    from paths. The temporary file is deleted whether ingestion succeeds or not.
    """
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(uploaded_file.name).suffix
        ) as handle:
            handle.write(uploaded_file.read())
            temp_path = handle.name
        documents = load_document(temp_path)
        # The temporary file's name is meaningless; title by what the user uploaded.
        stamp_title(documents, Path(uploaded_file.name).stem)
        get_repository().add_documents(documents, session_id)
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


def render_file_uploader(state: AppState, session_id: str) -> None:
    """Upload and ingest PDF, text or Markdown files."""
    st.markdown("**Upload files**")
    uploaded_files = st.file_uploader(
        "PDF, TXT or Markdown",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        key=f"uploader_{session_id}",
        label_visibility="collapsed",
    )
    if not st.button("Add files", use_container_width=True, key="add_files"):
        return
    if not uploaded_files:
        st.warning("No files selected.")
        return

    with st.spinner("Processing files…"):
        for uploaded_file in uploaded_files:
            if state.already_loaded(session_id, uploaded_file.name):
                st.info(f"Already loaded: {uploaded_file.name}")
                continue
            try:
                _ingest_upload(uploaded_file, session_id)
            except PaperPilotError as error:
                st.error(f"Failed: {uploaded_file.name} — {error}")
                continue
            state.mark_loaded(session_id, uploaded_file.name)
            st.success(f"Added: {uploaded_file.name}")
    st.rerun()


def render_url_loader(session_id: str) -> None:
    """Fetch and ingest one or more web pages."""
    st.markdown("**Web pages**")
    raw_urls = st.text_area(
        "URLs, one per line",
        key=f"urls_{session_id}",
        height=80,
        label_visibility="collapsed",
        placeholder=URL_PLACEHOLDER,
    )
    if not st.button("Load URLs", use_container_width=True, key="load_urls"):
        return

    urls = [url.strip() for url in raw_urls.splitlines() if url.strip()]
    if not urls:
        st.warning("Enter at least one URL.")
        return

    with st.spinner("Loading web pages…"):
        for url in urls:
            try:
                get_repository().add_documents(load_webpage(url), session_id)
            except PaperPilotError as error:
                st.error(f"Failed: {url[:60]} — {error}")
                continue
            st.success(f"Loaded: {url[:60]}")
    st.rerun()


def render_arxiv_loader(session_id: str) -> None:
    """Look up a paper on ArXiv by identifier or title and ingest it."""
    st.markdown("**ArXiv papers**")
    query = st.text_input(
        "Paper title or ArXiv ID",
        key=f"arxiv_{session_id}",
        label_visibility="collapsed",
        placeholder=ARXIV_PLACEHOLDER,
    )
    if not st.button("Load ArXiv paper", use_container_width=True, key="load_arxiv"):
        return
    if not query.strip():
        st.warning("Enter a paper title or ArXiv ID.")
        return

    with st.spinner("Loading from ArXiv…"):
        try:
            documents = load_arxiv(query.strip())
            get_repository().add_documents(documents, session_id)
        except PaperPilotError as error:
            st.error(f"Failed: {error}")
            return
        title = documents[0].metadata.get("title") if documents else query.strip()
    st.success(f"Loaded: {title}")
    st.rerun()


def render_loaded_documents(session_id: str) -> None:
    """List the papers currently available to this session."""
    st.divider()
    st.markdown("### Loaded documents")
    try:
        titles = get_repository().list_titles(session_id)
    except PaperPilotError as error:
        logger.warning("Could not list documents: %s", error)
        st.caption("Could not reach the document store — try refreshing.")
        return

    if not titles:
        st.caption("No documents loaded yet.")
        return
    for title in titles:
        st.markdown(f"- {title}")
