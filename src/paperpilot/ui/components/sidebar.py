"""The sidebar: session switching and document loading."""

from __future__ import annotations

import streamlit as st

from paperpilot.ui.components.documents import (
    render_arxiv_loader,
    render_file_uploader,
    render_loaded_documents,
    render_url_loader,
)
from paperpilot.ui.state import AppState


def _render_session_list(state: AppState) -> None:
    """List sessions newest first, highlighting the active one."""
    st.markdown("## Sessions")
    for meta in state.store.sorted_by_recency(state.sessions):
        is_active = meta.id == state.active_id
        if (
            st.button(
                meta.name,
                key=f"session_{meta.id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            )
            and not is_active
        ):
            state.switch_to(meta.id)
            st.rerun()


def render_sidebar(state: AppState) -> None:
    """Render the full sidebar for the active session."""
    with st.sidebar:
        if st.button("New chat", use_container_width=True, key="new_chat"):
            state.create_session()
            st.rerun()

        st.divider()
        _render_session_list(state)

        st.divider()
        st.markdown("## Documents")
        session_id = state.active_id
        render_file_uploader(state, session_id)
        render_url_loader(session_id)
        render_arxiv_loader(session_id)


def render_document_list(session_id: str) -> None:
    """Append the loaded-document list to the sidebar.

    Called last, after the chat has rendered. Listing papers is the one sidebar
    control that talks to the vector store, so a slow or unreachable store
    delays only this list instead of holding up the entire page.
    """
    with st.sidebar:
        render_loaded_documents(session_id)
