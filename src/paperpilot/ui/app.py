"""Streamlit entry point.

Kept to page setup and composition: bootstrap state, draw the sidebar, replay
the conversation, read the next message. Every decision behind those four steps
lives in a service or a component.
"""

from __future__ import annotations

import streamlit as st

from paperpilot.config import get_settings
from paperpilot.core.logging import configure_logging
from paperpilot.services.chat_service import get_graph
from paperpilot.ui.components import (
    render_chat_input,
    render_document_list,
    render_history,
    render_sidebar,
)
from paperpilot.ui.state import AppState

PAGE_TITLE = "PaperPilot"
PAGE_ICON = "📚"
HEADER = "📚 PaperPilot — Research Paper Assistant"
INTRO = (
    "**Ask questions** about your uploaded papers &nbsp;·&nbsp; "
    "**Verify claims** against recent literature &nbsp;·&nbsp; "
    "**Search the web** for the latest findings\n\n"
    "> Load documents from the sidebar, then start chatting below. "
    "Prefix a message with `/btw` to ask something off-topic without it "
    "entering this conversation's history."
)


@st.cache_resource
def _bootstrap():
    """Configure logging and compile the graph once per process.

    Streamlit reruns this script on every interaction; cache_resource keeps the
    compiled graph and its SQLite connection alive across those reruns instead
    of rebuilding them each time.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    return get_graph()


def main() -> None:
    """Render one pass of the application."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")

    _bootstrap()
    state = AppState()
    state.bootstrap()

    render_sidebar(state)

    st.title(HEADER)
    st.markdown(INTRO)
    st.divider()

    session_id = state.active_id
    render_history(state, session_id)
    render_chat_input(state, session_id)

    # Last, because it is the only control that queries the vector store.
    render_document_list(session_id)


main()
