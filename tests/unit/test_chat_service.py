"""Tests for the chat orchestration service."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from paperpilot.services.chat_service import ChatService, serialise_state


class FakeGraph:
    """Stands in for the compiled graph, recording what it was asked to run."""

    def __init__(self, chunks=None, values=None):
        self.chunks = chunks or []
        self.values = values or {}
        self.configs = []

    def stream(self, state, config, stream_mode):
        self.configs.append(config)
        self.last_state = state
        yield from self.chunks

    def get_state(self, config):
        self.configs.append(config)
        return type("Snapshot", (), {"values": self.values})()


def chunk(content, node):
    return type("Chunk", (), {"content": content})(), {"langgraph_node": node}


def test_only_answer_node_tokens_are_streamed():
    graph = FakeGraph(
        chunks=[
            chunk("routing noise", "router"),
            chunk("Hel", "generate_answer"),
            chunk("grader noise", "relevancy_check"),
            chunk("lo", "generate_answer"),
        ]
    )
    assert "".join(ChatService(graph=graph).stream_answer("q", "s1")) == "Hello"


def test_empty_chunks_are_skipped():
    graph = FakeGraph(chunks=[chunk("", "generate_answer"), chunk("x", "generate_answer")])
    assert "".join(ChatService(graph=graph).stream_answer("q", "s1")) == "x"


def test_the_session_scopes_the_checkpoint_thread():
    graph = FakeGraph()
    list(ChatService(graph=graph).stream_answer("q", "session-42"))
    assert graph.configs == [{"configurable": {"thread_id": "session-42"}}]


def test_answer_of_record_reads_state_when_nothing_streamed():
    graph = FakeGraph(values={"answer": "assembled in python"})
    assert ChatService(graph=graph).answer_of_record("s1") == "assembled in python"


def test_answer_of_record_falls_back_when_state_is_empty():
    service = ChatService(graph=FakeGraph(values={}))
    assert service.answer_of_record("s1") == "No response generated."


def test_history_maps_messages_to_chat_roles():
    graph = FakeGraph(
        values={
            "messages": [
                HumanMessage(content="question"),
                ToolMessage(content="tool noise", tool_call_id="1"),
                AIMessage(content="answer"),
            ]
        }
    )
    assert ChatService(graph=graph).history("s1") == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]


def test_history_of_an_unknown_session_is_empty():
    class BrokenGraph(FakeGraph):
        def get_state(self, config):
            raise RuntimeError("no such thread")

    assert ChatService(graph=BrokenGraph()).history("s1") == []


def test_serialised_state_truncates_long_content():
    values = {"messages": [HumanMessage(content="x" * 1000)]}
    assert len(serialise_state(values)["messages"][0]["content"]) == 300


def test_serialised_state_keeps_document_metadata():
    values = {"retrieved_docs": [Document(page_content="body", metadata={"title": "T"})]}
    assert serialise_state(values)["retrieved_docs"] == [
        {"content": "body", "metadata": {"title": "T"}}
    ]


def test_serialised_state_passes_scalars_through():
    assert serialise_state({"route": "retrieve", "rewrite_count": 2}) == {
        "route": "retrieve",
        "rewrite_count": 2,
    }


def test_serialised_state_handles_missing_collections():
    assert serialise_state({"messages": None, "retrieved_docs": None}) == {
        "messages": [],
        "retrieved_docs": [],
    }
