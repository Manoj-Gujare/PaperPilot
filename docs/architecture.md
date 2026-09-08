# Architecture

How a question becomes an answer, and why the pieces are arranged the way they are.

---

## Layers

Dependencies point one way: the interface depends on services, services on the graph, the graph on
retrieval and models, and everything on configuration and core. Nothing lower reaches upward.

| Layer | Package | Responsibility |
|---|---|---|
| Configuration | `paperpilot.config` | One validated settings object, read from the environment |
| Core | `paperpilot.core` | Errors, logging, structured-output schemas |
| Models | `paperpilot.llm` | Chat and embedding clients |
| Ingestion | `paperpilot.ingestion` | Sources in, chunks out |
| Retrieval | `paperpilot.retrieval` | Context out — from stored papers or the live web |
| Workflow | `paperpilot.graph` | The decision graph that answers a question |
| Services | `paperpilot.services` | Orchestration, session metadata, the `/btw` channel |
| Interface | `paperpilot.ui` | Streamlit composition and components |

---

## The workflow

```
query ─► router ─┬─ direct_answer ────────────────────────► generate_answer
                 │
                 ├─ retrieve ─► agent ─► tools ─► agent ─► relevancy_check
                 │                 ▲                    │        │
                 │                 └── query_rewrite ◄──┘        │
                 │                                               ▼
                 └─ verify_claim ─────────────────────────► generate_answer
```

### router

Classifies the message into `retrieve`, `verify_claim` or `direct_answer`. Classification failure
falls back to `retrieve`: answering from paper context is the safe default, whereas defaulting to a
direct answer would invent content the user reads as coming from their papers.

### agent and tools

The agent chooses its own tool and arguments — which query to search with, how many chunks to pull,
whether the answer lives in the papers or on the web. Two tools are bound:

- `retrieve_from_vectorstore` — semantic search over this session's papers
- `web_search` — live results through Tavily

Each returns a short `ToolMessage` summary plus a `Command` carrying the documents into state. The
summary keeps the message history well-formed; the `Command` keeps whole chunks out of a transcript
that is resent with every subsequent prompt.

### relevancy_check

Grades the first few retrieved chunks, truncated to snippets. It is a cheap gate on whether
retrieval worked at all — grading the full context would cost as much as answering. A grading
failure accepts the context: answering from real chunks beats discarding them because the grader
was unavailable.

### query_rewrite

Rewrites a query that retrieved nothing relevant, and clears the retrieved documents and the
attempt counter so the new query is graded on what *it* finds. Budgeted at
`max_query_rewrites` (default 1).

### verify_claim

Issues two searches, not one. A general search surfaces blog posts, retractions and news; an
`arxiv.org`-restricted search surfaces the academic work that actually supersedes a finding. Either
alone reliably misses one of those categories. The model may only quote titles and URLs that appear
verbatim in the results.

### generate_answer

The single exit. Grounded answer, formatted verdict, or direct answer, depending on the route. It
always produces a reply — a provider failure here yields an apology rather than an exception.

---

## Invariants

**Pending tool calls always run.** `route_after_agent` sends the turn to the tool node whenever
tool calls are pending, even at the retrieval cap. Routing away leaves an `AIMessage` carrying
`tool_call` ids that no `ToolMessage` answers; the checkpointer persists that, and every later turn
in the session replays a malformed history to the provider.

**At the cap, the agent has no tools.** `agent_node` binds tools only while attempts remain below
`max_retrieval_attempts`. Without that the model could emit one final unanswerable tool call.

**Counters reset per turn.** `initial_state` clears `retrieved_docs`, `retrieval_attempts`,
`rewrite_count` and the claim fields. Carrying them forward would let one question answer the next
from stale context, or exhaust the retry budget before the new question is tried once.

**Sessions are isolated in storage.** Each session owns a Qdrant collection named
`{collection_prefix}_{session_id}` and its own checkpointer thread. Isolation is a property of the
storage layer, not a filter every query has to remember.

**`/btw` touches nothing.** The side channel bypasses the vector store, the checkpointer and
session history entirely. A passing question mid-review should not become context the assistant
reasons from for the rest of the conversation.

---

## Failure behaviour

Every external dependency can fail, and none of them ends a turn:

| Failure | Behaviour |
|---|---|
| Routing | Falls back to retrieval |
| Vector search | Tool reports the outage; the agent continues with what it has |
| Web search | Same, and claim verification returns an "unverifiable" verdict |
| Relevancy grading | Accepts the retrieved context |
| Query rewrite | Retries the original query |
| Answer generation | Returns a plain apology |
| Session naming | Keeps the default name |
| Session index unreadable | Starts with an empty session list |

Loader, vector-store and search errors are typed (`DocumentLoadError`, `VectorStoreError`,
`WebSearchError`) and caught at the interface, so the user sees an explanation rather than a
stack trace.

---

## Performance choices

| Choice | Why |
|---|---|
| Embeddings cached to disk, keyed by model | Re-ingesting a paper someone already uploaded costs nothing; the model-name key prevents stale vectors of the wrong dimensionality |
| Graph compiled once behind `st.cache_resource` | Streamlit reruns the script on every interaction; recompiling each time would rebuild every node and reopen SQLite |
| Documents carried in state, not messages | Keeps the transcript small, since it is resent with every prompt |
| Only answer-node tokens streamed | The router, grader and rewriter also call the model; showing their output would show the machinery instead of the answer |
| Document listing rendered last | It is the only sidebar control that queries the vector store, so a slow store delays that list alone instead of the whole page |
| Uploads tracked per session | Streamlit resubmits its uploader contents on every rerun; without this each upload would be re-embedded repeatedly |
| Evaluation throttled to 3 workers | Higher concurrency fails on provider rate limits rather than finishing sooner |

---

## Chosen limits

| Limit | Default | Reasoning |
|---|---|---|
| `chunk_size` / `chunk_overlap` | 1000 / 200 | Small chunks retrieve precisely; the overlap keeps a sentence straddling a boundary reachable from both sides |
| `retrieval_top_k` | 4 | Enough context to answer without diluting focus or inflating cost |
| `max_retrieval_attempts` | 3 | Caps an ambiguous question's tool loop before it burns tokens indefinitely |
| `max_query_rewrites` | 1 | A second rewrite rarely helps; saying "I could not find this" is more useful than a third guess |
| `web_search_max_results` | 3 | Keeps side-channel and supplementary context light |
| `verification_max_results` | 5 per search | Two searches, ten results — enough for the model to find genuinely superseding work |
| `max_superseding_papers` | 3 | A readable answer, not a bibliography |
