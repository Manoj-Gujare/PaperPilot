# PaperPilot

An AI research assistant that lets you chat with your papers and automatically flags claims
superseded by newer research.

Load papers from a file, a URL or ArXiv into an isolated session, then ask questions about them.
PaperPilot routes each message: it answers from the papers, searches the live web, or checks
whether a claim has been overtaken by more recent work — and returns links to the papers that
overtook it.

---

## Features

| Feature | What it does |
|---|---|
| **Paper Q&A** | Retrieves relevant chunks from your papers and answers from them, grounded in what they actually say |
| **Claim verification** | Searches the web and ArXiv to decide whether a claim still holds, and links the papers that supersede it |
| **Web search** | Pulls live results in when a question is about current developments rather than the papers |
| **Direct answers** | Answers stable general-knowledge questions without retrieval or a web call |
| **`/btw` side channel** | Asks an off-topic question without it entering the conversation's history |
| **Multiple sessions** | Independent conversations, each with its own paper collection and its own vector-store collection |
| **Automatic titles** | Names each session from its first message |
| **Four ways to load a paper** | PDF, plain text, Markdown, a web URL, or an ArXiv ID or title |
| **Graph state inspector** | Expands every answer to show the path the workflow took to produce it |
| **Streaming replies** | Renders tokens as they arrive |

---

## Quick start

PaperPilot uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone https://github.com/Manoj-Gujare/PaperPilot.git
cd PaperPilot

uv sync --all-extras --dev     # install runtime, evaluation and dev dependencies
cp .env.example .env           # then fill in your API keys

uv run streamlit run app.py
```

Then open <http://localhost:8501>.

### With Docker

```bash
docker build -t paperpilot .
docker run --rm -p 8501:8501 --env-file .env paperpilot
```

---

## Using it

1. **Start a session.** One is created for you; **New chat** in the sidebar starts another.
2. **Load papers.** Upload PDF, TXT or Markdown files, paste URLs one per line, or enter an
   ArXiv ID (`1706.03762`) or title (`Attention Is All You Need`).
3. **Ask.** For example:
   - *What methodology does the paper use for evaluation?*
   - *Verify the claim that encoder-decoder models are the best approach for translation.*
   - *What are the latest developments in diffusion models?*
4. **Ask something off-topic** without polluting the conversation:
   ```
   /btw What is the difference between RLHF and DPO?
   ```

---

## Configuration

Every setting is declared in `src/paperpilot/config/settings.py` and can be overridden through the
environment or `.env`. Four values are required:

| Variable | Purpose | Where to get it |
|---|---|---|
| `OPENAI_API_KEY` | Chat completions and embeddings | [platform.openai.com](https://platform.openai.com) |
| `TAVILY_API_KEY` | Web search and claim verification | [tavily.com](https://tavily.com) |
| `QDRANT_URL` | Vector store endpoint | [cloud.qdrant.io](https://cloud.qdrant.io) |
| `QDRANT_API_KEY` | Vector store authentication | [cloud.qdrant.io](https://cloud.qdrant.io) |

Everything else has a default: model names, chunk size and overlap, retrieval `k`, retry budgets,
timeouts, storage paths and log level. See `.env.example` for the full list.

Invalid configuration fails at start-up rather than mid-request — a missing key, an unknown log
level, or a chunk overlap wider than the chunk size all raise before the app serves anything.

---

## How it works

```
src/paperpilot/
├── config/       Validated settings, loaded once from the environment
├── core/         Exception hierarchy, logging, structured-output schemas
├── llm/          Chat and embedding models, with a file-backed embedding cache
├── ingestion/    Chunking, file and web loaders, ArXiv lookup
├── retrieval/    Session-scoped Qdrant storage and web search
├── graph/        Workflow state, prompts, tools, nodes and edges
├── services/     Chat orchestration, session store, naming, /btw side channel
└── ui/           Streamlit entry point, state wrapper and components
evaluation/       Golden generation, metrics and the scoring CLI
tests/            Offline unit suite
```

The workflow:

```
query ─► router ─┬─ direct_answer ────────────────────────► generate_answer
                 │
                 ├─ retrieve ─► agent ─► tools ─► agent ─► relevancy_check
                 │                 ▲                    │        │
                 │                 └── query_rewrite ◄──┘        │
                 │                                               ▼
                 └─ verify_claim ─────────────────────────► generate_answer
```

`docs/architecture.md` explains each step, the decisions behind them, and the invariants the
workflow depends on.

---

## Development

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run pytest                # unit suite, runs offline
uv run pytest --cov=paperpilot
```

CI runs lint, a format check and the tests on every pull request against `main`.

The test suite needs no credentials and makes no network calls: external clients are replaced
with fakes and model calls are patched, so it exercises this codebase rather than a provider.

---

## Evaluation

```bash
uv run paperpilot-eval
uv run paperpilot-eval --document documents/your_paper.pdf --threshold 0.8
```

Scores retrieval quality (contextual precision, recall and relevancy) and generation quality
(answer relevancy, faithfulness) with [DeepEval](https://github.com/confident-ai/deepeval), and
writes a per-case report to `eval_results.json`. Goldens are synthesised on first run and cached
in `goldens.json`. Requires live credentials — it runs the real pipeline end to end.

---

## License

MIT
