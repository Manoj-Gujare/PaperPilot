# PaperPilot

An AI research assistant that lets you chat with your papers and automatically flags claims
superseded by newer research.

> **Status:** foundation layer. Configuration, logging, error handling, tooling and CI are in
> place; the RAG pipeline and the Streamlit interface land in the following changes.

---

## Project layout

```
src/paperpilot/
├── config/      Validated settings loaded once from the environment
└── core/        Exception hierarchy, logging setup, structured-output schemas
tests/           Pytest suite — runs offline, no credentials needed
```

## Getting started

PaperPilot uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone https://github.com/Manoj-Gujare/PaperPilot.git
cd PaperPilot

uv sync --all-extras --dev     # install runtime, evaluation and dev dependencies
cp .env.example .env           # then fill in your API keys
```

## Configuration

Every setting is declared in `src/paperpilot/config/settings.py` and can be overridden through
the environment or `.env`. Only four values are required:

| Variable | Purpose | Where to get it |
|---|---|---|
| `OPENAI_API_KEY` | Chat completions and embeddings | [platform.openai.com](https://platform.openai.com) |
| `TAVILY_API_KEY` | Web search and claim verification | [tavily.com](https://tavily.com) |
| `QDRANT_URL` | Vector store endpoint | [cloud.qdrant.io](https://cloud.qdrant.io) |
| `QDRANT_API_KEY` | Vector store authentication | [cloud.qdrant.io](https://cloud.qdrant.io) |

See `.env.example` for the full list of optional overrides — model names, chunk sizes, retrieval
limits and log level.

## Development

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run pytest                # test suite
uv run pytest --cov=paperpilot
```

CI runs the same three commands on every pull request against `main`.

## License

MIT
