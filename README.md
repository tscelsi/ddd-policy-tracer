# ddd-policy-tracer

DDD-first policy document acquisition prototype.

## Tooling

- We use `uv` as the project and package manager.
- Do not use `pip`, `pipenv`, `poetry`, or `requirements.txt` workflows for this repo.

## Common commands

- Install dependencies and create/update lockfile: `uv sync`
- Run tests: `uv run pytest`
- Run lint: `uv run ruff check . --fix`
- Run formatting check: `uv run ruff format --check .`
- Run type checks: `uv run mypy .`

## Running the project

- CLI entrypoint: `uv run python main.py`
- API: `uv run uvicorn ddd_policy_tracer.app:app --port 5000`

### Question completion API

The `POST /api/completion` endpoint runs hybrid retrieval and answer
synthesis. It returns the answer, inline citation text, cited source passages,
and an explicit abstention state when the evidence is insufficient.
See [RAG completion request flow](docs/RAG_COMPLETION_FLOW.md) for the
sequence diagram and module responsibilities.

```json
{
  "question": "Why does Helen Haines support the Three Day Guarantee Bill?",
  "limit": 10
}
```

The app loads `.env` from the repository root during startup. Set
`RAG_DATABASE_PATH` there to the SQLite database containing `document_chunks`
and `chunk_embeddings` (defaults to `data/new3.db`). Configure the local
Ollama instance with `OLLAMA_HOST` if it is not using the client default, and
select the answer model with `OLLAMA_MODEL` (defaults to `llama3.1:8b`).
Existing process environment variables take precedence over `.env`. The
Ollama server and selected model must be available when a completion request
is made.

Set `RAG_ANSWER_GENERATOR=openai` to use the OpenAI adapter instead. Provide
`OPENAI_API_KEY` and optionally `OPENAI_MODEL` (defaults to `gpt-4.1-mini`)
in `.env` or the process environment. Ollama remains the default provider.
The OpenAI SDK sends requests to OpenAI using that API key; do not commit the
key to the repository.

### Profiling completion requests

Each successful `POST /api/completion` response includes a `Server-Timing`
header with durations for `embedding`, `hybrid_retrieval`, `generation`, and
the total `completion` handler time. The application log also emits these
durations with the number of retrieved evidence chunks and whether the answer
abstained; it does not log the question or speech passages.

Inspect the header with:

```bash
curl -sS -D - -o /dev/null \
  -H 'Content-Type: application/json' \
  -d '{"question":"Why does Helen Haines support the Three Day Guarantee Bill?"}' \
  http://localhost:5000/api/completion
```

`generation` includes the Ollama request and answer/citation validation.
`hybrid_retrieval` combines vector search, BM25, and rank fusion. Compare
several requests after warm-up; the first model call may include Ollama model
loading time.
