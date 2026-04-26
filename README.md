# ddd-policy-tracer

DDD-first policy document acquisition prototype.

## Tooling

- We use `uv` as the project and package manager.
- Do not use `pip`, `pipenv`, `poetry`, or `requirements.txt` workflows for this repo.

## Common commands

- Install dependencies and create/update lockfile: `uv sync`
- Run tests: `uv run pytest`
- Run lint: `uv run ruff check .`
- Run formatting check: `uv run ruff format --check .`
- Run type checks: `uv run mypy .`

## Running the project

- CLI entrypoint: `uv run python main.py`
