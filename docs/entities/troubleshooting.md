# Entities Troubleshooting

This guide helps diagnose common failures in entities extraction and resolution.

## Turn on detailed logs

Use `--log-level DEBUG` on `entities.run`:

```bash
uv run python -m ddd_policy_tracer.analysis.entities.run \
  --all-chunks \
  --chunk-state-path data/chunks.jsonl \
  --entity-state-path data/entities.jsonl \
  --log-level DEBUG
```

You can also set `ENTITY_LOG_LEVEL=DEBUG`.

## Common failure modes

### 1) Chunk loading failures

Symptoms:

- `status=failed` for chunk with `chunk not found`
- logs mention chunk repository load/list errors

Checks:

- confirm `--chunk-state-path` points to a real JSONL file
- validate rows contain required chunk fields

### 2) Extraction runtime exception

Symptoms:

- report failure with exception message
- stack trace in logs from `EntitiesService`

Checks:

- run one chunk with `--chunk-id` to isolate input
- inspect problematic chunk text/encoding

### 3) Resolution artifacts missing

Symptoms:

- run fails or returns weak linking when catalog content is empty

Checks:

- run catalog import first (`entities-catalog import`)
- verify promoted entities exist in `entity_catalog`
- verify vectors file exists and contains non-empty `vectors` list

Note:

- Missing DB files/tables are auto-created.
- Missing vectors file is auto-created as empty.
- Auto-created empty artifacts are valid but will not improve linking until catalog has real rows.

### 4) No unresolved rows in review DB

Symptoms:

- `review_items` empty even with resolution enabled

Checks:

- unresolved statuses only (`needs_review`, `new_candidate`, `abstain`) are queued
- linked rows are intentionally excluded

### 5) Auto-promotion not moving items

Symptoms:

- `entities-catalog promote --auto-approve` reports zero promoted items

Checks:

- pending items must be `needs_review` for auto-approval
- `decision_score` must be >= configured min score
- top candidate margin must be >= configured min margin
- top candidate entity type must match mention entity type
- ambiguous items are expected to remain pending for manual review

### 6) Downstream canonicalization surprises

Symptoms:

- entities appear linked in claims unexpectedly

Checks:

- verify entities were run with resolution enabled if you expect explicit decision statuses
- inspect `metadata.decision_status` in `entities.jsonl`

## Quick diagnostics commands

List pending review items:

```bash
uv run python -m ddd_policy_tracer entities-review list \
  --review-db-path data/review.db \
  --state pending
```

Run auto-promotion with strict thresholds:

```bash
uv run python -m ddd_policy_tracer entities-catalog promote \
  --review-db-path data/review.db \
  --catalog-path data/entity_catalog.db \
  --vectors-path data/entity_vectors.json \
  --catalog-version catalog-v1 \
  --auto-approve \
  --auto-approve-min-score 0.93 \
  --auto-approve-min-margin 0.15
```

## Useful test targets

```bash
uv run pytest \
  tests/unit/analysis/test_entities_run.py \
  tests/unit/analysis/test_entities_service.py \
  tests/unit/analysis/test_entities_resolution.py \
  tests/unit/analysis/test_entities_review_queue.py
```
