# Entities Runbook

This runbook shows practical command sequences for running the entities pipeline.

## Prerequisites

- chunk input exists at `data/chunks.jsonl`
- optional curated seed exists at `data/entity_seed.jsonl`
- if running claim canonicalization/graph: `data/claims.jsonl` exists

## 1) Build or refresh runtime catalog

Use this when bootstrapping from curated seed data.

```bash
uv run python -m ddd_policy_tracer \
  entities-catalog import \
  --seed-path data/entity_seed.jsonl \
  --catalog-path data/entity_catalog.db \
  --vectors-path data/entity_vectors.json \
  --catalog-version catalog-v1
```

## 2) Run entities extraction

### Fast extraction only (no catalog resolution)

```bash
uv run python -m ddd_policy_tracer.analysis.entities.run \
  --all-chunks \
  --chunk-state-path data/chunks.jsonl \
  --entity-state-path data/entities.jsonl \
  --extractor robust-ensemble \
  --extractor-version robust-ensemble-v1 \
  --log-level INFO
```

### Extraction + resolution + review queue

```bash
uv run python -m ddd_policy_tracer.analysis.entities.run \
  --all-chunks \
  --chunk-state-path data/chunks.jsonl \
  --entity-state-path data/entities.jsonl \
  --extractor robust-ensemble \
  --extractor-version robust-ensemble-v1 \
  --catalog-path data/entity_catalog.db \
  --vectors-path data/entity_vectors.json \
  --review-db-path data/review.db \
  --log-level DEBUG
```

Note: catalog/vectors/review DB files auto-initialize if missing.

## 3) Review unresolved items

List pending:

```bash
uv run python -m ddd_policy_tracer \
  entities-review list \
  --review-db-path data/review.db \
  --state pending
```

Manual decision example:

```bash
uv run python -m ddd_policy_tracer \
  entities-review decide \
  --review-db-path data/review.db \
  --mention-entity-id entity_123 \
  --decision approved \
  --canonical-entity-key policy:clean-energy-act \
  --canonical-name "Clean Energy Act" \
  --entity-type POLICY \
  --aliases "Clean Energy Act,CEA"
```

Interactive one-by-one curation (recommended):

```bash
uv run python -m ddd_policy_tracer \
  entities-review curate \
  --review-db-path data/review.db \
  --chunk-state-path data/chunks.jsonl \
  --catalog-path data/entity_catalog.db \
  --vectors-path data/entity_vectors.json \
  --limit 50
```

Actions in curate mode:

- `m`: merge with an existing canonical entity
- `n`: create a new canonical entity
- `s`: skip cataloging this candidate

## 4) Promote manually approved items into catalog

```bash
uv run python -m ddd_policy_tracer \
  entities-catalog promote \
  --review-db-path data/review.db \
  --catalog-path data/entity_catalog.db \
  --vectors-path data/entity_vectors.json \
  --catalog-version catalog-v1
```

## 5) Rerun entities after promotion

This is the learning loop: reruns should produce more `linked` outcomes.

## 6) Canonicalize entities and claims

```bash
uv run python -m ddd_policy_tracer \
  canonicalize all \
  --entities-path data/entities.jsonl \
  --entities-canonical-path data/entities_canonical.jsonl \
  --claims-path data/claims.jsonl \
  --claims-canonical-path data/claims_canonical.jsonl \
  --entity-canonicalizer-version entity-v1 \
  --claim-canonicalizer-version claim-v1
```

## 7) Build graph artifacts

```bash
uv run python -m ddd_policy_tracer.analysis.graph.run \
  --chunks-path data/chunks.jsonl \
  --claims-path data/claims_canonical.jsonl \
  --entities-path data/entities_canonical.jsonl \
  --output-root data/graph_runs
```

## Quick checklist

- `data/entities.jsonl` exists and includes extracted mentions.
- If resolution enabled: rows include `metadata.decision_status`.
- If review queue enabled: `data/review.db` has `review_items` and `review_events`.
- `data/claims_canonical.jsonl` includes:
  - `linked_entities` (linked only)
  - `pending_entity_links` (unresolved)
- `data/graph_runs/latest/graph.json` exists.
