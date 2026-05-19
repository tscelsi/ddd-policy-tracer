# Entities Data Contracts

This page describes what the main entities files and tables contain.

## 1) Input

### `chunks.jsonl`

Minimum fields used by entities extraction:

- `chunk_id`
- `source_id`
- `source_document_id`
- `document_checksum`
- `chunk_text`

## 2) Runtime entity catalog (source of truth)

### `entity_catalog.db`

Table: `entity_catalog`

- `canonical_entity_key` (unique identity)
- `canonical_name`
- `entity_type`
- `aliases_json`
- `source` (for example: seed/review_queue)

Table: `catalog_metadata`

- `catalog_version`
- `seed_hash` (content hash of current catalog rows)

### `entity_vectors.json`

Built from `entity_catalog`.

Top-level keys:

- `catalog_version`
- `seed_hash`
- `vectors`

Each vector item:

- `canonical_entity_key`
- `entity_type`
- `tokens`

## 3) Extraction output

### `entities.jsonl`

Main mention fields:

- `entity_id`
- `chunk_id`
- `source_id`
- `source_document_id`
- `document_checksum`
- `start_char`, `end_char`
- `mention_text`, `normalized_mention_text`
- `entity_type`
- `confidence`
- `extractor_version`
- `canonical_entity_key` (nullable)
- `metadata` (nullable)

When resolution runs, `metadata` may include:

- `decision_status`
- `decision_score`
- `reason_codes`
- `selected_candidate_key`
- `top_candidates`
- `candidate_links` (probabilistic ranked candidates with confidence scores)
- `max_candidate_score`

## 4) Review queue

### `review.db`

Table: `review_items` (current state per unresolved mention)

- mention identity and context (`mention_entity_id`, chunk/source fields)
- mention span offsets (`start_char`, `end_char`) for sentence-window context display
- decision fields (`decision_status`, `decision_score`, candidates)
- review workflow state (`review_state`)
- editable canonical fields (`canonical_entity_key`, `canonical_name`, `canonical_aliases_json`)
- promotion tracking (`promoted_at`)

Review queue inclusion rule:

- Only unresolved and catalog-eligible mentions are queued.
- Catalog eligibility currently includes only `PERSON`, `ORG`, and `JURISDICTION` mentions.

Table: `review_events` (append-only audit trail)

- event rows for enqueue, manual decisions, auto-approval, and promotion
- deduplicated by `event_hash`

## 5) Canonicalization outputs

### `entities_canonical.jsonl`

Adds canonicalization metadata plus carried decision fields:

- `canonical_name`
- `canonical_entity_key`
- `canonicalization`
- `decision_status` (nullable)
- `decision_score` (nullable)
- `selected_candidate_key` (nullable)

### `claims_canonical.jsonl`

Two link arrays:

- `linked_entities`: immediate links (`linked` only)
- `pending_entity_links`: unresolved links (`needs_review`, `new_candidate`, `abstain`)

Linked and pending records can include link confidence and candidate link evidence:

- `linked_entities[].link_confidence`
- `pending_entity_links[].candidate_links`

## 6) Graph compatibility

Graph inputs accept claim rows containing both:

- `linked_entities`
- `pending_entity_links`

Current graph mention edges come from immediate links; pending links are retained for future backfill flows.
