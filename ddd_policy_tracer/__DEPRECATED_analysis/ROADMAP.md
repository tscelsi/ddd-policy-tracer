# Analysis Roadmap

This roadmap breaks the analysis pipeline into small vertical slices so we can ship incrementally with high confidence.

## Current status

- Stage 1 chunking is implemented.
  - Deterministic chunk IDs.
  - Version binding to `source_document_id` + `checksum`.
  - SQLite and filesystem chunk persistence.
  - Basic `chunk` CLI command.

## Recommended next start (lowest risk)

Start with claim extraction in a non-LLM, deterministic v1.

Why this first:

- Directly supports policy influence analysis.
- Works on current chunks immediately.
- Easier to audit and test than entity resolution at first.

## Phase plan

### Phase 1: Claim extraction v1 (non-LLM)

Goal: produce `ClaimCandidate` records from chunks with evidence spans.

Scope:

- Sentence segmentation.
- Rule-based claim-likelihood scoring.
- Basic claim typing (`descriptive`, `normative`, `forecast`, `quantitative`).
- Confidence score from deterministic features.
- SQLite/filesystem persistence.

Exit criteria:

- Deterministic outputs for same input chunk.
- Every claim includes `chunk_id` and offsets.
- Unit tests with gold fixtures pass.

### Phase 2: Entity extraction v1 (non-LLM)

Goal: extract entity mentions from chunks (`POLICY`, `ORG`, `PERSON`, `JURISDICTION`, `PROGRAM`).

Scope:

- Rule patterns and dictionaries.
- Optional lightweight NER model behind an adapter.
- Mention-level persistence with spans and confidence.

Exit criteria:

- Mention extraction precision acceptable on seed fixtures.
- No unresolved mention is silently canonicalized.

### Phase 3: Enrichment layer

Goal: add retrieval metadata for chunks and extracted objects.

Scope:

- Embeddings (chunk-level first).
- Topic tags.
- Extraction lineage fields (`extractor_version`, model id if used).

Exit criteria:

- Enrichment output can be regenerated from same inputs.

### Phase 4: Assertion and relation inference

Goal: map claims against policies (`supports`, `contradicts`, etc.) as `Assessment` records.

Scope:

- Candidate policy matching.
- Stance classification.
- Required evidence linkage for non-neutral stances.

Exit criteria:

- No non-neutral assessment without evidence chunks.

### Phase 5: Graph materialization and serving

Goal: build queryable graph and analyst workflows.

Scope:

- Upsert graph nodes/edges with provenance.
- Add canned query recipes and diagnostics.

Exit criteria:

- Core policy influence queries run with explainable evidence trails.

## Backlog of optional enhancements

- Clause-level claim extraction instead of sentence-level only.
- Active-learning loop for human-reviewed claim labels.
- LLM-assisted extraction fallback for low-confidence edge cases.
- Confidence calibration with held-out evaluation set.

## Recommended immediate tasks

1. Add `claim_models.py` with `ClaimCandidate` dataclass and strict fields.
2. Add `claim_extraction.py` with deterministic sentence + rules pipeline.
3. Add claim persistence adapters (SQLite and JSONL) mirroring chunk adapters.
4. Add `extract-claims` CLI command and test fixture corpus.
