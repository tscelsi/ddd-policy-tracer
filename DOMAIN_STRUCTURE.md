# Domain Structure

## Scope and subdomains

- Core: Document acquisition orchestration and normalization for policy-relevant documents.
- Supporting: Storage adapters (SQLite metadata store, disk raw artifact store), CLI operations, compliance throttling controls.
- Generic: Parsing libraries, HTTP client behavior, filesystem operations, checksum/hash utilities.

## Bounded contexts

### Document Acquisition
- Responsibility: Discover source URLs, fetch content, extract/normalize text, persist source document versions, and emit run/document events.
- Owns language: Source Document, Acquisition Run, normalized_text, source_document_id, parser chain, compliance policy.
- Does not own: Influence analysis, policy correlation, cross-source entity resolution/canonicalization.
- Integrations:
  - Upstream external source websites/sitemaps (Australia Institute first).
  - Persistence adapters: SQLite for metadata/version state, disk storage for raw artifacts.
  - Operator interface: CLI manual trigger.

### Analysis (Planned, now active)
- Responsibility: Transform `SourceDocumentVersion` records into chunks, extracted claims/entities, enrichment metadata, and graph-ready assertions.
- Owns language: Document Chunk, Claim Candidate, Entity Mention, Assessment, extraction lineage.
- Does not own: Source discovery/fetching and raw artifact capture.
- Integrations:
  - Upstream from Document Acquisition persisted versions.
  - Persistence adapters: SQLite/JSONL for chunks and extraction outputs.
  - Operator interface: CLI chunk/extract commands.
  - Downstream: knowledge graph materialization and analyst query services.

### Retrieval (new, minimal v1)
- Responsibility: Retrieve ranked evidence from indexed `DocumentChunk` records using vector, BM25, and hybrid search; apply typed speaker and date constraints before candidate ranking.
- Owns language: Retrieval Query, Retrieval Filters, Evidence Chunk, Similarity Score, Speaker Identity, Metadata Filter Intent.
- Does not own: filter-intent extraction, answer synthesis, extraction, canonicalization, or policy judgment beyond ranking matched chunks.
- Integrations:
  - Upstream from Document Acquisition and Analysis chunk persistence.
  - Persistence adapter: SQLite chunk records, sqlite-vec embeddings, and an in-memory FTS5 index.
  - Retrieval operations: filtered vector and BM25 candidate generation followed by reciprocal-rank fusion.

### Answering (application service)
- Responsibility: Extract supported retrieval constraints, resolve requested Hansard speakers, coordinate evidence retrieval, and synthesize cited answers or clarifications.
- Owns language: Completion Request, Metadata Filter Intent, Clarification Required, Answer Citation.
- Does not own: parliamentary speaker identity data, chunk persistence, or answer-provider infrastructure.
- Integrations:
  - Outbound to the configured Ollama or OpenAI provider for filter extraction and answer synthesis.
  - Outbound to the Hansard talker catalog only when speaker resolution is requested.
  - Outbound to Retrieval with the original question and validated `Retrieval Filters`.

## Context map (draft)

- External Source Web (Australia Institute) -> Document Acquisition: sitemap and document retrieval over HTTP (anti-corruption via source adapter + URL/content filters).
- Document Acquisition -> SQLite Adapter: repository/UoW persistence port implementation.
- Document Acquisition -> Disk Artifact Adapter: raw artifact storage port implementation.
- Operator CLI -> Document Acquisition: command-based manual run initiation.
- Document Acquisition -> Future Analysis Context: downstream consumption of normalized source document records/events (planned, not in v1 scope).
- Document Acquisition -> Analysis: consume persisted source document versions as analysis input.
- Analysis -> Chunk Persistence Adapter: persist `DocumentChunk` records for each document version.
- Analysis -> Extraction Persistence Adapter: persist `ClaimCandidate` and `EntityMention` records.
- Analysis -> Future Graph Context: emit graph-ready assertions and provenance metadata.
- Answering -> Retrieval: pass the original question and typed speaker/date constraints to candidate generation.
- Answering -> Hansard Talker Catalog: lazily resolve canonical names, first-last names, and unique surnames to stable talker IDs.
- Retrieval -> SQLite Chunk Store: apply valid constraints to vector and BM25 candidates before their respective limits.
- Retrieval -> Answering: return hybrid-ranked evidence for citation-backed synthesis.

## Aggregate candidates

### Document Acquisition
| Aggregate root | Purpose | Invariants | Consistency boundary |
| --- | --- | --- | --- |
| SourceDocument | Represent a source-scoped acquired document with versions, provenance, normalized text, and processing status. | `source_document_id` is normalized URL and stable within source; `done` requires non-empty `normalized_text` and no uncaught exceptions; each checksum change under same identity creates a new append-only version; each version must persist both `raw_content_ref` and normalized record; each version should persist `published_at` from sitemap `<lastmod>` when available; each version persists audit timestamps (`retrieved_at`, `created_at`, `updated_at`). | Per source identity + version append operation.
| AcquisitionRun | Represent a single manual execution boundary with outcome and counts. | Run has `source_id`, `started_at`; completion sets terminal run status (`completed`, `completed_with_failures`, `failed`) and `finished_at`; bulk ingestion allows partial per-URL success/failure. | Per run lifecycle and aggregate counters/status.

### Retrieval
| Aggregate root | Purpose | Invariants | Consistency boundary |
| --- | --- | --- | --- |
| Retrieval Filters (value object) | Constrain one completion's candidate search to speaker identities and/or an inclusive date range. | Speaker IDs are OR alternatives; speaker and date dimensions combine with AND; date bounds are Canberra calendar dates; requested constraints are never silently dropped. | Per completion request.
| Indexed Document Chunk (owned by Analysis) | Supply searchable text, provenance, metadata, and optional vector embedding to retrieval. | Filters apply to Hansard chunks before vector and BM25 candidate limits; returned records retain source metadata for citation. | Per indexed chunk snapshot.

## Commands and domain events (candidates)

- Command: `StartAcquisitionRun`
  - Starts a manual run for one source and initializes run state.
- Command: `BulkIngestSourceUrls`
  - Ingests many discovered URLs under one run with independent per-URL outcomes.
- Command: `MarkSourceDocumentFailed`
  - Records failed processing attempt metadata for a source document URL/identity.

- Event: `AcquisitionRunStarted`
- Event: `SourceDocumentIngested`
- Event: `SourceDocumentIngestionFailed`
- Event: `AcquisitionRunCompleted`

## Open questions

- What exact Australia Institute URL allowlist/excludelist patterns should be configured initially?
- Which concrete parser stack order should be default for PDF/HTML/text in v1?
- What retry/backoff defaults should be enforced for document fetch and parser failures?
- Should run-level counts include skipped URLs explicitly as a first-class metric in v1?

## Top risks

- Sitemap discovery may include significant non-target URLs, causing noisy ingestion without tuned filters.
- URL normalization mistakes can produce identity splits (duplicates) or accidental collisions.
- Parser fallback chains may yield inconsistent quality on malformed documents despite non-empty text success rule.
- Manual-only operation can reduce freshness discipline unless run cadence is operationally documented.
