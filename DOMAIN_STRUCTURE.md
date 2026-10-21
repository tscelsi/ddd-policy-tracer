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
- Responsibility: Ingest `SourceDocument` and `DocumentChunk` records into a graph database and answer similarity queries using chunk embedding vectors.
- Owns language: Graph Store, Chunk Embedding, Retrieval Query, Similarity Score, SourceDocument node, DocumentChunk node.
- Does not own: extraction, canonicalization, or policy judgment beyond ranking matched chunks.
- Integrations:
  - Upstream from Document Acquisition and Analysis chunk persistence.
  - Persistence adapter: FalkorDB graph instance for node/edge storage.
  - Retrieval operations: cosine-similarity ranking over `chunk.embeddings`.

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
- Retrieval -> FalkorDB Graph Store: persist `SourceDocument` and `DocumentChunk` nodes with `HAS_CHUNK` relationships and chunk embedding properties.
- Retrieval -> Similarity Query Service: rank stored `DocumentChunk` nodes by cosine similarity against a query embedding.

## Aggregate candidates

### Document Acquisition
| Aggregate root | Purpose | Invariants | Consistency boundary |
| --- | --- | --- | --- |
| SourceDocument | Represent a source-scoped acquired document with versions, provenance, normalized text, and processing status. | `source_document_id` is normalized URL and stable within source; `done` requires non-empty `normalized_text` and no uncaught exceptions; each checksum change under same identity creates a new append-only version; each version must persist both `raw_content_ref` and normalized record; each version should persist `published_at` from sitemap `<lastmod>` when available; each version persists audit timestamps (`retrieved_at`, `created_at`, `updated_at`). | Per source identity + version append operation.
| AcquisitionRun | Represent a single manual execution boundary with outcome and counts. | Run has `source_id`, `started_at`; completion sets terminal run status (`completed`, `completed_with_failures`, `failed`) and `finished_at`; bulk ingestion allows partial per-URL success/failure. | Per run lifecycle and aggregate counters/status.

### Retrieval
| Aggregate root | Purpose | Invariants | Consistency boundary |
| --- | --- | --- | --- |
| SourceDocument Graph Node | Represent an ingested source document as a graph node with provenance and text metadata. | Node key is `source_document_id`; one document node may own many chunk nodes; metadata must remain linked to the originating source version. | Per document identity in the graph.
| DocumentChunk Graph Node | Represent one chunk snapshot and its semantic embedding. | `chunk_id` is unique; a `chunk.embedding` array is optional but required for semantic search; each chunk must reference its parent source document via `HAS_CHUNK`. | Per chunk identity and parent document relationship.

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
