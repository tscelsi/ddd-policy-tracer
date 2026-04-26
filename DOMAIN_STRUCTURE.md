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

## Context map (draft)

- External Source Web (Australia Institute) -> Document Acquisition: sitemap and document retrieval over HTTP (anti-corruption via source adapter + URL/content filters).
- Document Acquisition -> SQLite Adapter: repository/UoW persistence port implementation.
- Document Acquisition -> Disk Artifact Adapter: raw artifact storage port implementation.
- Operator CLI -> Document Acquisition: command-based manual run initiation.
- Document Acquisition -> Future Analysis Context: downstream consumption of normalized source document records/events (planned, not in v1 scope).

## Aggregate candidates

### Document Acquisition
| Aggregate root | Purpose | Invariants | Consistency boundary |
| --- | --- | --- | --- |
| SourceDocument | Represent a source-scoped acquired document with versions, provenance, normalized text, and processing status. | `source_document_id` is normalized URL and stable within source; `done` requires non-empty `normalized_text` and no uncaught exceptions; each checksum change under same identity creates a new append-only version; each version must persist both `raw_content_ref` and normalized record. | Per source identity + version append operation.
| AcquisitionRun | Represent a single manual execution boundary with outcome and counts. | Run has `source_id`, `started_at`; completion sets terminal run status (`completed`, `completed_with_failures`, `failed`) and `finished_at`; bulk ingestion allows partial per-URL success/failure. | Per run lifecycle and aggregate counters/status.

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
