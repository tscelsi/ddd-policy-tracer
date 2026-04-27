## Status

Closed: v1 scope in this PRD is implemented.

## Problem Statement

We need a practical, well-architected data acquisition tool that can reliably collect policy-relevant documents from external publishers (starting with The Australia Institute), normalize extracted text into a consistent shape, and preserve provenance/history for future analysis. Today, there is no operationally simple pipeline that supports repeatable manual runs, versioned document history, and low-friction source onboarding without overengineering.

## Solution

Build a v1 Document Acquisition system focused on orchestration and reliability rather than perfect extraction quality. The system will run manually via CLI, discover candidate URLs from The Australia Institute sitemap, fetch raw artifacts, extract text through parser fallbacks, apply minimal normalization, and persist versioned `SourceDocument` records in SQLite while storing raw artifacts on disk. It will enforce source compliance controls and produce a small, explicit domain event stream for observability and future integrations.

## User Stories

1. As a research operator, I want to run acquisition manually from a CLI, so that I can control execution during early development.
2. As a research operator, I want to target a specific source in a run, so that I can ingest one publisher at a time.
3. As a research operator, I want to set a URL limit for a run, so that I can test safely on small batches.
4. As a research operator, I want a dry-run mode, so that I can inspect discovery behavior without persisting data.
5. As a platform engineer, I want sitemap-based discovery for The Australia Institute, so that source onboarding is simple and deterministic.
6. As a platform engineer, I want URL allowlist/excludelist filtering, so that non-document pages are skipped.
7. As a platform engineer, I want content-type validation, so that only ingestible documents are processed.
8. As a compliance-conscious team member, I want robots-aware and throttled fetching, so that source policies are respected.
9. As an operator, I want retries with backoff for transient fetch failures, so that runs are resilient.
10. As an operator, I want per-URL independent processing in bulk ingest, so that one failure does not block the rest.
11. As an operator, I want run-level aggregate statuses (`completed`, `completed_with_failures`, `failed`), so that outcomes are clear.
12. As a domain modeler, I want `SourceDocument` to be source-scoped, so that we avoid premature cross-source entity resolution.
13. As a domain modeler, I want normalized URL identity (`source_document_id`), so that duplicate detection is stable.
14. As a data engineer, I want tracking query params stripped but meaningful params retained, so that identity avoids noise without collisions.
15. As a data engineer, I want append-only versioning on checksum change, so that document history is preserved.
16. As a researcher, I want provenance fields (`published_at`, `retrieved_at`, source URL, fetch method), so that data lineage is auditable.
17. As a researcher, I want audit fields (`created_at`, `updated_at`) on persisted versions, so that record lifecycle is traceable.
18. As a researcher, I want raw artifacts retained on disk, so that extraction can be reproduced later.
19. As a data engineer, I want raw artifact references stored in SQLite, so that metadata storage stays lightweight.
20. As a product owner, I want minimal normalization only (no aggressive cleanup), so that we ship quickly and avoid semantic loss.
21. As a product owner, I want a simple success criterion (non-empty `normalized_text`, no uncaught exceptions), so that status rules stay clear.
22. As an operator, I want failed document attempts recorded with reason and retry count, so that troubleshooting is straightforward.
23. As a system integrator, I want core events emitted for run start/completion and per-document success/failure, so that future workflows can subscribe cleanly.
24. As an architect, I want clear ports for repository, artifact storage, fetcher, and parsers, so that adapters can evolve independently.
25. As an architect, I want one bounded context in v1 (`Document Acquisition`), so that implementation remains focused.
26. As a future contributor, I want a design that can add source #2 without rewriting core orchestration, so that onboarding remains fast.
27. As a QA engineer, I want deterministic tests around identity normalization and versioning, so that regressions are caught early.
28. As a QA engineer, I want service-layer tests with fakes for repositories and adapters, so that business behavior is verified quickly.
29. As a policy analyst (future consumer), I want consistent `SourceDocument` output shape across inputs, so that analysis code does not need per-source logic.
30. As a maintainer, I want parser fallback attempts and outcomes logged, so that extraction quality issues can be diagnosed.
31. As an operator, I want skipped URLs counted and visible, so that allowlist tuning is evidence-based.

## Implementation Decisions

- Architecture follows a DDD + hexagonal approach with one v1 bounded context: Document Acquisition.
- Domain aggregates:
  - `SourceDocument` as the core aggregate root.
  - `AcquisitionRun` as a lightweight operational aggregate.
- Domain invariants:
  - `source_document_id` is normalized URL and stable per source.
  - `done` requires non-empty `normalized_text` and no uncaught exceptions.
  - Changed checksum under same identity creates a new append-only version.
  - Each persisted version requires both normalized record and raw artifact reference.
  - Each persisted version stores `published_at` from sitemap `<lastmod>` when available.
  - Each persisted version stores audit timestamps (`retrieved_at`, `created_at`, `updated_at`).
- Command model for orchestration:
  - `StartAcquisitionRun`
  - `BulkIngestSourceUrls`
  - `MarkSourceDocumentFailed`
- Event model (minimal):
  - `AcquisitionRunStarted`
  - `SourceDocumentIngested`
  - `SourceDocumentIngestionFailed`
  - `AcquisitionRunCompleted`
- Status model:
  - SourceDocument: `pending`, `done`, `failed` (+ `error_reason`, `retry_count`).
  - AcquisitionRun: `completed`, `completed_with_failures`, `failed`.
- Source integration strategy for v1:
  - Concrete Australia Institute adapter.
  - Discovery via `sitemap.xml`.
  - Sitemap `<lastmod>` maps directly to `published_at`.
  - Conservative URL allowlist/excludelist + content-type checks.
- URL normalization strategy:
  - Normalize scheme/host casing.
  - Remove fragments.
  - Trim trailing slash (except root).
  - Keep query params by default but strip known tracking keys and empty params.
- Storage strategy:
  - SQLite for metadata, status, runs, and version records.
  - Disk storage for raw fetched artifacts with `raw_content_ref` pointers.
- Parsing/normalization strategy:
  - Best-effort parser chain by content type (PDF/HTML/text) with fallbacks.
  - Minimal text normalization only (UTF-8 normalization, whitespace/control char cleanup, preserve paragraph structure where possible).
- Operator interface:
  - Single CLI entrypoint for manual runs with `source`, optional `limit`, and optional `dry-run`.
- Assumptions:
  - Cross-source deduplication/entity resolution is not needed in v1.
  - The Australia Institute URLs are unique enough when normalized by agreed rules.
  - Manual operation is sufficient before scheduling is introduced.

## Testing Decisions

- Good tests validate external behavior and domain outcomes, not internal implementation details.
- Testing emphasis follows high-gear first:
  - Domain unit tests for invariants, identity normalization decisions, status transitions, and versioning behavior.
  - Service-layer unit tests with fake repository/artifact store/fetcher/parser ports for orchestration commands and event emission.
  - Contract-style adapter tests for SQLite repository and disk artifact adapter.
  - Focused integration tests for sitemap discovery + content-type filtering and parser-chain fallback behavior.
- Prior art in codebase:
  - No established test suite yet; create baseline project testing conventions and reusable fixtures with this feature.
- Python testing stack and quality gates:
  - `pytest` for unit/integration tests.
  - `ruff` for linting/format checks.
  - `mypy` for type checks on domain/service boundaries.
  - Coverage target assumption: start with meaningful coverage on domain/service modules (explicit numeric target to be set once CI baseline exists).
- TDD order:
  - Start with failing domain/service behavior tests.
  - Implement minimal code to satisfy one behavior at a time.
  - Refactor only with tests green.
- Additional testing assumptions:
  - Validate run aggregate statuses under mixed success/failure batches.
  - Validate idempotency behavior when same URL/checksum is reprocessed.
  - Validate version append behavior when checksum changes.

## Out of Scope

- Policy influence correlation, impact scoring, or amendment linkage.
- Cross-source canonical document matching/entity resolution.
- Automated scheduling/cron orchestration.
- Multi-tenant access control, auth, and external API surface.
- Advanced NLP/OCR tuning beyond best-effort parser fallback.
- Cloud object storage implementation (e.g., S3) in v1.
- Production-scale observability platform integrations beyond core domain events/logging.

## Further Notes

- Ubiquitous language is standardized around US spelling (`normalized_text`).
- This PRD intentionally optimizes for delivery speed and architectural cleanliness over extraction perfection.
- The design keeps adapter seams explicit so adding source #2 can drive the next level of abstraction (e.g., generalized source profiles) only when needed.
