# Ubiquitous Language

## Context: Document Acquisition (Core)

### Canonical terms

| Term | Definition | Aliases to avoid | Ambiguity flag |
| --- | --- | --- | --- |
| Document Acquisition | The bounded context responsible for discovering, fetching, extracting, normalizing, and storing source documents. | Ingestion Pipeline, Scraper System (as primary term) | No |
| Source | An external publisher or provider from which documents are acquired. | Website, Feed (when referring to the organization) | No |
| Source Adapter | A concrete adapter for discovering and retrieving documents from a particular source. | Generic SourceProfile (for v1) | No |
| Discovered Document | A as-of-yet unprocessed document that has been discovered but not processed. Contains only a URL and optional published_at date.| Source Document, Canonical Document, Unified Document (v1) | No |
| Source Document | A source-scoped document record with identity, provenance, normalized text, status, and version history. | Canonical Document, Unified Document (v1) | No |
| Published At (`published_at`) | Publication timestamp sourced from sitemap `<lastmod>` for the discovered source URL. | Content Publish Date (if not sourced from sitemap) | Yes (some sitemaps may be update times, not true publish times) |
| Retrieved At (`retrieved_at`) | UTC timestamp when acquisition successfully fetched the document artifact for persistence. | Downloaded At (if ambiguously tied to retries) | No |
| Created At (`created_at`) | UTC timestamp when a source document version record is first persisted. | Inserted At | No |
| Updated At (`updated_at`) | UTC timestamp for the latest persistence update to a source document version record. | Modified At (without persistence context) | No |
| Source Document Identity (`source_document_id`) | The normalized URL used as the stable identifier for a source document within a source. | Raw URL, Global Document ID | No |
| Normalized URL | A URL transformed by v1 rules to remove non-semantic noise while preserving uniqueness-relevant parts. | Clean URL | Yes (query-param handling can evolve per source) |
| Raw Content Artifact | The original fetched content (PDF, HTML, text) stored on disk. | Blob (if implying DB blob) | No |
| Raw Content Reference (`raw_content_ref`) | A filesystem path reference to the stored raw artifact. | Raw Payload | No |
| Normalized Text (`normalized_text`) | The extracted and minimally cleaned text string used by downstream analysis. | Parsed Text, Processed Body | No |
| Checksum | A deterministic content hash used to detect content changes and create new versions. | Fingerprint (if unclear whether URL or content) | No |
| Source Document Version | An append-only historical record of a source document state for a stable source identity. | Overwrite, Update-in-place | No |
| Acquisition Run | A single manual execution boundary for acquiring documents for a source, with aggregate status and counts. | Job, Batch (without explicit run identity) | No |
| Parser Chain | Ordered extraction strategies per content type, with fallback attempts on failure. | Smart Extractor (implies ML sophistication) | No |
| Compliance Policy | Per-source operational rules: robots awareness, throttling, user-agent, retries, and URL/path allow/deny behavior. | Best effort crawl | No |

## Supporting terms

| Term | Definition | Aliases to avoid | Ambiguity flag |
| --- | --- | --- | --- |
| Pending | A source document attempt has been queued or started but is not yet completed. | Processing (as status value) | No |
| Done | A source document meets v1 success criteria: non-empty `normalized_text` and no uncaught exceptions. | Success (as persisted status value) | No |
| Failed | A source document attempt did not complete successfully and has failure metadata. | Error (as status value) | No |
| Completed | An acquisition run completed with no failed items. | Done (run-level) | No |
| Completed With Failures | An acquisition run finished with at least one failed item and at least one successful item. | Partial | No |
| Failed (Run) | An acquisition run failed at the run level (for example startup/system failure) and could not complete normal processing. | Crashed | No |

## Candidate commands and events terminology

### Commands
- `StartAcquisitionRun`
- `BulkIngestSourceUrls`
- `MarkSourceDocumentFailed`

### Events
- `AcquisitionRunStarted`
- `SourceDocumentIngested`
- `SourceDocumentIngestionFailed`
- `AcquisitionRunCompleted`

## Key relationships

- A `Source` owns many `AcquisitionRun` instances.
- An `AcquisitionRun` processes many source URLs.
- A source URL resolves to one `Source Document Identity` (`source_document_id`) within a source.
- A stable `Source Document Identity` owns many append-only `Source Document Version` records.
- Each version references one `Raw Content Artifact` and one `normalized_text` payload.

## Example dialogue

"Start an acquisition run for `australia_institute` using sitemap discovery and a limit of 100 URLs."

"The run is `completed_with_failures`: 86 `SourceDocumentIngested`, 14 `SourceDocumentIngestionFailed`."

"For URL identity `https://example.org/report-x`, checksum changed, so a new source document version was appended and marked `done`."
