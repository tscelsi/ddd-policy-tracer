# Analysis Context

This folder owns downstream analysis of acquired `SourceDocumentVersion` records.

Current focus is a staged pipeline:

1. Chunk source document versions.
2. Extract structured claims and entities from chunks.
3. Enrich chunks and extracted objects with metadata (embeddings, topics, confidence, lineage).
4. Build auditable assertions and, later, a knowledge graph.

## Why this context exists

The acquisition context already gives strong provenance (`source_document_id`, `checksum`, `published_at`, `retrieved_at`, `raw_content_ref`).

The analysis context should preserve that provenance while adding:

- retrieval-friendly chunk structure,
- traceable extraction outputs,
- explainable influence and contradiction analysis.

## Design principles

- Evidence first: every extracted fact points back to chunk offsets.
- Version binding: every downstream record ties to document `checksum`.
- Determinism first: start with reproducible NLP/rule-based extraction.
- LLM optionality: use only as an enhancement layer, not a hard dependency.
- Rebuildability: relation inference is separate from graph materialization.

## Document map

- `PIPELINE.md`: end-to-end architecture, diagrams, stage contracts, and quality gates.
- `ROADMAP.md`: phased implementation plan with start recommendations.
