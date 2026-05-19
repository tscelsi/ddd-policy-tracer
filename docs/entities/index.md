# Entities Docs

This section explains the entities workflow from zero context.

If you only remember one idea, remember this:

- **The runtime entity catalog is the source of truth for known entities.**
- New mentions are extracted from text, then checked against the catalog.
- Catalog changes require explicit human review decisions.

## Read in this order

- [Workflow Overview](workflow.md): simple end-to-end mental model and diagram.
- [Runbook](runbook.md): exact commands to run the pipeline.
- [Data Contracts](data-contracts.md): what each file/table contains.
- [Troubleshooting](troubleshooting.md): what to do when runs fail.

## Current behavior (quick summary)

- One extractor strategy: `robust-ensemble`.
- Optional resolution mode (catalog + vectors): deterministic statuses.
- Optional resolution mode (catalog + vectors): probabilistic candidate retrieval plus threshold statuses.
- Statuses: `linked`, `needs_review`, `new_candidate`, `abstain`.
- Review queue in SQLite: `review_items`, `review_events`.
- Manual review decisions can edit canonical fields directly on `review_items`.
- Only `PERSON`, `ORG`, and `JURISDICTION` mentions are catalog-eligible.
- `POLICY` and `PROGRAM` mentions are still extracted, but not promoted to catalog.
- Promotion writes manually approved entities back into the runtime catalog and refreshes vectors.
- Claim canonicalization links only `linked` entities immediately and stores unresolved as `pending_entity_links`.
- Graph materialization uses weighted mentions links with `link_confidence` for thresholded querying.
