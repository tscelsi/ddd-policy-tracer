# Creating and reviewing retrieval evaluation cases

This guide covers the human workflow for adding query-to-evidence cases and
deciding which cases are trustworthy enough to include in retrieval
evaluation. It does not cover answer-quality evaluation:
`expected_answer_points` may remain in existing cases, but this workflow does
not generate, edit, validate, or score them.

## What a case means

An answerable case connects one query to one set of required evidence chunk
IDs. Every ID in `gold_evidence_chunk_ids` must be retrieved in the top 10 for
the case to pass. Extra retrieved chunks are allowed. The IDs are conjunctive:
they are all required, not interchangeable alternatives.

If either of two chunks would independently support a query, do not list both
as required. Make the query specific enough to target one evidence set, or
reject it as ambiguous.

Cases are stored in
[`tests/fixtures/rag_validation_cases.json`](../tests/fixtures/rag_validation_cases.json).
The main fields used by this workflow are:

| Field | Meaning |
|---|---|
| `case_id` | Unique case identifier. |
| `question` | Query sent to retrieval. |
| `answerable` | Whether the reviewed corpus contains the evidence. |
| `gold_evidence_chunk_ids` | Required chunk IDs for an answerable case; empty for an unanswerable case. |
| `review_status` | `draft`, `approved`, or `rejected`. Only approved cases enter evaluation. |
| `generation_provenance` | Generator provider, model, and prompt version, when generated. |
| `review_provenance` | Reviewer, decision, review timestamp, and optionally a note. |
| `abstention_reason` | For unanswerable cases, why the current corpus cannot answer. |
| `absence_confirmed_by` | Reviewer who explicitly confirmed corpus absence for an approved unanswerable case. |

Other case and fixture fields are preserved. In particular,
`expected_answer_points` are outside this retrieval workflow.

## Prerequisites

1. Make sure the chunk JSONL file reflects the corpus you intend to evaluate.
   The default path is `data/chunks_spacy.jsonl`.
2. Make sure the configured query-generation provider is available:
   - Ollama is the default. Start the server and make the configured model
     available; `OLLAMA_HOST` and `OLLAMA_MODEL` can override defaults.
   - To use OpenAI, set `RAG_ANSWER_GENERATOR=openai` and provide
     `OPENAI_API_KEY`; `OPENAI_MODEL` optionally selects the model.
   - Settings can be placed in the repository `.env` file or exported in the
     process environment. Existing environment variables take precedence.
3. Use a human reviewer who can verify both that the evidence supports the
   query and that the wording is specific and unambiguous.

The generator proposes two or three queries grounded in the selected passages.
Generated text is not ground truth: a person must review every case.

## Generate answerable cases

From the repository root, start the interactive chunk viewer in review mode:

```bash
uv run python scripts/visualise_chunks.py --review \
  --chunks-path data/chunks_spacy.jsonl
```

You can narrow the chunks shown with the available filters. For example:

```bash
uv run python scripts/visualise_chunks.py --review \
  --chunks-path data/chunks_spacy.jsonl \
  --source-document-id YOUR_DOCUMENT_ID
```

The viewer also accepts `--source-id`, `--chunk-id`, and `--contains`.
`--fixture-path PATH` selects a different case fixture; normally, use the
default so cases are added to the shared evaluation fixture.

In the viewer:

1. Navigate with **Enter** for the next chunk, **b** for the previous chunk,
   **Space** for a random chunk, and **q** to quit. Inspect the chunk text and
   its metadata (including source document, date,
   speaker, and available debate details).
2. Press **x** to select each chunk that is jointly necessary to support the
   question. Press **x** again to deselect a chunk. All selected chunks will
   become required IDs for each query variant.
3. Press **g** to ask the configured model to draft query variants.
4. Enter a reviewer name. Without one, the variants are not saved.
5. Review each query against the complete selected text and metadata:
   - **a — approve** only if the query is answerable from the selected
     evidence, clearly identifies the intended topic/speaker/date where
     needed, is not ambiguous, and genuinely adds useful coverage.
   - **e — edit** to improve or narrow the wording, then review the edited
     query again.
   - **r — reject** an ambiguous, unsupported, duplicated, or otherwise poor
     query; provide a rejection reason.
   - **d — draft** to save it without including it in evaluation.

Each variant is stored as its own case with the same selected evidence IDs.
Distinct variants may both be approved, but avoid approving near-duplicates
just to increase the case count. If the selected chunks do not all support the
question together, reject it or adjust the selection and generate again.

The fixture is rewritten atomically after each saved decision. If the process
stops, decisions already saved remain in the fixture; unsaved variants must be
reviewed again. The fixture records generation provenance and reviewer
provenance for approved and rejected cases.

## Review a saved draft

The interactive viewer currently reviews each generated variant immediately;
it does not have a command for reopening existing draft cases. To approve or
reject a saved draft later, edit that case in the fixture after independently
verifying its evidence IDs against the chunk data.

For approval, set `review_status` to `approved` and add `review_provenance`
with the reviewer and an ISO-8601 UTC timestamp:

```json
"review_status": "approved",
"review_provenance": {
  "reviewer": "Reviewer name",
  "status": "approved",
  "reviewed_at": "2026-11-18T12:00:00+00:00"
}
```

For rejection, set `review_status` to `rejected` and include a non-empty
`note` explaining the decision in `review_provenance`. Do not change a case to
approved merely to get it into the test run. If an approved query or its
evidence IDs change, review and approve it again.

## Add a confirmed unanswerable case

In review mode, press **u** to author an unanswerable case. This path is for a
specific query that a human has checked against the current corpus—not for a
query that simply returned no results.

Provide:

1. A clear query.
2. A reason the current corpus cannot answer it (for example, the relevant
   date is outside corpus coverage).
3. The exact confirmation `YES` when prompted, after checking corpus absence.
4. A reviewer name.
5. An explicit approval, rejection, or draft decision.

An unanswerable case has `answerable: false`, an empty
`gold_evidence_chunk_ids` list, and a non-empty `abstention_reason`. An
approved unanswerable case must also have `absence_confirmed_by`. These cases
are reported separately; they do not have a pass/fail threshold yet.

## Run retrieval evaluation

Only approved answerable cases are evaluated. The retriever must find every
required chunk within the top 10. The report lists each target's rank; a
missing chunk ID is reported as stale and fails the case, requiring human
re-review. Stale IDs are never silently remapped or skipped. The evaluation
also reports approved unanswerable cases separately. Cases are retrieved
through `RagCompletionService.retrieve`, including the configured LLM metadata
filtering and speaker resolution used by normal RAG queries.

Run the evaluation script against `data/new4.db`:

```bash
uv run python scripts/evaluate_rag_retrieval.py
```

> **Warning:** Running this evaluation calls the configured LLM for metadata
> filtering on every case. Hosted providers may incur usage fees.

Each run appends one JSON record per evaluated case to
`logs/rag_validation.log`, including the retrieved chunk IDs, ranks, passage
text, source metadata, and any retrieval error. Set `RAG_VALIDATION_LOG_PATH`
to write the log to a different location. The `logs/` directory is ignored by
Git. `--database`, `--fixture`, and `--log-file` can override the default paths.

An exception while retrieving one case is recorded as a failure and does not
stop later cases from running. The script exits with a non-zero status if any
answerable case fails, including cases with retrieval errors.

The run fails if there are no approved answerable cases, so approve at least
one valid answerable case before running it. Make sure the indexed database
matches the chunk data and corpus version used to select the gold IDs.

## Review checklist

Before approving an answerable case, confirm:

- The query asks about a fact actually stated in the selected chunk or chunks.
- Every listed chunk is needed for the query, and every listed chunk supports
  it.
- The query is sufficiently specific about the person, event, subject, and
  date when those details disambiguate the target.
- There is no plausible competing interpretation or interchangeable target
  chunk; narrow the wording or reject the case if there is.
- The question is meaningfully different from already-approved cases.
- The selected chunk IDs exist in the corpus version used for evaluation.

Before approving an unanswerable case, confirm:

- You searched or inspected the relevant current corpus, rather than assuming
  absence from a failed retrieval.
- The query is specific enough to establish what evidence would answer it.
- The `abstention_reason` describes the corpus limitation, and the confirmer is
  recorded.
