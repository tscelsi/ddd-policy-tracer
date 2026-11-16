Status: ready-for-agent

# Agentic metadata filters for parliamentary speech retrieval

## Problem Statement

Researchers often ask questions constrained to a particular speaker or time
period, but the current RAG completion flow searches the whole indexed Hansard
corpus. Relevant speeches can be crowded out by unrelated candidates, and
filtering only after retrieval would not recover matching chunks excluded by
the vector or BM25 candidate limits.

The user needs retrieval constraints inferred from the question and applied
consistently before candidate ranking, without changing the question used for
semantic and lexical search. If the system cannot safely apply a requested
constraint, it must ask for clarification or report an explicit error rather
than silently search more broadly.

## Solution

Add a bounded, structured metadata-filter extraction step to each completion
request. Use the already configured answer-generation provider (Ollama or
OpenAI) to extract supported date and speaker constraints, resolve speaker
names against the Hansard talker catalog, and pass typed constraints to both
vector and BM25 candidate generation.

Support inclusive Australia/Canberra calendar-date ranges and speaker
identities. Apply multiple speakers as OR and combine speaker and date
dimensions with AND. Constrain date filters to Hansard speech chunks. If intent
is unsupported, unclear, or a requested speaker is unknown or ambiguous, return
a structured clarification response without embedding, retrieval, or answer
generation. Preserve normal abstention when valid filters produce no evidence;
never retry without those filters.

### Agreed test seams

1. At the completion request/use-case boundary, use controlled extraction and
   speaker-resolution collaborators to verify that resolved filters reach
   retrieval and that clarification bypasses retrieval and generation.
2. At the SQLite hybrid-retrieval boundary, use a temporary database to verify
   that date and speaker constraints exclude nonmatching vector and BM25
   candidates before their candidate limits.

These seams test observable behavior while avoiding live LLM and MongoDB
dependencies.

## User Stories

1. As a researcher, I want to ask what a named member said about a topic, so
   that retrieval focuses on that member's speeches.
2. As a researcher, I want to identify a speaker by their canonical name, so
   that I do not need to know their internal Hansard identifier.
3. As a researcher, I want to identify a speaker by their first and last name,
   so that ordinary name forms select the correct talker.
4. As a researcher, I want to identify a speaker by a unique surname, so that
   concise questions can still narrow the search.
5. As a researcher, I want ambiguous surnames to trigger a clarification, so
   that the system does not attribute evidence to the wrong member.
6. As a researcher, I want unknown speaker names to trigger a clarification,
   so that the system does not silently return speeches by other people.
7. As a researcher, I want to request one or more speakers, so that matching
   chunks from any requested speaker can be considered.
8. As a researcher, I want to ask about speeches on a specific calendar date,
   so that retrieval includes speeches from that date only.
9. As a researcher, I want to specify a start and end date, so that retrieval
   covers the requested inclusive interval.
10. As a researcher, I want to specify only a start date or only an end date,
    so that open-ended time ranges are supported.
11. As a researcher, I want date phrases without a year to use the current
    Canberra calendar year, so that the request has a deterministic
    interpretation.
12. As a researcher, I want “last month” and “last year” to refer to the
    previous complete calendar period, so that the boundary is predictable.
13. As a researcher, I want “past N” date phrases to use a rolling range, so
    that recent-history requests end on the current Canberra date.
14. As a researcher, I want date bounds interpreted in Australia/Canberra time,
    so that local parliamentary dates remain inclusive across UTC offsets and
    daylight-saving changes.
15. As a researcher, I want date constraints to apply only to Hansard speech
    chunks, so that other source types are not incorrectly interpreted as
    parliamentary records.
16. As a researcher, I want a speaker and date constraint to apply together,
    so that results satisfy both parts of my request.
17. As a researcher, I want each requested speaker to be an alternative within
    the speaker filter, so that any of those speakers may match.
18. As a researcher, I want metadata constraints applied before vector
    candidate limits, so that matching chunks are not lost to unrelated nearest
    neighbors.
19. As a researcher, I want metadata constraints applied before BM25 candidate
    limits, so that matching lexical results are not displaced by unrelated
    chunks.
20. As a researcher, I want the original question preserved for embedding,
    lexical search, and answer generation, so that filter extraction does not
    distort the question's meaning.
21. As a researcher, I want ordinary content questions to proceed without
    filters, so that adding filter extraction does not change general search
    behavior.
22. As a researcher, I want an explicitly requested but unsupported filter to
    produce a clarification, so that the system does not quietly ignore my
    constraint.
23. As a researcher, I want unclear filter intent to produce a clarification,
    so that I can refine my request before evidence is retrieved.
24. As a researcher, I want an empty result under valid filters to remain empty,
    so that the system does not broaden the search without my consent.
25. As a researcher, I want the answer flow to abstain normally when filtered
    retrieval finds no evidence, so that it does not invent an answer.
26. As an API caller, I want clarification represented with an explicit
    `clarification_required` flag and question, so that I can render a useful
    follow-up prompt.
27. As an API caller, I want clarification responses to contain no citations or
    synthesized answer, so that response fields accurately reflect that no
    retrieval took place.
28. As an API caller, I want malformed structured filter output to fail
    explicitly, so that invalid constraints are not treated as an unfiltered
    request.
29. As an API caller, I want provider and Mongo connectivity failures surfaced
    as service errors, so that operational problems are distinguishable from
    valid empty results.
30. As an operator, I want speaker lookup to load the Mongo talker catalog only
    when needed and cache it for the process lifetime, so that ordinary
    questions avoid unnecessary Mongo work.
31. As an operator, I want filter-extraction time reported separately from
    embedding, retrieval, and generation, so that the additional completion
    cost can be observed.
32. As an operator, I want logs to report timings and outcome state without
    logging the question or speech passages, so that operational visibility
    does not expose query content.

## Implementation Decisions

- Treat metadata extraction as a single bounded structured-output request per
  completion. It is not an autonomous tool-using agent and does not rewrite the
  user's original question.
- Reuse the configured answer-generation provider and its selected model:
  Ollama or OpenAI. Require a strict, validated structured response.
- The extraction contract contains inclusive date bounds, requested speaker
  names, explicitly requested unsupported constraints, and an optional
  clarification question. Missing dates are null; name and unsupported-filter
  values are non-empty strings.
- Resolve relative periods against the current Australia/Canberra date. A date
  lacking a year uses the current Canberra year. “Last month” and “last year”
  mean the previous complete calendar period; “past N” means a rolling period.
- Use a typed retrieval-filter contract containing zero or more stable talker
  IDs and optional date bounds. Multiple talker IDs combine with OR; the speaker
  dimension and date dimension combine with AND.
- Interpret speaker names deterministically against a lazily loaded Mongo
  talker catalog. Accept normalized canonical-name, first-and-last-name, or
  unique-last-name matches. Cache the catalog until process restart. Unknown or
  ambiguous names require clarification, not a guessed identity.
- Apply date constraints only to Hansard rows. Convert inclusive Canberra
  calendar dates to UTC bounds at the persistence-query boundary, using a
  half-open interval for the ending date.
- Apply the same filters inside vector and BM25 candidate generation, before
  each candidate limit, then fuse the filtered rankings with the existing
  reciprocal-rank fusion behavior.
- Do not retry unfiltered if valid constraints yield no matches. Let the
  existing no-evidence abstention behavior handle the empty result.
- If extraction marks the request unclear or reports unsupported constraints,
  or speaker resolution needs clarification, return before embedding,
  retrieval, and synthesis.
- Expose clarification in the completion response with
  `clarification_required` and `clarification_question`; no answer or citations
  are returned for that path.
- Surface invalid structured extraction as an explicit client/service error and
  provider or Mongo connectivity failures as service-unavailable errors. Do not
  turn failures into success-shaped unfiltered retrieval.
- Normalize Hansard publication timestamps to UTC during ingestion and
  backfill, treating Mongo-naive timestamps as UTC, so stored values compare
  consistently with calculated date bounds.
- Report filter extraction and speaker resolution separately in completion
  timings. Avoid logging questions and passages.
- No new database columns are required for online filtering: use the existing
  typed `published_at` and `talker_id` chunk fields.

## Testing Decisions

- Tests should assert externally observable behavior through public contracts:
  the filter extracted from structured output, the constraints passed to
  retrieval, the chunks returned by filtered candidate searches, and the HTTP
  clarification/error response. Avoid asserting private helper structure or
  SQL text.
- Use controlled provider clients for Ollama and mocked HTTP responses for
  OpenAI. Do not require live LLM credentials or network access.
- Use a fake talker repository to verify deterministic name forms, ambiguity,
  unknown-name clarification, lazy catalog loading, and process-lifetime cache
  behavior without requiring MongoDB.
- Use a temporary SQLite database with representative vector embeddings and
  BM25 content to verify that filters are applied before candidate limits,
  speaker alternatives combine correctly, date and speaker constraints
  intersect, date filters exclude non-Hansard rows, and no-match behavior does
  not broaden retrieval.
- Test inclusive Canberra dates at UTC conversion boundaries, open-ended
  ranges, daylight-saving offsets, missing-year interpretation, and relative
  period semantics.
- Test completion behavior for no extracted filters, valid filters,
  unsupported constraints, unclear intent, ambiguous and unknown speakers,
  empty filtered results, and explicit provider failures.
- Test the HTTP contract for answer responses, clarification responses,
  provider errors, and timing headers.
- Prior art includes the existing completion API tests, SQLite hybrid
  retrieval tests, structured-output adapter tests, and Hansard ingestion
  metadata tests. Extend those behaviors rather than duplicating their
  infrastructure.

## Out of Scope

- Metadata constraints for party, electorate, house, chamber, bill IDs, debate
  category, or debate titles. Explicit requests for these unsupported
  dimensions must not be silently ignored.
- LLM-based retrieval rewriting, query expansion, reranking, or an autonomous
  tool-using agent.
- Changing the answer synthesis prompt, citation rules, or abstention policy,
  except to preserve abstention when filtered retrieval returns no evidence.
- Replacing SQLite, sqlite-vec, FTS5, or reciprocal-rank fusion.
- Live synchronization or refresh of the Mongo talker catalog during process
  lifetime.
- Automatically relaxing valid filters when they produce too few or no
  results.
- Running the Mongo-to-SQLite historical metadata backfill as part of a
  completion request.

## Further Notes

- This spec captures the confirmed planning decisions for the metadata-filter
  feature. Implementation work in the current branch has already introduced
  the extraction, resolution, retrieval, and clarification path; this spec
  records the intended behavior and regression criteria for that work.
- The live Mongo database and configured LLM providers were not exercised in
  the implementation validation. Provider adapters and repository behavior
  should continue to be verified with deterministic tests; deployment
  configuration still needs appropriate Mongo and provider availability.
- The project has no root glossary or ADR files for this area at the time of
  writing. Terms used here follow the existing Hansard, chunk, completion, and
  retrieval vocabulary.
