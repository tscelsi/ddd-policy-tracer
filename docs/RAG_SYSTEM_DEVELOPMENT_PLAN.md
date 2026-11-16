# Parliamentary Speeches RAG Development Plan

## Goal

Build a question-answering system over Australian parliamentary speeches that
answers from retrieved evidence, exposes where that evidence came from, and
abstains when the corpus does not support an answer.

The active retrieval implementation combines SQLite vector search with
sqlite-vec, FTS5 BM25, and reciprocal-rank fusion. Completion requests extract
date and speaker constraints with the configured answer provider, resolve
speaker names against the Hansard catalog on demand, and apply valid filters
before vector and BM25 candidate limits. This plan prioritizes measurable
retrieval quality and reliable evidence handling before adding more model
complexity.

## Current state and schema findings

The active retriever is
[`ddd_policy_tracer/retrieval/sqlite_retrieval.py`](../ddd_policy_tracer/retrieval/sqlite_retrieval.py).
It provides vector, BM25, and hybrid search. It does not currently expose
metadata filters, reranking, or answer generation. Its FTS5 index is copied
into memory when the retriever is initialized.

The schema declared by
[`ddd_policy_tracer/enrichment/chunking/adapters.py`](../ddd_policy_tracer/enrichment/chunking/adapters.py)
contains:

- `document_chunks`: chunk identity, source and document identity, document
  checksum, chunk order and offsets, text, topics, and metadata.
- `chunk_embeddings`: a sqlite-vec table with 1024-dimensional embeddings.
- `topics` and `metadata` are JSON text fields, not typed filter columns.

These are schema definitions in the repository; no populated SQLite database
was available in the repository to inspect directly. The retrieval filter
contract reads `published_at` and `talker_id` from existing chunk metadata, so
no SQLite columns are added.

Hansard ingestion already has richer data than the chunk record makes readily
filterable. [`SpeechPart`](../ddd_policy_tracer/ingestion/hansard/part_repository.py)
includes date, house, chamber, bill IDs, debate descriptors, talker ID, and
speech-part type. The Hansard conversion service persists a UTC `published_at`
timestamp and stable `talker_id` in source-document metadata, alongside the
talker record.
[`MarkdownChunker`](../ddd_policy_tracer/enrichment/chunking/chunking.py)
carries source-document metadata into chunk metadata under `doc_metadata`.
Online V1 filters support Canberra calendar-date ranges and speaker identities;
other explicitly requested dimensions return a clarification instead of being
silently ignored.

### Metadata filter behavior (implemented)

- The configured answer provider extracts strict date, speaker, unsupported,
  and clarification fields without rewriting the original question.
- Speaker names resolve by canonical name, first-and-last name, or unique
  surname. The Mongo talker catalog loads on first speaker lookup and remains
  cached for the process lifetime (`MONGO_URI` defaults to localhost).
- Inclusive Australia/Canberra date bounds are converted to UTC at the SQLite
  query boundary. Filters apply to Hansard chunks and constrain vector and
  BM25 candidate generation before each candidate limit.
- Unknown or ambiguous speakers, unclear intent, and unsupported constraints
  stop before embedding and retrieval. A valid filtered search with no evidence
  remains an abstention; it is never retried without filters.

## Recommended implementation order

### Phase 0 — Question and evidence contract

#### Audience and purpose

The primary users are researchers, journalists, and parliamentary staff
investigating what was said in Australian parliamentary speeches, by whom, and
when. The system reports what the parliamentary record supports; it is not an
independent policy judge.

#### V1 question families

- **Speaker lookup:** What did a named member say about an issue?
- **Debate discovery:** What was said during a debate on an issue or bill?
- **Comparison:** How did speakers, parties, or arguments differ across bills
  or time?
- **Chronology:** How did debate or arguments develop over time?
- **Explicit stance:** Did a speaker directly state support for or opposition to
  a proposal?

#### Grounding and attribution rules

- Synthesize only what the retrieved speeches support. Distinguish direct
  statements from synthesis, and do not present inferred motives or external
  facts as established by the parliamentary record.
- Label support or opposition only when the speech directly states a position.
  Otherwise report that no explicit position was found. Do not infer stance
  from arguments, party membership, or voting behavior.
- Attribute statements to individual speakers. Describe a party-wide pattern
  only when supported by multiple cited speeches, and disclose material
  disagreement within the party.
- Comparisons and chronology should describe dated, evidenced differences.
  Do not claim a speaker changed their mind or that one event caused another
  unless the record explicitly establishes it.

#### Answer and citation format

Give a concise answer with citations inline, followed by selected supporting
passages. Comparisons may use a compact table. Cite each substantive claim to a
verifiable passage. Show speaker, date, house/chamber, and a stable speech or
chunk reference when available.

#### Uncertainty and corpus coverage

Surface material disagreement and uncertainty rather than smoothing it over.
When the indexed corpus does not establish an answer, say so and suggest a
useful query refinement when possible; do not guess. Disclose the available
corpus date and source coverage. Absence from the index is not proof that
something was never said in Parliament.

**Exit criteria:** this question and evidence contract is the basis for the
evaluation questions and acceptance criteria in Phase 1.

### Phase 1 — Establish a retrieval and answer evaluation set

Create a small, version-controlled set of representative questions with
relevant chunk IDs or source passages, expected metadata constraints where
applicable, and unanswerable examples. Include exact-term queries, paraphrases,
named-speaker and bill queries, date-bounded questions, and questions needing
multiple passages.

An initial 25-case draft is in
[`tests/fixtures/rag_validation_cases.json`](../tests/fixtures/rag_validation_cases.json).
It covers the five Phase 0 question families, with 20 answerable cases and five
cases that should abstain. Its evidence IDs and labels are candidate
annotations and must be reviewed before reporting evaluation metrics.

Record a baseline for hybrid retrieval before changing ranking behavior.
Measure:

- **Recall@k**: whether relevant evidence appears in the candidate set.
- **MRR or nDCG@k**: how highly relevant evidence is ranked.
- **Latency**: retrieval and total response time.
- Once generation exists: citation precision/support and answer faithfulness,
  including abstention quality on unanswerable questions.

**Exit criteria:** the evaluation can be run repeatedly and produces comparable
metrics for the current retriever.

### Phase 2 — Preserve and filter parliamentary metadata

First retain the fields required for useful Hansard searches. Candidate
metadata includes:

- Speech date (`published_at` or an explicit date field).
- Speaker name and stable talker ID; party and electorate where present.
- House and chamber.
- Bill IDs.
- Debate category and debate titles.
- Speech-part type.

Keep stable provenance and use explicit types for fields that need filtering.
For frequently queried or range-filtered fields (especially date, house,
chamber, speaker/talker, and bill), prefer typed/indexed columns or a normalized
metadata table over repeatedly parsing JSON. Preserve extensibility for
infrequent metadata.

Add a typed filter contract to vector, BM25, and hybrid retrieval. Ensure that
filters constrain candidate generation, not merely the final top-k list. A
filter applied after nearest-neighbor top-k can return too few results even
when matching chunks exist farther down the unfiltered ranking. Define and
test behavior for absent metadata and combinations of filters.

**Exit criteria:** each supported field survives Hansard ingestion and chunking;
filtered vector, BM25, and hybrid searches return only matching chunks; the
evaluation set confirms filtered recall and expected empty-result behavior.

### Phase 3 — Add a second-stage reranker

Use the hybrid retriever for broad candidate generation, then rerank a larger
candidate pool with a query–passage cross-encoder. Keep the candidate count and
final result count separate. Make the reranker replaceable so local or hosted
models can be compared without changing the retrieval contract.

Evaluate the reranked results against the hybrid baseline. A reranker should
not be considered an improvement solely because a few hand-picked queries
look better: check Recall@k before reranking and MRR/nDCG after reranking, and
report latency and resource cost. Reranking cannot recover evidence missing
from the candidate pool, so monitor candidate recall.

**Exit criteria:** reranking improves a chosen ranking metric on the held-out
questions without exceeding an agreed latency/resource budget, with a way to
disable or bypass it.

### Phase 4 — Generate grounded answers and citations

Add a thin answer-generation use case after retrieval quality is measurable.
Pass selected evidence with provenance to the model and have it return
structured claims, each linked to one or more retrieved chunk IDs, or an
explicit abstention. Validate every cited ID against the supplied evidence,
then render inline citation numbers and the source list from stored metadata
rather than model-generated identifiers. Retain the exact source passage with
each citation so callers can display or verify it. For synthesis questions,
retrieve and retain multiple distinct passages rather than repeatedly
selecting near-duplicate chunks from one speech.

The first generation adapter is `OllamaAnswerGenerator`, which uses the
Ollama Python client and requests schema-constrained JSON. Its model and local
Ollama host are configurable; the default model is `llama3.1:8b`. The Ollama
server must be running with the selected model available. The generator is
injected into `AnswerSynthesizer`, keeping citation validation and rendering
independent of the model provider.

Support explicit abstention when the retrieved evidence is missing, weak, or
conflicting. Test citations against actual chunk text and source metadata, not
only the surface shape of generated answers.

**Exit criteria:** supported questions have verifiable citations; unsupported
questions reliably abstain; answer and citation metrics are included in the
evaluation report.

### Phase 5 — Improve context assembly and reliability

After the core path is evaluated:

- Reduce redundant overlapping chunks and consider adjacent-chunk expansion
  when it improves evidence completeness.
- Handle contradictory statements and temporal changes explicitly; preserve
  dates and avoid presenting older statements as current positions.
- Track embedding model, chunking configuration, index/schema version, and
  reranker/generator model versions for reproducibility.
- Measure index build/startup time, database size, retrieval latency, and
  memory use as the corpus grows. Revisit the in-memory FTS copy if its startup
  or memory cost becomes material.
- Add operational logging for query, filter, candidate counts, stage timings,
  and model versions without logging sensitive user data unnecessarily.

## Suggested vertical slices

1. **Evaluation baseline:** build test fixtures and a small judged dataset;
   record current hybrid results.
2. **Metadata preservation:** carry the selected Hansard fields from
   `SpeechPart` through `SourceDocument` and `DocumentChunk`, with regression
   tests.
3. **Filterable retrieval:** implement and test the same filter semantics for
   vector, BM25, and hybrid paths.
4. **Reranking:** add a reranker port/contract, one adapter, evaluation, and
   latency reporting.
5. **Grounded QA:** add answer orchestration, citations, and abstention tests.

## Decisions to make before implementation

- Which parliamentary metadata filters are required for the first release?
- Which question categories and reference examples define success?
- What are the acceptable latency and deployment constraints for a reranker
  and answer model?
- Should answers include verbatim quotations, concise paraphrases, or both?
- How should the system present conflicting or time-dependent statements?

## Definition of done

- Retrieval and answer changes are evaluated on a repeatable, representative
  question set.
- Metadata required by supported filters is preserved, typed where useful, and
  filterable consistently across retrieval methods.
- Reranking is retained only if it improves measured quality within the
  agreed operating budget.
- Generated answers cite traceable speech evidence and abstain when evidence
  is insufficient.
- Tests cover core retrieval behavior, filter combinations, provenance,
  citations, and abstention.
