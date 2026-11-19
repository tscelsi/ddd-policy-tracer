"""Review and evaluate query-to-evidence retrieval cases."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict, cast

from ddd_policy_tracer.completion.retrieval.service_layer import (
    ChunkIdentityIndex,
    RetrievedChunk,
)
from ddd_policy_tracer.completion.service_layer import RagCompletionService

ReviewStatus = Literal["draft", "approved", "rejected"]


class RetrievalCase(TypedDict, total=False):
    """Represent one reviewed query and its conjunctive evidence targets."""

    case_id: str
    question: str
    answerable: bool
    gold_evidence_chunk_ids: list[str]
    review_status: ReviewStatus
    abstention_reason: str


@dataclass(frozen=True)
class RetrievedChunkRecord:
    """Capture one ranked passage and its provenance for evaluation review."""

    rank: int
    chunk_id: str
    chunk_text: str
    source_document_id: str
    metadata: str | None


@dataclass(frozen=True)
class CaseEvaluation:
    """Report one answerable query's target ranks and corpus validity."""

    case_id: str
    passed: bool
    target_ranks: dict[str, int | None]
    stale_chunk_ids: tuple[str, ...]
    retrieved_chunks: tuple[RetrievedChunkRecord, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class EvaluationReport:
    """Separate answerable retrieval scores from unanswerable case review."""

    cases: tuple[CaseEvaluation, ...]
    unanswerable_case_ids: tuple[str, ...]

    @property
    def answerable_total(self) -> int:
        """Return the number of approved answerable queries evaluated."""
        return len(self.cases)

    @property
    def answerable_passed(self) -> int:
        """Return approved answerable queries with every target in the top ten."""
        return sum(case.passed for case in self.cases)

    @property
    def answerable_failed(self) -> int:
        """Return answerable retrieval failures, including stale annotations."""
        return self.answerable_total - self.answerable_passed


def validate_case(case: Mapping[str, object]) -> None:
    """Validate a query case's status and answerability evidence contract."""
    for field in ("case_id", "question"):
        value = case.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")

    answerable = case.get("answerable")
    if type(answerable) is not bool:
        raise ValueError("answerable must be a boolean")

    status = case.get("review_status")
    if status not in ("draft", "approved", "rejected"):
        raise ValueError("review_status must be draft, approved, or rejected")

    evidence_ids = case.get("gold_evidence_chunk_ids")
    if (
        not isinstance(evidence_ids, list)
        or any(not isinstance(chunk_id, str) or not chunk_id.strip() for chunk_id in evidence_ids)
        or len(evidence_ids) != len(set(evidence_ids))
    ):
        raise ValueError("gold_evidence_chunk_ids must contain unique non-empty chunk IDs")

    if answerable and not evidence_ids:
        raise ValueError("answerable cases require at least one gold_evidence_chunk_ids value")
    if not answerable:
        if evidence_ids:
            raise ValueError("unanswerable cases must not have gold_evidence_chunk_ids")
        reason = case.get("abstention_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("unanswerable cases require a human-reviewed abstention_reason")
        if status == "approved":
            confirmer = case.get("absence_confirmed_by")
            if not isinstance(confirmer, str) or not confirmer.strip():
                raise ValueError("approved unanswerable cases require absence_confirmed_by")

    provenance = case.get("review_provenance")
    if status in ("approved", "rejected"):
        if not isinstance(provenance, Mapping):
            raise ValueError(f"{status} cases require review_provenance")
        if (
            provenance.get("status") != status
            or not isinstance(
                provenance.get("reviewer"),
                str,
            )
            or not str(provenance.get("reviewer")).strip()
        ):
            raise ValueError(f"{status} cases require reviewer and matching review_provenance")
        if status == "rejected":
            note = provenance.get("note")
            if not isinstance(note, str) or not note.strip():
                raise ValueError("rejected cases require a rejection reason in review_provenance")

    generation_provenance = case.get("generation_provenance")
    if generation_provenance is not None:
        if not isinstance(generation_provenance, Mapping) or any(
            not isinstance(generation_provenance.get(field), str)
            or not str(generation_provenance.get(field)).strip()
            for field in ("provider", "model", "prompt_version")
        ):
            raise ValueError("generation_provenance requires provider, model, and prompt_version")


def transition_review_status(
    case: Mapping[str, object],
    *,
    status: ReviewStatus,
    reviewer: str,
    note: str | None = None,
) -> dict[str, object]:
    """Record an explicit human decision while preserving case annotations."""
    validate_case(case)
    if not reviewer.strip():
        raise ValueError("reviewer must not be empty")
    if status == "rejected" and (note is None or not note.strip()):
        raise ValueError("rejected cases require a rejection reason in note")

    updated = dict(case)
    updated["review_status"] = status
    provenance: dict[str, str] = {
        "reviewer": reviewer.strip(),
        "status": status,
        "reviewed_at": datetime.now(UTC).isoformat(),
    }
    if note is not None and note.strip():
        provenance["note"] = note.strip()
    updated["review_provenance"] = provenance
    validate_case(cast(Mapping[str, object], updated))
    return updated


def append_evaluation_log(report: EvaluationReport, log_path: Path) -> None:
    """Append one JSON record per case, including retrieved evidence and errors."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        for case in report.cases:
            log_file.write(
                json.dumps(
                    {
                        "case_id": case.case_id,
                        "passed": case.passed,
                        "target_ranks": case.target_ranks,
                        "stale_chunk_ids": case.stale_chunk_ids,
                        "error": case.error,
                        "retrieved_chunks": [
                            {
                                "rank": chunk.rank,
                                "chunk_id": chunk.chunk_id,
                                "chunk_text": chunk.chunk_text,
                                "source_document_id": chunk.source_document_id,
                                "metadata": chunk.metadata,
                            }
                            for chunk in case.retrieved_chunks
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
            )


async def evaluate_cases(
    cases: Sequence[Mapping[str, object]],
    *,
    completion_service: RagCompletionService,
    chunk_index: ChunkIdentityIndex,
) -> EvaluationReport:
    """Evaluate approved queries using metadata-filtered RAG evidence."""
    approved_answerable: list[Mapping[str, object]] = []
    approved_unanswerable: list[str] = []
    for case in cases:
        validate_case(case)
        if case["review_status"] != "approved":
            continue
        if case["answerable"]:
            approved_answerable.append(case)
        else:
            approved_unanswerable.append(str(case["case_id"]))

    if not approved_answerable:
        raise ValueError("No approved answerable retrieval cases are available to evaluate")

    annotated_ids = sorted(
        {
            chunk_id
            for case in approved_answerable
            for chunk_id in cast(list[str], case["gold_evidence_chunk_ids"])
        },
    )
    corpus_chunk_ids = chunk_index.existing_chunk_ids(annotated_ids)
    results: list[CaseEvaluation] = []
    for case in approved_answerable:
        case_id = str(case["case_id"])
        gold_ids = cast(list[str], case["gold_evidence_chunk_ids"])
        stale_ids = tuple(sorted(set(gold_ids) - corpus_chunk_ids))
        try:
            ranked_chunks: Sequence[RetrievedChunk] = await completion_service.retrieve(
                str(case["question"]),
                limit=10,
            )
        except Exception as exc:
            results.append(
                CaseEvaluation(
                    case_id=case_id,
                    passed=False,
                    target_ranks=dict.fromkeys(gold_ids),
                    stale_chunk_ids=stale_ids,
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )
            continue
        retrieved_chunks = tuple(
            RetrievedChunkRecord(
                rank=rank,
                chunk_id=chunk.chunk_id,
                chunk_text=chunk.chunk_text,
                source_document_id=chunk.source_document_id,
                metadata=chunk.metadata,
            )
            for rank, chunk in enumerate(ranked_chunks, start=1)
        )
        retrieved_ranks = {chunk.chunk_id: chunk.rank for chunk in retrieved_chunks}
        target_ranks = {chunk_id: retrieved_ranks.get(chunk_id) for chunk_id in gold_ids}
        passed = not stale_ids and all(rank is not None for rank in target_ranks.values())
        results.append(
            CaseEvaluation(
                case_id=case_id,
                passed=passed,
                target_ranks=target_ranks,
                stale_chunk_ids=stale_ids,
                retrieved_chunks=retrieved_chunks,
            ),
        )

    return EvaluationReport(
        cases=tuple(results),
        unanswerable_case_ids=tuple(approved_unanswerable),
    )
