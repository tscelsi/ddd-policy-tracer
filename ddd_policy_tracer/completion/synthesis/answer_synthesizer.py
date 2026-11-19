"""Synthesize answers from parliamentary evidence with traceable citations."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from ddd_policy_tracer.completion.synthesis.generators.base_generator import AbstractAnswerGenerator
from ddd_policy_tracer.completion.synthesis.models import (
    AnswerCitation,
    EvidenceChunk,
    SynthesizedAnswer,
)


class InvalidEvidenceReferenceError(ValueError):
    """Signal that generated claims cite evidence outside the retrieved passages."""


class AnswerSynthesizer:
    """Validate generated evidence references and render source-backed citations."""

    def __init__(self, generator: AbstractAnswerGenerator) -> None:
        """Use the supplied generator to synthesize answers from retrieved evidence."""
        self._generator = generator

    def synthesize(
        self,
        question: str,
        evidence: Sequence[EvidenceChunk],
    ) -> SynthesizedAnswer:
        """Return an answer whose inline citations resolve to retrieved speech passages."""
        if not question.strip():
            raise ValueError("question must not be empty")

        evidence_by_id = {chunk.chunk_id: chunk for chunk in evidence}
        if len(evidence_by_id) != len(evidence):
            raise ValueError("evidence must not contain duplicate chunk IDs")
        if not evidence:
            reason = "No evidence passages were retrieved."
            return SynthesizedAnswer(
                text=f"I cannot answer from the retrieved evidence. {reason}",
                citations=(),
                abstained=True,
                abstention_reason=reason,
            )

        draft = self._generator.generate(question, evidence)
        if draft.abstained:
            if draft.claims:
                raise ValueError("an abstaining answer must not contain claims")
            if not draft.abstention_reason or not draft.abstention_reason.strip():
                raise ValueError("an abstaining answer must include a reason")
            return SynthesizedAnswer(
                text=f"I cannot answer from the retrieved evidence. {draft.abstention_reason}",
                citations=(),
                abstained=True,
                abstention_reason=draft.abstention_reason,
            )
        if not draft.claims:
            raise ValueError("a non-abstaining answer must contain at least one claim")

        citation_numbers: dict[str, int] = {}
        rendered_claims: list[str] = []
        for claim in draft.claims:
            if not claim.text.strip():
                raise ValueError("answer claims must not be empty")
            if not claim.evidence_chunk_ids:
                raise InvalidEvidenceReferenceError("every answer claim must cite evidence")

            markers: list[str] = []
            claim_chunk_ids: set[str] = set()
            for chunk_id in claim.evidence_chunk_ids:
                chunk = evidence_by_id.get(chunk_id)
                if chunk is None:
                    raise InvalidEvidenceReferenceError(
                        f"answer claim cites evidence that was not retrieved: {chunk_id}",
                    )
                if chunk_id in claim_chunk_ids:
                    continue
                claim_chunk_ids.add(chunk_id)
                if chunk_id not in citation_numbers:
                    citation_numbers[chunk_id] = len(citation_numbers) + 1
                markers.append(f"[{citation_numbers[chunk_id]}]")
            rendered_claims.append(f"{claim.text.strip()} {' '.join(markers)}")

        citations = tuple(
            _citation(number, evidence_by_id[chunk_id])
            for chunk_id, number in citation_numbers.items()
        )
        answer_text = " ".join(rendered_claims)
        answer_text += "\n\nSources:\n" + "\n".join(
            _render_citation(citation) for citation in citations
        )
        return SynthesizedAnswer(
            text=answer_text,
            citations=citations,
            abstained=False,
        )


def _citation(number: int, chunk: EvidenceChunk) -> AnswerCitation:
    """Build citation details from the evidence record rather than generated text."""
    date, house, chamber = _document_context(chunk.source_document_id)
    return AnswerCitation(
        number=number,
        chunk_id=chunk.chunk_id,
        source_document_id=chunk.source_document_id,
        passage=chunk.chunk_text,
        speaker=_speaker_name(chunk.metadata),
        date=date,
        house=house,
        chamber=chamber,
    )


def _speaker_name(metadata: str | None) -> str | None:
    """Read the recorded speaker name from Hansard chunk metadata."""
    if metadata is None:
        return None
    parsed = json.loads(metadata)
    if not isinstance(parsed, dict):
        return None
    document_metadata = parsed.get("doc_metadata")
    if not isinstance(document_metadata, dict):
        return None
    talker = document_metadata.get("talker")
    if not isinstance(talker, dict):
        return None
    name = talker.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    first_name = talker.get("first_name")
    last_name = talker.get("last_name")
    if isinstance(first_name, str) and isinstance(last_name, str):
        full_name = f"{first_name.strip()} {last_name.strip()}".strip()
        return full_name or None
    return None


def _document_context(source_document_id: str) -> tuple[str | None, str | None, str | None]:
    """Extract date, house, and chamber when encoded in the Hansard document ID."""
    parts = source_document_id.split("_")
    date_index = next(
        (index for index, part in enumerate(parts) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", part)),
        None,
    )
    date = parts[date_index] if date_index is not None else None
    if date_index is None or date_index + 2 >= len(parts):
        return date, None, None

    house = {
        "hor": "House of Representatives",
        "sen": "Senate",
    }.get(parts[date_index + 1])
    chamber_code = parts[date_index + 2]
    chamber = chamber_code.replace("-", " ").title() if house and chamber_code != "none" else None
    return date, house, chamber


def _render_citation(citation: AnswerCitation) -> str:
    """Render a numbered source line from persisted citation provenance."""
    details = [
        citation.speaker,
        citation.date,
        (
            f"{citation.house} ({citation.chamber})"
            if citation.house and citation.chamber
            else citation.house
        ),
    ]
    source_details = [detail for detail in details if detail]
    source_details.extend(
        [
            f"source `{citation.source_document_id}`",
            f"chunk `{citation.chunk_id}`",
        ],
    )
    return f"[{citation.number}] " + " | ".join(source_details)
