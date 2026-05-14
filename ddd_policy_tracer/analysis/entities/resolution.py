"""Deterministic entity judge and resolver status policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Protocol

from .models import EntityMention
from .retrieval import RetrievalCandidate, RetrievalResult

DecisionStatus = Literal["linked", "needs_review", "new_candidate", "abstain"]


@dataclass(frozen=True)
class CandidateEvidence:
    """Capture candidate ranking evidence used by resolver decisions."""

    canonical_entity_key: str
    canonical_name: str
    entity_type: str
    fused_score: float
    lexical_score: float
    vector_score: float


@dataclass(frozen=True)
class ResolutionDecision:
    """Represent deterministic resolution output for one mention."""

    mention_entity_id: str
    decision_status: DecisionStatus
    decision_score: float
    reason_codes: list[str]
    selected_candidate_key: str | None
    top_candidates: list[CandidateEvidence]


class MentionRetriever(Protocol):
    """Retrieve ranked catalog candidates for one mention."""

    def retrieve(
        self,
        *,
        mention_text: str,
        normalized_mention_text: str,
        mention_entity_type: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Return deterministic candidate ranking for one mention."""


@dataclass(frozen=True)
class DeterministicEntityJudge:
    """Score retrieval outputs into deterministic resolver statuses."""

    linked_threshold: float = 0.85
    review_threshold: float = 0.60

    def evaluate(
        self,
        *,
        mention: EntityMention,
        retrieval_result: RetrievalResult,
    ) -> ResolutionDecision:
        """Evaluate one mention against retrieval candidates and thresholds."""
        evidence = [_candidate_to_evidence(candidate) for candidate in retrieval_result.candidates[:3]]
        if not retrieval_result.candidates:
            return ResolutionDecision(
                mention_entity_id=mention.entity_id,
                decision_status="abstain",
                decision_score=0.0,
                reason_codes=["no_candidates"],
                selected_candidate_key=None,
                top_candidates=evidence,
            )

        top_candidate = retrieval_result.candidates[0]
        score, reason_codes = self._calibrated_score(
            mention=mention,
            top_candidate=top_candidate,
            second_candidate=(
                retrieval_result.candidates[1]
                if len(retrieval_result.candidates) > 1
                else None
            ),
        )

        if score >= self.linked_threshold:
            return ResolutionDecision(
                mention_entity_id=mention.entity_id,
                decision_status="linked",
                decision_score=score,
                reason_codes=[*reason_codes, "threshold_linked"],
                selected_candidate_key=top_candidate.canonical_entity_key,
                top_candidates=evidence,
            )
        if score >= self.review_threshold:
            return ResolutionDecision(
                mention_entity_id=mention.entity_id,
                decision_status="needs_review",
                decision_score=score,
                reason_codes=[*reason_codes, "threshold_review"],
                selected_candidate_key=top_candidate.canonical_entity_key,
                top_candidates=evidence,
            )
        if _is_valid_new_candidate(mention):
            return ResolutionDecision(
                mention_entity_id=mention.entity_id,
                decision_status="new_candidate",
                decision_score=score,
                reason_codes=[*reason_codes, "threshold_new_candidate"],
                selected_candidate_key=None,
                top_candidates=evidence,
            )
        return ResolutionDecision(
            mention_entity_id=mention.entity_id,
            decision_status="abstain",
            decision_score=score,
            reason_codes=[*reason_codes, "threshold_abstain"],
            selected_candidate_key=None,
            top_candidates=evidence,
        )

    def _calibrated_score(
        self,
        *,
        mention: EntityMention,
        top_candidate: RetrievalCandidate,
        second_candidate: RetrievalCandidate | None,
    ) -> tuple[float, list[str]]:
        """Build calibrated deterministic score and reason trace codes."""
        score = top_candidate.fused_score
        reasons = ["base_fused_score"]

        if top_candidate.entity_type == mention.entity_type:
            score += 0.05
            reasons.append("entity_type_match")
        else:
            score -= 0.10
            reasons.append("entity_type_mismatch")

        if top_candidate.canonical_name.casefold() == mention.normalized_mention_text.casefold():
            score += 0.05
            reasons.append("exact_name_match")

        if second_candidate is None:
            score += 0.05
            reasons.append("single_candidate")
        else:
            margin = top_candidate.fused_score - second_candidate.fused_score
            if margin >= 0.15:
                score += 0.05
                reasons.append("clear_margin")
            elif margin < 0.05:
                score -= 0.05
                reasons.append("ambiguous_margin")

        return max(0.0, min(1.0, round(score, 6))), reasons


@dataclass(frozen=True)
class EntityResolutionService:
    """Resolve mentions with hybrid retrieval and deterministic judging."""

    retriever: MentionRetriever
    judge: DeterministicEntityJudge

    def resolve_mentions(self, *, mentions: list[EntityMention], top_k: int = 5) -> list[EntityMention]:
        """Resolve each mention and attach review-ready decision metadata."""
        resolved: list[EntityMention] = []
        for mention in mentions:
            retrieval_result = self.retriever.retrieve(
                mention_text=mention.mention_text,
                normalized_mention_text=mention.normalized_mention_text,
                mention_entity_type=mention.entity_type,
                top_k=top_k,
            )
            decision = self.judge.evaluate(
                mention=mention,
                retrieval_result=retrieval_result,
            )
            resolved.append(apply_resolution_decision(mention=mention, decision=decision))
        return resolved


def apply_resolution_decision(
    *,
    mention: EntityMention,
    decision: ResolutionDecision,
) -> EntityMention:
    """Apply one resolution decision to mention metadata and canonical key."""
    metadata = dict(mention.metadata or {})
    metadata["decision_status"] = decision.decision_status
    metadata["decision_score"] = decision.decision_score
    metadata["reason_codes"] = list(decision.reason_codes)
    metadata["selected_candidate_key"] = decision.selected_candidate_key
    metadata["top_candidates"] = [
        {
            "canonical_entity_key": candidate.canonical_entity_key,
            "canonical_name": candidate.canonical_name,
            "entity_type": candidate.entity_type,
            "fused_score": candidate.fused_score,
            "lexical_score": candidate.lexical_score,
            "vector_score": candidate.vector_score,
        }
        for candidate in decision.top_candidates
    ]
    canonical_key = mention.canonical_entity_key
    if decision.decision_status == "linked":
        canonical_key = decision.selected_candidate_key
    return replace(
        mention,
        canonical_entity_key=canonical_key,
        metadata=metadata,
    )


def _candidate_to_evidence(candidate: RetrievalCandidate) -> CandidateEvidence:
    """Map one retrieval candidate to compact serialized evidence payload."""
    return CandidateEvidence(
        canonical_entity_key=candidate.canonical_entity_key,
        canonical_name=candidate.canonical_name,
        entity_type=candidate.entity_type,
        fused_score=candidate.fused_score,
        lexical_score=candidate.lexical_score,
        vector_score=candidate.vector_score,
    )


def _is_valid_new_candidate(mention: EntityMention) -> bool:
    """Decide whether mention text quality qualifies as a new candidate."""
    normalized = mention.normalized_mention_text.strip()
    if len(normalized) < 4:
        return False
    if not any(character.isalpha() for character in normalized):
        return False
    return mention.confidence >= 0.5
