"""Deterministic entity judge and resolver status policy."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from .models import CATALOG_ELIGIBLE_ENTITY_TYPES, EntityMention
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


@dataclass(frozen=True)
class CatalogEligibilityDecision:
    """Capture catalog eligibility outcome and diagnostic reason codes."""

    is_eligible: bool
    reason_codes: list[str]


@dataclass(frozen=True)
class CatalogEligibilityPolicy:
    """Decide if one mention is suitable for human-reviewed catalog curation."""

    generic_ambiguous_terms: tuple[str, ...] = (
        "government",
        "department",
        "ministry",
        "agency",
        "commission",
        "council",
        "authority",
        "administration",
        "program",
        "policy",
        "plan",
        "act",
        "bill",
    )

    def evaluate(self, *, mention: EntityMention) -> CatalogEligibilityDecision:
        """Evaluate mention text and type against catalog eligibility invariants."""
        reasons: list[str] = []
        if mention.entity_type not in CATALOG_ELIGIBLE_ENTITY_TYPES:
            reasons.append("ineligible_entity_type")

        normalized = mention.normalized_mention_text.strip()
        lowered = normalized.casefold()

        if len(normalized) < 3:
            reasons.append("too_short")
        if any(character.isdigit() for character in normalized):
            reasons.append("numeric_like")
        if not any(character.isalpha() for character in normalized):
            reasons.append("no_alpha")
        if lowered in self.generic_ambiguous_terms:
            reasons.append("ambiguous_generic")
        if _looks_like_non_unique_org_phrase(normalized):
            reasons.append("ambiguous_generic_org")
        if not _is_proper_noun_surface(mention.mention_text):
            reasons.append("not_proper_noun_surface")

        return CatalogEligibilityDecision(
            is_eligible=not reasons,
            reason_codes=sorted(set(reasons)),
        )


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

    linked_threshold: float = 0.90
    new_candidate_threshold: float = 0.60
    eligibility_policy: CatalogEligibilityPolicy = CatalogEligibilityPolicy()

    def evaluate(
        self,
        *,
        mention: EntityMention,
        retrieval_result: RetrievalResult,
    ) -> ResolutionDecision:
        """Evaluate one mention against retrieval candidates and thresholds."""
        eligibility = self.eligibility_policy.evaluate(mention=mention)
        if not eligibility.is_eligible:
            return ResolutionDecision(
                mention_entity_id=mention.entity_id,
                decision_status="abstain",
                decision_score=0.0,
                reason_codes=[*eligibility.reason_codes, "catalog_ineligible"],
                selected_candidate_key=None,
                top_candidates=[],
            )

        evidence = [
            _candidate_to_evidence(candidate) for candidate in retrieval_result.candidates[:3]
        ]
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
        score = top_candidate.fused_score
        if score >= self.linked_threshold:
            return ResolutionDecision(
                mention_entity_id=mention.entity_id,
                decision_status="linked",
                decision_score=score,
                reason_codes=["threshold_linked"],
                selected_candidate_key=top_candidate.canonical_entity_key,
                top_candidates=evidence,
            )
        if score >= self.new_candidate_threshold:
            return ResolutionDecision(
                mention_entity_id=mention.entity_id,
                decision_status="needs_review",
                decision_score=score,
                reason_codes=["threshold_review"],
                selected_candidate_key=None,
                top_candidates=evidence,
            )
        if _is_valid_new_candidate(mention):
            return ResolutionDecision(
                mention_entity_id=mention.entity_id,
                decision_status="new_candidate",
                decision_score=score,
                reason_codes=["threshold_new_candidate"],
                selected_candidate_key=None,
                top_candidates=evidence,
            )
        return ResolutionDecision(
            mention_entity_id=mention.entity_id,
            decision_status="abstain",
            decision_score=score,
            reason_codes=["threshold_abstain"],
            selected_candidate_key=None,
            top_candidates=evidence,
        )


@dataclass(frozen=True)
class EntityResolutionService:
    """Resolve mentions with hybrid retrieval and deterministic judging."""

    retriever: MentionRetriever
    judge: DeterministicEntityJudge

    def resolve_mentions(
        self,
        *,
        mentions: list[EntityMention],
        top_k: int = 5,
    ) -> list[EntityMention]:
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
    metadata["candidate_links"] = [
        {
            "canonical_entity_key": candidate.canonical_entity_key,
            "canonical_name": candidate.canonical_name,
            "entity_type": candidate.entity_type,
            "confidence_score": candidate.fused_score,
            "lexical_score": candidate.lexical_score,
            "vector_score": candidate.vector_score,
        }
        for candidate in decision.top_candidates
    ]
    metadata["max_candidate_score"] = (
        max(candidate.fused_score for candidate in decision.top_candidates)
        if decision.top_candidates
        else 0.0
    )
    eligibility = CatalogEligibilityPolicy().evaluate(mention=mention)
    metadata["catalog_eligible"] = eligibility.is_eligible
    metadata["catalog_eligibility_reasons"] = eligibility.reason_codes
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


def _is_proper_noun_surface(value: str) -> bool:
    """Check whether mention surface resembles a proper noun phrase."""
    tokens = re.findall(r"[A-Za-z][A-Za-z'&/-]*", value)
    if not tokens:
        return False
    significant = [token for token in tokens if token.casefold() not in {"of", "the", "and", "for"}]
    if not significant:
        return False
    return all(token[0].isupper() for token in significant)


def _looks_like_non_unique_org_phrase(normalized_value: str) -> bool:
    """Detect generic organization phrases that are commonly ambiguous."""
    lowered = normalized_value.casefold()
    tokens = [token for token in re.split(r"\s+", lowered) if token]
    if not tokens:
        return True
    if len(tokens) == 1 and tokens[0] in {
        "government",
        "department",
        "ministry",
        "agency",
        "commission",
        "council",
        "authority",
    }:
        return True
    if lowered.startswith("the ") and len(tokens) <= 2:
        return True
    return False
