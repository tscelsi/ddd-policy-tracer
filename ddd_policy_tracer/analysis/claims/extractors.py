"""Rule-based claim extraction strategies for one document chunk."""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256

from ddd_policy_tracer.analysis.chunking_models import DocumentChunk

from .models import ClaimCandidate
from .ports import ClaimExtractor

_SENTENCE_END_RE = re.compile(r"[.!?]+")
_QUANTITATIVE_RE = re.compile(
    r"("  # start capture for readability
    r"\b\d+(?:[.,]\d+)?%\b"
    r"|\$\s?\d+(?:[.,]\d+)?"
    r"|\b\d+(?:[.,]\d+)?\s*(?:billion|million|tonnes|tons|kt|mt)\b"
    r"|\b\d{4}\b"
    r")",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class RuleBasedClaimExtractorConfig:
    """Configure weighted cue scoring for deterministic extraction."""

    threshold: float = 0.8
    policy_modality_weight: float = 1.0
    quantitative_weight: float = 0.8
    causal_impact_weight: float = 0.6
    attribution_reporting_weight: float = 0.3
    extractor_version: str = "rules-v1"


@dataclass(frozen=True)
class RuleBasedSentenceClaimExtractor(ClaimExtractor):
    """Extract sentence-bounded claim candidates using weighted cues."""

    config: RuleBasedClaimExtractorConfig = RuleBasedClaimExtractorConfig()

    def extract(self, *, chunk: DocumentChunk) -> list[ClaimCandidate]:
        """Return deterministic claim candidates for one source chunk."""
        claims: list[ClaimCandidate] = []
        for sentence_text, start_char, end_char in _split_sentences_with_offsets(
            chunk.chunk_text,
        ):
            score = self._score_sentence(sentence_text)
            if score < self.config.threshold:
                continue

            normalized_claim_text = _normalize_claim_text(sentence_text)
            claim_id = _build_claim_id(
                chunk_id=chunk.chunk_id,
                start_char=start_char,
                end_char=end_char,
                normalized_claim_text=normalized_claim_text,
                extractor_version=self.config.extractor_version,
            )
            claims.append(
                ClaimCandidate(
                    claim_id=claim_id,
                    chunk_id=chunk.chunk_id,
                    source_id=chunk.source_id,
                    source_document_id=chunk.source_document_id,
                    document_checksum=chunk.document_checksum,
                    start_char=start_char,
                    end_char=end_char,
                    evidence_text=sentence_text,
                    normalized_claim_text=normalized_claim_text,
                    confidence=min(1.0, score),
                    claim_type=None,
                    extractor_version=self.config.extractor_version,
                ),
            )
        return claims

    def count_processed_sentences(self, *, chunk: DocumentChunk) -> int:
        """Count sentence units processed from one chunk payload."""
        return len(_split_sentences_with_offsets(chunk.chunk_text))

    def _score_sentence(self, sentence_text: str) -> float:
        """Score one sentence using weighted cue category matches."""
        lowered = sentence_text.lower()
        score = 0.0

        if _contains_phrase(lowered, _POLICY_MODALITY_CUES):
            score += self.config.policy_modality_weight
        if _contains_quantitative_cue(sentence_text):
            score += self.config.quantitative_weight
        if _contains_phrase(lowered, _CAUSAL_IMPACT_CUES):
            score += self.config.causal_impact_weight
        if _contains_phrase(lowered, _ATTRIBUTION_REPORTING_CUES):
            score += self.config.attribution_reporting_weight

        return score


_POLICY_MODALITY_CUES = (
    "must",
    "should",
    "need to",
    "required to",
    "shall",
    "have to",
    "ought to",
)

_CAUSAL_IMPACT_CUES = (
    "leads to",
    "lead to",
    "results in",
    "result in",
    "causes",
    "cause",
    "increases",
    "increase",
    "reduces",
    "reduce",
    "decreases",
    "decrease",
    "weakens",
    "weaken",
    "improves",
    "improve",
)

_ATTRIBUTION_REPORTING_CUES = (
    "argues",
    "argue",
    "finds",
    "find",
    "shows",
    "show",
    "reports",
    "report",
    "states",
    "state",
    "according to",
)


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    """Return true when text includes at least one phrase cue."""
    return any(phrase in text for phrase in phrases)


def _contains_quantitative_cue(text: str) -> bool:
    """Return true when text includes numeric/quantitative claim cues."""
    return _QUANTITATIVE_RE.search(text) is not None


def _split_sentences_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Split text into sentence-like units with stable character offsets."""
    segments: list[tuple[str, int, int]] = []
    cursor = 0
    text_length = len(text)

    while cursor < text_length:
        match = _SENTENCE_END_RE.search(text, cursor)
        if match is None:
            end = text_length
        else:
            end = match.end()

        raw_segment = text[cursor:end]
        stripped = raw_segment.strip()
        if stripped:
            leading_ws = len(raw_segment) - len(raw_segment.lstrip())
            trailing_ws = len(raw_segment) - len(raw_segment.rstrip())
            start_char = cursor + leading_ws
            end_char = end - trailing_ws
            segments.append((text[start_char:end_char], start_char, end_char))

        cursor = end

    return segments


def _normalize_claim_text(sentence_text: str) -> str:
    """Apply minimal whitespace normalization to one claim sentence."""
    return " ".join(sentence_text.split())


def _build_claim_id(
    *,
    chunk_id: str,
    start_char: int,
    end_char: int,
    normalized_claim_text: str,
    extractor_version: str,
) -> str:
    """Create deterministic claim identity tied to sentence and version."""
    raw_id = (
        f"{chunk_id}|{start_char}|{end_char}|{normalized_claim_text}|"
        f"{extractor_version}"
    )
    digest = sha256(raw_id.encode("utf-8")).hexdigest()
    return f"claim_{digest[:16]}"
