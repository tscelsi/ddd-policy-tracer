"""Rule-based claim extraction strategies for one document chunk."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Protocol

import httpx

from ddd_policy_tracer.analysis.chunking_models import DocumentChunk

from .models import ClaimCandidate
from .ports import ClaimExtractor

OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"


class HttpClient(Protocol):
    """Minimal HTTP client protocol for LLM extraction transport."""

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> httpx.Response:
        """Send one JSON HTTP POST request and return response."""


@dataclass(frozen=True)
class LLMClaimExtractorConfig:
    """Configure an LLM-backed claim extraction strategy."""

    model: str = "gpt-4.1-mini"
    temperature: float = 0.0
    max_claims_per_chunk: int = 8
    request_timeout_seconds: float = 60.0
    api_url: str = OPENAI_CHAT_COMPLETIONS_URL
    extractor_version: str = "llm-v1"
    api_key_env_var: str = "OPENAI_API_KEY"


@dataclass(frozen=True)
class LLMClaimExtractor(ClaimExtractor):
    """Extract claim candidates by prompting an LLM with chunk text."""

    config: LLMClaimExtractorConfig = LLMClaimExtractorConfig()
    http_client: HttpClient | None = None

    def extract(self, *, chunk: DocumentChunk) -> list[ClaimCandidate]:
        """Return claim candidates extracted from one chunk via an LLM."""
        api_key = os.environ.get(self.config.api_key_env_var)
        if api_key is None:
            raise ValueError(
                f"{self.config.api_key_env_var} must be set for LLM extraction",
            )

        claims = self._request_claim_strings(api_key=api_key, chunk=chunk)
        candidates: list[ClaimCandidate] = []
        used_spans: set[tuple[int, int]] = set()

        for claim_text in claims:
            normalized_claim = _normalize_claim_text(claim_text)
            if not normalized_claim:
                continue

            offsets = _find_claim_offsets(
                chunk_text=chunk.chunk_text,
                claim_text=claim_text,
                used_spans=used_spans,
            )
            if offsets is None:
                continue

            start_char, end_char = offsets
            used_spans.add(offsets)
            evidence_text = chunk.chunk_text[start_char:end_char]
            claim_id = _build_claim_id(
                chunk_id=chunk.chunk_id,
                start_char=start_char,
                end_char=end_char,
                normalized_claim_text=_normalize_claim_text(evidence_text),
                extractor_version=self.config.extractor_version,
            )
            candidates.append(
                ClaimCandidate(
                    claim_id=claim_id,
                    chunk_id=chunk.chunk_id,
                    source_id=chunk.source_id,
                    source_document_id=chunk.source_document_id,
                    document_checksum=chunk.document_checksum,
                    start_char=start_char,
                    end_char=end_char,
                    evidence_text=evidence_text,
                    normalized_claim_text=_normalize_claim_text(evidence_text),
                    confidence=1.0,
                    claim_type=None,
                    extractor_version=self.config.extractor_version,
                ),
            )

        return candidates

    def count_processed_sentences(self, *, chunk: DocumentChunk) -> int:
        """Count sentence units in chunk text for report semantics."""
        return len(_split_sentences_with_offsets(chunk.chunk_text))

    def _request_claim_strings(
        self,
        *,
        api_key: str,
        chunk: DocumentChunk,
    ) -> list[str]:
        """Call LLM endpoint and return raw claim strings."""
        prompt = (
            "Extract policy-relevant claims from the text. "
            "Return only claims that are exact substrings from input text. "
            "Return JSON object with key claims as array of strings. "
            f"Limit to at most {self.config.max_claims_per_chunk} claims."
        )
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict claim extraction assistant. "
                        "Do not paraphrase."
                    ),
                },
                {
                    "role": "user",
                    "content": f"{prompt}\n\nTEXT:\n{chunk.chunk_text}",
                },
            ],
            "response_format": {"type": "json_object"},
        }

        if self.http_client is None:
            with httpx.Client(
                timeout=self.config.request_timeout_seconds,
            ) as client:
                response = client.post(
                    self.config.api_url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
        else:
            response = self.http_client.post(
                self.config.api_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

        response.raise_for_status()
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            claims = parsed.get("claims", [])
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid LLM response format for claim extraction") from exc

        if not isinstance(claims, list):
            return []
        return [str(claim).strip() for claim in claims if str(claim).strip()]

_QUANTITATIVE_RE = re.compile(
    r"("  # start capture for readability
    r"\b\d+(?:[.,]\d+)?%\b"
    r"|\$\s?\d+(?:[.,]\d+)?"
    r"|\b\d+(?:[.,]\d+)?\s*(?:billion|million|tonnes|tons|kt|mt)\b"
    r"|\b\d{4}\b"
    r")",
    flags=re.IGNORECASE,
)
_SKIP_HEADING_RE = re.compile(
    r"^(table|figure|fig\.?|appendix|chapter|section)\s+"
    r"[A-Za-z0-9\-]+(?:\s+[A-Za-z0-9\-]+)*(?:[:.-]+)?$",
    flags=re.IGNORECASE,
)
_SKIP_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z\s]{0,40}:$")


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
            if _is_skippable_sentence(sentence_text):
                continue
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

_SKIP_EXACT_SENTENCES = {
    "acknowledgements",
    "acknowledgment",
    "references",
    "reference",
    "table",
    "tables",
    "figure",
    "figures",
    "quote",
    "quotes",
}


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    """Return true when text includes at least one phrase cue."""
    return any(phrase in text for phrase in phrases)


def _is_skippable_sentence(sentence_text: str) -> bool:
    """Return true when sentence should be excluded from claim extraction."""
    normalized = _normalize_claim_text(sentence_text)
    lowered = normalized.casefold().rstrip(".:;,-")

    if lowered in _SKIP_EXACT_SENTENCES:
        return True
    if _SKIP_HEADING_RE.match(normalized) is not None:
        return True
    if _SKIP_LABEL_RE.match(normalized) is not None:
        return True
    if normalized.endswith(":") or normalized.endswith(":."):
        return True
    if "|" in normalized or "\t" in normalized:
        return True
    if normalized.startswith(">"):
        return True
    if _is_quoted_block(normalized):
        return True
    return False


def _is_quoted_block(text: str) -> bool:
    """Return true when sentence appears to be a standalone quoted block."""
    trimmed = text.strip().rstrip(".:;,-")
    if len(trimmed) < 2:
        return False
    quote_pairs = (('"', '"'), ("'", "'"), ("\u201c", "\u201d"))
    return any(trimmed.startswith(start) and trimmed.endswith(end) for start, end in quote_pairs)


def _contains_quantitative_cue(text: str) -> bool:
    """Return true when text includes numeric/quantitative claim cues."""
    return _QUANTITATIVE_RE.search(text) is not None


def _split_sentences_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Split text into sentence-like units with stable character offsets."""
    segments: list[tuple[str, int, int]] = []
    sentence_start = 0

    for index, char in enumerate(text):
        if char not in ".!?":
            continue
        if _is_decimal_point(text=text, index=index):
            continue

        end = index + 1
        raw_segment = text[sentence_start:end]
        stripped = raw_segment.strip()
        if stripped:
            leading_ws = len(raw_segment) - len(raw_segment.lstrip())
            trailing_ws = len(raw_segment) - len(raw_segment.rstrip())
            start_char = sentence_start + leading_ws
            end_char = end - trailing_ws
            segments.append((text[start_char:end_char], start_char, end_char))
        sentence_start = end

    if sentence_start < len(text):
        raw_segment = text[sentence_start:]
        stripped = raw_segment.strip()
        if stripped:
            leading_ws = len(raw_segment) - len(raw_segment.lstrip())
            trailing_ws = len(raw_segment) - len(raw_segment.rstrip())
            start_char = sentence_start + leading_ws
            end_char = len(text) - trailing_ws
            segments.append((text[start_char:end_char], start_char, end_char))

    return segments


def _is_decimal_point(*, text: str, index: int) -> bool:
    """Return true when one dot is part of a numeric decimal token."""
    if text[index] != ".":
        return False
    if index == 0 or index >= len(text) - 1:
        return False
    return text[index - 1].isdigit() and text[index + 1].isdigit()


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
    raw_id = f"{chunk_id}|{start_char}|{end_char}|{normalized_claim_text}|{extractor_version}"
    digest = sha256(raw_id.encode("utf-8")).hexdigest()
    return f"claim_{digest[:16]}"


def _find_claim_offsets(
    *,
    chunk_text: str,
    claim_text: str,
    used_spans: set[tuple[int, int]],
) -> tuple[int, int] | None:
    """Find claim offsets in chunk text and avoid duplicate span matches."""
    start = chunk_text.find(claim_text)
    if start == -1:
        normalized_claim = _normalize_claim_text(claim_text)
        if not normalized_claim:
            return None
        start = chunk_text.find(normalized_claim)
        if start == -1:
            return None
        claim_text = normalized_claim

    candidate = (start, start + len(claim_text))
    if candidate not in used_spans:
        return candidate

    search_start = start + 1
    while True:
        next_start = chunk_text.find(claim_text, search_start)
        if next_start == -1:
            return None
        candidate = (next_start, next_start + len(claim_text))
        if candidate not in used_spans:
            return candidate
        search_start = next_start + 1
