"""Rule-based claim extraction strategies for one document chunk."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

import httpx
import joblib

from ddd_policy_tracer.analysis.chunks.chunking_models import DocumentChunk

from .models import ClaimCandidate
from .ports import ClaimExtractor

OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
_ML_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
_HF_GENERATOR_CACHE: dict[str, HuggingFaceGenerator] = {}


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


class HuggingFaceGenerator(Protocol):
    """Generate one text2text output payload from input chunk text."""

    def __call__(self, text: str, *, max_new_tokens: int) -> object:
        """Return pipeline generation output for one input string."""


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
            candidates.append(
                _build_claim_candidate(
                    chunk=chunk,
                    start_char=start_char,
                    end_char=end_char,
                    evidence_text=evidence_text,
                    confidence=1.0,
                    extractor_version=self.config.extractor_version,
                ),
            )

        return candidates

    def count_processed_sentences(self, *, chunk: DocumentChunk) -> int:
        """Count sentence units in chunk text for report semantics."""
        return _count_chunk_sentences(chunk)

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
                    "content": ("You are a strict claim extraction assistant. Do not paraphrase."),
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
            claims = _parse_claims_from_json_content(content)
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid LLM response format for claim extraction") from exc

        if not isinstance(claims, list):
            return []
        return [str(claim).strip() for claim in claims if str(claim).strip()]


@dataclass(frozen=True)
class OllamaClaimExtractorConfig:
    """Configure a local Ollama-backed claim extraction strategy."""

    model: str = "llama3.1:8b"
    max_claims_per_chunk: int = 8
    request_timeout_seconds: float = 60.0
    api_url: str = OLLAMA_CHAT_URL
    extractor_version: str = "ollama-v1"


@dataclass(frozen=True)
class OllamaClaimExtractor(ClaimExtractor):
    """Extract claim candidates by prompting a local Ollama model."""

    config: OllamaClaimExtractorConfig = OllamaClaimExtractorConfig()
    http_client: HttpClient | None = None

    def extract(self, *, chunk: DocumentChunk) -> list[ClaimCandidate]:
        """Return claim candidates extracted from one chunk via Ollama."""
        claims = self._request_claim_strings(chunk=chunk)
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
            candidates.append(
                _build_claim_candidate(
                    chunk=chunk,
                    start_char=start_char,
                    end_char=end_char,
                    evidence_text=evidence_text,
                    confidence=1.0,
                    extractor_version=self.config.extractor_version,
                ),
            )

        return candidates

    def count_processed_sentences(self, *, chunk: DocumentChunk) -> int:
        """Count sentence units in chunk text for report semantics."""
        return _count_chunk_sentences(chunk)

    def _request_claim_strings(self, *, chunk: DocumentChunk) -> list[str]:
        """Call Ollama endpoint and return raw claim strings."""
        prompt = (
            "Extract policy-relevant claims from the text. "
            "Return only claims that are exact substrings from input text. "
            "Return JSON object with key claims as array of strings. "
            f"Limit to at most {self.config.max_claims_per_chunk} claims."
        )
        payload = {
            "model": self.config.model,
            "stream": False,
            "format": "json",
            "messages": [
                {
                    "role": "system",
                    "content": ("You are a strict claim extraction assistant. Do not paraphrase."),
                },
                {
                    "role": "user",
                    "content": f"{prompt}\n\nTEXT:\n{chunk.chunk_text}",
                },
            ],
        }

        if self.http_client is None:
            with httpx.Client(
                timeout=self.config.request_timeout_seconds,
            ) as client:
                response = client.post(
                    self.config.api_url,
                    headers={
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
        else:
            response = self.http_client.post(
                self.config.api_url,
                headers={
                    "Content-Type": "application/json",
                },
                json=payload,
            )

        response.raise_for_status()
        data = response.json()
        try:
            content = data["message"]["content"]
            claims = _parse_claims_from_json_content(content)
        except (KeyError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid Ollama response format for claim extraction") from exc

        if not isinstance(claims, list):
            return []
        return [str(claim).strip() for claim in claims if str(claim).strip()]


def _parse_claims_from_json_content(content: str) -> list[object]:
    """Parse claims list from JSON object text content."""
    parsed = json.loads(content)
    claims = parsed.get("claims", []) if isinstance(parsed, dict) else []
    return claims if isinstance(claims, list) else []


@dataclass(frozen=True)
class HuggingFaceClaimExtractorConfig:
    """Configure Hugging Face seq2seq claim extraction behavior."""

    model_name: str = "Babelscape/t5-base-summarization-claim-extractor"
    max_new_tokens: int = 256
    extractor_version: str = "hf-t5-babelscape-v1"


@dataclass(frozen=True)
class HuggingFaceClaimExtractor(ClaimExtractor):
    """Extract claim candidates using a Hugging Face text2text model."""

    config: HuggingFaceClaimExtractorConfig = HuggingFaceClaimExtractorConfig()
    generator: HuggingFaceGenerator | None = None

    def extract(self, *, chunk: DocumentChunk) -> list[ClaimCandidate]:
        """Generate claims from one chunk and map them back to source offsets."""
        generator = self.generator or self._build_generator()
        output = generator(chunk.chunk_text, max_new_tokens=self.config.max_new_tokens)
        generated_text = _extract_generated_text(output)
        claim_strings = [
            sentence_text
            for sentence_text, _, _ in _split_sentences_with_offsets(generated_text)
            if sentence_text.strip()
        ]

        candidates: list[ClaimCandidate] = []
        used_spans: set[tuple[int, int]] = set()
        for claim_text in claim_strings:
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
            candidates.append(
                _build_claim_candidate(
                    chunk=chunk,
                    start_char=start_char,
                    end_char=end_char,
                    evidence_text=evidence_text,
                    confidence=1.0,
                    extractor_version=self.config.extractor_version,
                ),
            )
        return candidates

    def count_processed_sentences(self, *, chunk: DocumentChunk) -> int:
        """Count sentence-like units from the processed chunk text."""
        return _count_chunk_sentences(chunk)

    def _build_generator(self) -> HuggingFaceGenerator:
        """Build a Hugging Face seq2seq generator callable lazily."""
        cached = _HF_GENERATOR_CACHE.get(self.config.model_name)
        if cached is not None:
            return cached

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ValueError(
                "transformers is required for HuggingFaceClaimExtractor",
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)

        def _generate(text: str, *, max_new_tokens: int) -> object:
            """Generate one claim text output payload for one input text."""
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return [{"generated_text": generated_text}]

        _HF_GENERATOR_CACHE[self.config.model_name] = _generate
        return _generate


@dataclass(frozen=True)
class MLClaimExtractorConfig:
    """Configure loading and threshold behavior for ML claim extraction."""

    model_path: Path = Path(__file__).resolve().parent / "ml" / "claims_model.joblib"
    decision_threshold_override: float | None = None
    extractor_version: str = "ml-v1"


@dataclass(frozen=True)
class MLClaimExtractor(ClaimExtractor):
    """Extract free-form claim spans using a token-level ML model artifact."""

    config: MLClaimExtractorConfig = MLClaimExtractorConfig()

    def extract(self, *, chunk: DocumentChunk) -> list[ClaimCandidate]:
        """Return claim candidates predicted from chunk token probabilities."""
        artifact = self._load_artifact()
        model = artifact.get("model")
        if not isinstance(model, dict):
            raise ValueError("claims model artifact missing 'model' bundle")

        raw_threshold = artifact.get("decision_threshold", 0.5)
        model_threshold = float(raw_threshold) if isinstance(raw_threshold, int | float) else 0.5
        decision_threshold = (
            self.config.decision_threshold_override
            if self.config.decision_threshold_override is not None
            else model_threshold
        )

        spans = _predict_ml_spans(
            model=model,
            chunk_text=chunk.chunk_text,
            decision_threshold=decision_threshold,
        )
        claims: list[ClaimCandidate] = []
        for start_char, end_char, confidence in spans:
            evidence_text = chunk.chunk_text[start_char:end_char]
            claims.append(
                _build_claim_candidate(
                    chunk=chunk,
                    start_char=start_char,
                    end_char=end_char,
                    evidence_text=evidence_text,
                    confidence=confidence,
                    extractor_version=self.config.extractor_version,
                ),
            )
        return claims

    def count_processed_sentences(self, *, chunk: DocumentChunk) -> int:
        """Count sentence units in chunk text for report semantics."""
        return _count_chunk_sentences(chunk)

    def _load_artifact(self) -> dict[str, object]:
        """Load serialized claims model artifact from configured path."""
        if not self.config.model_path.exists():
            raise ValueError(f"claims model file not found: {self.config.model_path}")
        artifact = joblib.load(self.config.model_path)
        if not isinstance(artifact, dict):
            raise ValueError("claims model artifact must be a dict")
        return artifact


def _count_chunk_sentences(chunk: DocumentChunk) -> int:
    """Count sentence-like units from one chunk text payload."""
    return len(_split_sentences_with_offsets(chunk.chunk_text))


def _build_claim_candidate(
    *,
    chunk: DocumentChunk,
    start_char: int,
    end_char: int,
    evidence_text: str,
    confidence: float,
    extractor_version: str,
) -> ClaimCandidate:
    """Create one canonical claim candidate from aligned chunk offsets."""
    normalized_claim_text = _normalize_claim_text(evidence_text)
    claim_id = _build_claim_id(
        chunk_id=chunk.chunk_id,
        start_char=start_char,
        end_char=end_char,
        normalized_claim_text=normalized_claim_text,
        extractor_version=extractor_version,
    )
    return ClaimCandidate(
        claim_id=claim_id,
        chunk_id=chunk.chunk_id,
        source_id=chunk.source_id,
        source_document_id=chunk.source_document_id,
        document_checksum=chunk.document_checksum,
        start_char=start_char,
        end_char=end_char,
        evidence_text=evidence_text,
        normalized_claim_text=normalized_claim_text,
        confidence=confidence,
        claim_type=None,
        extractor_version=extractor_version,
    )


def _predict_ml_spans(
    *,
    model: dict[str, object],
    chunk_text: str,
    decision_threshold: float,
) -> list[tuple[int, int, float]]:
    """Predict claim spans with confidence from token-level BIO probabilities."""
    tokens = _tokenize_with_offsets(chunk_text)
    if not tokens:
        return []

    features = [_token_features(tokens=tokens, index=index) for index in range(len(tokens))]
    vectorizer = model.get("vectorizer")
    b_classifier = model.get("b_classifier")
    i_classifier = model.get("i_classifier")
    if vectorizer is None or b_classifier is None or i_classifier is None:
        raise ValueError("claims model bundle missing vectorizer or classifiers")

    x_matrix = vectorizer.transform(features)
    b_probs = _positive_class_probabilities(classifier=b_classifier, x_matrix=x_matrix)
    i_probs = _positive_class_probabilities(classifier=i_classifier, x_matrix=x_matrix)

    tags: list[str] = ["O"] * len(tokens)
    scores: list[float] = [0.0] * len(tokens)
    for index in range(len(tokens)):
        b_score = b_probs[index]
        i_score = i_probs[index]
        if b_score >= decision_threshold:
            tags[index] = "B"
            scores[index] = b_score
        elif i_score >= decision_threshold:
            tags[index] = "I"
            scores[index] = i_score

    spans: list[tuple[int, int, float]] = []
    current_start: int | None = None
    current_end: int | None = None
    current_scores: list[float] = []
    for index, tag in enumerate(tags):
        token_start, token_end = tokens[index][1], tokens[index][2]
        if tag == "B":
            if current_start is not None and current_end is not None and current_scores:
                spans.append(
                    (current_start, current_end, sum(current_scores) / len(current_scores)),
                )
            current_start = token_start
            current_end = token_end
            current_scores = [scores[index]]
            continue

        if tag == "I" and current_start is not None:
            current_end = token_end
            current_scores.append(scores[index])
            continue

        if current_start is not None and current_end is not None and current_scores:
            spans.append((current_start, current_end, sum(current_scores) / len(current_scores)))
        current_start = None
        current_end = None
        current_scores = []

    if current_start is not None and current_end is not None and current_scores:
        spans.append((current_start, current_end, sum(current_scores) / len(current_scores)))
    return _dedupe_ml_spans(spans)


def _dedupe_ml_spans(spans: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    """Dedupe predicted spans while preserving insertion order."""
    seen: set[tuple[int, int]] = set()
    deduped: list[tuple[int, int, float]] = []
    for start_char, end_char, confidence in spans:
        key = (start_char, end_char)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((start_char, end_char, confidence))
    return deduped


def _tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Tokenize text and return token strings with absolute char offsets."""
    tokens: list[tuple[str, int, int]] = []
    for match in _ML_TOKEN_RE.finditer(text):
        tokens.append((match.group(0), match.start(), match.end()))
    return tokens


def _token_features(
    *,
    tokens: list[tuple[str, int, int]],
    index: int,
) -> dict[str, object]:
    """Build token-level contextual features aligned with training pipeline."""
    token_text = tokens[index][0]
    prev_text = tokens[index - 1][0] if index > 0 else "<START>"
    next_text = tokens[index + 1][0] if index < len(tokens) - 1 else "<END>"
    return {
        "token": token_text.casefold(),
        "token_is_title": token_text.istitle(),
        "token_is_upper": token_text.isupper(),
        "token_is_digit": token_text.isdigit(),
        "token_has_digit": any(char.isdigit() for char in token_text),
        "token_has_percent": "%" in token_text,
        "token_prefix_3": token_text[:3].casefold(),
        "token_suffix_3": token_text[-3:].casefold(),
        "prev_token": prev_text.casefold(),
        "next_token": next_text.casefold(),
    }


def _positive_class_probabilities(*, classifier: object, x_matrix: object) -> list[float]:
    """Return class-1 probabilities for a fitted sklearn classifier."""
    probabilities = classifier.predict_proba(x_matrix)
    classes = getattr(classifier, "classes_", None)
    if classes is None:
        return [0.0 for _ in range(len(probabilities))]
    classes_list = list(classes)
    if 1 not in classes_list:
        return [0.0 for _ in range(len(probabilities))]
    class_index = classes_list.index(1)
    return [float(row[class_index]) for row in probabilities]


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

            claims.append(
                _build_claim_candidate(
                    chunk=chunk,
                    start_char=start_char,
                    end_char=end_char,
                    evidence_text=sentence_text,
                    confidence=min(1.0, score),
                    extractor_version=self.config.extractor_version,
                ),
            )
        return claims

    def count_processed_sentences(self, *, chunk: DocumentChunk) -> int:
        """Count sentence units processed from one chunk payload."""
        return _count_chunk_sentences(chunk)

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


def _extract_generated_text(output: object) -> str:
    """Extract one generated text string from HF pipeline output payload."""
    if isinstance(output, str):
        return output
    if isinstance(output, list) and output:
        first_item = output[0]
        if isinstance(first_item, dict):
            value = first_item.get("generated_text")
            if isinstance(value, str):
                return value
    return ""
