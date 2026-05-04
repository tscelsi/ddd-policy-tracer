"""Rule-based entity extraction strategies for one document chunk."""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from ddd_policy_tracer.analysis.chunking_models import DocumentChunk

from .models import EntityMention, EntityType
from .ports import EntityExtractor

_POLICY_CUES = (
    "policy",
    "act",
    "bill",
    "regulation",
    "strategy",
    "framework",
    "target",
)
_PROGRAM_CUES = ("program", "initiative", "scheme", "fund", "package", "plan")
_ORG_CUES = (
    "institute",
    "department",
    "ministry",
    "commission",
    "council",
    "agency",
    "university",
    "bank",
    "foundation",
)
_PERSON_TITLES = ("Dr", "Prof", "Mr", "Ms", "Mrs")

_POLICY_RE = re.compile(
    r"\b((?!(?:The|A|An)\b)[A-Z][A-Za-z0-9'&/-]*(?:\s+[A-Z][A-Za-z0-9'&/-]*){0,7}\s+"
    r"(?:Policy|Act|Bill|Regulation|Strategy|Framework|Target))\b",
)
_PROGRAM_RE = re.compile(
    r"\b((?!(?:The|A|An)\b)[A-Z][A-Za-z0-9'&/-]*(?:\s+[A-Z][A-Za-z0-9'&/-]*){0,7}\s+"
    r"(?:Program|Initiative|Scheme|Fund|Package|Plan))\b",
)
_ORG_RE = re.compile(
    r"\b((?!(?:The|A|An)\b)[A-Z][A-Za-z0-9'&/-]*(?:\s+[A-Z][A-Za-z0-9'&/-]*){0,7}\s+"
    r"(?:Institute|Department|Ministry|Commission|Council|Agency|University|Bank|Foundation))\b",
)
_PERSON_TITLE_RE = re.compile(r"\b(?:Dr|Prof|Mr|Ms|Mrs)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")
_PERSON_NAME_RE = re.compile(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b")
_SENTENCE_PERSON_STOP_SUFFIX = (
    "Institute",
    "Department",
    "Ministry",
    "Commission",
    "Council",
    "Agency",
    "University",
    "Bank",
    "Foundation",
    "Policy",
    "Act",
    "Bill",
    "Regulation",
    "Strategy",
    "Framework",
    "Target",
    "Program",
    "Initiative",
    "Scheme",
    "Fund",
    "Package",
    "Plan",
)

_QUOTE_PAIRS = (("\"", "\""), ("'", "'"), ("\u201c", "\u201d"))


@dataclass(frozen=True)
class RuleBasedEntityExtractorConfig:
    """Configure deterministic entity extraction thresholds and precedence."""

    extractor_version: str = "rules-v1"
    policy_threshold: float = 1.0
    program_threshold: float = 1.0
    org_threshold: float = 1.0
    person_threshold: float = 1.0
    jurisdiction_threshold: float = 1.0
    cue_weight: float = 1.0
    boundary_weight: float = 0.4
    title_weight: float = 0.6
    precedence: tuple[EntityType, ...] = (
        "POLICY",
        "PROGRAM",
        "ORG",
        "PERSON",
        "JURISDICTION",
    )
    jurisdictions_path: Path | None = None


@dataclass(frozen=True)
class RuleBasedSentenceEntityExtractor(EntityExtractor):
    """Extract sentence-bounded entity mentions using deterministic rules."""

    config: RuleBasedEntityExtractorConfig = RuleBasedEntityExtractorConfig()

    def extract(self, *, chunk: DocumentChunk) -> list[EntityMention]:
        """Return deterministic entity mentions for one source chunk."""
        candidates: list[EntityMention] = []
        jurisdictions = self._load_jurisdictions()
        precedence_rank = {
            entity_type: index
            for index, entity_type in enumerate(self.config.precedence)
        }
        for sentence_text, start_char, _end_char in _split_sentences_with_offsets(
            chunk.chunk_text,
        ):
            sentence_candidates = self._extract_sentence_candidates(
                sentence_text=sentence_text,
                chunk=chunk,
                sentence_start_char=start_char,
                jurisdictions=jurisdictions,
            )
            kept = self._resolve_overlaps(
                sentence_candidates,
                precedence_rank=precedence_rank,
            )
            candidates.extend(kept)

        return candidates

    def count_processed_sentences(self, *, chunk: DocumentChunk) -> int:
        """Count sentence units processed from one chunk payload."""
        return len(_split_sentences_with_offsets(chunk.chunk_text))

    def _extract_sentence_candidates(
        self,
        *,
        sentence_text: str,
        chunk: DocumentChunk,
        sentence_start_char: int,
        jurisdictions: tuple[str, ...],
    ) -> list[EntityMention]:
        """Extract deterministic entity candidates from one sentence."""
        raw_candidates: list[EntityMention] = []
        raw_candidates.extend(
            self._match_pattern_candidates(
                chunk=chunk,
                sentence_text=sentence_text,
                sentence_start_char=sentence_start_char,
                pattern=_POLICY_RE,
                entity_type="POLICY",
                threshold=self.config.policy_threshold,
                cue_terms=_POLICY_CUES,
            ),
        )
        raw_candidates.extend(
            self._match_pattern_candidates(
                chunk=chunk,
                sentence_text=sentence_text,
                sentence_start_char=sentence_start_char,
                pattern=_PROGRAM_RE,
                entity_type="PROGRAM",
                threshold=self.config.program_threshold,
                cue_terms=_PROGRAM_CUES,
            ),
        )
        raw_candidates.extend(
            self._match_pattern_candidates(
                chunk=chunk,
                sentence_text=sentence_text,
                sentence_start_char=sentence_start_char,
                pattern=_ORG_RE,
                entity_type="ORG",
                threshold=self.config.org_threshold,
                cue_terms=_ORG_CUES,
            ),
        )
        raw_candidates.extend(
            self._match_person_candidates(
                chunk=chunk,
                sentence_text=sentence_text,
                sentence_start_char=sentence_start_char,
            ),
        )
        raw_candidates.extend(
            self._match_jurisdiction_candidates(
                chunk=chunk,
                sentence_text=sentence_text,
                sentence_start_char=sentence_start_char,
                jurisdictions=jurisdictions,
            ),
        )
        return raw_candidates

    def _resolve_overlaps(
        self,
        candidates: list[EntityMention],
        *,
        precedence_rank: dict[EntityType, int],
    ) -> list[EntityMention]:
        """Resolve overlapping mentions by confidence, precedence, and span length."""
        kept: list[EntityMention] = []
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                -candidate.confidence,
                precedence_rank.get(candidate.entity_type, 999),
                -(candidate.end_char - candidate.start_char),
                candidate.start_char,
            ),
        )

        for candidate in ordered:
            if any(_spans_overlap(candidate, existing) for existing in kept):
                continue
            kept.append(candidate)

        return sorted(kept, key=lambda candidate: (candidate.start_char, candidate.end_char))

    def _match_pattern_candidates(
        self,
        *,
        chunk: DocumentChunk,
        sentence_text: str,
        sentence_start_char: int,
        pattern: re.Pattern[str],
        entity_type: EntityType,
        threshold: float,
        cue_terms: tuple[str, ...],
    ) -> list[EntityMention]:
        """Create type-specific candidates from one regex pattern match set."""
        candidates: list[EntityMention] = []
        for match in pattern.finditer(sentence_text):
            mention_text = match.group(1)
            normalized = _normalize_mention_text(mention_text)
            confidence = self._score_mention(
                mention_text=mention_text,
                normalized_mention_text=normalized,
                cue_terms=cue_terms,
                has_title=False,
            )
            if confidence < threshold:
                continue
            start_char = sentence_start_char + match.start(1)
            end_char = sentence_start_char + match.end(1)
            candidates.append(
                _build_entity_mention(
                    chunk=chunk,
                    start_char=start_char,
                    end_char=end_char,
                    mention_text=chunk.chunk_text[start_char:end_char],
                    normalized_mention_text=normalized,
                    entity_type=entity_type,
                    confidence=confidence,
                    extractor_version=self.config.extractor_version,
                ),
            )
        return candidates

    def _match_person_candidates(
        self,
        *,
        chunk: DocumentChunk,
        sentence_text: str,
        sentence_start_char: int,
    ) -> list[EntityMention]:
        """Extract person candidates using title-aware and name pattern rules."""
        candidates: list[EntityMention] = []
        candidate_groups = [
            (match.group(0), match.start(0), match.end(0), True)
            for match in _PERSON_TITLE_RE.finditer(sentence_text)
        ]
        if not candidate_groups:
            candidate_groups = [
                (match.group(1), match.start(1), match.end(1), False)
                for match in _PERSON_NAME_RE.finditer(sentence_text)
            ]

        for mention_text, start, end, has_title in candidate_groups:
            normalized = _normalize_mention_text(mention_text)
            if any(normalized.endswith(suffix) for suffix in _SENTENCE_PERSON_STOP_SUFFIX):
                continue

            confidence = self._score_mention(
                mention_text=mention_text,
                normalized_mention_text=normalized,
                cue_terms=(),
                has_title=has_title,
            )
            if confidence < self.config.person_threshold:
                continue

            start_char = sentence_start_char + start
            end_char = sentence_start_char + end
            candidates.append(
                _build_entity_mention(
                    chunk=chunk,
                    start_char=start_char,
                    end_char=end_char,
                    mention_text=chunk.chunk_text[start_char:end_char],
                    normalized_mention_text=normalized,
                    entity_type="PERSON",
                    confidence=confidence,
                    extractor_version=self.config.extractor_version,
                ),
            )

        return candidates

    def _match_jurisdiction_candidates(
        self,
        *,
        chunk: DocumentChunk,
        sentence_text: str,
        sentence_start_char: int,
        jurisdictions: tuple[str, ...],
    ) -> list[EntityMention]:
        """Extract jurisdiction candidates from deterministic gazetteer terms."""
        candidates: list[EntityMention] = []
        lowered_sentence = sentence_text.casefold()
        for jurisdiction in jurisdictions:
            pattern = re.compile(rf"\b{re.escape(jurisdiction)}\b", flags=re.IGNORECASE)
            for match in pattern.finditer(sentence_text):
                mention_text = match.group(0)
                normalized = _normalize_mention_text(mention_text)
                confidence = self._score_mention(
                    mention_text=mention_text,
                    normalized_mention_text=normalized,
                    cue_terms=(jurisdiction.casefold(),),
                    has_title=False,
                )
                if confidence < self.config.jurisdiction_threshold:
                    continue

                if mention_text.casefold() not in lowered_sentence:
                    continue

                start_char = sentence_start_char + match.start(0)
                end_char = sentence_start_char + match.end(0)
                candidates.append(
                    _build_entity_mention(
                        chunk=chunk,
                        start_char=start_char,
                        end_char=end_char,
                        mention_text=chunk.chunk_text[start_char:end_char],
                        normalized_mention_text=normalized,
                        entity_type="JURISDICTION",
                        confidence=confidence,
                        extractor_version=self.config.extractor_version,
                    ),
                )
        return candidates

    def _score_mention(
        self,
        *,
        mention_text: str,
        normalized_mention_text: str,
        cue_terms: tuple[str, ...],
        has_title: bool,
    ) -> float:
        """Score one mention candidate using deterministic weighted cues."""
        score = 0.0
        lowered = mention_text.casefold()
        if any(cue in lowered for cue in cue_terms):
            score += self.config.cue_weight
        if _is_boundary_clean(normalized_mention_text):
            score += self.config.boundary_weight
        if has_title:
            score += self.config.title_weight
        return min(1.0, score)

    def _load_jurisdictions(self) -> tuple[str, ...]:
        """Load jurisdiction gazetteer terms from configured resource path."""
        path = self.config.jurisdictions_path or _default_jurisdictions_path()
        content = path.read_text(encoding="utf-8")
        return tuple(line.strip() for line in content.splitlines() if line.strip())


def _default_jurisdictions_path() -> Path:
    """Return default in-repo jurisdiction resource path for v1."""
    return Path(__file__).resolve().parent / "resources" / "jurisdictions.txt"


def _normalize_mention_text(mention_text: str) -> str:
    """Apply minimal whitespace normalization to one mention string."""
    return " ".join(mention_text.split())


def _is_boundary_clean(mention_text: str) -> bool:
    """Return true when mention is not wrapped in standalone quote blocks."""
    trimmed = mention_text.strip().rstrip(".:;,-")
    if len(trimmed) < 2:
        return True
    return not any(
        trimmed.startswith(start) and trimmed.endswith(end)
        for start, end in _QUOTE_PAIRS
    )


def _build_entity_mention(
    *,
    chunk: DocumentChunk,
    start_char: int,
    end_char: int,
    mention_text: str,
    normalized_mention_text: str,
    entity_type: EntityType,
    confidence: float,
    extractor_version: str,
) -> EntityMention:
    """Build one deterministic entity mention domain record."""
    entity_id = _build_entity_id(
        chunk_id=chunk.chunk_id,
        start_char=start_char,
        end_char=end_char,
        entity_type=entity_type,
        extractor_version=extractor_version,
    )
    return EntityMention(
        entity_id=entity_id,
        chunk_id=chunk.chunk_id,
        source_id=chunk.source_id,
        source_document_id=chunk.source_document_id,
        document_checksum=chunk.document_checksum,
        start_char=start_char,
        end_char=end_char,
        mention_text=mention_text,
        normalized_mention_text=normalized_mention_text,
        entity_type=entity_type,
        confidence=confidence,
        extractor_version=extractor_version,
    )


def _build_entity_id(
    *,
    chunk_id: str,
    start_char: int,
    end_char: int,
    entity_type: EntityType,
    extractor_version: str,
) -> str:
    """Create deterministic entity identity tied to span, type, and version."""
    raw_id = f"{chunk_id}|{start_char}|{end_char}|{entity_type}|{extractor_version}"
    digest = sha256(raw_id.encode("utf-8")).hexdigest()
    return f"entity_{digest[:16]}"


def _spans_overlap(left: EntityMention, right: EntityMention) -> bool:
    """Return true when two mention character spans overlap."""
    return left.start_char < right.end_char and right.start_char < left.end_char


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
