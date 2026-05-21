"""Rule-based entity extraction strategies for one document chunk."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from ddd_policy_tracer.analysis.chunks.chunking_models import DocumentChunk

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

_QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("\u201c", "\u201d"))
_LOGGER = logging.getLogger(__name__)


class SpacyEntityLike(Protocol):
    """Represent minimal spaCy entity span interface for extraction."""

    start_char: int
    end_char: int
    text: str
    label_: str


class SpacySentenceLike(Protocol):
    """Represent minimal spaCy sentence span interface for counting."""

    text: str


class SpacyDocLike(Protocol):
    """Represent minimal spaCy doc interface consumed by extractor."""

    ents: list[SpacyEntityLike]
    sents: list[SpacySentenceLike]


class SpacyNlpLike(Protocol):
    """Represent minimal spaCy nlp callable for document parsing."""

    def __call__(self, text: str) -> SpacyDocLike:
        """Parse one text string and return a spaCy-like document."""

    def pipe(self, texts: list[str]) -> list[SpacyDocLike]:
        """Parse many text strings and yield spaCy-like documents."""


@dataclass(frozen=True)
class SpacyEntityExtractorConfig:
    """Configure spaCy plus fastcoref entity extraction behavior."""

    extractor_version: str = "spacy-v1"
    spacy_model: str = "en_core_web_sm"


@dataclass(frozen=True)
class SpacyEntityExtractor(EntityExtractor):
    """Extract PROPN entities with spaCy."""

    config: SpacyEntityExtractorConfig = SpacyEntityExtractorConfig()
    nlp: SpacyNlpLike | None = None

    def extract(self, *, chunk: DocumentChunk) -> list[EntityMention]:
        """Return entity mentions from spaCy labels and optional coreference clusters."""
        nlp = self.nlp or _build_spacy_nlp(self.config)
        doc = nlp(chunk.chunk_text)
        entities = self._extract_from_doc(chunk=chunk, doc=doc)
        return entities

    def extract_many(self, *, chunks: list[DocumentChunk]) -> list[EntityMention]:
        """Return entity mentions for many chunks using spaCy nlp.pipe."""
        if not chunks:
            return []
        nlp = self.nlp or _build_spacy_nlp(self.config)
        docs = nlp.pipe([chunk.chunk_text for chunk in chunks])

        entities: list[EntityMention] = []
        for chunk, doc in zip(chunks, docs, strict=False):
            chunk_entities = self._extract_from_doc(chunk=chunk, doc=doc)
            entities.extend(chunk_entities)
        return entities

    def _extract_from_doc(self, *, chunk: DocumentChunk, doc: SpacyDocLike) -> list[EntityMention]:
        """Map one parsed doc into strict v1 entity mention records."""
        canonical_by_span = _canonical_by_span_from_doc(doc)
        entities: list[EntityMention] = []
        for entity in doc.ents:
            entity_type = _map_spacy_entity_type(
                label=entity.label_,
                mention_text=entity.text,
            )
            if entity_type is None:
                continue

            mention_text = chunk.chunk_text[entity.start_char : entity.end_char]
            normalized = _normalize_mention_text(mention_text)
            canonical_text = canonical_by_span.get((entity.start_char, entity.end_char))
            canonical_key = (
                _normalize_mention_text(canonical_text).casefold()
                if canonical_text is not None and canonical_text.strip()
                else None
            )
            pos_labels = [
                token.pos_
                for token in doc
                if token.idx >= entity.start_char and token.idx < entity.end_char
            ]
            ent_is_proper_noun = any(token == "PROPN" for token in pos_labels)
            # check if PROPN as pos
            if ent_is_proper_noun:
                entities.append(
                    _build_entity_mention(
                        chunk=chunk,
                        start_char=entity.start_char,
                        end_char=entity.end_char,
                        mention_text=mention_text,
                        normalized_mention_text=normalized,
                        entity_type=entity_type,
                        confidence=1.0,
                        extractor_version=self.config.extractor_version,
                        canonical_entity_key=canonical_key,
                        metadata={
                            "spacy_label": entity.label_,
                            "coref_canonical": canonical_text,
                            "pos_labels": pos_labels,
                        }
                        if canonical_text is not None
                        else {"spacy_label": entity.label_, "pos_labels": pos_labels},
                    ),
                )
        return entities

    def count_processed_sentences(self, *, chunk: DocumentChunk) -> int:
        """Count sentence units from spaCy parsing for report semantics."""
        nlp = self.nlp or _build_spacy_nlp(self.config)
        doc = nlp(chunk.chunk_text)
        try:
            return len(list(doc.sents))
        except Exception:
            return len(_split_sentences_with_offsets(chunk.chunk_text))


def _build_spacy_nlp(config: SpacyEntityExtractorConfig) -> SpacyNlpLike:
    """Build spaCy pipeline and attach fastcoref when configured and available."""
    import spacy

    try:
        nlp = spacy.load(config.spacy_model)
    except OSError:
        _LOGGER.warning(
            "spaCy model '%s' not found; falling back to blank 'en' pipeline",
            config.spacy_model,
        )
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
    return nlp


def _map_spacy_entity_type(*, label: str, mention_text: str) -> EntityType | None:
    """Map spaCy label and mention cues to one strict v1 entity type."""
    lowered = mention_text.casefold()
    if any(cue in lowered for cue in _POLICY_CUES):
        return "POLICY"
    if any(cue in lowered for cue in _PROGRAM_CUES):
        return "PROGRAM"
    if label == "ORG":
        return "ORG"
    if label == "PERSON":
        return "PERSON"
    if label in {"GPE", "LOC", "NORP"}:
        return "JURISDICTION"
    return None


def _map_gazetteer_entity_type(*, cue: str) -> EntityType:
    """Map one cue term to strict entity type used by v1 schema."""
    lowered = cue.casefold()
    if lowered in _POLICY_CUES:
        return "POLICY"
    if lowered in _PROGRAM_CUES:
        return "PROGRAM"
    return "ORG"


def _group_mentions_by_chunk(mentions: list[EntityMention]) -> dict[str, list[EntityMention]]:
    """Group extracted mentions by chunk identifier for batch merge."""
    grouped: dict[str, list[EntityMention]] = {}
    for mention in mentions:
        grouped.setdefault(mention.chunk_id, []).append(mention)
    return grouped


def _coref_mentions(cluster: object) -> list[tuple[int, int, str]]:
    """Extract span mentions from one fastcoref cluster object."""
    mentions_obj = getattr(cluster, "mentions", None)
    if not isinstance(mentions_obj, list):
        return []

    mentions: list[tuple[int, int, str]] = []
    for mention in mentions_obj:
        start_char = getattr(mention, "start_char", None)
        end_char = getattr(mention, "end_char", None)
        text = getattr(mention, "text", None)
        if not isinstance(start_char, int) or not isinstance(end_char, int):
            continue
        if not isinstance(text, str) or not text.strip():
            continue
        mentions.append((start_char, end_char, text))
    return mentions


def _canonical_by_span_from_doc(doc: SpacyDocLike) -> dict[tuple[int, int], str]:
    """Build canonical mention text by span from fastcoref clusters when present."""
    by_span: dict[tuple[int, int], str] = {}
    extensions = getattr(doc, "_", None)
    clusters = getattr(extensions, "coref_clusters", None)
    if not isinstance(clusters, list):
        return by_span
    for cluster in clusters:
        mentions = _coref_mentions(cluster)
        if not mentions:
            continue
        canonical_text = mentions[0][2]
        for start_char, end_char, _text in mentions:
            by_span[(start_char, end_char)] = canonical_text
    return by_span


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
        trimmed.startswith(start) and trimmed.endswith(end) for start, end in _QUOTE_PAIRS
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
    canonical_entity_key: str | None = None,
    metadata: dict[str, object] | None = None,
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
        canonical_entity_key=canonical_entity_key,
        metadata=metadata,
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
