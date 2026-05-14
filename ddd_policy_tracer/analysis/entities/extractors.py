"""Rule-based entity extraction strategies for one document chunk."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol, cast

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
_LOGGER = logging.getLogger(__name__)


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

    def extract_many(self, *, chunks: list[DocumentChunk]) -> list[EntityMention]:
        """Extract entity mentions for many chunks with deterministic batching."""
        entities: list[EntityMention] = []
        for chunk in chunks:
            entities.extend(self.extract(chunk=chunk))
        return entities

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
class SpacyFastCorefEntityExtractorConfig:
    """Configure spaCy plus fastcoref entity extraction behavior."""

    extractor_version: str = "spacy-fastcoref-v1"
    spacy_model: str = "en_core_web_sm"
    enable_coref: bool = True
    fallback_to_rule_extractor: bool = True


@dataclass(frozen=True)
class SpacyFastCorefEntityExtractor(EntityExtractor):
    """Extract entities with spaCy NER and enrich canonical keys via fastcoref."""

    config: SpacyFastCorefEntityExtractorConfig = SpacyFastCorefEntityExtractorConfig()
    nlp: SpacyNlpLike | None = None

    def extract(self, *, chunk: DocumentChunk) -> list[EntityMention]:
        """Return entity mentions from spaCy labels and optional coreference clusters."""
        nlp = self.nlp or _build_spacy_fastcoref_nlp(self.config)
        doc = nlp(chunk.chunk_text)
        entities = self._extract_from_doc(chunk=chunk, doc=doc)
        if entities:
            return entities
        if self.config.fallback_to_rule_extractor:
            return self._fallback_rule_extract(chunk=chunk)
        return []

    def extract_many(self, *, chunks: list[DocumentChunk]) -> list[EntityMention]:
        """Return entity mentions for many chunks using spaCy nlp.pipe."""
        if not chunks:
            return []
        nlp = self.nlp or _build_spacy_fastcoref_nlp(self.config)
        docs = nlp.pipe([chunk.chunk_text for chunk in chunks])

        entities: list[EntityMention] = []
        for chunk, doc in zip(chunks, docs, strict=False):
            chunk_entities = self._extract_from_doc(chunk=chunk, doc=doc)
            if not chunk_entities and self.config.fallback_to_rule_extractor:
                chunk_entities = self._fallback_rule_extract(chunk=chunk)
            entities.extend(chunk_entities)
        return entities

    def _extract_from_doc(self, *, chunk: DocumentChunk, doc: SpacyDocLike) -> list[EntityMention]:
        """Map one parsed doc into strict v1 entity mention records."""
        canonical_by_span = _build_coref_canonical_lookup(doc)

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
                    }
                    if canonical_text is not None
                    else {"spacy_label": entity.label_},
                ),
            )
        return entities

    def _fallback_rule_extract(self, *, chunk: DocumentChunk) -> list[EntityMention]:
        """Fallback to deterministic rules when spaCy returns no mapped entities."""
        rule_extractor = RuleBasedSentenceEntityExtractor(
            config=RuleBasedEntityExtractorConfig(
                extractor_version=f"{self.config.extractor_version}-fallback-rules",
            ),
        )
        return rule_extractor.extract(chunk=chunk)

    def count_processed_sentences(self, *, chunk: DocumentChunk) -> int:
        """Count sentence units from spaCy parsing for report semantics."""
        nlp = self.nlp or _build_spacy_fastcoref_nlp(self.config)
        doc = nlp(chunk.chunk_text)
        try:
            return len(list(doc.sents))
        except Exception:
            return len(_split_sentences_with_offsets(chunk.chunk_text))


@dataclass(frozen=True)
class RobustEnsembleEntityExtractorConfig:
    """Configure the robust single-path mention generation strategy."""

    extractor_version: str = "robust-ensemble-v1"
    source_priority: tuple[str, ...] = ("spacy", "gazetteer", "rule")


@dataclass(frozen=True)
class RobustEnsembleEntityExtractor(EntityExtractor):
    """Extract mentions with an ensemble and deterministic overlap merge."""

    config: RobustEnsembleEntityExtractorConfig = RobustEnsembleEntityExtractorConfig()
    rule_extractor: EntityExtractor | None = None
    spacy_extractor: EntityExtractor | None = None

    def extract(self, *, chunk: DocumentChunk) -> list[EntityMention]:
        """Combine mention channels and resolve conflicts deterministically."""
        rule_mentions = self._rule_extractor().extract(chunk=chunk)
        spacy_mentions = self._spacy_extractor().extract(chunk=chunk)
        gazetteer_mentions = self._extract_gazetteer_dependency_mentions(chunk=chunk)
        merged = self._merge_mentions(
            rule_mentions=rule_mentions,
            spacy_mentions=spacy_mentions,
            gazetteer_mentions=gazetteer_mentions,
        )
        return [self._with_runtime_version(mention=mention) for mention in merged]

    def extract_many(self, *, chunks: list[DocumentChunk]) -> list[EntityMention]:
        """Run extraction for many chunks using deterministic ordering."""
        entities: list[EntityMention] = []
        for chunk in chunks:
            entities.extend(self.extract(chunk=chunk))
        return entities

    def count_processed_sentences(self, *, chunk: DocumentChunk) -> int:
        """Count processed sentence-like units for reporting semantics."""
        return len(_split_sentences_with_offsets(chunk.chunk_text))

    def _rule_extractor(self) -> EntityExtractor:
        """Return the configured or default rule-based mention channel."""
        if self.rule_extractor is not None:
            return self.rule_extractor
        return RuleBasedSentenceEntityExtractor(
            config=RuleBasedEntityExtractorConfig(
                extractor_version=f"{self.config.extractor_version}-rule",
            ),
        )

    def _spacy_extractor(self) -> EntityExtractor:
        """Return the configured or default spaCy/coref mention channel."""
        if self.spacy_extractor is not None:
            return self.spacy_extractor
        return SpacyFastCorefEntityExtractor(
            config=SpacyFastCorefEntityExtractorConfig(
                extractor_version=f"{self.config.extractor_version}-spacy",
                fallback_to_rule_extractor=False,
            ),
        )

    def _extract_gazetteer_dependency_mentions(self, *, chunk: DocumentChunk) -> list[EntityMention]:
        """Extract extra mentions from gazetteer-like boundary patterns."""
        candidates: list[EntityMention] = []
        for sentence_text, start_char, _end_char in _split_sentences_with_offsets(chunk.chunk_text):
            for cue in _POLICY_CUES + _PROGRAM_CUES + _ORG_CUES:
                pattern = re.compile(
                    rf"\b([A-Z][A-Za-z0-9'&/-]*(?:\s+[A-Z][A-Za-z0-9'&/-]*){{0,6}}\s+{cue.title()})\b",
                )
                for match in pattern.finditer(sentence_text):
                    mention_text = match.group(1)
                    normalized = _normalize_mention_text(mention_text)
                    entity_type = _map_gazetteer_entity_type(cue=cue)
                    mention_start = start_char + match.start(1)
                    mention_end = start_char + match.end(1)
                    candidates.append(
                        _build_entity_mention(
                            chunk=chunk,
                            start_char=mention_start,
                            end_char=mention_end,
                            mention_text=chunk.chunk_text[mention_start:mention_end],
                            normalized_mention_text=normalized,
                            entity_type=entity_type,
                            confidence=0.9,
                            extractor_version=f"{self.config.extractor_version}-gazetteer",
                            metadata={"channel": "gazetteer"},
                        ),
                    )
        return candidates

    def _merge_mentions(
        self,
        *,
        rule_mentions: list[EntityMention],
        spacy_mentions: list[EntityMention],
        gazetteer_mentions: list[EntityMention],
    ) -> list[EntityMention]:
        """Merge channels with deterministic overlap and dedup resolution."""
        rank = {source: index for index, source in enumerate(self.config.source_priority)}

        def source_for(mention: EntityMention) -> str:
            metadata = mention.metadata or {}
            channel = metadata.get("channel")
            if isinstance(channel, str):
                return channel
            if "spacy" in mention.extractor_version:
                return "spacy"
            if "gazetteer" in mention.extractor_version:
                return "gazetteer"
            return "rule"

        ordered = sorted(
            [*spacy_mentions, *gazetteer_mentions, *rule_mentions],
            key=lambda mention: (
                -mention.confidence,
                rank.get(source_for(mention), 999),
                mention.start_char,
                mention.end_char,
                mention.entity_type,
            ),
        )
        kept: list[EntityMention] = []
        for candidate in ordered:
            if any(_spans_overlap(candidate, existing) for existing in kept):
                continue
            kept.append(candidate)

        unique: dict[tuple[int, int, str, str], EntityMention] = {}
        for mention in kept:
            unique[(
                mention.start_char,
                mention.end_char,
                mention.entity_type,
                mention.normalized_mention_text.casefold(),
            )] = mention
        return sorted(
            unique.values(),
            key=lambda mention: (mention.start_char, mention.end_char, mention.entity_type),
        )

    def _with_runtime_version(self, *, mention: EntityMention) -> EntityMention:
        """Normalize mention ids and versions to one runtime strategy label."""
        entity_id = _build_entity_id(
            chunk_id=mention.chunk_id,
            start_char=mention.start_char,
            end_char=mention.end_char,
            entity_type=mention.entity_type,
            extractor_version=self.config.extractor_version,
        )
        metadata = dict(cast(dict[str, object], mention.metadata or {}))
        if "channel" not in metadata:
            if "spacy" in mention.extractor_version:
                metadata["channel"] = "spacy"
            elif "gazetteer" in mention.extractor_version:
                metadata["channel"] = "gazetteer"
            else:
                metadata["channel"] = "rule"
        return EntityMention(
            entity_id=entity_id,
            chunk_id=mention.chunk_id,
            source_id=mention.source_id,
            source_document_id=mention.source_document_id,
            document_checksum=mention.document_checksum,
            start_char=mention.start_char,
            end_char=mention.end_char,
            mention_text=mention.mention_text,
            normalized_mention_text=mention.normalized_mention_text,
            entity_type=mention.entity_type,
            confidence=mention.confidence,
            extractor_version=self.config.extractor_version,
            canonical_entity_key=mention.canonical_entity_key,
            metadata=metadata,
        )


def _build_spacy_fastcoref_nlp(config: SpacyFastCorefEntityExtractorConfig) -> SpacyNlpLike:
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

    if config.enable_coref:
        try:
            import fastcoref  # noqa: F401

            if "fastcoref" not in nlp.pipe_names:
                nlp.add_pipe("fastcoref")
        except (ImportError, ValueError, AttributeError):
            _LOGGER.warning(
                "fastcoref component unavailable; continuing without coreference resolution",
            )
            return nlp
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


def _build_coref_canonical_lookup(doc: SpacyDocLike) -> dict[tuple[int, int], str]:
    """Build mention-span to canonical text lookup from fastcoref clusters."""
    lookup: dict[tuple[int, int], str] = {}
    doc_extensions = getattr(doc, "_", None)
    clusters = getattr(doc_extensions, "coref_clusters", None)
    if not isinstance(clusters, list):
        return lookup

    for cluster in clusters:
        mentions = _coref_mentions(cluster)
        if not mentions:
            continue
        canonical_text = mentions[0][2]
        for start_char, end_char, _mention_text in mentions:
            lookup[(start_char, end_char)] = canonical_text
    return lookup


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
