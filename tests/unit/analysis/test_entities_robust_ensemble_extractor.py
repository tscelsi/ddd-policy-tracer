"""Unit tests for robust ensemble mention generation behavior."""

from __future__ import annotations

from dataclasses import dataclass

from ddd_policy_tracer.analysis.chunks.chunking_models import DocumentChunk
from ddd_policy_tracer.analysis.entities import EntityMention, EntityType
from ddd_policy_tracer.analysis.entities.extractors import (
    RobustEnsembleEntityExtractor,
    RobustEnsembleEntityExtractorConfig,
)


def _chunk(text: str) -> DocumentChunk:
    """Build one representative chunk fixture for ensemble tests."""
    return DocumentChunk(
        chunk_id="chunk_1",
        source_id="australia_institute",
        source_document_id="https://example.org/report-1",
        document_checksum="checksum-1",
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        chunk_text=text,
    )


def _mention(
    *,
    chunk: DocumentChunk,
    start_char: int,
    end_char: int,
    mention_text: str,
    entity_type: EntityType,
    confidence: float,
    extractor_version: str,
    channel: str,
) -> EntityMention:
    """Build one mention fixture with explicit source channel metadata."""
    return EntityMention(
        entity_id=f"entity_{start_char}_{end_char}_{entity_type}",
        chunk_id=chunk.chunk_id,
        source_id=chunk.source_id,
        source_document_id=chunk.source_document_id,
        document_checksum=chunk.document_checksum,
        start_char=start_char,
        end_char=end_char,
        mention_text=mention_text,
        normalized_mention_text=mention_text,
        entity_type=entity_type,
        confidence=confidence,
        extractor_version=extractor_version,
        metadata={"channel": channel},
    )


@dataclass(frozen=True)
class StubExtractor:
    """Return fixed mentions for deterministic ensemble testing."""

    mentions: list[EntityMention]

    def extract(self, *, chunk: DocumentChunk) -> list[EntityMention]:
        """Return configured mentions for one chunk."""
        _ = chunk
        return list(self.mentions)

    def extract_many(self, *, chunks: list[DocumentChunk]) -> list[EntityMention]:
        """Return configured mentions for each chunk in order."""
        return [mention for _ in chunks for mention in self.mentions]

    def count_processed_sentences(self, *, chunk: DocumentChunk) -> int:
        """Return one sentence count to satisfy extractor interface."""
        _ = chunk
        return 1


def test_robust_ensemble_resolves_overlap_by_source_priority() -> None:
    """Prefer spaCy mention over overlapping rule mention at equal confidence."""
    text = "The Clean Energy Act was discussed."
    chunk = _chunk(text)
    rule = _mention(
        chunk=chunk,
        start_char=4,
        end_char=20,
        mention_text="Clean Energy Act",
        entity_type="POLICY",
        confidence=0.9,
        extractor_version="robust-ensemble-v1-rule",
        channel="rule",
    )
    spacy = _mention(
        chunk=chunk,
        start_char=4,
        end_char=20,
        mention_text="Clean Energy Act",
        entity_type="POLICY",
        confidence=0.9,
        extractor_version="robust-ensemble-v1-spacy",
        channel="spacy",
    )

    extractor = RobustEnsembleEntityExtractor(
        config=RobustEnsembleEntityExtractorConfig(extractor_version="robust-ensemble-v1"),
        rule_extractor=StubExtractor([rule]),
        spacy_extractor=StubExtractor([spacy]),
    )

    entities = extractor.extract(chunk=chunk)

    assert len(entities) == 1
    assert entities[0].extractor_version == "robust-ensemble-v1"
    assert entities[0].metadata == {"channel": "spacy"}


def test_robust_ensemble_extract_many_is_deterministic() -> None:
    """Return stable mention ordering across repeated batch extractions."""
    chunk = _chunk("Queensland Program by Australia Institute.")
    spacy = _mention(
        chunk=chunk,
        start_char=0,
        end_char=18,
        mention_text="Queensland Program",
        entity_type="PROGRAM",
        confidence=0.95,
        extractor_version="robust-ensemble-v1-spacy",
        channel="spacy",
    )
    extractor = RobustEnsembleEntityExtractor(
        config=RobustEnsembleEntityExtractorConfig(extractor_version="robust-ensemble-v1"),
        rule_extractor=StubExtractor([]),
        spacy_extractor=StubExtractor([spacy]),
    )

    first = extractor.extract_many(chunks=[chunk, chunk])
    second = extractor.extract_many(chunks=[chunk, chunk])

    assert [entity.entity_id for entity in first] == [entity.entity_id for entity in second]
