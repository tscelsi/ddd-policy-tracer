"""Unit tests for rule-based sentence entity extraction behavior."""

from __future__ import annotations

from ddd_policy_tracer.analysis.chunking_models import DocumentChunk
from ddd_policy_tracer.analysis.entities import (
    RuleBasedEntityExtractorConfig,
    RuleBasedSentenceEntityExtractor,
)


def _chunk(text: str) -> DocumentChunk:
    """Build one representative chunk fixture for extractor tests."""
    return DocumentChunk(
        chunk_id="chunk_test",
        source_id="australia_institute",
        source_document_id="https://example.org/report-1",
        document_checksum="checksum-1",
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        chunk_text=text,
    )


def test_rule_based_entity_extractor_emits_strict_v1_types() -> None:
    """Emit only strict entity types from deterministic pattern cues."""
    extractor = RuleBasedSentenceEntityExtractor()
    chunk = _chunk(
        "The Clean Energy Act was discussed by Australia Institute in Queensland.",
    )

    entities = extractor.extract(chunk=chunk)

    assert [entity.entity_type for entity in entities] == ["POLICY", "ORG", "JURISDICTION"]
    assert [entity.mention_text for entity in entities] == [
        "Clean Energy Act",
        "Australia Institute",
        "Queensland",
    ]


def test_rule_based_entity_extractor_filters_by_threshold() -> None:
    """Skip candidates when configured type threshold exceeds confidence."""
    extractor = RuleBasedSentenceEntityExtractor(
        RuleBasedEntityExtractorConfig(policy_threshold=1.1),
    )
    chunk = _chunk("The Clean Energy Act should reduce emissions.")

    entities = extractor.extract(chunk=chunk)

    assert entities == []


def test_rule_based_entity_extractor_resolves_overlap_by_precedence() -> None:
    """Keep policy mention over overlapping org mention by precedence."""
    extractor = RuleBasedSentenceEntityExtractor(
        RuleBasedEntityExtractorConfig(
            org_threshold=0.0,
            precedence=("POLICY", "ORG", "PROGRAM", "PERSON", "JURISDICTION"),
        ),
    )
    chunk = _chunk("The Climate Institute Policy must be implemented.")

    entities = extractor.extract(chunk=chunk)

    assert len(entities) == 1
    assert entities[0].entity_type == "POLICY"
    assert entities[0].mention_text == "Climate Institute Policy"


def test_rule_based_entity_extractor_uses_deterministic_entity_id() -> None:
    """Generate stable IDs from chunk/span/type/version fields."""
    extractor = RuleBasedSentenceEntityExtractor()
    chunk = _chunk("The Clean Energy Act applies in Australia.")

    first = extractor.extract(chunk=chunk)
    second = extractor.extract(chunk=chunk)

    assert len(first) >= 1
    assert [entity.entity_id for entity in first] == [entity.entity_id for entity in second]


def test_rule_based_entity_extractor_counts_processed_sentences() -> None:
    """Count sentence units for service report semantics."""
    extractor = RuleBasedSentenceEntityExtractor()
    chunk = _chunk("One sentence. Two sentences? Three sentences!")

    processed_sentences = extractor.count_processed_sentences(chunk=chunk)

    assert processed_sentences == 3


def test_rule_based_entity_extractor_offsets_align_with_chunk_text() -> None:
    """Emit offsets that exactly map to mention text slices."""
    extractor = RuleBasedSentenceEntityExtractor()
    text = "Intro sentence. The Clean Energy Act was discussed by Australia Institute."
    chunk = _chunk(text)

    entities = extractor.extract(chunk=chunk)

    assert len(entities) >= 2
    for entity in entities:
        assert text[entity.start_char : entity.end_char] == entity.mention_text
