"""Unit tests for rule-based sentence claim extraction behavior."""

from ddd_policy_tracer.analysis.chunking_models import DocumentChunk
from ddd_policy_tracer.analysis.claims import (
    RuleBasedClaimExtractorConfig,
    RuleBasedSentenceClaimExtractor,
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


def test_rule_based_extractor_emits_claim_for_policy_modality_sentence() -> None:
    """Emit one claim when sentence includes strong modality cues."""
    extractor = RuleBasedSentenceClaimExtractor()
    chunk = _chunk("Government should ban new coal projects.")

    claims = extractor.extract(chunk=chunk)

    assert len(claims) == 1
    claim = claims[0]
    assert claim.chunk_id == chunk.chunk_id
    assert claim.source_document_id == chunk.source_document_id
    assert claim.document_checksum == chunk.document_checksum
    assert claim.evidence_text == "Government should ban new coal projects."
    assert claim.normalized_claim_text == claim.evidence_text
    assert claim.confidence == 1.0
    assert claim.claim_type is None
    assert claim.extractor_version == "rules-v1"


def test_rule_based_extractor_uses_weighted_cues_and_global_threshold() -> None:
    """Emit no claim when sentence score stays below configured threshold."""
    extractor = RuleBasedSentenceClaimExtractor(
        RuleBasedClaimExtractorConfig(threshold=0.8),
    )
    chunk = _chunk("The report shows mixed public sentiment.")

    claims = extractor.extract(chunk=chunk)

    assert claims == []


def test_rule_based_extractor_processes_one_claim_per_sentence() -> None:
    """Limit extraction output to one claim candidate per sentence."""
    extractor = RuleBasedSentenceClaimExtractor(
        RuleBasedClaimExtractorConfig(threshold=0.2),
    )
    chunk = _chunk(
        "Policy should change now."
        " It leads to better outcomes and reduces costs by 10%.",
    )

    claims = extractor.extract(chunk=chunk)

    assert len(claims) == 2
    assert claims[0].evidence_text == "Policy should change now."
    assert (
        claims[1].evidence_text
        == "It leads to better outcomes and reduces costs by 10%."
    )


def test_rule_based_extractor_requires_at_least_one_cue() -> None:
    """Skip sentence when no configured cue categories are detected."""
    extractor = RuleBasedSentenceClaimExtractor(
        RuleBasedClaimExtractorConfig(threshold=0.1),
    )
    chunk = _chunk("This sentence has plain prose without cues")

    claims = extractor.extract(chunk=chunk)

    assert claims == []


def test_rule_based_extractor_counts_processed_sentences() -> None:
    """Count sentence units for service report semantics."""
    extractor = RuleBasedSentenceClaimExtractor()
    chunk = _chunk("One sentence. Two sentences? Three sentences!")

    processed_sentences = extractor.count_processed_sentences(chunk=chunk)

    assert processed_sentences == 3


def test_rule_based_extractor_skips_acknowledgements_headings_and_labels() -> (
    None
):
    """Skip configured boilerplate headings/labels from extraction output."""
    extractor = RuleBasedSentenceClaimExtractor(
        RuleBasedClaimExtractorConfig(threshold=0.1),
    )
    chunk = _chunk(
        "Acknowledgements."
        " References."
        " Table 1:."
        " Key Findings:."
        " Policy should reduce emissions by 4.9%.",
    )

    claims = extractor.extract(chunk=chunk)

    assert [claim.evidence_text for claim in claims] == [
        "Policy should reduce emissions by 4.9%.",
    ]
    assert len(claims) == 1
    assert claims[0].evidence_text == "Policy should reduce emissions by 4.9%."


def test_rule_based_extractor_skips_table_and_quote_block_sentences() -> None:
    """Skip table-like and standalone quoted block sentence candidates."""
    extractor = RuleBasedSentenceClaimExtractor(
        RuleBasedClaimExtractorConfig(threshold=0.1),
    )
    chunk = _chunk(
        "Revenue | Cost | Margin."
        " \"Government should ban new coal projects\"."
        " The government should ban new coal projects.",
    )

    claims = extractor.extract(chunk=chunk)

    assert len(claims) == 1
    assert (
        claims[0].evidence_text
        == "The government should ban new coal projects."
    )


def test_rule_based_extractor_offsets_align_with_chunk_text() -> None:
    """Emit offsets that exactly map to sentence evidence text slices."""
    extractor = RuleBasedSentenceClaimExtractor()
    text = (
        "Intro sentence. "
        "Government should reduce emissions by 4.9% each year."
    )
    chunk = _chunk(text)

    claims = extractor.extract(chunk=chunk)

    assert len(claims) == 1
    claim = claims[0]
    assert text[claim.start_char : claim.end_char] == claim.evidence_text
