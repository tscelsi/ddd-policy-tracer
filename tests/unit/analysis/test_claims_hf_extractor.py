"""Unit tests for Hugging Face-backed claim extraction behavior."""

from __future__ import annotations

from ddd_policy_tracer.analysis.chunks.chunking_models import DocumentChunk
from ddd_policy_tracer.analysis.claims import (
    HuggingFaceClaimExtractor,
    HuggingFaceClaimExtractorConfig,
)


def _chunk(text: str) -> DocumentChunk:
    """Build one representative chunk fixture for extractor tests."""
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


def test_hf_extractor_maps_generated_claims_to_chunk_spans() -> None:
    """Extract sentence claims and map valid offsets to claim candidates."""
    chunk = _chunk(
        "Government should ban new coal projects. "
        "Policy should reduce emissions by 4.9%.",
    )

    def fake_generator(text: str, max_new_tokens: int) -> list[dict[str, str]]:
        """Return deterministic generated claim sentences for one chunk."""
        assert text == chunk.chunk_text
        assert max_new_tokens == 64
        return [
            {
                "generated_text": (
                    "Government should ban new coal projects. "
                    "Policy should reduce emissions by 4.9%."
                ),
            },
        ]

    extractor = HuggingFaceClaimExtractor(
        config=HuggingFaceClaimExtractorConfig(max_new_tokens=64),
        generator=fake_generator,
    )

    claims = extractor.extract(chunk=chunk)

    assert len(claims) == 2
    assert claims[0].evidence_text == "Government should ban new coal projects."
    assert claims[1].evidence_text == "Policy should reduce emissions by 4.9%."
    assert claims[0].extractor_version == "hf-t5-babelscape-v1"


def test_hf_extractor_skips_generated_claims_missing_from_chunk() -> None:
    """Drop generated claim strings that cannot be aligned to chunk text."""
    chunk = _chunk("Government should ban new coal projects.")

    def fake_generator(_: str, max_new_tokens: int) -> list[dict[str, str]]:
        """Return one paraphrased claim not present in source chunk."""
        assert max_new_tokens == 256
        return [{"generated_text": "Government should ban coal immediately."}]

    extractor = HuggingFaceClaimExtractor(generator=fake_generator)

    claims = extractor.extract(chunk=chunk)

    assert claims == []
