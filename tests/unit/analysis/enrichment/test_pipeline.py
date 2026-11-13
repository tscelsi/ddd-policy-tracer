from ddd_policy_tracer.enrichment.chunking.models import DocumentChunk


def _sample_chunk(*, document_checksum: str = "checksum-1", text_length: int = 10) -> DocumentChunk:
    """Build one representative chunk record fixture."""
    return DocumentChunk(
        chunk_id="chunk_abc",
        source_id="australia_institute",
        source_document_id="https://example.org/report-1",
        document_checksum=document_checksum,
        chunk_index=0,
        start_char=0,
        end_char=10,
        chunk_text="a" * text_length,
    )


def test_filter_by_length_layer() -> None:
    """Verify that the FilterByLengthLayer correctly filters chunks by length."""
    from ddd_policy_tracer.enrichment.pipeline.layers import FilterByLengthLayer

    layer = FilterByLengthLayer(min_length=5, max_length=10)
    too_small_chunk = _sample_chunk(text_length=4)  # Should be filtered out (length 4)
    too_large_chunk = _sample_chunk(text_length=11)  # Should be filtered out (length 11)
    valid_chunk = _sample_chunk(text_length=10)  # Should
    assert layer.call(chunks=[too_small_chunk, too_large_chunk, valid_chunk]) == [valid_chunk]


async def test_pipeline_runs() -> None:
    """Verify that the Pipeline correctly runs multiple layers in sequence."""
    from ddd_policy_tracer.enrichment.pipeline.layers import FilterByLengthLayer
    from ddd_policy_tracer.enrichment.pipeline.pipeline_base import Pipeline

    layer1 = FilterByLengthLayer(min_length=5, max_length=10)
    layer2 = FilterByLengthLayer(min_length=8, max_length=12)
    pipeline = Pipeline(layers=[layer1, layer2])

    chunk1 = _sample_chunk(text_length=4)  # Filtered out by layer1
    chunk2 = _sample_chunk(text_length=7)  # Passes layer1, filtered out by layer2
    chunk3 = _sample_chunk(text_length=10)  # Passes both layers
    result = await pipeline.run(chunks=[chunk1, chunk2, chunk3])
    assert result == [chunk3]
