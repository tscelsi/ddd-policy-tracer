from ddd_policy_tracer.enrichment.chunking.models import DocumentChunk


def _sample_chunk(*, document_checksum: str = "checksum-1") -> DocumentChunk:
    """Build one representative chunk record fixture."""
    return DocumentChunk(
        chunk_id="chunk_abc",
        source_id="australia_institute",
        source_document_id="https://example.org/report-1",
        document_checksum=document_checksum,
        chunk_index=0,
        start_char=0,
        end_char=10,
        chunk_text="abcdefghij",
    )
