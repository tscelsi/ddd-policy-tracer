"""Deterministic chunking service for versioned source documents."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import spacy

from ddd_policy_tracer.discovery.domain import SourceDocumentVersion

from .chunking_models import ChunkingConfig, DocumentChunk

_NLP = spacy.blank("en")
if "sentencizer" not in _NLP.pipe_names:
    _NLP.add_pipe("sentencizer")


@dataclass(frozen=True)
class _SentenceSpan:
    """Represent one sentence span within normalized document text."""

    start_char: int
    end_char: int


def chunk_document_version(
    *,
    version: SourceDocumentVersion,
    config: ChunkingConfig | None = None,
) -> list[DocumentChunk]:
    """Split one document version into stable sentence-based chunks."""
    active_config = config or ChunkingConfig()
    _validate_chunking_config(active_config)

    normalized_text = version.normalized_text.strip()
    if not normalized_text:
        return []

    sentences = _segment_sentences(normalized_text)
    chunks: list[DocumentChunk] = []
    index = 0
    start_sentence_index = 0

    while start_sentence_index < len(sentences):
        end_sentence_index = _select_chunk_end_sentence(
            sentences=sentences,
            start_sentence_index=start_sentence_index,
            chunk_size_chars=active_config.chunk_size_chars,
        )
        start_char = sentences[start_sentence_index].start_char
        end_char = sentences[end_sentence_index - 1].end_char
        chunk_text = normalized_text[start_char:end_char]
        chunk_id = _build_chunk_id(
            source_id=version.source_id,
            source_document_id=version.source_document_id,
            checksum=version.checksum,
            chunk_index=index,
            start_char=start_char,
            end_char=end_char,
        )
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                source_id=version.source_id,
                source_document_id=version.source_document_id,
                document_checksum=version.checksum,
                chunk_index=index,
                start_char=start_char,
                end_char=end_char,
                chunk_text=chunk_text,
            ),
        )
        index += 1

        if end_sentence_index >= len(sentences):
            break

        start_sentence_index = _select_next_chunk_start_sentence(
            sentences=sentences,
            current_start_sentence_index=start_sentence_index,
            current_end_sentence_index=end_sentence_index,
            chunk_overlap_chars=active_config.chunk_overlap_chars,
        )

    return chunks


def _segment_sentences(normalized_text: str) -> list[_SentenceSpan]:
    """Segment normalized text into sentence spans with stable offsets."""
    doc = _NLP(normalized_text)
    sentences = [
        _SentenceSpan(start_char=sent.start_char, end_char=sent.end_char)
        for sent in doc.sents
        if sent.text.strip()
    ]
    if sentences:
        return sentences
    return [_SentenceSpan(start_char=0, end_char=len(normalized_text))]


def _select_chunk_end_sentence(
    *,
    sentences: list[_SentenceSpan],
    start_sentence_index: int,
    chunk_size_chars: int,
) -> int:
    """Return the exclusive end sentence index for one chunk window."""
    chunk_start_char = sentences[start_sentence_index].start_char
    end_sentence_index = start_sentence_index + 1

    while end_sentence_index < len(sentences):
        candidate_end_char = sentences[end_sentence_index].end_char
        candidate_size = candidate_end_char - chunk_start_char
        if candidate_size > chunk_size_chars:
            break
        end_sentence_index += 1

    return end_sentence_index


def _select_next_chunk_start_sentence(
    *,
    sentences: list[_SentenceSpan],
    current_start_sentence_index: int,
    current_end_sentence_index: int,
    chunk_overlap_chars: int,
) -> int:
    """Return the next start sentence index using sentence-level overlap."""
    if chunk_overlap_chars <= 0:
        return current_end_sentence_index

    overlap_start_index = current_end_sentence_index
    retained_chars = 0

    for sentence_index in range(
        current_end_sentence_index - 1,
        current_start_sentence_index,
        -1,
    ):
        sentence = sentences[sentence_index]
        sentence_size = sentence.end_char - sentence.start_char
        if retained_chars + sentence_size > chunk_overlap_chars:
            if overlap_start_index == current_end_sentence_index:
                overlap_start_index = sentence_index
            break
        retained_chars += sentence_size
        overlap_start_index = sentence_index

    return overlap_start_index


def _validate_chunking_config(config: ChunkingConfig) -> None:
    """Reject chunking settings that cannot make forward progress."""
    if config.chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars must be greater than zero")
    if config.chunk_overlap_chars < 0:
        raise ValueError("chunk_overlap_chars must be zero or greater")
    if config.chunk_overlap_chars >= config.chunk_size_chars:
        raise ValueError(
            "chunk_overlap_chars must be less than chunk_size_chars",
        )


def _build_chunk_id(
    *,
    source_id: str,
    source_document_id: str,
    checksum: str,
    chunk_index: int,
    start_char: int,
    end_char: int,
) -> str:
    """Create a stable identifier for one chunk of one document version."""
    raw_id = f"{source_id}|{source_document_id}|{checksum}|{chunk_index}|{start_char}|{end_char}"
    digest = sha256(raw_id.encode("utf-8")).hexdigest()
    return f"chunk_{digest[:16]}"
