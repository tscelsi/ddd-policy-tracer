"""Deterministic chunking service for versioned source documents."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Protocol, TypedDict

import spacy
import spacy.tokens
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from ddd_policy_tracer.enrichment.embedding import EmbeddingService
from ddd_policy_tracer.ingestion.domain import SourceDocument

from .models import ChunkingConfig, DocumentChunk

_NLP = spacy.blank("en")
if "sentencizer" not in _NLP.pipe_names:
    _NLP.add_pipe("sentencizer")

MD_SPLIT_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]
MAX_CHUNK_SIZE_CHARS = 1500


@dataclass(frozen=True)
class _SentenceSpan:
    """Represent one sentence span within normalized document text."""

    start_char: int
    end_char: int


class _ChunkProvenance(TypedDict):
    """Type the source fields copied onto each document chunk."""

    metadata: dict[str, Any] | None
    published_at: str | None
    speaker_name: str | None
    talker_id: str | None
    party: str | None
    electorate: str | None
    house: str | None
    chamber: str | None
    bill_ids: list[str] | None
    debate_category: str | None
    debate_titles: list[str] | None
    speech_part_type: str | None


def get_chunker(source_id: str) -> Chunker:
    """Return one chunker instance for a given source."""
    if source_id in ["australia_institute", "committees"]:
        return MarkdownChunker(
            separators=["\n\n", "\n", " "],
        )
    elif source_id == "hansard":
        return MarkdownChunker(
            headers_to_split_on=[
                ("#;;#;;#;;", "__blank__"),
            ],  # won't find pattern so will default to splitting into equal parts
        )
    return SpacyChunker()


class Chunker(Protocol):
    """Chunk one source document version into text spans for claim extraction."""

    config: ChunkingConfig

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        """Initialise one SpacyChunker with chunking settings."""
        self.config = config or ChunkingConfig()

    def set_config(self, config: ChunkingConfig) -> None:
        """Update chunking settings on one SpacyChunker."""
        self.config = config

    def chunk_document(
        self,
        *,
        doc: SourceDocument,
    ) -> list[DocumentChunk]:
        """Chunk one source document version into text spans for claim extraction."""
        ...


class SpacyChunker(Chunker):
    """Chunk one source document version into text spans for claim extraction using spaCy."""

    def chunk_document(
        self,
        *,
        doc: SourceDocument,
    ) -> list[DocumentChunk]:
        """Split one document into stable sentence-based chunks."""
        _validate_chunking_config(self.config)

        normalized_text = doc.normalized_text.strip()
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
                chunk_size_chars=self.config.chunk_size_chars,
            )
            start_char = sentences[start_sentence_index].start_char
            end_char = sentences[end_sentence_index - 1].end_char
            chunk_text = normalized_text[start_char:end_char]
            chunk_id = build_chunk_id(
                source_id=doc.source_id,
                source_document_id=doc.source_document_id,
                checksum=doc.checksum,
                chunk_index=index,
                start_char=start_char,
                end_char=end_char,
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source_id=doc.source_id,
                    source_document_id=doc.source_document_id,
                    document_checksum=doc.checksum,
                    chunk_index=index,
                    start_char=start_char,
                    end_char=end_char,
                    chunk_text=chunk_text,
                    **_chunk_provenance(doc),
                ),
            )
            index += 1

            if end_sentence_index >= len(sentences):
                break

            start_sentence_index = _select_next_chunk_start_sentence(
                sentences=sentences,
                current_start_sentence_index=start_sentence_index,
                current_end_sentence_index=end_sentence_index,
                chunk_overlap_chars=self.config.chunk_overlap_chars,
            )

        return chunks


class MarkdownChunker(Chunker):
    """Chunk one source document version into markdown-based chunks using header structure."""

    def __init__(
        self,
        max_chunk_size_chars: int = 1500,
        separators: list[str] | None = None,
        headers_to_split_on: list[tuple[str, str]] | None = None,
    ) -> None:
        """Initialise one MarkdownChunker with a max chunk size."""
        super().__init__()
        self.max_chunk_size_chars = max_chunk_size_chars
        self._splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on or MD_SPLIT_HEADERS,
            strip_headers=False,
        )
        self._char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size_chars,
            chunk_overlap=0,
            separators=separators or ["\n\n", "\n", " ", ""],
        )

    def chunk_document(
        self,
        *,
        doc: SourceDocument,
    ) -> list[DocumentChunk]:
        """Chunk one document into markdown-based chunks."""
        chunks = self._splitter.split_text(doc.normalized_text)
        chunk_objs: list[DocumentChunk] = []
        start_char = 0  # 30
        for i, chunk in enumerate(chunks):
            if len(chunk.page_content) > self.max_chunk_size_chars:
                # split into equal-ish smaller parts to avoid OOM errors with embedding model
                n_sections = len(chunk.page_content) // self.max_chunk_size_chars + 1
                rough_char_split_size = len(chunk.page_content) // n_sections
                spacy_doc = _NLP(chunk.page_content)
                curr_span: spacy.tokens.Span | None = None
                for sent in spacy_doc.sents:
                    if curr_span is None:
                        curr_span = sent
                        continue
                    elif sent.end_char - curr_span.start_char >= rough_char_split_size:
                        end_char = start_char + curr_span.end_char
                        chunk_id = build_chunk_id(
                            source_id=doc.source_id,
                            source_document_id=doc.source_document_id,
                            checksum=doc.checksum,
                            chunk_index=i,
                            start_char=start_char,
                            end_char=end_char,
                        )
                        chunk_objs.append(
                            DocumentChunk(
                                chunk_id=chunk_id,
                                source_id=doc.source_id,
                                source_document_id=doc.source_document_id,
                                document_checksum=doc.checksum,
                                chunk_index=i,
                                start_char=start_char,
                                end_char=end_char,
                                chunk_text=curr_span.text.strip(),
                                **_chunk_provenance(doc),
                            ),
                        )
                        start_char = end_char + 1
                        curr_span = sent
                    else:
                        curr_span = spacy.tokens.Span(
                            spacy_doc,
                            start=curr_span.start,
                            end=sent.end,
                        )
                if curr_span is not None and curr_span.text.strip():
                    end_char = start_char + curr_span.end_char
                    chunk_id = build_chunk_id(
                        source_id=doc.source_id,
                        source_document_id=doc.source_document_id,
                        checksum=doc.checksum,
                        chunk_index=i,
                        start_char=start_char,
                        end_char=end_char,
                    )
                    chunk_objs.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            source_id=doc.source_id,
                            source_document_id=doc.source_document_id,
                            document_checksum=doc.checksum,
                            chunk_index=i,
                            start_char=start_char,
                            end_char=end_char,
                            chunk_text=curr_span.text.strip(),
                            **_chunk_provenance(doc),
                        ),
                    )
                start_char = end_char + 1
                continue
            else:
                end_char = start_char + len(chunk.page_content)
                chunk_id = build_chunk_id(
                    source_id=doc.source_id,
                    source_document_id=doc.source_document_id,
                    checksum=doc.checksum,
                    chunk_index=i,
                    start_char=start_char,
                    end_char=end_char,
                )
                chunk_objs.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        source_id=doc.source_id,
                        source_document_id=doc.source_document_id,
                        document_checksum=doc.checksum,
                        chunk_index=i,
                        start_char=start_char,
                        end_char=end_char,
                        chunk_text=chunk.page_content.strip(),
                        **_chunk_provenance(doc),
                    ),
                )
                start_char = end_char + 1
        return chunk_objs


class SemanticChunker(Chunker):
    # TODO: fix max chunk size to avoid OOM errors with embedding model.
    """Class for chunking a document into smaller, semantically similar chunks."""

    def __init__(self, max_iterations: int = 3, min_sentences_in_chunk: int = 3) -> None:
        """Initialise one SemanticChunker with a Spacy model and embedding service."""
        super().__init__()
        self.nlp = spacy.load("en_core_web_sm")
        self.embed_service = EmbeddingService()
        self.max_iterations = max_iterations
        self.min_sentences_in_chunk = min_sentences_in_chunk

    def set_threshold(self, threshold: float) -> None:
        """Set the distance threshold for merging chunks."""
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self.BREAKPOINT_DISTANCE_THRESHOLD = threshold

    def chunk_document(
        self,
        *,
        doc: SourceDocument,
    ) -> list[DocumentChunk]:
        """Chunk one document into stable sentence-based chunks.

        This is done by iteratively merging adjacent chunks with high semantic similarity.
        """
        if self.min_sentences_in_chunk < 1 or self.min_sentences_in_chunk > self.MAX_CHUNK_SENTS:
            raise ValueError(
                f"min_sentences_in_chunk must be at least 1 and at most {self.MAX_CHUNK_SENTS}",
            )
        spacy_doc = self.nlp(doc.normalized_text)
        sents = list(spacy_doc.sents)
        chunks: list[DocumentChunk] = []
        _min_sentences_in_chunk = min(self.min_sentences_in_chunk, len(sents))
        for i in range(0, len(sents), _min_sentences_in_chunk):
            end_i = min(i + _min_sentences_in_chunk - 1, len(sents) - 1)
            doc_chunk = DocumentChunk(
                chunk_id=build_chunk_id(
                    source_id=doc.source_id,
                    source_document_id=doc.source_document_id,
                    checksum=doc.checksum,
                    chunk_index=i,
                    start_char=sents[i].start_char,
                    end_char=sents[end_i].end_char,
                ),
                source_id=doc.source_id,
                source_document_id=doc.source_document_id,
                document_checksum=doc.checksum,
                chunk_index=i,
                start_char=sents[i].start_char,
                end_char=sents[end_i].end_char,
                chunk_text=spacy_doc.text[sents[i].start_char : sents[end_i].end_char].strip(),
                **_chunk_provenance(doc),
            )
            chunks.append(doc_chunk)
        i = 0
        while i < self.max_iterations:
            merged_chunks, has_changed = self._complete_pass(chunks)
            if not has_changed:
                break
            chunks = merged_chunks
            i += 1
        return chunks

    def _complete_pass(self, chunks: list[DocumentChunk]) -> tuple[list[DocumentChunk], bool]:
        """Complete one pass of semantic chunking by merging chunks below similarity threshold."""
        embs = self.embed_service.embed_many(
            [c.chunk_text for c in chunks],
            prompt_name="document",
        )
        dists = self._calculate_adjacent_distances(embs)
        merged_chunks: list[DocumentChunk] = []
        curr_chunk = chunks[0]
        for i, dist in enumerate(dists):
            if dist < self.BREAKPOINT_DISTANCE_THRESHOLD:
                # merge chunks when similarity is below threshold
                curr_chunk = DocumentChunk(
                    chunk_id=curr_chunk.chunk_id,
                    source_id=curr_chunk.source_id,
                    source_document_id=curr_chunk.source_document_id,
                    document_checksum=curr_chunk.document_checksum,
                    chunk_index=curr_chunk.chunk_index,
                    start_char=curr_chunk.start_char,
                    end_char=chunks[i + 1].end_char,
                    chunk_text=curr_chunk.chunk_text + " " + chunks[i + 1].chunk_text,
                    metadata=curr_chunk.metadata,
                    published_at=curr_chunk.published_at,
                    speaker_name=curr_chunk.speaker_name,
                    talker_id=curr_chunk.talker_id,
                    party=curr_chunk.party,
                    electorate=curr_chunk.electorate,
                    house=curr_chunk.house,
                    chamber=curr_chunk.chamber,
                    bill_ids=curr_chunk.bill_ids,
                    debate_category=curr_chunk.debate_category,
                    debate_titles=curr_chunk.debate_titles,
                    speech_part_type=curr_chunk.speech_part_type,
                )
            else:
                merged_chunks.append(curr_chunk)
                curr_chunk = chunks[i + 1]
        if curr_chunk:
            merged_chunks.append(curr_chunk)
        # merge sentences
        return merged_chunks, len(merged_chunks) != len(chunks)

    def _calculate_adjacent_distances(self, embeddings: list[list[float]]) -> list[float]:
        """Calculate L2 distance between adjacent chunk embeddings."""
        sims = []
        for i in range(len(embeddings) - 1):
            similarity = self.embed_service.similarity(embeddings[i], embeddings[i + 1])
            sims.append(similarity)
        return sims


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


def _chunk_provenance(doc: SourceDocument) -> _ChunkProvenance:
    """Map source provenance into explicit chunk fields while retaining source metadata."""
    metadata = doc.metadata or {}
    talker = metadata.get("talker")
    talker_data = talker if isinstance(talker, dict) else {}
    bill_ids = metadata.get("bill_ids")
    debate_titles = metadata.get("debate_titles")
    return {
        "metadata": {"doc_metadata": doc.metadata} if doc.metadata else None,
        "published_at": doc.published_at,
        "speaker_name": _optional_string(talker_data.get("name")),
        "talker_id": _optional_string(metadata.get("talker_id", talker_data.get("id"))),
        "party": _optional_string(talker_data.get("party")),
        "electorate": _optional_string(talker_data.get("electorate")),
        "house": _optional_string(metadata.get("house")),
        "chamber": _optional_string(metadata.get("chamber")),
        "bill_ids": _string_list(bill_ids),
        "debate_category": _optional_string(metadata.get("debate_category")),
        "debate_titles": _string_list(debate_titles),
        "speech_part_type": _optional_string(metadata.get("speech_part_type")),
    }


def _optional_string(value: object) -> str | None:
    """Return a metadata value only when it is a string."""
    return value if isinstance(value, str) else None


def _string_list(value: object) -> list[str] | None:
    """Return a copied list containing only string metadata values."""
    if not isinstance(value, list):
        return None
    return [item for item in value if isinstance(item, str)]


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


def build_chunk_id(
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
