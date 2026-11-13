"""Pipeline layers for enriching document chunks with embeddings and topic classifications."""

import logging
from collections.abc import Callable
from typing import Literal, Protocol

from gliner2 import GLiNER2  # requires gliner2[local]

from ddd_policy_tracer.enrichment.chunking.models import DocumentChunk, Topic
from ddd_policy_tracer.enrichment.chunking.service_layer import generate_embeddings
from ddd_policy_tracer.enrichment.embedding import EmbeddingService
from ddd_policy_tracer.enrichment.topics import topics_sm
from ddd_policy_tracer.utils.events.local import LocalPublisher

logger = logging.getLogger(__name__)


class BaseLayer(Protocol):
    """Base class for enrichment layers in the pipeline."""

    _should_save_embeddings_after: bool = True
    _should_save_objects_after: bool = True

    def call(
        self,
        chunks: list[DocumentChunk],
        event_publisher: LocalPublisher | None = None,
    ) -> list[DocumentChunk]:
        """Process a list of DocumentChunk records and return enriched versions."""
        return []

    def should_save(self, type: Literal["embeddings", "object"]) -> bool:
        """Indicate whether the pipeline should persist chunks after this layer."""
        if type == "embeddings":
            return self._should_save_embeddings_after
        elif type == "object":
            return self._should_save_objects_after
        raise ValueError(f"Unsupported save type: {type}")


class Filter(BaseLayer):
    """Filter chunks based on a provided condition."""

    def __init__(
        self,
        condition: Callable[[DocumentChunk], bool],
    ) -> None:
        """Initialize the filter layer with a condition to evaluate."""
        self._condition = condition
        self._should_save_after = False  # No need to persist after filtering.

    def call(
        self,
        chunks: list[DocumentChunk],
        event_publisher: LocalPublisher | None = None,
    ) -> list[DocumentChunk]:
        """Filter chunks based on the provided condition."""
        _ = event_publisher  # Not used in this layer, but included for interface consistency
        return [chunk for chunk in chunks if self._condition(chunk)]


class Embed(BaseLayer):
    """Generate embeddings for chunks using an embedding service."""

    def __init__(self, embedding_service: EmbeddingService) -> None:
        """Initialize the embedding layer with a service to generate embeddings."""
        self._embedding_service = embedding_service
        self._should_save_after = True  # We want to persist the generated embeddings.

    def call(
        self,
        chunks: list[DocumentChunk],
        event_publisher: LocalPublisher | None = None,
    ) -> list[DocumentChunk]:
        """Generate embeddings for each chunk and return enriched chunks."""
        _ = event_publisher  # Not used in this layer, but included for interface consistency
        chunks = generate_embeddings(
            chunks=chunks,
            embedding_service=self._embedding_service,
        )
        return chunks


class ClassifyTopics(BaseLayer):
    """Tag chunks with topics using GLiNER, a zero-shot topic classification model."""

    def __init__(
        self,
        batch_size: int = 8,
        include_confidence: bool = True,
        cls_threshold: float = 0.5,
        topics: dict[str, str] = topics_sm,
    ) -> None:
        """Initialize the topic classification layer."""
        self._extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
        self._batch_size = batch_size
        self._include_confidence = include_confidence
        self._cls_threshold = cls_threshold
        self._should_save_after = True  # We want to persist the assigned topics.
        self._topics = topics

    def call(
        self,
        chunks: list[DocumentChunk],
        event_publisher: LocalPublisher | None = None,
    ) -> list[DocumentChunk]:
        """Classify each chunk into topics and return enriched chunks."""
        _ = event_publisher  # Not used in this layer, but included for interface consistency
        classification_results = self._extractor.batch_classify_text(
            [c.chunk_text for c in chunks],
            {
                "topics": {
                    "labels": self._topics,
                    "multi_label": True,
                    "cls_threshold": self._cls_threshold,
                },
            },
            batch_size=self._batch_size,
            include_confidence=self._include_confidence,
        )
        for idx, (chunk, result) in enumerate(zip(chunks, classification_results, strict=True)):
            formatted_topics: list[Topic] = [
                {"topic": t["label"], "confidence": t["confidence"]} for t in result["topics"]
            ]
            sorted_topics = sorted(formatted_topics, key=lambda x: x["confidence"], reverse=True)
            chunks[idx] = chunk.update_topics(sorted_topics)
        return chunks
