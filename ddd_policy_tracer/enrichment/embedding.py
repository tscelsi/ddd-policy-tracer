"""Utility functions for generating text embeddings using a sentence transformer model."""

import logging
from typing import Literal, Protocol

from sentence_transformers import SentenceTransformer
from torch import Tensor

logger = logging.getLogger(__name__)


class AbstractEmbeddingService(Protocol):
    """Protocol for embedding service implementations."""

    def embed_one(
        self,
        text: str,
        prompt_name: Literal["query", "document"] = "document",
        return_tensor: bool = False,
    ) -> list[float] | Tensor:
        """Generate embeddings for a single document."""
        ...

    def embed_many(
        self,
        texts: list[str],
        prompt_name: Literal["query", "document"] = "document",
        return_tensor: bool = False,
    ) -> list[list[float]] | Tensor:
        """Generate embeddings for many documents."""
        ...

    def similarity(
        self,
        query_embedding: list[float],
        document_embedding: list[float],
    ) -> float:
        """Compute similarity between one query embedding and one document embedding."""
        ...

    def similarity_many(
        self,
        query_embeddings: list[list[float]],
        document_embeddings: list[list[float]],
    ) -> list[list[float]]:
        """Compute similarity between many query embeddings and many document embeddings."""
        ...


class EmbeddingService(AbstractEmbeddingService):
    """Service for embedding catalog entities using a sentence transformer model."""

    _instance: "EmbeddingService | None" = None
    _batch_size: int = 16

    @staticmethod
    def get_instance() -> "EmbeddingService":
        """Get a singleton instance of the EmbedService."""
        if EmbeddingService._instance is None:
            EmbeddingService._instance = EmbeddingService()
        return EmbeddingService._instance

    def __init__(self) -> None:
        """Initialize the embedding model for catalog entity representation."""
        self.model = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B",
            # model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
            # tokenizer_kwargs={"padding_side": "left"},
        )

    def embed_one(
        self,
        text: str,
        prompt_name: Literal["query", "document"] = "document",
        return_tensor: bool = False,
    ) -> list[float] | Tensor:
        """Generate embeddings for a single document."""
        embs = self.model.encode(text, prompt_name=prompt_name)  # type: ignore
        if return_tensor:
            return embs
        return embs.tolist()  # type: ignore

    def embed_many(
        self,
        texts: list[str],
        prompt_name: Literal["query", "document"] = "document",
        return_tensor: bool = False,
    ) -> list[list[float]] | Tensor:
        """Generate embeddings for many documents."""
        emb_tens = self.model.encode(  # type: ignore
            texts,
            prompt_name=prompt_name,
        )
        if return_tensor:
            return emb_tens
        return emb_tens.tolist()  # type: ignore

    def similarity(
        self,
        query_embedding: list[float],
        document_embedding: list[float],
    ) -> float:
        """Compute similarity between one query embedding and one document embedding."""
        return self.model.similarity([query_embedding], [document_embedding])[0][0]  # type: ignore

    def similarity_many(
        self,
        query_embeddings: list[list[float]],
        document_embeddings: list[list[float]],
    ) -> list[list[float]]:
        """Compute similarity between many query embeddings and many document embeddings."""
        return self.model.similarity(query_embeddings, document_embeddings).tolist()  # type: ignore
