"""Orchestrate parliamentary speech retrieval and grounded answer synthesis."""

from __future__ import annotations

from ddd_policy_tracer.answering.service_layer import AnswerSynthesizer, SynthesizedAnswer
from ddd_policy_tracer.retrieval.service_layer import (
    HybridChunkRetriever,
    QueryEmbedder,
    retrieve_ranked_evidence,
)


class RagCompletionService:
    """Run hybrid evidence retrieval followed by citation-backed synthesis."""

    def __init__(
        self,
        *,
        retriever: HybridChunkRetriever,
        embedding_service: QueryEmbedder,
        synthesizer: AnswerSynthesizer,
    ) -> None:
        """Accept reusable retrieval and answer-generation dependencies."""
        self._retriever = retriever
        self._embedding_service = embedding_service
        self._synthesizer = synthesizer

    def complete(self, question: str, *, limit: int = 10) -> SynthesizedAnswer:
        """Retrieve ranked speech evidence and synthesize a cited answer."""
        evidence = retrieve_ranked_evidence(
            query=question,
            retriever=self._retriever,
            embedding_service=self._embedding_service,
            limit=limit,
        )
        return self._synthesizer.synthesize(question=question, evidence=evidence)
