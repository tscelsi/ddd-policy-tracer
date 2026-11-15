"""Orchestrate parliamentary speech retrieval and grounded answer synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from ddd_policy_tracer.answering.service_layer import AnswerSynthesizer, SynthesizedAnswer
from ddd_policy_tracer.retrieval.service_layer import (
    HybridChunkRetriever,
    QueryEmbedder,
    embed_query,
)


@dataclass(frozen=True)
class CompletionTimings:
    """Measure the major synchronous stages of one completion request."""

    embedding_ms: float
    hybrid_retrieval_ms: float
    generation_ms: float


@dataclass(frozen=True)
class CompletionRun:
    """Return one synthesized answer with its stage timings and evidence count."""

    answer: SynthesizedAnswer
    timings: CompletionTimings
    evidence_count: int


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

    def complete(self, question: str, *, limit: int = 10) -> CompletionRun:
        """Retrieve and synthesize a cited answer while measuring each major stage."""
        stage_started = perf_counter()
        query_embedding = embed_query(
            question,
            embedding_service=self._embedding_service,
        )
        embedding_ms = (perf_counter() - stage_started) * 1000

        stage_started = perf_counter()
        evidence = self._retriever.search_hybrid(
            query=question,
            query_embedding=query_embedding,
            limit=limit,
        )
        hybrid_retrieval_ms = (perf_counter() - stage_started) * 1000

        stage_started = perf_counter()
        answer = self._synthesizer.synthesize(question=question, evidence=evidence)
        generation_ms = (perf_counter() - stage_started) * 1000

        return CompletionRun(
            answer=answer,
            timings=CompletionTimings(
                embedding_ms=embedding_ms,
                hybrid_retrieval_ms=hybrid_retrieval_ms,
                generation_ms=generation_ms,
            ),
            evidence_count=len(evidence),
        )
