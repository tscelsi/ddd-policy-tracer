"""Orchestrate parliamentary speech retrieval and grounded answer synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from ddd_policy_tracer.answering.metadata_filter_extraction import MetadataFilterExtractor
from ddd_policy_tracer.answering.service_layer import AnswerSynthesizer, SynthesizedAnswer
from ddd_policy_tracer.answering.speaker_resolution import SpeakerResolver
from ddd_policy_tracer.retrieval.service_layer import (
    HybridChunkRetriever,
    QueryEmbedder,
    RetrievalFilters,
    embed_query,
)


@dataclass(frozen=True)
class CompletionTimings:
    """Measure the major synchronous stages of one completion request."""

    filter_extraction_ms: float
    speaker_resolution_ms: float
    embedding_ms: float
    hybrid_retrieval_ms: float
    generation_ms: float


@dataclass(frozen=True)
class CompletionRun:
    """Return one synthesized answer with its stage timings and evidence count."""

    answer: SynthesizedAnswer | None
    timings: CompletionTimings
    evidence_count: int
    clarification_question: str | None = None

    def __post_init__(self) -> None:
        """Keep clarification outcomes separate from answer and abstention outcomes."""
        if self.answer is None and self.clarification_question is None:
            raise ValueError("completion run must contain an answer or clarification")
        if self.answer is not None and self.clarification_question is not None:
            raise ValueError("completion run cannot contain both an answer and clarification")
        if self.clarification_question is not None and not self.clarification_question.strip():
            raise ValueError("clarification_question must not be empty")

    @property
    def clarification_required(self) -> bool:
        """Report whether retrieval stopped to clarify an unsupported or unclear filter."""
        return self.clarification_question is not None


class RagCompletionService:
    """Run hybrid evidence retrieval followed by citation-backed synthesis."""

    def __init__(
        self,
        *,
        retriever: HybridChunkRetriever,
        embedding_service: QueryEmbedder,
        synthesizer: AnswerSynthesizer,
        filter_extractor: MetadataFilterExtractor | None = None,
        speaker_resolver: SpeakerResolver | None = None,
    ) -> None:
        """Accept the reusable retrieval, answer, and optional metadata-filter collaborators."""
        self._retriever = retriever
        self._embedding_service = embedding_service
        self._synthesizer = synthesizer
        self._filter_extractor = filter_extractor
        self._speaker_resolver = speaker_resolver

    def complete(self, question: str, *, limit: int = 10) -> CompletionRun:
        """Resolve constraints, retrieve matching evidence, and synthesize a cited answer."""
        filters: RetrievalFilters | None = None
        filter_extraction_ms = 0.0
        speaker_resolution_ms = 0.0
        if self._filter_extractor is not None:
            stage_started = perf_counter()
            intent = self._filter_extractor.extract_filters(question)
            filter_extraction_ms = (perf_counter() - stage_started) * 1000
            clarification = _clarification_for_intent(intent.unsupported_constraints)
            if intent.clarification_question is not None:
                clarification = intent.clarification_question
            if clarification is not None:
                return _clarification_run(
                    clarification,
                    filter_extraction_ms=filter_extraction_ms,
                )

            talker_ids: tuple[str, ...] = ()
            if intent.speaker_names:
                if self._speaker_resolver is None:
                    raise RuntimeError("Speaker resolution is required for speaker filters")
                stage_started = perf_counter()
                resolution = self._speaker_resolver.resolve(intent.speaker_names)
                speaker_resolution_ms = (perf_counter() - stage_started) * 1000
                if resolution.clarification_question is not None:
                    return _clarification_run(
                        resolution.clarification_question,
                        filter_extraction_ms=filter_extraction_ms,
                        speaker_resolution_ms=speaker_resolution_ms,
                    )
                if not resolution.talker_ids:
                    raise RuntimeError("Speaker resolution returned no identities or clarification")
                talker_ids = resolution.talker_ids
            resolved_filters = RetrievalFilters(
                talker_ids=talker_ids,
                start_date=intent.start_date,
                end_date=intent.end_date,
            )
            if resolved_filters.has_constraints:
                filters = resolved_filters

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
            filters=filters,
        )
        hybrid_retrieval_ms = (perf_counter() - stage_started) * 1000

        stage_started = perf_counter()
        answer = self._synthesizer.synthesize(question=question, evidence=evidence)
        generation_ms = (perf_counter() - stage_started) * 1000

        return CompletionRun(
            answer=answer,
            timings=CompletionTimings(
                filter_extraction_ms=filter_extraction_ms,
                speaker_resolution_ms=speaker_resolution_ms,
                embedding_ms=embedding_ms,
                hybrid_retrieval_ms=hybrid_retrieval_ms,
                generation_ms=generation_ms,
            ),
            evidence_count=len(evidence),
        )


def _clarification_for_intent(unsupported: tuple[str, ...]) -> str | None:
    """Ask the caller to remove constraints that retrieval cannot safely apply."""
    if not unsupported:
        return None
    dimensions = ", ".join(unsupported)
    return (
        f"I can filter by date and speaker, but not by {dimensions}. "
        "Could you ask without that constraint?"
    )


def _clarification_run(
    question: str,
    *,
    filter_extraction_ms: float,
    speaker_resolution_ms: float = 0.0,
) -> CompletionRun:
    """Return a clarification without creating an answer or performing retrieval."""
    return CompletionRun(
        answer=None,
        timings=CompletionTimings(
            filter_extraction_ms=filter_extraction_ms,
            speaker_resolution_ms=speaker_resolution_ms,
            embedding_ms=0.0,
            hybrid_retrieval_ms=0.0,
            generation_ms=0.0,
        ),
        evidence_count=0,
        clarification_question=question,
    )
