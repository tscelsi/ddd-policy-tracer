"""Orchestrate filter extraction, parliamentary retrieval, and grounded answers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from zoneinfo import ZoneInfo

from ddd_policy_tracer.answering.service_layer import AnswerSynthesizer, SynthesizedAnswer
from ddd_policy_tracer.retrieval.metadata_filters import (
    MetadataFilter,
    MetadataFilterExtractor,
    SpeakerNameResolver,
)
from ddd_policy_tracer.retrieval.service_layer import (
    HybridChunkRetriever,
    QueryEmbedder,
    embed_query,
)

_CANBERRA = ZoneInfo("Australia/Canberra")


@dataclass(frozen=True)
class CompletionTimings:
    """Measure metadata extraction, embedding, retrieval, and synthesis stages."""

    metadata_filter_ms: float
    embedding_ms: float
    hybrid_retrieval_ms: float
    generation_ms: float


@dataclass(frozen=True)
class CompletionRun:
    """Return a synthesized answer or a clarification with stage timings."""

    answer: SynthesizedAnswer | None
    timings: CompletionTimings
    evidence_count: int
    clarification_question: str | None = None

    @property
    def clarification_required(self) -> bool:
        """Report whether metadata constraints require user clarification."""
        return self.clarification_question is not None


class RagCompletionService:
    """Run metadata filtering before hybrid evidence retrieval and synthesis."""

    def __init__(
        self,
        *,
        retriever: HybridChunkRetriever,
        embedding_service: QueryEmbedder,
        synthesizer: AnswerSynthesizer,
        metadata_filter_extractor: MetadataFilterExtractor,
        speaker_name_resolver: SpeakerNameResolver,
    ) -> None:
        """Accept reusable ports for extraction, speaker resolution, retrieval, and answers."""
        self._retriever = retriever
        self._embedding_service = embedding_service
        self._synthesizer = synthesizer
        self._metadata_filter_extractor = metadata_filter_extractor
        self._speaker_name_resolver = speaker_name_resolver

    async def complete(self, question: str, *, limit: int = 10) -> CompletionRun:
        """Extract constraints, clarify unresolved intent, and answer from filtered evidence."""
        stage_started = perf_counter()
        reference_date = datetime.now(_CANBERRA).date()
        extraction = await asyncio.to_thread(
            self._metadata_filter_extractor.extract,
            question,
            reference_date=reference_date,
        )
        if extraction.clarification_question is not None:
            return self._clarification_run(
                extraction.clarification_question,
                metadata_filter_ms=(perf_counter() - stage_started) * 1000,
            )
        if extraction.unsupported_filters:
            unsupported = ", ".join(extraction.unsupported_filters)
            return self._clarification_run(
                "I can currently narrow Hansard searches by speaker and date, "
                f"but not by {unsupported}. Could you rephrase your request?",
                metadata_filter_ms=(perf_counter() - stage_started) * 1000,
            )

        speaker_resolution = await self._speaker_name_resolver.resolve(
            extraction.speaker_names,
        )
        if speaker_resolution.clarification_question is not None:
            return self._clarification_run(
                speaker_resolution.clarification_question,
                metadata_filter_ms=(perf_counter() - stage_started) * 1000,
            )
        if extraction.speaker_names and not speaker_resolution.talker_ids:
            raise RuntimeError(
                "Speaker resolver returned no IDs or clarification for requested names.",
            )
        filters = MetadataFilter(
            talker_ids=speaker_resolution.talker_ids,
            date_from=extraction.date_from,
            date_to=extraction.date_to,
        )
        metadata_filter_ms = (perf_counter() - stage_started) * 1000

        stage_started = perf_counter()
        query_embedding = await asyncio.to_thread(
            embed_query,
            question,
            embedding_service=self._embedding_service,
        )
        embedding_ms = (perf_counter() - stage_started) * 1000

        stage_started = perf_counter()
        evidence = await asyncio.to_thread(
            self._retriever.search_hybrid,
            query=question,
            query_embedding=query_embedding,
            limit=limit,
            filters=filters,
        )
        hybrid_retrieval_ms = (perf_counter() - stage_started) * 1000

        stage_started = perf_counter()
        answer = await asyncio.to_thread(
            self._synthesizer.synthesize,
            question=question,
            evidence=evidence,
        )
        generation_ms = (perf_counter() - stage_started) * 1000

        return CompletionRun(
            answer=answer,
            timings=CompletionTimings(
                metadata_filter_ms=metadata_filter_ms,
                embedding_ms=embedding_ms,
                hybrid_retrieval_ms=hybrid_retrieval_ms,
                generation_ms=generation_ms,
            ),
            evidence_count=len(evidence),
        )

    @staticmethod
    def _clarification_run(
        question: str,
        *,
        metadata_filter_ms: float,
    ) -> CompletionRun:
        """Create a clarification result without embedding, retrieval, or generation."""
        return CompletionRun(
            answer=None,
            timings=CompletionTimings(
                metadata_filter_ms=metadata_filter_ms,
                embedding_ms=0.0,
                hybrid_retrieval_ms=0.0,
                generation_ms=0.0,
            ),
            evidence_count=0,
            clarification_question=question,
        )
