"""Orchestrate filter extraction, parliamentary retrieval, and grounded answers."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from ddd_policy_tracer.completion.filtering.metadata_filter_extractor import (
    AbstractMetadataFilterExtractor,
)
from ddd_policy_tracer.completion.filtering.metadata_filters import (
    MetadataFilter,
)
from ddd_policy_tracer.completion.filtering.resolvers.bill_name_resolver import (
    AbstractBillNameResolver,
)
from ddd_policy_tracer.completion.filtering.resolvers.speaker_name_resolver import (
    AbstractSpeakerNameResolver,
)
from ddd_policy_tracer.completion.retrieval.service_layer import (
    HybridChunkRetriever,
    QueryEmbedder,
    RetrievedChunk,
    embed_query,
)
from ddd_policy_tracer.completion.synthesis.answer_synthesizer import (
    AnswerSynthesizer,
    SynthesizedAnswer,
)
from ddd_policy_tracer.utils.logger import get_logger

logger = get_logger(__name__)

_CANBERRA = ZoneInfo("Australia/Canberra")


class UnsupportedFilterError(ValueError):
    """Raise when a user-specified filter cannot be applied to the Hansard corpus."""

    pass


class ClarificationRequiredError(ValueError):
    """Raise when a user-specified filter cannot be resolved without clarification."""

    pass


@dataclass(frozen=True)
class CompletionRun:
    """Return a synthesized answer or a clarification with stage timings."""

    answer: SynthesizedAnswer | None
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
        metadata_filter_extractor: AbstractMetadataFilterExtractor,
        speaker_name_resolver: AbstractSpeakerNameResolver,
        bill_name_resolver: AbstractBillNameResolver,
    ) -> None:
        """Accept reusable ports for extraction, speaker resolution, retrieval, and answers."""
        self._retriever = retriever
        self._embedding_service = embedding_service
        self._synthesizer = synthesizer
        self._metadata_filter_extractor = metadata_filter_extractor
        self._speaker_name_resolver = speaker_name_resolver
        self._bill_name_resolver = bill_name_resolver

    async def retrieve(
        self,
        question: str,
        *,
        limit: int = 10,
    ) -> Sequence[RetrievedChunk]:
        """Return the ranked evidence for a question."""
        reference_date = datetime.now(_CANBERRA).date()
        # Extract metadata constraints from the question, or raise if clarification is needed.
        logger.info(f"Extracting metadata filters for query '{question}'")
        extraction = await asyncio.to_thread(
            self._metadata_filter_extractor.extract,
            question,
            reference_date=reference_date,
        )
        if extraction.clarification_question is not None:
            raise ClarificationRequiredError(extraction.clarification_question)
        if extraction.unsupported_filters:
            unsupported = ", ".join(extraction.unsupported_filters)
            raise UnsupportedFilterError(
                "I can currently narrow Hansard searches by speaker and date, "
                f"but not by {unsupported}. Could you rephrase your request?",
            )
        # Resolve speaker names to talker IDs for filtering, or raise if clarification is needed.
        logger.info(f"Resolving speaker names for query '{question}': {extraction.speaker_names}")
        speaker_resolution = await self._speaker_name_resolver.resolve(
            extraction.speaker_names,
        )
        if speaker_resolution.clarification_question is not None:
            raise ClarificationRequiredError(speaker_resolution.clarification_question)
        if extraction.speaker_names and not speaker_resolution.talker_ids:
            raise RuntimeError(
                "Speaker resolver returned no IDs or clarification for requested names.",
            )
        # Resolve bill names to bill IDs for filtering, or raise if clarification is needed.
        logger.info(f"Resolving bill names for query '{question}': {extraction.bill_names}")
        bill_resolution = await self._bill_name_resolver.resolve(
            extraction.bill_names,
        )
        if bill_resolution.clarification_question is not None:
            raise ClarificationRequiredError(bill_resolution.clarification_question)
        if extraction.bill_names and not bill_resolution.bill_ids:
            raise RuntimeError(
                "Bill resolver returned no IDs or clarification for requested names.",
            )
        filters = MetadataFilter(
            talker_ids=speaker_resolution.talker_ids,
            date_from=extraction.date_from,
            date_to=extraction.date_to,
            bill_ids=bill_resolution.bill_ids,
        )
        logger.info(f"Metadata filters extracted for query '{question}': {filters}")
        query_embedding = await asyncio.to_thread(
            embed_query,
            question,
            embedding_service=self._embedding_service,
        )
        logger.info("Query embedding generated")
        logger.info(f"Searching hybrid evidence for query '{question}'")
        evidence = await asyncio.to_thread(
            self._retriever.search_hybrid,
            query=question,
            query_embedding=query_embedding,
            limit=limit,
            filters=filters,
        )
        logger.info(f"Retrieved {len(evidence)} evidence passages")
        logger.debug(f"Evidence passages: {evidence}")
        return evidence

    async def completion(self, question: str, *, limit: int = 10) -> CompletionRun:
        """Extract constraints, clarify unresolved intent, and answer from filtered evidence."""
        logger.info(f"Generating completion for query '{question}'")
        try:
            evidence = await self.retrieve(question, limit=limit)
        except (ClarificationRequiredError, UnsupportedFilterError) as exc:
            return self._clarification_run(
                str(exc),
            )
        logger.info(f"Retrieved {len(evidence)} evidence passages for query '{question}'")
        answer = await asyncio.to_thread(
            self._synthesizer.synthesize,
            question=question,
            evidence=evidence,
        )
        logger.info(f"Answer generated for query '{question}': {answer}")
        return CompletionRun(
            answer=answer,
            evidence_count=len(evidence),
        )

    @staticmethod
    def _clarification_run(question: str) -> CompletionRun:
        """Create a clarification result without embedding, retrieval, or generation."""
        return CompletionRun(
            answer=None,
            evidence_count=0,
            clarification_question=question,
        )
