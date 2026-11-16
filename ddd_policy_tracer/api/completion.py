"""HTTP adapter for complete parliamentary speech question answering."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from ollama import ResponseError
from openai import APIError as OpenAIError
from pydantic import BaseModel, ConfigDict, Field
from pymongo.errors import PyMongoError

from ddd_policy_tracer.answering.completion_service import CompletionRun, RagCompletionService
from ddd_policy_tracer.answering.metadata_filter_extraction import MetadataFilterExtractionError
from ddd_policy_tracer.answering.service_layer import SynthesizedAnswer

logger = logging.getLogger(__name__)


class CompletionRequest(BaseModel):
    """Represent a question and retrieval limit submitted for completion."""

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=4000)
    limit: int = Field(default=10, ge=1, le=50)


class CitationResponse(BaseModel):
    """Expose a source passage and its stored parliamentary provenance."""

    number: int
    chunk_id: str
    source_document_id: str
    passage: str
    speaker: str | None
    date: str | None
    house: str | None
    chamber: str | None


class CompletionResponse(BaseModel):
    """Return a synthesized answer, citations, and abstention state."""

    answer: str | None
    citations: list[CitationResponse]
    abstained: bool
    abstention_reason: str | None
    clarification_required: bool
    clarification_question: str | None

    @classmethod
    def from_answer(cls, answer: SynthesizedAnswer) -> CompletionResponse:
        """Convert the answer-domain result into its HTTP response representation."""
        return cls(
            answer=answer.text,
            citations=[
                CitationResponse(
                    number=citation.number,
                    chunk_id=citation.chunk_id,
                    source_document_id=citation.source_document_id,
                    passage=citation.passage,
                    speaker=citation.speaker,
                    date=citation.date,
                    house=citation.house,
                    chamber=citation.chamber,
                )
                for citation in answer.citations
            ],
            abstained=answer.abstained,
            abstention_reason=answer.abstention_reason,
            clarification_required=False,
            clarification_question=None,
        )

    @classmethod
    def from_run(cls, run: CompletionRun) -> CompletionResponse:
        """Convert a completion result into answer or clarification HTTP fields."""
        if run.clarification_required:
            return cls(
                answer=None,
                citations=[],
                abstained=False,
                abstention_reason=None,
                clarification_required=True,
                clarification_question=run.clarification_question,
            )
        if run.answer is None:
            raise RuntimeError("completion result has neither an answer nor a clarification")
        return cls.from_answer(run.answer)


def get_completion_service(request: Request) -> RagCompletionService:
    """Resolve the reusable RAG completion use case from application lifespan state."""
    service = getattr(request.state, "rag_completion_service", None)
    if not isinstance(service, RagCompletionService):
        raise HTTPException(
            status_code=503,
            detail="Question answering is not configured for this application.",
        )
    return service


router = APIRouter(tags=["completion"])


@router.post("/completion", response_model=CompletionResponse)
def create_completion(
    payload: CompletionRequest,
    completion_service: Annotated[RagCompletionService, Depends(get_completion_service)],
    response: Response,
) -> CompletionResponse:
    """Retrieve speech evidence and return an answer with source-backed citations."""
    request_started = perf_counter()
    try:
        run = completion_service.complete(payload.question, limit=payload.limit)
    except MetadataFilterExtractionError as exc:
        logger.warning("Answer generation provider returned invalid metadata filters")
        raise HTTPException(
            status_code=502,
            detail="The answer generation provider returned invalid metadata filters.",
        ) from exc
    except ValueError as exc:
        logger.warning("Completion rejected invalid structured input")
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except (httpx.HTTPError, OpenAIError, ResponseError) as exc:
        logger.warning("Answer generation provider is unavailable")
        raise HTTPException(
            status_code=503,
            detail="The answer generation provider is unavailable.",
        ) from exc
    except PyMongoError as exc:
        logger.warning("Completion dependency is unavailable")
        raise HTTPException(
            status_code=503,
            detail="The Hansard speaker catalog is unavailable.",
        ) from exc
    completion_response = CompletionResponse.from_run(run)
    total_ms = (perf_counter() - request_started) * 1000
    timings = run.timings
    response.headers["Server-Timing"] = ", ".join(
        (
            f"filter_extraction;dur={timings.filter_extraction_ms:.3f}",
            f"speaker_resolution;dur={timings.speaker_resolution_ms:.3f}",
            f"embedding;dur={timings.embedding_ms:.3f}",
            f"hybrid_retrieval;dur={timings.hybrid_retrieval_ms:.3f}",
            f"generation;dur={timings.generation_ms:.3f}",
            f"completion;dur={total_ms:.3f}",
        ),
    )
    logger.info(
        "completion_timing",
        extra={
            "embedding_ms": timings.embedding_ms,
            "filter_extraction_ms": timings.filter_extraction_ms,
            "speaker_resolution_ms": timings.speaker_resolution_ms,
            "hybrid_retrieval_ms": timings.hybrid_retrieval_ms,
            "generation_ms": timings.generation_ms,
            "completion_ms": total_ms,
            "evidence_count": run.evidence_count,
            "abstained": run.answer.abstained if run.answer is not None else False,
            "clarification_required": run.clarification_required,
        },
    )
    return completion_response
