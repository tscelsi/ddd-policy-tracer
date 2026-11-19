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

from ddd_policy_tracer.completion.service_layer import CompletionRun, RagCompletionService
from ddd_policy_tracer.completion.synthesis.answer_synthesizer import SynthesizedAnswer
from ddd_policy_tracer.completion.filtering.metadata_filters import MetadataFilterExtractionError

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
    """Return an answer or clarification, citations, and abstention state."""

    answer: str | None
    citations: list[CitationResponse]
    abstained: bool
    abstention_reason: str | None
    clarification_required: bool = False
    clarification_question: str | None = None

    @classmethod
    def from_run(cls, run: CompletionRun) -> CompletionResponse:
        """Convert an answer or clarification result into its HTTP representation."""
        if run.answer is None:
            if run.clarification_question is None:
                raise ValueError("Completion run must contain an answer or clarification")
            return cls(
                answer=None,
                citations=[],
                abstained=False,
                abstention_reason=None,
                clarification_required=True,
                clarification_question=run.clarification_question,
            )
        answer: SynthesizedAnswer = run.answer
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
async def create_completion(
    payload: CompletionRequest,
    completion_service: Annotated[RagCompletionService, Depends(get_completion_service)],
    response: Response,
) -> CompletionResponse:
    """Retrieve speech evidence and return an answer with source-backed citations."""
    request_started = perf_counter()
    try:
        run = await completion_service.completion(payload.question, limit=payload.limit)
    except MetadataFilterExtractionError as exc:
        logger.exception("Metadata filter extraction returned an invalid response")
        raise HTTPException(
            status_code=502,
            detail="The metadata filter provider returned an invalid response.",
        ) from exc
    except ValueError as exc:
        logger.exception("Answer generation failed due to invalid input")
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except (httpx.HTTPError, OpenAIError, ResponseError, PyMongoError) as exc:
        logger.exception("A completion provider or data source is unavailable")
        raise HTTPException(
            status_code=503,
            detail="The answer generation provider is unavailable.",
        ) from exc
    completion_response = CompletionResponse.from_run(run)
    total_ms = (perf_counter() - request_started) * 1000
    logger.info(
        "completion_timing",
        extra={
            "completion_ms": total_ms,
            "evidence_count": run.evidence_count,
            "abstained": run.answer.abstained if run.answer is not None else False,
            "clarification_required": run.clarification_required,
        },
    )
    return completion_response
