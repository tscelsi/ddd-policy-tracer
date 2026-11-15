"""HTTP adapter for complete parliamentary speech question answering."""

from __future__ import annotations

import logging
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from ollama import ResponseError
from pydantic import BaseModel, ConfigDict, Field

from ddd_policy_tracer.answering.completion_service import RagCompletionService
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

    answer: str
    citations: list[CitationResponse]
    abstained: bool
    abstention_reason: str | None

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
def create_completion(
    payload: CompletionRequest,
    completion_service: Annotated[RagCompletionService, Depends(get_completion_service)],
) -> CompletionResponse:
    """Retrieve speech evidence and return an answer with source-backed citations."""
    try:
        answer = completion_service.complete(payload.question, limit=payload.limit)
    except ValueError as exc:
        logger.exception("Answer generation failed due to invalid input")
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except (httpx.HTTPError, ResponseError) as exc:
        logger.exception("Answer generation provider is unavailable")
        raise HTTPException(
            status_code=503,
            detail="The answer generation provider is unavailable.",
        ) from exc
    return CompletionResponse.from_answer(answer)
