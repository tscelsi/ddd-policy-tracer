"""Models for answer synthesis and structured output validation."""

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


class EvidenceChunk(Protocol):
    """Expose retrieved speech text and provenance for answer synthesis."""

    @property
    def chunk_id(self) -> str:
        """Return the stable identity of the evidence passage."""
        ...

    @property
    def chunk_text(self) -> str:
        """Return the speech passage used to support an answer."""
        ...

    @property
    def source_document_id(self) -> str:
        """Return the stable identity of the source speech document."""
        ...

    @property
    def metadata(self) -> str | None:
        """Return speaker and source metadata stored with the passage."""
        ...


class AnswerClaim(BaseModel):
    """Represent one answer statement and the evidence passages supporting it."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    text: str
    evidence_chunk_ids: tuple[str, ...] = Field(min_length=1, strict=False)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Require and normalize the generated answer statement."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("claim text must not be empty")
        return normalized

    @field_validator("evidence_chunk_ids")
    @classmethod
    def validate_evidence_chunk_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Require every claim citation to identify a non-empty evidence chunk."""
        if any(not chunk_id for chunk_id in values):
            raise ValueError("evidence chunk IDs must not be empty")
        return values


class AnswerDraft(BaseModel):
    """Represent structured claims returned by an answer-generation adapter."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    claims: tuple[AnswerClaim, ...] = Field(default=(), strict=False)
    abstained: bool = False
    abstention_reason: str | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_response_shape(cls, value: object, info: ValidationInfo) -> object:
        """Require the complete provider response shape when parsing model output."""
        context = info.context
        if context is not None and context.get("provider_response"):
            expected_fields = {"claims", "abstained", "abstention_reason"}
            if not isinstance(value, dict) or set(value) != expected_fields:
                raise ValueError("answer response must contain exactly the required fields")
        return value

    @field_validator("abstention_reason")
    @classmethod
    def validate_abstention_reason(cls, value: str | None) -> str | None:
        """Normalize an abstention explanation and reject blank strings."""
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("abstention reason must not be empty")
        return normalized

    @model_validator(mode="after")
    def validate_contract(self, info: ValidationInfo) -> "AnswerDraft":
        """Keep claims and abstention explanations consistent with the answer state."""
        if self.abstained:
            if self.claims or self.abstention_reason is None:
                raise ValueError("abstentions require a reason and must not contain claims")
        elif self.abstention_reason is not None:
            raise ValueError("answers require claims and must not contain an abstention reason")
        elif not self.claims and info.context and info.context.get("provider_response"):
            raise ValueError("non-abstaining answers require at least one claim")
        return self


@dataclass(frozen=True)
class AnswerCitation:
    """Bind a rendered citation to its stored source passage and provenance."""

    number: int
    chunk_id: str
    source_document_id: str
    passage: str
    speaker: str | None
    date: str | None
    house: str | None
    chamber: str | None


@dataclass(frozen=True)
class SynthesizedAnswer:
    """Return rendered answer text and the sources behind its citations."""

    text: str
    citations: tuple[AnswerCitation, ...]
    abstained: bool
    abstention_reason: str | None = None
