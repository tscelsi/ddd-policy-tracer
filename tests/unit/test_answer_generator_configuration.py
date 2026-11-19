"""Tests for selecting the application's answer-generation adapter."""

from __future__ import annotations

import pytest

from ddd_policy_tracer.completion._evaluation.query_variants import (
    OllamaQueryVariantDrafter,
    OpenAIQueryVariantDrafter,
)
from ddd_policy_tracer.app import create_answer_generator, create_query_variant_drafter
from ddd_policy_tracer.completion.synthesis.generators.ollama_generator import OllamaAnswerGenerator
from ddd_policy_tracer.completion.synthesis.generators.openai_generator import OpenAIAnswerGenerator


def test_ollama_remains_the_default_answer_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep existing app deployments on Ollama unless another adapter is selected."""
    monkeypatch.delenv("RAG_ANSWER_GENERATOR", raising=False)

    generator = create_answer_generator()

    assert isinstance(generator, OllamaAnswerGenerator)


def test_openai_answer_generator_can_be_selected_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build the OpenAI adapter with configured credentials and model selection."""
    monkeypatch.setenv("RAG_ANSWER_GENERATOR", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")

    generator = create_answer_generator()

    assert isinstance(generator, OpenAIAnswerGenerator)
    assert generator.config.model == "test-model"


def test_unknown_answer_generator_configuration_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject unsupported provider names instead of silently choosing a fallback."""
    monkeypatch.setenv("RAG_ANSWER_GENERATOR", "unknown")

    with pytest.raises(ValueError, match="RAG_ANSWER_GENERATOR"):
        create_answer_generator()


def test_query_drafter_uses_the_configured_ollama_answer_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep local query drafting on the application's configured provider and model."""
    monkeypatch.setenv("RAG_ANSWER_GENERATOR", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "review-model")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")

    drafter = create_query_variant_drafter()

    assert isinstance(drafter, OllamaQueryVariantDrafter)
    assert drafter.model == "review-model"


def test_query_drafter_uses_the_configured_openai_answer_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the same OpenAI model selection as answer generation."""
    monkeypatch.setenv("RAG_ANSWER_GENERATOR", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "review-model")

    drafter = create_query_variant_drafter()

    assert isinstance(drafter, OpenAIQueryVariantDrafter)
    assert drafter.model == "review-model"
