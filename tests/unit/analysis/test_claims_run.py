"""Unit tests for claims run module extractor selection behavior."""

from ddd_policy_tracer.analysis.claims import (
    LLMClaimExtractor,
    RuleBasedSentenceClaimExtractor,
)
from ddd_policy_tracer.analysis.claims.run import _build_extractor


def test_build_extractor_returns_rule_based_strategy() -> None:
    """Build rule extractor when strategy is configured as rule."""
    extractor = _build_extractor(
        extractor_kind="rule",
        rule_threshold=0.9,
        llm_model="gpt-4.1-mini",
        llm_temperature=0.0,
    )

    assert isinstance(extractor, RuleBasedSentenceClaimExtractor)
    assert extractor.config.threshold == 0.9


def test_build_extractor_returns_llm_strategy() -> None:
    """Build LLM extractor when strategy is configured as llm."""
    extractor = _build_extractor(
        extractor_kind="llm",
        rule_threshold=0.8,
        llm_model="gpt-4.1-mini",
        llm_temperature=0.2,
    )

    assert isinstance(extractor, LLMClaimExtractor)
    assert extractor.config.model == "gpt-4.1-mini"
    assert extractor.config.temperature == 0.2
