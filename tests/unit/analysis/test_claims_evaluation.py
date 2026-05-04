"""Unit tests for claim evaluation extractor strategy selection."""

from ddd_policy_tracer.analysis.claims.evaluation.evaluate_extractor import (
    _build_extractor,
)
from ddd_policy_tracer.analysis.claims.extractors import (
    LLMClaimExtractor,
    RuleBasedSentenceClaimExtractor,
)


def test_build_extractor_returns_rule_for_rule_kind() -> None:
    """Build rule-based extractor when evaluator requests rule strategy."""
    extractor = _build_extractor(
        extractor_kind="rule",
        threshold=0.9,
        llm_model="gpt-4.1-mini",
        llm_temperature=0.0,
        llm_max_claims_per_chunk=8,
    )

    assert isinstance(extractor, RuleBasedSentenceClaimExtractor)
    assert extractor.config.threshold == 0.9


def test_build_extractor_returns_llm_for_llm_kind() -> None:
    """Build LLM extractor when evaluator requests llm strategy."""
    extractor = _build_extractor(
        extractor_kind="llm",
        threshold=0.8,
        llm_model="gpt-4.1-mini",
        llm_temperature=0.2,
        llm_max_claims_per_chunk=5,
    )

    assert isinstance(extractor, LLMClaimExtractor)
    assert extractor.config.model == "gpt-4.1-mini"
    assert extractor.config.temperature == 0.2
    assert extractor.config.max_claims_per_chunk == 5
