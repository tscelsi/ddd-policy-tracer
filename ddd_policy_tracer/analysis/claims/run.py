"""Script entrypoint for running one concrete claims extraction flow."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Literal

from ddd_policy_tracer.utils.events.local import LocalPublisher

from .adapters import FilesystemChunkRepository, FilesystemClaimRepository
from .extractors import (
    LLMClaimExtractor,
    LLMClaimExtractorConfig,
    RuleBasedClaimExtractorConfig,
    RuleBasedSentenceClaimExtractor,
)
from .models import ClaimExtractionReport
from .ports import ClaimExtractor
from .service_layer import ClaimsService


def run(
    *,
    chunk_id: str,
    chunk_state_path: Path,
    claim_state_path: Path,
    extractor_kind: Literal["rule", "llm"] = "rule",
    rule_threshold: float = 0.8,
    llm_model: str = "gpt-4.1-mini",
    llm_temperature: float = 0.0,
) -> ClaimExtractionReport:
    """Run one concrete claims extraction for a single chunk identifier."""
    extractor = _build_extractor(
        extractor_kind=extractor_kind,
        rule_threshold=rule_threshold,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
    )
    service = ClaimsService(
        chunk_repository=FilesystemChunkRepository(chunk_state_path),
        claim_repository=FilesystemClaimRepository(claim_state_path),
        extractor=extractor,
        event_publisher=LocalPublisher(),
    )
    return service.extract_claims_for_chunk(chunk_id=chunk_id)


def _build_extractor(
    *,
    extractor_kind: Literal["rule", "llm"],
    rule_threshold: float,
    llm_model: str,
    llm_temperature: float,
) -> ClaimExtractor:
    """Build one configured extractor strategy from CLI options."""
    if extractor_kind == "llm":
        return LLMClaimExtractor(
            config=LLMClaimExtractorConfig(
                model=llm_model,
                temperature=llm_temperature,
            ),
        )
    return RuleBasedSentenceClaimExtractor(
        RuleBasedClaimExtractorConfig(threshold=rule_threshold),
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for claims extraction script."""
    parser = argparse.ArgumentParser(
        prog="claims-run",
        description="Run claims extraction for one chunk id.",
    )
    parser.add_argument(
        "--chunk-id",
        required=True,
        help="Chunk identifier to process.",
    )
    parser.add_argument(
        "--chunk-state-path",
        default="data/chunks.jsonl",
        help="Path to chunk JSONL state.",
    )
    parser.add_argument(
        "--claim-state-path",
        default="data/claims.jsonl",
        help="Path to claims JSONL state.",
    )
    parser.add_argument(
        "--extractor",
        choices=["rule", "llm"],
        default=None,
        help="Extractor strategy override. Falls back to CLAIMS_EXTRACTOR env var.",
    )
    parser.add_argument(
        "--rule-threshold",
        type=float,
        default=0.8,
        help="Threshold used by rule-based extractor.",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4.1-mini",
        help="Model used by LLM extractor.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.0,
        help="Temperature used by LLM extractor.",
    )
    return parser


def main() -> int:
    """Run claims extraction from command-line arguments."""
    parser = _build_parser()
    args = parser.parse_args()
    env_extractor = os.environ.get("CLAIMS_EXTRACTOR", "rule")
    extractor_kind = args.extractor or env_extractor
    if extractor_kind not in {"rule", "llm"}:
        raise ValueError("CLAIMS_EXTRACTOR must be either 'rule' or 'llm'")

    report = run(
        chunk_id=args.chunk_id,
        chunk_state_path=Path(args.chunk_state_path),
        claim_state_path=Path(args.claim_state_path),
        extractor_kind=extractor_kind,
        rule_threshold=args.rule_threshold,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
    )
    sys.stdout.write(f"{report}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
