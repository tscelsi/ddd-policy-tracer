"""Script entrypoint for running claims extraction workflows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Literal

from ddd_policy_tracer.utils.events.local import LocalPublisher

from .adapters import FilesystemChunkRepository, FilesystemClaimRepository
from .extractors import (
    LLMClaimExtractor,
    LLMClaimExtractorConfig,
    OllamaClaimExtractor,
    OllamaClaimExtractorConfig,
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
    extractor_kind: Literal["rule", "llm", "ollama"] = "rule",
    rule_threshold: float = 0.8,
    llm_model: str = "gpt-4.1-mini",
    llm_temperature: float = 0.0,
) -> ClaimExtractionReport:
    """Run one concrete claims extraction for a single chunk identifier."""
    service = _build_service(
        chunk_state_path=chunk_state_path,
        claim_state_path=claim_state_path,
        extractor_kind=extractor_kind,
        rule_threshold=rule_threshold,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
    )
    return service.extract_claims_for_chunk(chunk_id=chunk_id)


def run_bulk(
    *,
    chunk_state_path: Path,
    claim_state_path: Path,
    extractor_kind: Literal["rule", "llm", "ollama"] = "rule",
    rule_threshold: float = 0.8,
    llm_model: str = "gpt-4.1-mini",
    llm_temperature: float = 0.0,
) -> list[ClaimExtractionReport]:
    """Run claims extraction for all chunks found in one chunk dataset."""
    service = _build_service(
        chunk_state_path=chunk_state_path,
        claim_state_path=claim_state_path,
        extractor_kind=extractor_kind,
        rule_threshold=rule_threshold,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
    )
    chunk_ids = _load_chunk_ids(chunk_state_path=chunk_state_path)
    return service.extract_claims_for_chunks(chunk_ids=chunk_ids)


def _build_service(
    *,
    chunk_state_path: Path,
    claim_state_path: Path,
    extractor_kind: Literal["rule", "llm", "ollama"],
    rule_threshold: float,
    llm_model: str,
    llm_temperature: float,
) -> ClaimsService:
    """Build claims service dependencies from CLI configuration."""
    extractor = _build_extractor(
        extractor_kind=extractor_kind,
        rule_threshold=rule_threshold,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
    )
    return ClaimsService(
        chunk_repository=FilesystemChunkRepository(chunk_state_path),
        claim_repository=FilesystemClaimRepository(claim_state_path),
        extractor=extractor,
        event_publisher=LocalPublisher(),
    )


def _load_chunk_ids(*, chunk_state_path: Path) -> list[str]:
    """Load unique chunk identifiers from one JSONL chunk dataset."""
    if not chunk_state_path.exists():
        return []

    seen: set[str] = set()
    chunk_ids: list[str] = []
    for raw_line in chunk_state_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        if not isinstance(payload, dict):
            continue
        raw_chunk_id = payload.get("chunk_id")
        if not isinstance(raw_chunk_id, str) or not raw_chunk_id.strip():
            continue
        if raw_chunk_id in seen:
            continue
        seen.add(raw_chunk_id)
        chunk_ids.append(raw_chunk_id)
    return chunk_ids


def _build_extractor(
    *,
    extractor_kind: Literal["rule", "llm", "ollama"],
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
    if extractor_kind == "ollama":
        return OllamaClaimExtractor(
            config=OllamaClaimExtractorConfig(
                model=llm_model,
            ),
        )
    return RuleBasedSentenceClaimExtractor(
        RuleBasedClaimExtractorConfig(threshold=rule_threshold),
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for claims extraction script."""
    parser = argparse.ArgumentParser(
        prog="claims-run",
        description="Run claims extraction for one chunk or all chunks.",
    )
    parser.add_argument(
        "--chunk-id",
        default=None,
        help="Chunk identifier to process.",
    )
    parser.add_argument(
        "--all-chunks",
        action="store_true",
        help="Process all chunk ids found in chunk-state-path.",
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
        choices=["rule", "llm", "ollama"],
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
    if extractor_kind not in {"rule", "llm", "ollama"}:
        raise ValueError("CLAIMS_EXTRACTOR must be one of: 'rule', 'llm', 'ollama'")

    chunk_state_path = Path(args.chunk_state_path)
    claim_state_path = Path(args.claim_state_path)

    if args.all_chunks:
        reports = run_bulk(
            chunk_state_path=chunk_state_path,
            claim_state_path=claim_state_path,
            extractor_kind=extractor_kind,
            rule_threshold=args.rule_threshold,
            llm_model=args.llm_model,
            llm_temperature=args.llm_temperature,
        )
        summary = {
            "processed_chunks": len(reports),
            "completed_chunks": sum(1 for report in reports if report.status == "completed"),
            "failed_chunks": sum(1 for report in reports if report.status == "failed"),
            "claims_extracted": sum(report.claims_extracted for report in reports),
        }
        sys.stdout.write(json.dumps(summary, ensure_ascii=True) + "\n")
        return 0

    if args.chunk_id is None:
        parser.error("--chunk-id is required unless --all-chunks is set")

    report = run(
        chunk_id=args.chunk_id,
        chunk_state_path=chunk_state_path,
        claim_state_path=claim_state_path,
        extractor_kind=extractor_kind,
        rule_threshold=args.rule_threshold,
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
    )
    sys.stdout.write(f"{report}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
