"""Script entrypoint for running entities extraction workflows."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Literal

from ddd_policy_tracer.utils.events.local import LocalPublisher

from .adapters import FilesystemChunkRepository, FilesystemEntityRepository
from .extractors import (
    RuleBasedEntityExtractorConfig,
    RuleBasedSentenceEntityExtractor,
    SpacyFastCorefEntityExtractor,
    SpacyFastCorefEntityExtractorConfig,
)
from .models import EntityExtractionReport
from .service_layer import EntitiesService


def run(
    *,
    chunk_id: str,
    chunk_state_path: Path,
    entity_state_path: Path,
    extractor_kind: Literal["rule", "spacy-fastcoref"] = "rule",
    extractor_version: str = "rules-v1",
) -> EntityExtractionReport:
    """Run one concrete entities extraction for a single chunk identifier."""
    service = _build_service(
        chunk_state_path=chunk_state_path,
        entity_state_path=entity_state_path,
        extractor_kind=extractor_kind,
        extractor_version=extractor_version,
    )
    return service.extract_entities_for_chunk(chunk_id=chunk_id)


def run_bulk(
    *,
    chunk_state_path: Path,
    entity_state_path: Path,
    extractor_kind: Literal["rule", "spacy-fastcoref"] = "rule",
    extractor_version: str = "rules-v1",
) -> list[EntityExtractionReport]:
    """Run entities extraction for all chunks available in one dataset."""
    service = _build_service(
        chunk_state_path=chunk_state_path,
        entity_state_path=entity_state_path,
        extractor_kind=extractor_kind,
        extractor_version=extractor_version,
    )
    return service.extract_entities_for_all_chunks()


def _build_service(
    *,
    chunk_state_path: Path,
    entity_state_path: Path,
    extractor_kind: Literal["rule", "spacy-fastcoref"],
    extractor_version: str,
) -> EntitiesService:
    """Build entities service dependencies from CLI configuration."""
    if extractor_kind == "spacy-fastcoref":
        extractor = SpacyFastCorefEntityExtractor(
            config=SpacyFastCorefEntityExtractorConfig(
                extractor_version=extractor_version,
            ),
        )
    else:
        extractor = RuleBasedSentenceEntityExtractor(
            RuleBasedEntityExtractorConfig(extractor_version=extractor_version),
        )
    return EntitiesService(
        chunk_repository=FilesystemChunkRepository(chunk_state_path),
        entity_repository=FilesystemEntityRepository(entity_state_path),
        extractor=extractor,
        event_publisher=LocalPublisher(),
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for entities extraction script."""
    parser = argparse.ArgumentParser(
        prog="entities-run",
        description="Run entities extraction for one chunk or all chunks.",
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
        "--entity-state-path",
        default="data/entities.jsonl",
        help="Path to entities JSONL state.",
    )
    parser.add_argument(
        "--extractor",
        choices=["rule", "spacy-fastcoref"],
        default=os.environ.get("ENTITY_EXTRACTOR", "rule"),
        help="Extractor strategy to use for entity extraction.",
    )
    parser.add_argument(
        "--extractor-version",
        default=os.environ.get("ENTITY_EXTRACTOR_VERSION", "rules-v1"),
        help="Extractor version tag for deterministic entity ids.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=os.environ.get("ENTITY_LOG_LEVEL", "WARNING"),
        help="Logging level for entity extraction runtime diagnostics.",
    )
    return parser


def _configure_logging(*, log_level: str) -> None:
    """Configure CLI logging for extractor diagnostics and fallbacks."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.WARNING),
        format="%(levelname)s:%(name)s:%(message)s",
    )


def main() -> int:
    """Run entities extraction from command-line arguments."""
    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(log_level=args.log_level)
    chunk_state_path = Path(args.chunk_state_path)
    entity_state_path = Path(args.entity_state_path)

    if args.all_chunks:
        reports = run_bulk(
            chunk_state_path=chunk_state_path,
            entity_state_path=entity_state_path,
            extractor_kind=args.extractor,
            extractor_version=args.extractor_version,
        )
        summary = {
            "processed_chunks": len(reports),
            "completed_chunks": sum(1 for report in reports if report.status == "completed"),
            "failed_chunks": sum(1 for report in reports if report.status == "failed"),
            "entities_extracted": sum(report.entities_extracted for report in reports),
        }
        sys.stdout.write(json.dumps(summary, ensure_ascii=True) + "\n")
        return 0

    if args.chunk_id is None:
        parser.error("--chunk-id is required unless --all-chunks is set")

    report = run(
        chunk_id=args.chunk_id,
        chunk_state_path=chunk_state_path,
        entity_state_path=entity_state_path,
        extractor_kind=args.extractor,
        extractor_version=args.extractor_version,
    )
    sys.stdout.write(f"{report}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
