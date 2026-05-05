"""Script entrypoint for running one concrete entities extraction flow."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ddd_policy_tracer.utils.events.local import LocalPublisher

from .adapters import FilesystemChunkRepository, FilesystemEntityRepository
from .extractors import RuleBasedEntityExtractorConfig, RuleBasedSentenceEntityExtractor
from .models import EntityExtractionReport
from .service_layer import EntitiesService


def run(
    *,
    chunk_id: str,
    chunk_state_path: Path,
    entity_state_path: Path,
    extractor_version: str = "rules-v1",
) -> EntityExtractionReport:
    """Run one concrete entities extraction for a single chunk identifier."""
    extractor = RuleBasedSentenceEntityExtractor(
        RuleBasedEntityExtractorConfig(extractor_version=extractor_version),
    )
    service = EntitiesService(
        chunk_repository=FilesystemChunkRepository(chunk_state_path),
        entity_repository=FilesystemEntityRepository(entity_state_path),
        extractor=extractor,
        event_publisher=LocalPublisher(),
    )
    return service.extract_entities_for_chunk(chunk_id=chunk_id)


def _build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for entities extraction script."""
    parser = argparse.ArgumentParser(
        prog="entities-run",
        description="Run entities extraction for one chunk id.",
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
        "--entity-state-path",
        default="data/entities.jsonl",
        help="Path to entities JSONL state.",
    )
    parser.add_argument(
        "--extractor-version",
        default=os.environ.get("ENTITY_EXTRACTOR_VERSION", "rules-v1"),
        help="Extractor version tag for deterministic entity ids.",
    )
    return parser


def main() -> int:
    """Run entities extraction from command-line arguments."""
    parser = _build_parser()
    args = parser.parse_args()
    report = run(
        chunk_id=args.chunk_id,
        chunk_state_path=Path(args.chunk_state_path),
        entity_state_path=Path(args.entity_state_path),
        extractor_version=args.extractor_version,
    )
    sys.stdout.write(f"{report}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
