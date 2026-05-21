"""Script entrypoint for running entities extraction workflows."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from ddd_policy_tracer.analysis.chunks.chunking_models import DocumentChunk
from ddd_policy_tracer.utils.events.local import LocalPublisher

from .adapters import FilesystemChunkRepository, FilesystemEntityRepository
from .extractors import (
    SpacyEntityExtractor,
    SpacyEntityExtractorConfig,
)
from .models import EntityExtractionReport, EntityMention
from .ports import EntityExtractor
from .resolution import DeterministicEntityJudge, EntityResolutionService
from .retrieval import HybridCatalogRetriever
from .review_queue import SQLiteReviewQueueRepository
from .service_layer import EntitiesService


def run(
    *,
    chunk_id: str,
    chunk_state_path: Path,
    entity_state_path: Path,
    db_path: Path | None = None,
    vectors_path: Path | None = None,
) -> EntityExtractionReport:
    """Run one concrete entities extraction for a single chunk identifier."""
    service = _build_service(
        chunk_state_path=chunk_state_path,
        entity_state_path=entity_state_path,
        vectors_path=vectors_path,
        db_path=db_path,
    )
    return service.extract_entities_for_chunk(chunk_id=chunk_id)


def run_bulk(
    *,
    chunk_state_path: Path,
    entity_state_path: Path,
    db_path: Path | None = None,
    vectors_path: Path | None = None,
) -> list[EntityExtractionReport]:
    """Run entities extraction for all chunks available in one dataset."""
    service = _build_service(
        chunk_state_path=chunk_state_path,
        entity_state_path=entity_state_path,
        db_path=db_path,
        vectors_path=vectors_path,
    )
    return service.extract_entities_for_all_chunks()


def _build_service(
    *,
    chunk_state_path: Path,
    entity_state_path: Path,
    vectors_path: Path | None,
    db_path: Path | None = None,
) -> EntitiesService:
    """Build entities service dependencies from CLI configuration."""
    # if no db_path is provided, we will just run the extractor without resolution capabilities
    extractor = SpacyEntityExtractor(
        config=SpacyEntityExtractorConfig(
            extractor_version="v1-spacy",
        ),
    )
    if db_path is not None:
        review_repository = SQLiteReviewQueueRepository(sqlite_path=db_path)
        extractor = _ResolvedEntityExtractor(
            mention_extractor=extractor,
            resolution_service=EntityResolutionService(
                retriever=HybridCatalogRetriever(
                    catalog_path=db_path,
                    vectors_path=vectors_path,
                ),
                judge=DeterministicEntityJudge(),
            ),
            review_repository=review_repository,
        )

    return EntitiesService(
        chunk_repository=FilesystemChunkRepository(chunk_state_path),
        entity_repository=FilesystemEntityRepository(entity_state_path),
        extractor=extractor,
        event_publisher=LocalPublisher(),
    )


class _ResolvedEntityExtractor:
    """Compose mention extraction with deterministic catalog resolution."""

    def __init__(
        self,
        *,
        mention_extractor: EntityExtractor,
        resolution_service: EntityResolutionService,
        review_repository: SQLiteReviewQueueRepository | None,
    ) -> None:
        """Store extractor collaborators for integrated resolution runs."""
        self._mention_extractor = mention_extractor
        self._resolution_service = resolution_service
        self._review_repository = review_repository

    def extract(self, *, chunk: DocumentChunk) -> list[EntityMention]:
        """Extract and resolve mentions for one chunk payload."""
        mentions = self._mention_extractor.extract(chunk=chunk)
        resolved = self._resolution_service.resolve_mentions(mentions=mentions)
        self._enqueue_review_items(mentions=resolved)
        return resolved

    def extract_many(self, *, chunks: list[DocumentChunk]) -> list[EntityMention]:
        """Extract and resolve mentions for many chunks in one pass."""
        mentions = self._mention_extractor.extract_many(chunks=chunks)
        resolved = self._resolution_service.resolve_mentions(mentions=mentions)
        self._enqueue_review_items(mentions=resolved)
        return resolved

    def count_processed_sentences(self, *, chunk: DocumentChunk) -> int:
        """Delegate sentence counting to underlying mention extractor."""
        return self._mention_extractor.count_processed_sentences(chunk=chunk)

    def _enqueue_review_items(self, *, mentions: list[EntityMention]) -> None:
        """Persist unresolved resolution outcomes into review queue storage."""
        if self._review_repository is None:
            return
        self._review_repository.enqueue_resolver_decisions(mentions=mentions)


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
        "--vectors-path",
        default=None,
        help="Optional sidecar vectors path for entity resolution.",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="""Optional SQLite path for CatalogEntity entries.
        Required for resolution capabilities.""",
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
            db_path=Path(args.db_path) if args.db_path is not None else None,
            vectors_path=Path(args.vectors_path) if args.vectors_path is not None else None,
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
        db_path=Path(args.db_path) if args.db_path is not None else None,
        vectors_path=Path(args.vectors_path) if args.vectors_path is not None else None,
    )
    sys.stdout.write(f"{report}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
