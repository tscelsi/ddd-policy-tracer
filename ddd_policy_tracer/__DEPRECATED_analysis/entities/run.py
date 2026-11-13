"""Script entrypoint for running entities extraction workflows."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from ddd_policy_tracer.__DEPRECATED_analysis.entities._catalog.adapters import SQLiteCatalogEntityRepository
from ddd_policy_tracer.enrichment.embedding import EmbeddingService
from ddd_policy_tracer.utils.db_manager import get_db_async
from ddd_policy_tracer.utils.events.local import LocalPublisher

from .adapters import FilesystemChunkRepository, FilesystemEntityRepository
from .extractors import (
    ResolvedEntityExtractor,
)
from .models import EntityExtractionReport
from .resolution import DeterministicEntityJudge, EntityResolutionService
from .retrieval import EmbeddingRetriever
from .review_queue import SQLiteReviewQueueRepository
from .service_layer import EntitiesService


async def run(
    *,
    chunk_id: str,
    chunk_state_path: Path,
    entity_state_path: Path,
    db_path: Path | None = None,
) -> EntityExtractionReport:
    """Run one concrete entities extraction for a single chunk identifier."""
    service = await _build_service(
        chunk_state_path=chunk_state_path,
        entity_state_path=entity_state_path,
        db_path=db_path,
    )
    return await service.extract_entities_for_chunk(chunk_id=chunk_id)


async def run_bulk(
    *,
    chunk_state_path: Path,
    entity_state_path: Path,
    db_path: Path | None = None,
) -> list[EntityExtractionReport]:
    """Run entities extraction for all chunks available in one dataset."""
    service = await _build_service(
        chunk_state_path=chunk_state_path,
        entity_state_path=entity_state_path,
        db_path=db_path,
    )
    return await service.extract_entities_for_all_chunks()


async def _build_service(
    *,
    chunk_state_path: Path,
    entity_state_path: Path,
    db_path: Path | None = None,
) -> EntitiesService:
    """Build entities service dependencies from CLI configuration."""
    # if no db_path is provided, we will just run the extractor without resolution capabilities
    chunk_repo = FilesystemChunkRepository(state_path=chunk_state_path)
    entity_repo = FilesystemEntityRepository(state_path=entity_state_path)
    embed_service = EmbeddingService.get_instance()
    catalog_entity_repo = SQLiteCatalogEntityRepository()
    await catalog_entity_repo.init()
    event_publisher = LocalPublisher()
    if db_path is not None:
        review_repository = SQLiteReviewQueueRepository(sqlite_path=db_path)
        extractor = ResolvedEntityExtractor(
            resolution_service=EntityResolutionService(
                retriever=EmbeddingRetriever(
                    chunk_repo=chunk_repo,
                    embed_service=embed_service,
                    catalog_entity_repo=catalog_entity_repo,
                ),
                judge=DeterministicEntityJudge(),
            ),
            review_repository=review_repository,
        )

    return EntitiesService(
        chunk_repository=chunk_repo,
        entity_repository=entity_repo,
        extractor=extractor,
        event_publisher=event_publisher,
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


async def main() -> int:
    """Run entities extraction from command-line arguments."""
    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(log_level=args.log_level)
    chunk_state_path = Path(args.chunk_state_path)
    entity_state_path = Path(args.entity_state_path)

    if args.all_chunks:
        reports = await run_bulk(
            chunk_state_path=chunk_state_path,
            entity_state_path=entity_state_path,
            db_path=Path(args.db_path) if args.db_path is not None else None,
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

    report = await run(
        chunk_id=args.chunk_id,
        chunk_state_path=chunk_state_path,
        entity_state_path=entity_state_path,
        db_path=Path(args.db_path) if args.db_path is not None else None,
    )
    sys.stdout.write(f"{report}\n")
    db = await get_db_async()
    await db.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
