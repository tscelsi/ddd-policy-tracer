"""CLI entrypoint for analysis chunking and canonicalization operations."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from sys import stdout
from typing import TextIO

from ddd_policy_tracer.enrichment.chunking.adapters import SQLiteDocumentChunkRepository
from ddd_policy_tracer.ingestion.service_layer import (
    get_source_document_versions,
)
from ddd_policy_tracer.utils.db_manager import DbManager, get_db_async

from ..enrichment.chunking.service_layer import chunk_and_persist_documents
from .entities.catalog import import_seed_catalog, promote_review_items
from .entities.review_queue import SQLiteReviewQueueRepository


def run_cli(argv: Sequence[str], *, stdout: TextIO) -> int:
    """Execute analysis commands from the operator CLI surface."""
    parser = argparse.ArgumentParser(
        prog="ddd-policy-tracer",
        description="Run analysis operations on acquired source documents.",
    )
    subparsers = parser.add_subparsers(dest="command")

    chunk_parser = subparsers.add_parser(
        "chunk",
        help="Chunk acquired source document versions",
    )
    chunk_parser.add_argument(
        "--source",
        required=True,
        help="Source identifier to chunk (for example australia_institute).",
    )
    chunk_parser.add_argument(
        "--state-path",
        required=True,
        help="Path to acquisition state (SQLite DB or JSONL file).",
    )
    entities_catalog_parser = subparsers.add_parser(
        "entities-catalog",
        help="Manage runtime entity catalog operations",
    )
    entities_catalog_subparsers = entities_catalog_parser.add_subparsers(
        dest="entities_catalog_command",
        required=True,
    )
    entities_catalog_import_parser = entities_catalog_subparsers.add_parser(
        "import",
        help="Import curated seed rows into runtime entity catalog.",
    )
    entities_catalog_import_parser.add_argument(
        "--seed-path",
        required=True,
        help="Path to curated seed JSONL data.",
    )
    entities_catalog_import_parser.add_argument(
        "--catalog-path",
        required=True,
        help="Path to runtime SQLite catalog database.",
    )
    entities_catalog_import_parser.add_argument(
        "--vectors-path",
        required=True,
        help="Path to sidecar vectors JSON artifact.",
    )
    entities_catalog_import_parser.add_argument(
        "--catalog-version",
        default="catalog-v1",
        help="Catalog version metadata value for compatibility checks.",
    )

    entities_catalog_promote_parser = entities_catalog_subparsers.add_parser(
        "promote",
        help="Promote approved review items into runtime catalog.",
    )
    entities_catalog_promote_parser.add_argument(
        "--review-db-path",
        required=True,
        help="Path to SQLite review queue database.",
    )
    entities_catalog_promote_parser.add_argument(
        "--catalog-path",
        required=True,
        help="Path to runtime SQLite catalog database.",
    )
    entities_catalog_promote_parser.add_argument(
        "--vectors-path",
        required=True,
        help="Path to sidecar vectors JSON artifact.",
    )
    entities_catalog_promote_parser.add_argument(
        "--catalog-version",
        default="catalog-v1",
        help="Catalog version metadata value for compatibility checks.",
    )
    entities_catalog_promote_parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically approve obvious pending review items before promotion.",
    )
    entities_catalog_promote_parser.add_argument(
        "--auto-approve-min-score",
        type=float,
        default=0.93,
        help="Minimum decision score threshold for auto-approval.",
    )
    entities_catalog_promote_parser.add_argument(
        "--auto-approve-min-margin",
        type=float,
        default=0.15,
        help="Minimum fused-score margin between top candidates for auto-approval.",
    )

    entities_review_parser = subparsers.add_parser(
        "entities-review",
        help="Review queue operations for unresolved entity decisions.",
    )
    entities_review_subparsers = entities_review_parser.add_subparsers(
        dest="entities_review_command",
        required=True,
    )
    entities_review_list_parser = entities_review_subparsers.add_parser(
        "list",
        help="List review queue items.",
    )
    entities_review_list_parser.add_argument(
        "--review-db-path",
        required=True,
        help="Path to SQLite review queue database.",
    )
    entities_review_list_parser.add_argument(
        "--state",
        choices=["pending", "approved", "rejected", "auto_approved", "promoted"],
        default=None,
        help="Optional review state filter.",
    )

    entities_review_decide_parser = entities_review_subparsers.add_parser(
        "decide",
        help="Record explicit decision for one review item.",
    )
    entities_review_decide_parser.add_argument(
        "--review-db-path",
        required=True,
        help="Path to SQLite review queue database.",
    )
    entities_review_decide_parser.add_argument(
        "--mention-entity-id",
        required=True,
        help="Mention entity identifier in review queue.",
    )
    entities_review_decide_parser.add_argument(
        "--decision",
        required=True,
        choices=["approved", "rejected"],
        help="Review decision outcome.",
    )
    entities_review_decide_parser.add_argument(
        "--canonical-entity-key",
        default=None,
        help="Canonical key override used for approved items.",
    )
    entities_review_decide_parser.add_argument(
        "--canonical-name",
        default=None,
        help="Canonical name override used for approved items.",
    )
    entities_review_decide_parser.add_argument(
        "--entity-type",
        default=None,
        help="Entity type override used for approved items.",
    )
    entities_review_decide_parser.add_argument(
        "--aliases",
        default="",
        help="Comma-separated canonical aliases override.",
    )

    args = parser.parse_args(list(argv))

    if args.command == "entities-catalog":
        if args.entities_catalog_command == "import":
            report = import_seed_catalog(
                seed_path=Path(args.seed_path),
                catalog_path=Path(args.catalog_path),
                vectors_path=Path(args.vectors_path),
                catalog_version=args.catalog_version,
            )
            stdout.write(
                json.dumps(
                    {
                        "catalog_version": report.catalog_version,
                        "seed_hash": report.seed_hash,
                        "processed_records": report.processed_records,
                        "inserted_records": report.inserted_records,
                        "vectors_written": report.vectors_written,
                    },
                    ensure_ascii=True,
                )
                + "\n",
            )
            return 0

        report = promote_review_items(
            review_db_path=Path(args.review_db_path),
            catalog_path=Path(args.catalog_path),
            vectors_path=Path(args.vectors_path),
            catalog_version=args.catalog_version,
            auto_approve=args.auto_approve,
            min_score=args.auto_approve_min_score,
            min_margin=args.auto_approve_min_margin,
        )
        stdout.write(
            json.dumps(
                {
                    "auto_approved_items": report.auto_approved_items,
                    "promoted_items": report.promoted_items,
                    "inserted_records": report.inserted_records,
                    "updated_records": report.updated_records,
                    "vectors_written": report.vectors_written,
                },
                ensure_ascii=True,
            )
            + "\n",
        )
        return 0

    if args.command == "entities-review":
        repository = SQLiteReviewQueueRepository(sqlite_path=Path(args.review_db_path))
        if args.entities_review_command == "list":
            items = repository.list_review_items(state=args.state)
            stdout.write(json.dumps({"items": items}, ensure_ascii=True) + "\n")
            return 0

        aliases = [alias.strip() for alias in str(args.aliases).split(",") if alias.strip()]
        report = repository.decide_item(
            mention_entity_id=args.mention_entity_id,
            decision=args.decision,
            canonical_entity_key=args.canonical_entity_key,
            canonical_name=args.canonical_name,
            entity_type=args.entity_type,
            aliases=aliases,
        )
        stdout.write(
            json.dumps(
                {
                    "mention_entity_id": report.mention_entity_id,
                    "review_state": report.review_state,
                    "updated_rows": report.updated_rows,
                },
                ensure_ascii=True,
            )
            + "\n",
        )
        return 0

    if args.command != "chunk":
        parser.print_help(file=stdout)
        return 2

    asyncio.run(run_chunking(state_path=args.state_path, source_id=args.source))
    return 0


async def run_chunking(
    state_path: Path,
    source_id: str,
) -> None:
    """Run chunking for one source and return a report of outcomes."""
    versions = get_source_document_versions(
        state_path=Path(state_path),
        source_id=source_id,
    )
    repository = SQLiteDocumentChunkRepository(db_manager=DbManager.get_instance())
    await repository.init()
    report = await chunk_and_persist_documents(
        repository=repository,
        documents=versions,
    )
    db = await get_db_async()
    await db.close()
    stdout.write(
        " ".join(
            [
                f"source={source_id}",
                f"processed_documents={report.processed_documents}",
                f"chunked_documents={report.chunked_documents}",
                f"skipped_documents={report.skipped_documents}",
            ],
        )
        + "\n",
    )
