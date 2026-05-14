"""CLI entrypoint for analysis chunking and canonicalization operations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from ddd_policy_tracer.analysis.chunks.chunking import SpacyChunker
from ddd_policy_tracer.discovery.service_layer import (
    get_source_document_versions,
)

from .canonicalization.run import main as canonicalization_main
from .entities.catalog import import_seed_catalog
from .chunks.chunking_models import ChunkingConfig
from .chunks.service_layer import chunk_and_persist_document_versions


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
    chunk_parser.add_argument(
        "--chunk-state-path",
        required=True,
        help="Path where chunk records will be persisted.",
    )
    chunk_parser.add_argument(
        "--chunk-size-chars",
        type=int,
        default=1200,
        help="Maximum characters per chunk.",
    )
    chunk_parser.add_argument(
        "--chunk-overlap-chars",
        type=int,
        default=200,
        help="Overlap characters between adjacent chunks.",
    )

    canonicalize_parser = subparsers.add_parser(
        "canonicalize",
        help="Run analysis canonicalization workflows",
    )
    canonicalize_parser.add_argument(
        "canonicalization_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to canonicalization subcommands.",
    )

    entities_catalog_parser = subparsers.add_parser(
        "entities-catalog",
        help="Manage runtime entity catalog operations",
    )
    entities_catalog_parser.add_argument(
        "--seed-path",
        required=True,
        help="Path to curated seed JSONL data.",
    )
    entities_catalog_parser.add_argument(
        "--catalog-path",
        required=True,
        help="Path to runtime SQLite catalog database.",
    )
    entities_catalog_parser.add_argument(
        "--vectors-path",
        required=True,
        help="Path to sidecar vectors JSON artifact.",
    )
    entities_catalog_parser.add_argument(
        "--catalog-version",
        default="catalog-v1",
        help="Catalog version metadata value for compatibility checks.",
    )

    args = parser.parse_args(list(argv))
    if args.command == "canonicalize":
        forwarded = list(args.canonicalization_args)
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        return canonicalization_main(forwarded)

    if args.command == "entities-catalog":
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

    if args.command != "chunk":
        parser.print_help(file=stdout)
        return 2

    versions = get_source_document_versions(
        state_path=Path(args.state_path),
        source_id=args.source,
    )
    config = ChunkingConfig(
        chunk_size_chars=args.chunk_size_chars,
        chunk_overlap_chars=args.chunk_overlap_chars,
    )
    chunker = SpacyChunker(config=config)
    report = chunk_and_persist_document_versions(
        documents=versions,
        chunker=chunker,
        state_path=Path(args.chunk_state_path),
    )
    stdout.write(
        " ".join(
            [
                f"source={args.source}",
                f"processed_documents={report.processed_documents}",
                f"chunked_documents={report.chunked_documents}",
                f"skipped_documents={report.skipped_documents}",
                f"persisted_chunks={report.persisted_chunks}",
            ],
        )
        + "\n",
    )
    return 0
