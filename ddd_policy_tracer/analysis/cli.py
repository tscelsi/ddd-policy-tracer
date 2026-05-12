"""CLI entrypoint for analysis chunking operations."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from ddd_policy_tracer.analysis.chunks.chunking import SpacyChunker
from ddd_policy_tracer.discovery.service_layer import (
    get_source_document_versions,
)

from .chunks.chunking_models import ChunkingConfig
from .chunks.service_layer import chunk_and_persist_document_versions


def run_cli(argv: Sequence[str], *, stdout: TextIO) -> int:
    """Execute chunking commands from the operator CLI surface."""
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

    args = parser.parse_args(list(argv))
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
