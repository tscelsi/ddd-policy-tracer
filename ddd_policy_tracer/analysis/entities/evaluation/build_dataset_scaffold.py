"""Build an entity evaluation dataset scaffold from chunk JSONL input."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def run(
    *,
    chunks_path: Path,
    output_path: Path,
    sample_size: int,
    seed: int,
) -> None:
    """Write sampled chunk records with empty gold entity labels for review."""
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero")

    chunks = _load_chunks(chunks_path)
    if not chunks:
        raise ValueError("No chunks found in chunks_path")

    active_sample_size = min(sample_size, len(chunks))
    rng = random.Random(seed)  # noqa: S311 - deterministic non-crypto sampling
    sampled_chunks = rng.sample(chunks, k=active_sample_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in sampled_chunks:
            record = {
                "chunk_id": chunk["chunk_id"],
                "source_id": chunk["source_id"],
                "source_document_id": chunk["source_document_id"],
                "document_checksum": chunk["document_checksum"],
                "chunk_text": chunk["chunk_text"],
                "gold_entities": [],
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    """Load chunk records from append-only JSONL chunk state."""
    if not chunks_path.exists():
        raise ValueError("chunks_path does not exist")

    chunks: list[dict[str, Any]] = []
    content = chunks_path.read_text(encoding="utf-8")
    for raw_line in content.splitlines():
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        chunks.append(payload)
    return chunks


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for entity dataset scaffold generation."""
    parser = argparse.ArgumentParser(
        prog="entities-eval-build-dataset-scaffold",
        description="Build entity evaluation dataset scaffold from chunks.",
    )
    parser.add_argument(
        "--chunks-path",
        required=True,
        help="Path to chunk JSONL produced by analysis chunking.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to write scaffold dataset JSONL.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=300,
        help="Number of chunks to sample for review.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    return parser


def main() -> int:
    """Run scaffold generation from command-line arguments."""
    parser = _build_parser()
    args = parser.parse_args()
    run(
        chunks_path=Path(args.chunks_path),
        output_path=Path(args.output_path),
        sample_size=args.sample_size,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
