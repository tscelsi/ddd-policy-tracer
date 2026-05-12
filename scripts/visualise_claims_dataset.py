"""Interactive CLI for viewing claims silver dataset rows and extracted claims."""

# ruff: noqa: T201, S605, S311

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ClaimSpan:
    """Represent one extracted claim span in a chunk."""

    start_char: int
    end_char: int


@dataclass(frozen=True)
class ChunkClaimsRow:
    """Represent one chunk and its extracted claim spans."""

    chunk_id: str
    source_id: str
    source_document_id: str
    document_checksum: str
    chunk_text: str
    claims: tuple[ClaimSpan, ...]


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments for claims dataset visualization."""
    parser = argparse.ArgumentParser(
        description=(
            "Visualise claims silver dataset JSONL. "
            "Press space for random chunk, b for previous, q to quit."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/claims_dataset.jsonl"),
        help="Path to claims dataset JSONL file.",
    )
    parser.add_argument(
        "--source-id",
        default=None,
        help="Optional filter by source_id.",
    )
    parser.add_argument(
        "--source-document-id",
        default=None,
        help="Optional filter by source_document_id.",
    )
    parser.add_argument(
        "--chunk-id",
        default=None,
        help="Optional filter by chunk_id.",
    )
    parser.add_argument(
        "--contains",
        default=None,
        help="Optional case-insensitive text filter on chunk_text.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start viewing from this result index.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=100,
        help="Text wrap width.",
    )
    return parser.parse_args(argv)


def load_claim_rows(path: Path) -> list[ChunkClaimsRow]:
    """Load and group claim rows by chunk_id from a JSONL dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Claims dataset file not found: {path}")

    grouped: dict[str, ChunkClaimsRow] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number} in {path}") from exc

            row = _payload_to_row(payload)
            existing = grouped.get(row.chunk_id)
            if existing is None:
                grouped[row.chunk_id] = row
                continue

            merged_claims = tuple(
                sorted({*existing.claims, *row.claims}, key=lambda c: c.start_char),
            )
            grouped[row.chunk_id] = ChunkClaimsRow(
                chunk_id=existing.chunk_id,
                source_id=existing.source_id,
                source_document_id=existing.source_document_id,
                document_checksum=existing.document_checksum,
                chunk_text=existing.chunk_text,
                claims=merged_claims,
            )

    return list(grouped.values())


def _payload_to_row(payload: dict[str, Any]) -> ChunkClaimsRow:
    """Build one typed row from decoded JSON payload."""
    claims_payload = payload.get("silver_claims", [])
    claim_spans: list[ClaimSpan] = []
    if isinstance(claims_payload, list):
        for claim in claims_payload:
            if not isinstance(claim, dict):
                continue
            start_char = claim.get("start_char")
            end_char = claim.get("end_char")
            if isinstance(start_char, int) and isinstance(end_char, int):
                if start_char >= 0 and end_char > start_char:
                    claim_spans.append(ClaimSpan(start_char=start_char, end_char=end_char))

    return ChunkClaimsRow(
        chunk_id=str(payload["chunk_id"]),
        source_id=str(payload["source_id"]),
        source_document_id=str(payload["source_document_id"]),
        document_checksum=str(payload["document_checksum"]),
        chunk_text=str(payload["chunk_text"]),
        claims=tuple(sorted(claim_spans, key=lambda c: c.start_char)),
    )


def filter_rows(
    rows: list[ChunkClaimsRow],
    *,
    source_id: str | None,
    source_document_id: str | None,
    chunk_id: str | None,
    contains: str | None,
) -> list[ChunkClaimsRow]:
    """Filter claim rows by optional user criteria."""
    filtered = rows
    if source_id is not None:
        filtered = [row for row in filtered if row.source_id == source_id]
    if source_document_id is not None:
        filtered = [row for row in filtered if row.source_document_id == source_document_id]
    if chunk_id is not None:
        filtered = [row for row in filtered if row.chunk_id == chunk_id]
    if contains is not None:
        needle = contains.casefold()
        filtered = [row for row in filtered if needle in row.chunk_text.casefold()]
    return filtered


def clear_screen() -> None:
    """Clear terminal screen for paged dataset display."""
    os.system("cls" if os.name == "nt" else "clear")


def render_row(*, row: ChunkClaimsRow, index: int, total: int, width: int) -> None:
    """Print one chunk plus extracted claims with offsets and text."""
    clear_screen()
    print(f"Chunk {index + 1}/{total}")
    print(f"chunk_id: {row.chunk_id}")
    print(f"source_id: {row.source_id}")
    print(f"source_document_id: {row.source_document_id}")
    print(f"document_checksum: {row.document_checksum}")
    print(f"claims_extracted: {len(row.claims)}")
    print("-" * min(width, 120))
    wrapped = textwrap.fill(
        row.chunk_text,
        width=max(20, width),
        replace_whitespace=False,
        drop_whitespace=False,
    )
    print(wrapped)
    print("-" * min(width, 120))

    if not row.claims:
        print("No claims extracted for this chunk.")
    else:
        for claim_index, claim in enumerate(row.claims, start=1):
            claim_text = row.chunk_text[claim.start_char : claim.end_char]
            wrapped_claim = textwrap.fill(claim_text, width=max(20, width - 8))
            print(
                f"[{claim_index}] ({claim.start_char}, {claim.end_char})"
                f" len={claim.end_char - claim.start_char}",
            )
            print(f"    {wrapped_claim}")

    print("-" * min(width, 120))
    print("Controls: [space]=random  Enter=next  b=back  q=quit")


def read_command() -> str:
    """Read one keyboard command, preferring single-key input on TTY."""
    if not sys.stdin.isatty():
        return "q"

    if os.name != "nt":
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            char = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return char

    import msvcrt

    return msvcrt.getch().decode("utf-8", errors="ignore")


def run(argv: list[str]) -> int:
    """Run the interactive claims dataset visualizer CLI."""
    args = parse_args(argv)
    try:
        rows = load_claim_rows(args.dataset_path)
    except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    rows = filter_rows(
        rows,
        source_id=args.source_id,
        source_document_id=args.source_document_id,
        chunk_id=args.chunk_id,
        contains=args.contains,
    )
    if not rows:
        print("No claim rows matched your filters.")
        return 0

    index = min(max(args.start_index, 0), len(rows) - 1)
    while True:
        render_row(row=rows[index], index=index, total=len(rows), width=args.width)
        command = read_command()

        if command in {"q", "Q", "\x03"}:
            break
        if command == " ":
            if len(rows) > 1:
                next_index = index
                while next_index == index:
                    next_index = random.randrange(len(rows))
                index = next_index
            continue
        if command in {"\r", "\n"}:
            if index < len(rows) - 1:
                index += 1
            continue
        if command in {"b", "B", "\x7f"}:
            if index > 0:
                index -= 1
            continue

    return 0


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
