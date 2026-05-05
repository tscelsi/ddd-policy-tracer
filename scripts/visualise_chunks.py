"""Interactive CLI for viewing persisted chunk text from JSONL state."""

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
class ChunkRow:
    """Represent one chunk row loaded from JSONL."""

    chunk_id: str
    source_id: str
    source_document_id: str
    chunk_index: int
    start_char: int
    end_char: int
    chunk_text: str


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments for chunk visualization."""
    parser = argparse.ArgumentParser(
        description=(
            "Visualise chunk text from a JSONL chunk repository. "
            "Press space for random chunk, b for previous, q to quit."
        ),
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path("data/chunks_spacy.jsonl"),
        help="Path to chunk JSONL file.",
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


def load_chunks(path: Path) -> list[ChunkRow]:
    """Load chunk rows from one JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    rows: list[ChunkRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue

            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number} in {path}",
                ) from exc

            rows.append(_row_from_payload(payload))

    return rows


def _row_from_payload(payload: dict[str, Any]) -> ChunkRow:
    """Build one ChunkRow from a decoded JSON object."""
    return ChunkRow(
        chunk_id=str(payload["chunk_id"]),
        source_id=str(payload["source_id"]),
        source_document_id=str(payload["source_document_id"]),
        chunk_index=int(payload["chunk_index"]),
        start_char=int(payload["start_char"]),
        end_char=int(payload["end_char"]),
        chunk_text=str(payload["chunk_text"]),
    )


def filter_chunks(
    rows: list[ChunkRow],
    *,
    source_id: str | None,
    source_document_id: str | None,
    chunk_id: str | None,
    contains: str | None,
) -> list[ChunkRow]:
    """Filter chunk rows by optional user criteria."""
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
    """Clear the terminal screen for paged chunk display."""
    os.system("cls" if os.name == "nt" else "clear")


def render_chunk(*, row: ChunkRow, index: int, total: int, width: int) -> None:
    """Print one chunk view with metadata and wrapped text."""
    clear_screen()
    print(f"Chunk {index + 1}/{total}")
    print(f"chunk_id: {row.chunk_id}")
    print(f"source_id: {row.source_id}")
    print(f"source_document_id: {row.source_document_id}")
    print(
        f"chunk_index: {row.chunk_index}  range: {row.start_char}-{row.end_char}",
    )
    print("-" * min(width, 120))
    wrapped = textwrap.fill(
        row.chunk_text,
        width=max(20, width),
        replace_whitespace=False,
        drop_whitespace=False,
    )
    print(wrapped)
    print("-" * min(width, 120))
    print("Controls: [space]=random  b=back  q=quit")


def read_command() -> str:
    """Read one keyboard command, preferring single-key input on TTY."""
    if not sys.stdin.isatty():
        # Non-interactive fallback for pipes/CI.
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

    # Windows fallback.
    import msvcrt

    return msvcrt.getch().decode("utf-8", errors="ignore")


def run(argv: list[str]) -> int:
    """Run the interactive chunk visualizer CLI."""
    args = parse_args(argv)
    try:
        rows = load_chunks(args.chunks_path)
    except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    rows = filter_chunks(
        rows,
        source_id=args.source_id,
        source_document_id=args.source_document_id,
        chunk_id=args.chunk_id,
        contains=args.contains,
    )
    if not rows:
        print("No chunks matched your filters.")
        return 0

    index = min(max(args.start_index, 0), len(rows) - 1)
    while True:
        render_chunk(
            row=rows[index],
            index=index,
            total=len(rows),
            width=args.width,
        )
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
