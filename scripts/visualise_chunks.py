"""Interactive CLI for viewing persisted chunk text from JSONL state."""

# ruff: noqa: T201

from __future__ import annotations

import argparse
import json
import os
import random
import re
import stat
import sys
import tempfile
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    from ddd_policy_tracer.answering.query_variants import QueryVariantDrafter

DEFAULT_FIXTURE_PATH = Path("tests/fixtures/rag_validation_cases.json")


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
    metadata: Mapping[str, object]


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
    parser.add_argument(
        "--review",
        action="store_true",
        help="Enable chunk selection and retrieval-case drafting/review.",
    )
    parser.add_argument(
        "--fixture-path",
        type=Path,
        default=DEFAULT_FIXTURE_PATH,
        help="Case JSON fixture to update in review mode.",
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
    metadata = payload.get("metadata")
    if isinstance(metadata, str):
        try:
            decoded_metadata = json.loads(metadata)
            metadata = (
                decoded_metadata if isinstance(decoded_metadata, dict) else {"value": metadata}
            )
        except json.JSONDecodeError:
            metadata = {"value": metadata}
    if not isinstance(metadata, dict):
        metadata = {}
    selected_metadata: dict[str, object] = dict(metadata)
    for field in (
        "published_at",
        "speaker_name",
        "talker_id",
        "party",
        "electorate",
        "house",
        "chamber",
        "bill_ids",
        "debate_category",
        "debate_titles",
        "speech_part_type",
    ):
        value = payload.get(field)
        if value is not None:
            selected_metadata[field] = value
    return ChunkRow(
        chunk_id=str(payload["chunk_id"]),
        source_id=str(payload["source_id"]),
        source_document_id=str(payload["source_document_id"]),
        chunk_index=int(payload["chunk_index"]),
        start_char=int(payload["start_char"]),
        end_char=int(payload["end_char"]),
        chunk_text=str(payload["chunk_text"]),
        metadata=selected_metadata,
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
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def render_chunk(
    *,
    row: ChunkRow,
    index: int,
    total: int,
    width: int,
    selected: bool = False,
    review: bool = False,
) -> None:
    """Print one chunk view with metadata and wrapped text."""
    clear_screen()
    print(f"Chunk {index + 1}/{total}{' [SELECTED]' if selected else ''}")
    print(f"chunk_id: {row.chunk_id}")
    print(f"source_id: {row.source_id}")
    print(f"source_document_id: {row.source_document_id}")
    print(
        f"chunk_index: {row.chunk_index}  range: {row.start_char}-{row.end_char}",
    )
    if row.metadata:
        print("metadata:")
        for key, value in row.metadata.items():
            print(f"  {key}: {value}")
    print("-" * min(width, 120))
    wrapped = textwrap.fill(
        row.chunk_text,
        width=max(20, width),
        replace_whitespace=False,
        drop_whitespace=False,
    )
    print(wrapped)
    print("-" * min(width, 120))
    controls = "[space]=random  enter=next  b=back  q=quit"
    if review:
        controls += "  x=select  g=draft questions  u=unanswerable case"
    print(f"Controls: {controls}")


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

    return cast(str, msvcrt.getch().decode("utf-8", errors="ignore"))  # type: ignore[attr-defined]


def _read_fixture(path: Path) -> dict[str, Any]:
    """Load the shared retrieval-case fixture without dropping existing fields."""
    with path.open(encoding="utf-8") as fixture_file:
        fixture = json.load(fixture_file)
    if not isinstance(fixture, dict) or not isinstance(fixture.get("cases"), list):
        raise ValueError(f"Case fixture must be a JSON object with a cases array: {path}")
    return fixture


def _save_case(fixture: dict[str, Any], case: dict[str, object], path: Path) -> None:
    """Append one reviewed candidate to the existing fixture."""
    from ddd_policy_tracer.retrieval.evaluation import validate_case

    validate_case(case)
    fixture["cases"].append(case)
    contents = json.dumps(fixture, ensure_ascii=False, indent=2) + "\n"
    try:
        existing_mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        existing_mode = None

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(contents)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        if existing_mode is not None:
            os.chmod(temporary_path, existing_mode)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print(f"Saved {case['case_id']} with review_status={case['review_status']}.")


def _new_case_id(question: str, fixture: Mapping[str, Any]) -> str:
    """Create a stable readable identifier unique within the fixture."""
    stem = re.sub(r"[^a-z0-9]+", "-", question.casefold()).strip("-")[:48].rstrip("-")
    stem = stem or "retrieval-query"
    existing_ids = {case["case_id"] for case in fixture["cases"] if isinstance(case, dict)}
    candidate = f"retrieval-{stem}"
    suffix = 2
    while candidate in existing_ids:
        candidate = f"retrieval-{stem}-{suffix}"
        suffix += 1
    return candidate


def _review_decision(
    case: dict[str, object],
    *,
    reviewer: str,
) -> dict[str, object]:
    """Let a person edit, approve, reject, or retain a candidate as a draft."""
    from ddd_policy_tracer.retrieval.evaluation import transition_review_status

    while True:
        print(f"\nQuery: {case['question']}")
        decision = (
            input(
                "Review distinct, unambiguous query: [a]pprove / [e]dit / [r]eject / [d]raft: ",
            )
            .strip()
            .lower()
        )
        if decision == "e":
            question = input("Edited query: ").strip()
            if question:
                case["question"] = question
            continue
        if decision == "r":
            reason = input("Rejection reason: ").strip()
            if reason:
                return transition_review_status(
                    case,
                    status="rejected",
                    reviewer=reviewer,
                    note=reason,
                )
            print("A rejection reason is required.")
            continue
        if decision == "a":
            return transition_review_status(
                case,
                status="approved",
                reviewer=reviewer,
                note="Human confirmed the query unambiguously targets the selected evidence.",
            )
        if decision == "d":
            return transition_review_status(case, status="draft", reviewer=reviewer)
        print("Choose a, e, r, or d.")


def _review_selected_chunks(
    selected_rows: Sequence[ChunkRow],
    *,
    fixture: dict[str, Any],
    fixture_path: Path,
    drafter: QueryVariantDrafter,
) -> None:
    """Draft distinct query candidates and preserve explicit human decisions."""
    from ddd_policy_tracer.answering.query_variants import EvidencePassage

    if not selected_rows:
        print("Select one or more chunks with x before drafting questions.")
        return
    print("\nSelected evidence (all chunk IDs will be required together):")
    for row in selected_rows:
        print(
            f"\n{row.chunk_id} — {row.source_document_id} "
            f"(source: {row.source_id}, metadata: {dict(row.metadata)})",
        )
        print(textwrap.fill(row.chunk_text, width=100))
    evidence = [
        EvidencePassage(
            chunk_id=row.chunk_id,
            chunk_text=row.chunk_text,
            source_document_id=row.source_document_id,
            metadata=row.metadata,
        )
        for row in selected_rows
    ]
    try:
        draft = drafter.draft(evidence)
    except Exception as exc:
        print(f"Query-variant generation failed ({type(exc).__name__}): {exc}")
        return
    reviewer = input("Reviewer name: ").strip()
    if not reviewer:
        print("No reviewer supplied; generated queries were not saved.")
        return
    print(
        f"Generated by {draft.provider}/{draft.model} using prompt {draft.prompt_version}. "
        "Compare variants for genuine distinctness. If a query cannot unambiguously "
        "target every selected chunk, reject it or narrow the selection and redraft; "
        "never substitute evidence IDs. Each remains a draft until explicitly approved.",
    )
    for index, query in enumerate(draft.queries, start=1):
        print(f"Variant {index}: {query}")
    for query in draft.queries:
        case: dict[str, object] = {
            "case_id": _new_case_id(query, fixture),
            "family": "evidence_retrieval",
            "question": query,
            "answerable": True,
            "gold_evidence_chunk_ids": [row.chunk_id for row in selected_rows],
            "review_status": "draft",
            "generation_provenance": {
                "provider": draft.provider,
                "model": draft.model,
                "prompt_version": draft.prompt_version,
            },
        }
        reviewed = _review_decision(case, reviewer=reviewer)
        _save_case(fixture, reviewed, fixture_path)


def _author_unanswerable_case(*, fixture: dict[str, Any], fixture_path: Path) -> None:
    """Save a human-authored query only after explicit corpus-absence confirmation."""
    question = input("Unanswerable query: ").strip()
    reason = input("Why is this absent from the indexed corpus? ").strip()
    if not question or not reason:
        print("A query and an absence reason are required.")
        return
    confirmation = input("Type YES to confirm corpus absence: ").strip()
    if confirmation != "YES":
        print("Corpus absence was not confirmed; no case was saved.")
        return
    reviewer = input("Reviewer name: ").strip()
    if not reviewer:
        print("No reviewer supplied; no case was saved.")
        return
    case: dict[str, object] = {
        "case_id": _new_case_id(question, fixture),
        "family": "evidence_retrieval",
        "question": question,
        "answerable": False,
        "gold_evidence_chunk_ids": [],
        "abstention_reason": reason,
        "absence_confirmed_by": reviewer,
        "review_status": "draft",
    }
    _save_case(
        fixture,
        _review_decision(case, reviewer=reviewer),
        fixture_path,
    )


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

    fixture: dict[str, Any] | None = None
    if args.review:
        try:
            fixture = _read_fixture(args.fixture_path)
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        from ddd_policy_tracer.app import load_app_environment

        load_app_environment()

    index = min(max(args.start_index, 0), len(rows) - 1)
    selected_ids: set[str] = set()
    while True:
        render_chunk(
            row=rows[index],
            index=index,
            total=len(rows),
            width=args.width,
            selected=rows[index].chunk_id in selected_ids,
            review=args.review,
        )
        command = read_command()

        if command in {"q", "Q", "\x03"}:
            break
        if command == " ":
            if len(rows) > 1:
                next_index = index
                while next_index == index:
                    next_index = random.SystemRandom().randrange(len(rows))
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
        if args.review and command in {"x", "X"}:
            chunk_id = rows[index].chunk_id
            if chunk_id in selected_ids:
                selected_ids.remove(chunk_id)
            else:
                selected_ids.add(chunk_id)
            continue
        if args.review and command in {"g", "G"} and fixture is not None:
            try:
                from ddd_policy_tracer.app import create_query_variant_drafter

                _review_selected_chunks(
                    [row for row in rows if row.chunk_id in selected_ids],
                    fixture=fixture,
                    fixture_path=args.fixture_path,
                    drafter=create_query_variant_drafter(),
                )
            except Exception as exc:
                print(f"Query-review workflow failed ({type(exc).__name__}): {exc}")
            continue
        if args.review and command in {"u", "U"} and fixture is not None:
            try:
                _author_unanswerable_case(fixture=fixture, fixture_path=args.fixture_path)
            except (ValueError, OSError) as exc:
                print(f"Unanswerable-case workflow failed: {exc}")
            continue

    return 0


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
