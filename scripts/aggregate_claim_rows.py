"""Aggregate claim-per-row JSONL into one chunk-level row with merged spans."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path


def run(*, input_path: Path, output_path: Path) -> dict[str, int]:
    """Aggregate duplicate chunk rows and merge unique sorted claim spans."""
    if not input_path.exists():
        raise ValueError("input_path does not exist")

    grouped: OrderedDict[str, dict[str, object]] = OrderedDict()
    input_rows = 0

    for raw_line in input_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        if not isinstance(payload, dict):
            continue
        input_rows += 1

        chunk_id = str(payload.get("chunk_id", "")).strip()
        if not chunk_id:
            continue

        existing = grouped.get(chunk_id)
        if existing is None:
            existing = dict(payload)
            existing["silver_claims"] = []
            grouped[chunk_id] = existing

        existing_claims = existing.get("silver_claims")
        if not isinstance(existing_claims, list):
            existing_claims = []
            existing["silver_claims"] = existing_claims

        raw_claims = payload.get("silver_claims")
        if not isinstance(raw_claims, list):
            continue
        chunk_text = str(existing.get("chunk_text", ""))
        chunk_len = len(chunk_text)

        for claim in raw_claims:
            if not isinstance(claim, dict):
                continue
            start_char = claim.get("start_char")
            end_char = claim.get("end_char")
            if not isinstance(start_char, int) or not isinstance(end_char, int):
                continue
            if start_char < 0 or end_char <= start_char or end_char > chunk_len:
                continue
            existing_claims.append({"start_char": start_char, "end_char": end_char})

    output_rows = 0
    total_spans = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in grouped.values():
            raw_spans = row.get("silver_claims")
            valid_spans: list[tuple[int, int]] = []
            if isinstance(raw_spans, list):
                for span in raw_spans:
                    if not isinstance(span, dict):
                        continue
                    start_char = span.get("start_char")
                    end_char = span.get("end_char")
                    if (
                        isinstance(start_char, int)
                        and isinstance(end_char, int)
                        and end_char > start_char
                    ):
                        valid_spans.append((start_char, end_char))
            deduped_spans = sorted(set(valid_spans), key=lambda value: (value[0], value[1]))
            row["silver_claims"] = [
                {"start_char": start_char, "end_char": end_char}
                for start_char, end_char in deduped_spans
            ]

            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            output_rows += 1
            total_spans += len(deduped_spans)

    return {
        "input_rows": input_rows,
        "output_rows": output_rows,
        "total_spans": total_spans,
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for claims row aggregation utility."""
    parser = argparse.ArgumentParser(
        prog="aggregate-claim-rows",
        description="Aggregate claim-per-row JSONL into chunk-level rows.",
    )
    parser.add_argument("--input", required=True, help="Input claim-per-row JSONL path")
    parser.add_argument("--output", required=True, help="Output chunk-level JSONL path")
    return parser


def main() -> int:
    """Run row aggregation utility from CLI arguments."""
    parser = _build_parser()
    args = parser.parse_args()
    summary = run(input_path=Path(args.input), output_path=Path(args.output))
    sys.stdout.write(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
