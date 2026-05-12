"""Script entrypoint for Stage 5 graph scaffold generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .contracts import GraphThresholds
from .service_layer import GraphScaffoldResult, scaffold_graph_artifacts


def run(
    *,
    chunks_path: Path,
    claims_path: Path,
    entities_path: Path,
    output_root: Path,
    claim_confidence_min: float = 0.6,
    entity_confidence_min: float = 0.7,
) -> GraphScaffoldResult:
    """Run one Stage 5 scaffold execution from explicit input artifacts."""
    _validate_input_path(path=chunks_path, label="chunks")
    _validate_input_path(path=claims_path, label="claims")
    _validate_input_path(path=entities_path, label="entities")
    return scaffold_graph_artifacts(
        chunks_path=chunks_path,
        claims_path=claims_path,
        entities_path=entities_path,
        output_root=output_root,
        thresholds=GraphThresholds(
            claim_confidence_min=claim_confidence_min,
            entity_confidence_min=entity_confidence_min,
        ),
    )


def _validate_input_path(*, path: Path, label: str) -> None:
    """Validate existence and JSONL row shape for one input path."""
    if not path.exists():
        raise ValueError(f"{label} input path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{label} input path is not a file: {path}")

    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{label} input has invalid JSON at line {index}: {path}",
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{label} input line {index} is not a JSON object: {path}")


def _build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for Stage 5 scaffold execution."""
    parser = argparse.ArgumentParser(
        prog="graph-run",
        description="Generate Stage 5 graph scaffold artifacts from JSONL inputs.",
    )
    parser.add_argument("--chunks-path", required=True, help="Path to chunks JSONL input.")
    parser.add_argument("--claims-path", required=True, help="Path to claims JSONL input.")
    parser.add_argument("--entities-path", required=True, help="Path to entities JSONL input.")
    parser.add_argument(
        "--output-root",
        default="data/graph_runs",
        help="Output root directory for timestamped graph run artifacts.",
    )
    parser.add_argument(
        "--claim-confidence-min",
        type=float,
        default=0.6,
        help="Minimum claim confidence for filtered graph output metadata.",
    )
    parser.add_argument(
        "--entity-confidence-min",
        type=float,
        default=0.7,
        help="Minimum entity confidence for filtered graph output metadata.",
    )
    return parser


def main() -> int:
    """Run graph scaffold generation from command-line arguments."""
    parser = _build_parser()
    args = parser.parse_args()
    result = run(
        chunks_path=Path(args.chunks_path),
        claims_path=Path(args.claims_path),
        entities_path=Path(args.entities_path),
        output_root=Path(args.output_root),
        claim_confidence_min=args.claim_confidence_min,
        entity_confidence_min=args.entity_confidence_min,
    )
    sys.stdout.write(
        " ".join(
            [
                f"run_directory={result.run_directory}",
                f"latest_directory={result.latest_directory}",
                f"schema_version={result.summary.schema_version}",
            ],
        )
        + "\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
