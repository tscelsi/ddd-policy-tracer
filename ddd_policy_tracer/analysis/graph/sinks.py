"""Sink ports and JSON artifact sink implementation for graph outputs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .contracts import GraphArtifact, GraphSummary


class GraphSink:
    """Define the graph sink port for materialized Stage 5 artifacts."""

    def write_full_graph(self, *, artifact: GraphArtifact) -> None:
        """Persist full graph artifact data."""
        raise NotImplementedError

    def write_filtered_graph(self, *, artifact: GraphArtifact) -> None:
        """Persist filtered graph artifact data."""
        raise NotImplementedError

    def write_summary(self, *, summary: GraphSummary) -> None:
        """Persist summary metadata for one graph materialization run."""
        raise NotImplementedError


class JsonArtifactSink(GraphSink):
    """Write graph artifacts to JSON files within a run output folder."""

    def __init__(self, *, output_dir: Path) -> None:
        """Bind sink to one output directory and ensure it exists."""
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def write_full_graph(self, *, artifact: GraphArtifact) -> None:
        """Write full graph contract payload as JSON."""
        self._write_json(file_name="graph.json", payload=asdict(artifact))

    def write_filtered_graph(self, *, artifact: GraphArtifact) -> None:
        """Write filtered graph contract payload as JSON."""
        self._write_json(file_name="graph.filtered.json", payload=asdict(artifact))

    def write_summary(self, *, summary: GraphSummary) -> None:
        """Write summary contract payload as JSON."""
        self._write_json(file_name="summary.json", payload=asdict(summary))

    def _write_json(self, *, file_name: str, payload: dict[str, object]) -> None:
        """Serialize payload to JSON with deterministic formatting."""
        target = self._output_dir / file_name
        target.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
