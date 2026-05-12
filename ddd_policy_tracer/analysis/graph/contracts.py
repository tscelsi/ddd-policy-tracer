"""Graph contract dataclasses for Stage 5 artifact generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

GRAPH_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class GraphThresholds:
    """Represent confidence thresholds used for filtered graph output."""

    claim_confidence_min: float = 0.6
    entity_confidence_min: float = 0.7


@dataclass(frozen=True)
class GraphNode:
    """Represent one graph node in the materialized output contract."""

    id: str
    type: str
    label: str
    properties: dict[str, Any]


@dataclass(frozen=True)
class GraphEdge:
    """Represent one graph edge in the materialized output contract."""

    id: str
    type: str
    source: str
    target: str
    properties: dict[str, Any]


@dataclass(frozen=True)
class GraphArtifact:
    """Represent one complete graph artifact payload."""

    schema_version: str
    generated_at: str
    thresholds: GraphThresholds
    stats: dict[str, int]
    nodes: list[GraphNode]
    edges: list[GraphEdge]


@dataclass(frozen=True)
class GraphSummary:
    """Capture execution metadata and counters for one run."""

    schema_version: str
    generated_at: str
    inputs: dict[str, str]
    output_directory: str
    latest_directory: str
    thresholds: GraphThresholds
    max_anomalies: int
    anomaly_count: int
    exit_code: int
    stats: dict[str, int]
