"""Graph materialization logic for publisher-to-claim relationships."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from .contracts import GraphEdge, GraphNode
from .repositories import ClaimRecord

_PUBLISHER_LABEL_MAP = {
    "australia_institute": "The Australia Institute",
    "lowy_institute": "Lowy Institute",
}


@dataclass(frozen=True)
class MaterializedGraph:
    """Represent graph nodes and edges produced by one materialization pass."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    stats: dict[str, int]


def materialize_publisher_claim_graph(*, claims: list[ClaimRecord]) -> MaterializedGraph:
    """Materialize deterministic publisher and claim graph components."""
    nodes_by_id: dict[str, GraphNode] = {}
    edges_by_id: dict[str, GraphEdge] = {}

    for claim in claims:
        publisher_id = _publisher_node_id(source_id=claim.source_id)
        claim_node_id = claim.claim_id

        nodes_by_id.setdefault(
            publisher_id,
            GraphNode(
                id=publisher_id,
                type="PublisherOrganization",
                label=_publisher_label(source_id=claim.source_id),
                properties={
                    "source_id": claim.source_id,
                },
            ),
        )
        nodes_by_id.setdefault(
            claim_node_id,
            GraphNode(
                id=claim_node_id,
                type="Claim",
                label=claim.normalized_claim_text,
                properties={
                    "claim_id": claim.claim_id,
                    "chunk_id": claim.chunk_id,
                    "source_id": claim.source_id,
                    "source_document_id": claim.source_document_id,
                    "document_checksum": claim.document_checksum,
                    "start_char": claim.start_char,
                    "end_char": claim.end_char,
                    "evidence_text": claim.evidence_text,
                    "normalized_claim_text": claim.normalized_claim_text,
                    "confidence": claim.confidence,
                    "claim_type": claim.claim_type,
                    "extractor_version": claim.extractor_version,
                },
            ),
        )

        raised_edge_id = _edge_id(
            edge_type="RAISED",
            source=publisher_id,
            target=claim_node_id,
            scope_key=f"{claim.source_document_id}|{claim.document_checksum}|{claim.chunk_id}",
        )
        edges_by_id.setdefault(
            raised_edge_id,
            GraphEdge(
                id=raised_edge_id,
                type="RAISED",
                source=publisher_id,
                target=claim_node_id,
                properties={
                    "source_id": claim.source_id,
                    "source_document_id": claim.source_document_id,
                    "document_checksum": claim.document_checksum,
                    "chunk_id": claim.chunk_id,
                    "claim_id": claim.claim_id,
                },
            ),
        )

    nodes = sorted(nodes_by_id.values(), key=lambda node: node.id)
    edges = sorted(edges_by_id.values(), key=lambda edge: edge.id)
    stats = {
        "publisher_nodes": sum(1 for node in nodes if node.type == "PublisherOrganization"),
        "claim_nodes": sum(1 for node in nodes if node.type == "Claim"),
        "raised_edges": len(edges),
    }
    return MaterializedGraph(nodes=nodes, edges=edges, stats=stats)


def _publisher_node_id(*, source_id: str) -> str:
    """Build deterministic publisher organization node identifier."""
    return f"publisher:{_hash_value(source_id)}"


def _publisher_label(*, source_id: str) -> str:
    """Resolve a readable publisher label from source identifier."""
    mapped = _PUBLISHER_LABEL_MAP.get(source_id)
    if mapped is not None:
        return mapped
    return source_id.replace("_", " ").title()


def _edge_id(*, edge_type: str, source: str, target: str, scope_key: str) -> str:
    """Build deterministic edge identifier from graph relationship inputs."""
    raw = f"{edge_type}|{source}|{target}|{scope_key}"
    return f"edge:{_hash_value(raw)}"


def _hash_value(value: str) -> str:
    """Hash one value to a stable ASCII-safe identifier suffix."""
    return sha256(value.encode("utf-8")).hexdigest()
