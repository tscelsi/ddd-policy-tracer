"""Filtering rules for thresholded connected-triad graph artifacts."""

from __future__ import annotations

from .contracts import GraphArtifact, GraphEdge, GraphNode, GraphThresholds


def build_filtered_graph(
    *,
    artifact: GraphArtifact,
    thresholds: GraphThresholds,
) -> GraphArtifact:
    """Create a thresholded connected-triad graph artifact from full graph data."""
    claim_nodes_by_id = {
        node.id: node for node in artifact.nodes if node.type == "Claim"
    }

    mentions_edges = [edge for edge in artifact.edges if edge.type == "MENTIONS"]
    qualified_mentions = [
        edge
        for edge in mentions_edges
        if _qualifies_mentions_edge(edge=edge, thresholds=thresholds, claims=claim_nodes_by_id)
    ]
    connected_claim_ids = {edge.source for edge in qualified_mentions}
    connected_entity_ids = {edge.target for edge in qualified_mentions}

    qualified_raised = [
        edge
        for edge in artifact.edges
        if edge.type == "RAISED" and edge.target in connected_claim_ids
    ]
    connected_publisher_ids = {edge.source for edge in qualified_raised}

    selected_node_ids = connected_claim_ids | connected_entity_ids | connected_publisher_ids
    filtered_nodes = [
        node
        for node in artifact.nodes
        if node.id in selected_node_ids
    ]
    filtered_edges = qualified_raised + qualified_mentions

    stats = {
        **artifact.stats,
        "filtered_nodes": len(filtered_nodes),
        "filtered_edges": len(filtered_edges),
        "filtered_claim_nodes": len(connected_claim_ids),
        "filtered_mentioned_entity_nodes": len(connected_entity_ids),
        "disconnected_claim_nodes": sum(
            1
            for node in artifact.nodes
            if node.type == "Claim" and node.id not in connected_claim_ids
        ),
        "disconnected_mentioned_entity_nodes": sum(
            1
            for node in artifact.nodes
            if node.type == "MentionedEntity" and node.id not in connected_entity_ids
        ),
    }

    return GraphArtifact(
        schema_version=artifact.schema_version,
        generated_at=artifact.generated_at,
        thresholds=thresholds,
        stats=stats,
        nodes=sorted(filtered_nodes, key=lambda node: node.id),
        edges=sorted(filtered_edges, key=lambda edge: edge.id),
    )


def _qualifies_mentions_edge(
    *,
    edge: GraphEdge,
    thresholds: GraphThresholds,
    claims: dict[str, GraphNode],
) -> bool:
    """Return true when one mentions edge satisfies confidence thresholds."""
    claim_node = claims.get(edge.source)
    if claim_node is None:
        return False

    claim_confidence = _as_float(claim_node.properties.get("confidence"))
    entity_confidence = _as_float(edge.properties.get("entity_confidence"))
    return (
        claim_confidence >= thresholds.claim_confidence_min
        and entity_confidence >= thresholds.entity_confidence_min
    )


def _as_float(value: object) -> float:
    """Convert one numeric-like object into float for threshold comparisons."""
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0
