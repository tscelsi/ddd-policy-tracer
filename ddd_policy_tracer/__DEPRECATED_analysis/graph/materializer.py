"""Graph materialization logic for publisher-to-claim relationships."""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256

from .contracts import GraphEdge, GraphNode
from .repositories import ClaimRecord, EntityRecord

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


def materialize_graph_with_entities(
    *,
    claims: list[ClaimRecord],
    entities: list[EntityRecord],
) -> MaterializedGraph:
    """Materialize publisher, claim, and mentioned-entity graph components."""
    base_graph = materialize_publisher_claim_graph(claims=claims)
    nodes_by_id = {node.id: node for node in base_graph.nodes}
    edges_by_id = {edge.id: edge for edge in base_graph.edges}

    claims_by_chunk: dict[str, list[ClaimRecord]] = {}
    for claim in claims:
        claims_by_chunk.setdefault(claim.chunk_id, []).append(claim)

    canonical_links_by_claim_id: dict[str, set[str]] = {}
    weighted_links_by_claim_id: dict[str, dict[str, float]] = {}
    for claim in claims:
        linked_entities = claim.linked_entities or []
        linked_keys = {
            str(linked.get("canonical_entity_key"))
            for linked in linked_entities
            if isinstance(linked, dict) and isinstance(linked.get("canonical_entity_key"), str)
        }
        canonical_links_by_claim_id[claim.claim_id] = linked_keys
        weighted_links: dict[str, float] = {}
        for linked in linked_entities:
            if not isinstance(linked, dict):
                continue
            key = linked.get("canonical_entity_key")
            confidence = linked.get("link_confidence")
            if not isinstance(key, str):
                continue
            if isinstance(confidence, (int, float)):
                weighted_links[key] = float(confidence)
                continue
            weighted_links.setdefault(key, 1.0)
        for pending in claim.pending_entity_links or []:
            if not isinstance(pending, dict):
                continue
            for candidate in _candidate_links_from_pending(pending):
                existing = weighted_links.get(candidate["canonical_entity_key"], 0.0)
                weighted_links[candidate["canonical_entity_key"]] = max(
                    existing,
                    candidate["confidence_score"],
                )
        weighted_links_by_claim_id[claim.claim_id] = weighted_links

    for entity in entities:
        normalized_value = (
            entity.canonical_entity_key
            if entity.canonical_entity_key is not None
            else entity.normalized_mention_text
        )
        entity_node_id = _mentioned_entity_node_id(
            entity_type=entity.entity_type,
            normalized_mention_text=normalized_value,
        )
        nodes_by_id.setdefault(
            entity_node_id,
            GraphNode(
                id=entity_node_id,
                type="MentionedEntity",
                label=entity.canonical_name or entity.mention_text,
                properties={
                    "entity_type": entity.entity_type,
                    "normalized_mention_text": entity.normalized_mention_text,
                    "canonical_entity_key": entity.canonical_entity_key,
                    "canonical_name": entity.canonical_name,
                },
            ),
        )

        chunk_claims = claims_by_chunk.get(entity.chunk_id, [])
        for claim in chunk_claims:
            linked_keys = canonical_links_by_claim_id.get(claim.claim_id, set())
            weighted_links = weighted_links_by_claim_id.get(claim.claim_id, {})
            if (
                linked_keys
                and entity.canonical_entity_key not in linked_keys
                and entity.canonical_entity_key not in weighted_links
            ):
                continue
            if not _entity_matches_claim_text(
                claim=claim,
                entity=entity,
                has_explicit_link=(
                    bool(linked_keys)
                    and entity.canonical_entity_key in linked_keys
                )
                or (
                    entity.canonical_entity_key is not None
                    and entity.canonical_entity_key in weighted_links
                ),
            ):
                continue

            link_confidence = _resolve_link_confidence(
                entity=entity,
                weighted_links=weighted_links,
            )

            mentions_edge_id = _edge_id(
                edge_type="MENTIONS",
                source=claim.claim_id,
                target=entity_node_id,
                scope_key=(
                    f"{entity.chunk_id}|{entity.start_char}|{entity.end_char}|"
                    f"{claim.start_char}|{claim.end_char}"
                ),
            )
            edges_by_id.setdefault(
                mentions_edge_id,
                GraphEdge(
                    id=mentions_edge_id,
                    type="MENTIONS",
                    source=claim.claim_id,
                    target=entity_node_id,
                    properties={
                        "source_id": claim.source_id,
                        "source_document_id": claim.source_document_id,
                        "document_checksum": claim.document_checksum,
                        "chunk_id": claim.chunk_id,
                        "claim_id": claim.claim_id,
                        "entity_type": entity.entity_type,
                        "entity_mention_text": entity.mention_text,
                        "normalized_mention_text": entity.normalized_mention_text,
                        "entity_confidence": entity.confidence,
                        "link_confidence": link_confidence,
                        "claim_confidence": claim.confidence,
                        "link_method": "chunk_cooccurrence_text_match",
                    },
                ),
            )

    nodes = sorted(nodes_by_id.values(), key=lambda node: node.id)
    edges = sorted(edges_by_id.values(), key=lambda edge: edge.id)
    stats = {
        "publisher_nodes": sum(1 for node in nodes if node.type == "PublisherOrganization"),
        "claim_nodes": sum(1 for node in nodes if node.type == "Claim"),
        "mentioned_entity_nodes": sum(1 for node in nodes if node.type == "MentionedEntity"),
        "raised_edges": sum(1 for edge in edges if edge.type == "RAISED"),
        "mentions_edges": sum(1 for edge in edges if edge.type == "MENTIONS"),
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


def _mentioned_entity_node_id(*, entity_type: str, normalized_mention_text: str) -> str:
    """Build deterministic mentioned-entity node ID from type-sensitive key."""
    key = f"{entity_type}|{normalized_mention_text}"
    return f"entity:{_hash_value(key)}"


def _entity_matches_claim_text(
    *,
    claim: ClaimRecord,
    entity: EntityRecord,
    has_explicit_link: bool,
) -> bool:
    """Return true when an entity appears in claim text by two-pass rules."""
    if has_explicit_link:
        return True
    entity_text = entity.mention_text.strip()
    if not entity_text:
        return False

    if entity_text.lower() in claim.evidence_text.lower():
        return True

    normalized_claim = _normalize_for_match(claim.normalized_claim_text)
    normalized_entity = _normalize_for_match(entity.normalized_mention_text)
    return bool(normalized_entity) and normalized_entity in normalized_claim


def _normalize_for_match(value: str) -> str:
    """Normalize text by lowercasing and removing punctuation/extra spacing."""
    lowered = value.lower()
    no_punctuation = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return " ".join(no_punctuation.split())


def _resolve_link_confidence(*, entity: EntityRecord, weighted_links: dict[str, float]) -> float:
    """Resolve one link confidence value for entity->claim mentions edge."""
    if entity.canonical_entity_key is not None and entity.canonical_entity_key in weighted_links:
        return weighted_links[entity.canonical_entity_key]
    if entity.candidate_links:
        top = _top_candidate_confidence(entity.candidate_links)
        if top is not None:
            return top
    return float(entity.confidence)


def _top_candidate_confidence(candidate_links: list[dict[str, object]]) -> float | None:
    """Return top confidence among candidate links on one entity record."""
    top: float | None = None
    for candidate in candidate_links:
        if not isinstance(candidate, dict):
            continue
        confidence = candidate.get("confidence_score")
        if not isinstance(confidence, (int, float)):
            continue
        score = float(confidence)
        if top is None or score > top:
            top = score
    return top


def _candidate_links_from_pending(pending: dict[str, object]) -> list[dict[str, float | str]]:
    """Extract candidate-link confidence records from one pending link payload."""
    raw = pending.get("candidate_links")
    if not isinstance(raw, list):
        return []
    links: list[dict[str, float | str]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        key = entry.get("canonical_entity_key")
        confidence = entry.get("confidence_score")
        if not isinstance(key, str) or not key.strip():
            continue
        if not isinstance(confidence, (int, float)):
            continue
        links.append(
            {
                "canonical_entity_key": key,
                "confidence_score": float(confidence),
            },
        )
    return links
