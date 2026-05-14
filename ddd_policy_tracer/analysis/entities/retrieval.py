"""Hybrid retrieval adapter for runtime entity catalog resolution."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RetrievalCandidate:
    """Represent one ranked catalog candidate for one entity mention."""

    canonical_entity_key: str
    canonical_name: str
    entity_type: str
    fused_score: float
    lexical_score: float
    vector_score: float
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class RetrievalResult:
    """Represent ranked retrieval output for one mention query."""

    mention_text: str
    normalized_mention_text: str
    mention_entity_type: str
    candidates: list[RetrievalCandidate]


@dataclass(frozen=True)
class HybridCatalogRetriever:
    """Retrieve top-k catalog candidates by fused lexical and vector signals."""

    catalog_path: Path
    vectors_path: Path
    lexical_weight: float = 0.6
    vector_weight: float = 0.4

    def retrieve(
        self,
        *,
        mention_text: str,
        normalized_mention_text: str,
        mention_entity_type: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Return deterministic fused top-k candidates for one mention."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_tokens = _tokenize(normalized_mention_text or mention_text)
        vectors_by_key = _load_vectors(self.vectors_path)
        rows = _load_catalog_rows(self.catalog_path)

        scored: list[RetrievalCandidate] = []
        for row in rows:
            aliases = _parse_aliases(row["aliases_json"])
            row_tokens = _tokenize(" ".join([row["canonical_name"], *aliases]))
            lexical = _jaccard_similarity(query_tokens, row_tokens)
            vector_tokens = vectors_by_key.get(row["canonical_entity_key"], row_tokens)
            vector = _jaccard_similarity(query_tokens, vector_tokens)
            type_bonus = 0.1 if row["entity_type"] == mention_entity_type else 0.0
            fused = min(1.0, self.lexical_weight * lexical + self.vector_weight * vector + type_bonus)
            scored.append(
                RetrievalCandidate(
                    canonical_entity_key=row["canonical_entity_key"],
                    canonical_name=row["canonical_name"],
                    entity_type=row["entity_type"],
                    fused_score=round(fused, 6),
                    lexical_score=round(lexical, 6),
                    vector_score=round(vector, 6),
                    diagnostics={
                        "query_tokens": sorted(query_tokens),
                        "catalog_tokens": sorted(row_tokens),
                        "vector_tokens": sorted(vector_tokens),
                        "type_bonus": type_bonus,
                    },
                ),
            )

        ranked = sorted(
            scored,
            key=lambda candidate: (
                -candidate.fused_score,
                -candidate.lexical_score,
                -candidate.vector_score,
                candidate.canonical_entity_key,
            ),
        )[:top_k]
        return RetrievalResult(
            mention_text=mention_text,
            normalized_mention_text=normalized_mention_text,
            mention_entity_type=mention_entity_type,
            candidates=ranked,
        )


def _load_catalog_rows(catalog_path: Path) -> list[dict[str, str]]:
    """Load catalog rows from runtime SQLite repository."""
    with sqlite3.connect(catalog_path) as connection:
        rows = connection.execute(
            """
            SELECT canonical_entity_key, canonical_name, entity_type, aliases_json
            FROM entity_catalog
            ORDER BY canonical_entity_key ASC
            """,
        ).fetchall()
    return [
        {
            "canonical_entity_key": str(row[0]),
            "canonical_name": str(row[1]),
            "entity_type": str(row[2]),
            "aliases_json": str(row[3]),
        }
        for row in rows
    ]


def _load_vectors(vectors_path: Path) -> dict[str, set[str]]:
    """Load sidecar vector tokens indexed by canonical entity key."""
    payload = json.loads(vectors_path.read_text(encoding="utf-8"))
    vectors = payload.get("vectors")
    if not isinstance(vectors, list):
        return {}

    by_key: dict[str, set[str]] = {}
    for vector in vectors:
        if not isinstance(vector, dict):
            continue
        key = vector.get("canonical_entity_key")
        tokens = vector.get("tokens")
        if not isinstance(key, str) or not isinstance(tokens, list):
            continue
        normalized = {token for token in tokens if isinstance(token, str) and token.strip()}
        by_key[key] = normalized
    return by_key


def _parse_aliases(raw_aliases_json: str) -> list[str]:
    """Parse alias JSON payload into normalized alias strings."""
    try:
        parsed = json.loads(raw_aliases_json)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    aliases = [alias for alias in parsed if isinstance(alias, str) and alias.strip()]
    return aliases


def _tokenize(value: str) -> set[str]:
    """Tokenize one text value into lowercase alphanumeric words."""
    return {
        token
        for token in "".join(character if character.isalnum() else " " for character in value.casefold()).split()
        if token
    }


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    """Compute token Jaccard similarity for ranking signal fusion."""
    if not left or not right:
        return 0.0
    intersection = len(left.intersection(right))
    union = len(left.union(right))
    if union == 0:
        return 0.0
    return intersection / union
