"""Unit tests for hybrid catalog retrieval adapter behavior."""

from __future__ import annotations

import json
from pathlib import Path

from ddd_policy_tracer.analysis.entities import HybridCatalogRetriever, import_seed_catalog


def _write_seed(path: Path) -> None:
    """Write seed rows that exercise ranking and type signals."""
    rows = [
        {
            "canonical_entity_key": "policy:clean-energy-act",
            "canonical_name": "Clean Energy Act",
            "entity_type": "POLICY",
            "aliases": ["CEA"],
        },
        {
            "canonical_entity_key": "org:australia-institute",
            "canonical_name": "Australia Institute",
            "entity_type": "ORG",
            "aliases": ["The Australia Institute"],
        },
        {
            "canonical_entity_key": "program:clean-energy-program",
            "canonical_name": "Clean Energy Program",
            "entity_type": "PROGRAM",
            "aliases": ["CEP"],
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def test_hybrid_retriever_returns_fused_topk_candidates(tmp_path: Path) -> None:
    """Return deterministic top-k ranked candidates from hybrid signals."""
    seed_path = tmp_path / "seed.jsonl"
    catalog_path = tmp_path / "catalog.db"
    vectors_path = tmp_path / "vectors.json"
    _write_seed(seed_path)
    import_seed_catalog(
        seed_path=seed_path,
        catalog_path=catalog_path,
        vectors_path=vectors_path,
        catalog_version="catalog-v1",
    )
    retriever = HybridCatalogRetriever(catalog_path=catalog_path, vectors_path=vectors_path)

    result = retriever.retrieve(
        mention_text="Clean Energy Act",
        normalized_mention_text="clean energy act",
        mention_entity_type="POLICY",
        top_k=2,
    )

    assert len(result.candidates) == 2
    assert result.candidates[0].canonical_entity_key == "policy:clean-energy-act"
    assert result.candidates[0].fused_score >= result.candidates[1].fused_score
    assert "query_tokens" in result.candidates[0].diagnostics


def test_hybrid_retriever_produces_deterministic_order_across_runs(tmp_path: Path) -> None:
    """Keep candidate ordering stable for repeated identical queries."""
    seed_path = tmp_path / "seed.jsonl"
    catalog_path = tmp_path / "catalog.db"
    vectors_path = tmp_path / "vectors.json"
    _write_seed(seed_path)
    import_seed_catalog(
        seed_path=seed_path,
        catalog_path=catalog_path,
        vectors_path=vectors_path,
        catalog_version="catalog-v1",
    )
    retriever = HybridCatalogRetriever(catalog_path=catalog_path, vectors_path=vectors_path)

    first = retriever.retrieve(
        mention_text="Clean Energy",
        normalized_mention_text="clean energy",
        mention_entity_type="PROGRAM",
        top_k=3,
    )
    second = retriever.retrieve(
        mention_text="Clean Energy",
        normalized_mention_text="clean energy",
        mention_entity_type="PROGRAM",
        top_k=3,
    )

    assert [candidate.canonical_entity_key for candidate in first.candidates] == [
        candidate.canonical_entity_key for candidate in second.candidates
    ]
