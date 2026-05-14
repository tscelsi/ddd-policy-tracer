"""Runtime entity catalog bootstrap and persistence operations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CatalogImportReport:
    """Summarize one runtime catalog import execution result."""

    catalog_version: str
    seed_hash: str
    processed_records: int
    inserted_records: int
    vectors_written: int


def import_seed_catalog(
    *,
    seed_path: Path,
    catalog_path: Path,
    vectors_path: Path,
    catalog_version: str,
) -> CatalogImportReport:
    """Import curated seed entities into runtime catalog and vectors."""
    rows = _load_seed_rows(seed_path)
    seed_hash = _hash_seed_rows(rows)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    vectors_path.parent.mkdir(parents=True, exist_ok=True)

    inserted_records = 0
    with _connect(catalog_path) as connection:
        _initialize_schema(connection)
        for row in rows:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO entity_catalog (
                    canonical_entity_key,
                    canonical_name,
                    entity_type,
                    aliases_json,
                    source
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    row["canonical_entity_key"],
                    row["canonical_name"],
                    row["entity_type"],
                    json.dumps(row["aliases"], ensure_ascii=True),
                    "seed",
                ),
            )
            inserted_records += cursor.rowcount
        connection.execute(
            """
            INSERT OR REPLACE INTO catalog_metadata (
                key,
                value
            ) VALUES (?, ?), (?, ?)
            """,
            (
                "catalog_version",
                catalog_version,
                "seed_hash",
                seed_hash,
            ),
        )

    vectors = [_row_to_vector_record(row) for row in rows]
    payload = {
        "catalog_version": catalog_version,
        "seed_hash": seed_hash,
        "vectors": vectors,
    }
    vectors_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return CatalogImportReport(
        catalog_version=catalog_version,
        seed_hash=seed_hash,
        processed_records=len(rows),
        inserted_records=inserted_records,
        vectors_written=len(vectors),
    )


def get_catalog_metadata(*, catalog_path: Path) -> dict[str, str]:
    """Load metadata key-value pairs from runtime catalog state."""
    if not catalog_path.exists():
        return {}

    with _connect(catalog_path) as connection:
        _initialize_schema(connection)
        rows = connection.execute(
            """
            SELECT key, value
            FROM catalog_metadata
            ORDER BY key ASC
            """,
        ).fetchall()
    return {row[0]: row[1] for row in rows}


def _connect(path: Path) -> sqlite3.Connection:
    """Open one SQLite connection for catalog operations."""
    return sqlite3.connect(path)


def _initialize_schema(connection: sqlite3.Connection) -> None:
    """Create runtime catalog tables and uniqueness constraints."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS entity_catalog (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_entity_key TEXT NOT NULL,
            canonical_name TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            aliases_json TEXT NOT NULL,
            source TEXT NOT NULL,
            UNIQUE (canonical_entity_key)
        )
        """,
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS catalog_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """,
    )


def _load_seed_rows(seed_path: Path) -> list[dict[str, Any]]:
    """Load normalized seed rows from curated JSONL input."""
    if not seed_path.exists():
        raise ValueError("seed_path does not exist")

    rows: list[dict[str, Any]] = []
    for line in seed_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        key = str(payload.get("canonical_entity_key", "")).strip()
        name = str(payload.get("canonical_name", "")).strip()
        entity_type = str(payload.get("entity_type", "")).strip().upper()
        aliases = _normalize_aliases(payload.get("aliases"))
        if not key or not name or not entity_type:
            raise ValueError("seed row is missing required catalog fields")
        rows.append(
            {
                "canonical_entity_key": key,
                "canonical_name": name,
                "entity_type": entity_type,
                "aliases": aliases,
            },
        )
    if not rows:
        raise ValueError("seed_path contains no usable rows")
    return rows


def _normalize_aliases(raw_aliases: object) -> list[str]:
    """Normalize alias values into a deterministic distinct list."""
    if not isinstance(raw_aliases, list):
        return []
    deduped: list[str] = []
    seen: set[str] = set()
    for alias in raw_aliases:
        if not isinstance(alias, str):
            continue
        normalized = " ".join(alias.split())
        if not normalized:
            continue
        marker = normalized.casefold()
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(normalized)
    return deduped


def _hash_seed_rows(rows: list[dict[str, Any]]) -> str:
    """Build deterministic seed fingerprint for compatibility checks."""
    canonical_rows = [
        {
            "canonical_entity_key": row["canonical_entity_key"],
            "canonical_name": row["canonical_name"],
            "entity_type": row["entity_type"],
            "aliases": row["aliases"],
        }
        for row in sorted(rows, key=lambda item: item["canonical_entity_key"])
    ]
    serialized = json.dumps(canonical_rows, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _row_to_vector_record(row: dict[str, Any]) -> dict[str, Any]:
    """Build deterministic sidecar vector payload for one catalog row."""
    text = " ".join([row["canonical_name"], *row["aliases"]]).strip()
    tokens = [token for token in text.casefold().split() if token]
    unique_tokens = sorted(set(tokens))
    return {
        "canonical_entity_key": row["canonical_entity_key"],
        "entity_type": row["entity_type"],
        "tokens": unique_tokens,
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build parser for runtime catalog import command."""
    parser = argparse.ArgumentParser(
        prog="entities-catalog-import",
        description="Import curated entity seed data into runtime catalog storage.",
    )
    parser.add_argument(
        "--seed-path",
        required=True,
        help="Path to curated seed JSONL data.",
    )
    parser.add_argument(
        "--catalog-path",
        required=True,
        help="Path to runtime SQLite catalog database.",
    )
    parser.add_argument(
        "--vectors-path",
        required=True,
        help="Path to sidecar vectors JSON artifact.",
    )
    parser.add_argument(
        "--catalog-version",
        default="catalog-v1",
        help="Catalog version metadata value for compatibility checks.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run runtime catalog import from CLI arguments."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    report = import_seed_catalog(
        seed_path=Path(args.seed_path),
        catalog_path=Path(args.catalog_path),
        vectors_path=Path(args.vectors_path),
        catalog_version=args.catalog_version,
    )
    summary = {
        "catalog_version": report.catalog_version,
        "seed_hash": report.seed_hash,
        "processed_records": report.processed_records,
        "inserted_records": report.inserted_records,
        "vectors_written": report.vectors_written,
    }
    sys.stdout.write(json.dumps(summary, ensure_ascii=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
