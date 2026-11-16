"""Backfill typed Hansard provenance fields onto persisted SQLite chunks."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
from enum import Enum
from pathlib import Path
from typing import Any

from pymongo import AsyncMongoClient

from ddd_policy_tracer.ingestion.hansard.part_repository import (
    MongoPartRepository,
    SpeechPart,
)
from ddd_policy_tracer.ingestion.hansard.talker_repository import (
    MongoTalkerRepository,
    Talker,
)
from ddd_policy_tracer.paths import DATA_DIR
from ddd_policy_tracer.utils.time_helpers import utc_isoformat


def build_source_metadata(part: SpeechPart, talker: Talker) -> dict[str, object]:
    """Translate a Hansard speech part and talker into retained document metadata."""
    return {
        "published_at": utc_isoformat(part.date),
        "talker": talker.model_dump(),
        "talker_id": part.talker_id,
        "house": _enum_or_string_value(part.house, field_name="house"),
        "chamber": _enum_or_string_value(part.chamber, field_name="chamber"),
        "bill_ids": part.bill_ids,
        "debate_category": part.debate_category,
        "debate_titles": [
            title for title in (part.subdebate_1_title, part.subdebate_2_title) if title
        ],
        "speech_part_type": _enum_or_string_value(
            part.speech_part_type,
            field_name="speech_part_type",
        ),
    }


def _enum_or_string_value(value: object, *, field_name: str) -> str:
    """Return a Mongo field from either an Enum object or its stored string value."""
    if isinstance(value, Enum):
        enum_value = value.value
        if isinstance(enum_value, str):
            return enum_value
    elif isinstance(value, str):
        return value
    raise TypeError(
        f"Expected {field_name} to be an enum or string, got {type(value).__name__}.",
    )


def merge_chunk_metadata(existing: str | None, source_metadata: dict[str, object]) -> str:
    """Merge fresh source provenance into the chunk's existing JSON metadata."""
    if existing is None:
        chunk_metadata: dict[str, object] = {}
    else:
        decoded = json.loads(existing)
        if not isinstance(decoded, dict):
            raise ValueError("Chunk metadata must contain a JSON object.")
        chunk_metadata = decoded

    doc_metadata = chunk_metadata.get("doc_metadata")
    if doc_metadata is None:
        merged_doc_metadata: dict[str, object] = {}
    elif isinstance(doc_metadata, dict):
        merged_doc_metadata = doc_metadata
    else:
        raise ValueError("Chunk doc_metadata must contain a JSON object.")

    chunk_metadata["doc_metadata"] = {**merged_doc_metadata, **source_metadata}
    return json.dumps(chunk_metadata, ensure_ascii=True)


async def backfill_hansard_chunks(
    *,
    database_path: Path,
    mongo_uri: str,
    mongo_database: str,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """Merge Mongo Hansard provenance into matching persisted speech chunks."""
    if not await asyncio.to_thread(database_path.is_file):
        raise FileNotFoundError(f"Chunk database does not exist: {database_path}")

    client = AsyncMongoClient[Any](mongo_uri)
    try:
        parts = await MongoPartRepository(client, mongo_database).list_all_speech_parts()
        talkers = await MongoTalkerRepository(client, mongo_database).list_talkers()
    finally:
        await client.close()

    talker_map = {talker.id: talker for talker in talkers if talker.id is not None}
    return await asyncio.to_thread(
        _backfill_database,
        database_path,
        parts,
        talker_map,
        dry_run,
    )


def _backfill_database(
    database_path: Path,
    parts: list[SpeechPart],
    talker_map: dict[str, Talker],
    dry_run: bool,
) -> tuple[int, int, int]:
    """Migrate the chunk schema and merge provenance into matching Hansard rows."""
    updates: list[tuple[object, ...]] = []
    unmatched_parts = 0
    updated_documents = 0
    connection = sqlite3.connect(database_path)
    try:
        connection.row_factory = sqlite3.Row
        with connection:
            _ensure_metadata_columns(connection)
            for part in parts:
                talker = talker_map.get(part.talker_id)
                if talker is None or talker.id is None:
                    unmatched_parts += 1
                    continue

                source_document_id = f"{talker.id}_{part.id}"
                source_metadata = build_source_metadata(part, talker)
                chunk_rows = connection.execute(
                    """
                    SELECT chunk_id, metadata
                    FROM document_chunks
                    WHERE source_id = 'hansard' AND source_document_id = ?
                    """,
                    (source_document_id,),
                ).fetchall()
                if not chunk_rows:
                    unmatched_parts += 1
                    continue

                updated_documents += 1
                for row in chunk_rows:
                    updates.append(
                        (
                            utc_isoformat(part.date),
                            talker.name,
                            part.talker_id,
                            talker.party,
                            talker.electorate,
                            source_metadata["house"],
                            source_metadata["chamber"],
                            json.dumps(part.bill_ids) if part.bill_ids is not None else None,
                            part.debate_category,
                            json.dumps(source_metadata["debate_titles"]),
                            source_metadata["speech_part_type"],
                            merge_chunk_metadata(row["metadata"], source_metadata),
                            row["chunk_id"],
                        ),
                    )

            if updates and not dry_run:
                connection.executemany(
                    """
                    UPDATE document_chunks
                    SET published_at = ?,
                        speaker_name = ?,
                        talker_id = ?,
                        party = ?,
                        electorate = ?,
                        house = ?,
                        chamber = ?,
                        bill_ids = ?,
                        debate_category = ?,
                        debate_titles = ?,
                        speech_part_type = ?,
                        metadata = ?
                    WHERE chunk_id = ?
                    """,
                    updates,
                )
        return updated_documents, len(updates), unmatched_parts
    finally:
        connection.close()


def _ensure_metadata_columns(connection: sqlite3.Connection) -> None:
    """Add the typed provenance columns and indexes if the chunk table is older."""
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(document_chunks)")}
    if not columns:
        raise ValueError("The SQLite database does not contain a document_chunks table.")

    new_columns = (
        ("published_at", "TEXT"),
        ("speaker_name", "TEXT"),
        ("talker_id", "TEXT"),
        ("party", "TEXT"),
        ("electorate", "TEXT"),
        ("house", "TEXT"),
        ("chamber", "TEXT"),
        ("bill_ids", "TEXT"),
        ("debate_category", "TEXT"),
        ("debate_titles", "TEXT"),
        ("speech_part_type", "TEXT"),
    )
    for name, column_type in new_columns:
        if name not in columns:
            connection.execute(
                f"ALTER TABLE document_chunks ADD COLUMN {name} {column_type}",
            )

    indexes = (
        "CREATE INDEX IF NOT EXISTS idx_document_chunks_published_at "
        "ON document_chunks (published_at)",
        "CREATE INDEX IF NOT EXISTS idx_document_chunks_speaker_name "
        "ON document_chunks (speaker_name)",
        "CREATE INDEX IF NOT EXISTS idx_document_chunks_talker_id ON document_chunks (talker_id)",
        "CREATE INDEX IF NOT EXISTS idx_document_chunks_house ON document_chunks (house)",
        "CREATE INDEX IF NOT EXISTS idx_document_chunks_chamber ON document_chunks (chamber)",
    )
    for statement in indexes:
        connection.execute(statement)


async def _run(args: argparse.Namespace) -> None:
    """Run the metadata backfill and report its matching and update counts."""
    updated_documents, updated_chunks, unmatched_parts = await backfill_hansard_chunks(
        database_path=Path(args.database),
        mongo_uri=args.mongo_uri,
        mongo_database=args.mongo_database,
        dry_run=args.dry_run,
    )
    action = "Would update" if args.dry_run else "Updated"
    sys.stdout.write(
        f"{action} {updated_chunks} chunks across {updated_documents} speeches; "
        f"{unmatched_parts} speech parts had no matching chunk or talker.\n",
    )


def main() -> None:
    """Parse local database settings and start the Hansard metadata backfill."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        default=os.environ.get("RAG_DATABASE_PATH", str(DATA_DIR / "new4.db")),
        help="SQLite chunk database (defaults to RAG_DATABASE_PATH or data/new4.db)",
    )
    parser.add_argument(
        "--mongo-uri",
        default=os.environ.get("HANSARD_MONGO_URI", "mongodb://localhost:27017"),
        help="MongoDB connection URI",
    )
    parser.add_argument(
        "--mongo-database",
        default=os.environ.get("HANSARD_MONGO_DATABASE", "tmp"),
        help="MongoDB database containing parts and talkers",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report matching chunks without updating their metadata",
    )
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
