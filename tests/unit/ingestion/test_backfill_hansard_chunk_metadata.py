"""Verify Mongo Hansard provenance is merged into chunk metadata."""

import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from ddd_policy_tracer.ingestion.hansard.part_repository import (
    ChamberType,
    HouseType,
    PartType,
    SpeechPart,
    SpeechPartType,
)
from ddd_policy_tracer.ingestion.hansard.talker_repository import Talker
from scripts.backfill_hansard_chunk_metadata import (
    _backfill_database,
    build_source_metadata,
    merge_chunk_metadata,
)


def _speech_part() -> SpeechPart:
    """Build one speech part containing all metadata fields."""
    return SpeechPart(
        id="part-1",
        date=datetime(2024, 6, 1, tzinfo=UTC),
        bill_ids=["bill-1"],
        house=HouseType.SENATE,
        chamber=ChamberType.HOR_MAIN,
        type=PartType.SPEECH,
        debate_category="Bills",
        debate_seq=1,
        subdebate_1_title="Second reading",
        subdebate_1_info=None,
        subdebate_1_seq=1,
        subdebate_2_title=None,
        subdebate_2_info=None,
        subdebate_2_seq=None,
        speech_id="speech-1",
        speech_seq=1,
        part_seq=1,
        talker_id="talker-1",
        speech_content="Speech text.",
        speech_part_type=SpeechPartType.SPEECH,
    )


def _talker() -> Talker:
    """Build one talker with party and electorate provenance."""
    return Talker(
        id="talker-1",
        name="Member A",
        first_name="Member",
        last_name="A",
        electorate="Example Electorate",
        party="Example Party",
    )


def test_build_source_metadata_captures_speech_and_talker_fields() -> None:
    """Retain the speech's source fields and speaker identity."""
    metadata = build_source_metadata(_speech_part(), _talker())

    assert metadata["published_at"] == "2024-06-01T00:00:00+00:00"
    assert metadata["talker_id"] == "talker-1"
    assert metadata["house"] == "senate"
    assert metadata["chamber"] == "main"
    assert metadata["bill_ids"] == ["bill-1"]
    assert metadata["debate_category"] == "Bills"
    assert metadata["debate_titles"] == ["Second reading"]
    assert metadata["speech_part_type"] == "speech"
    assert metadata["talker"] == _talker().model_dump()


def test_build_source_metadata_accepts_mongo_string_enum_fields() -> None:
    """Translate Mongo's raw enum strings without assuming Enum instances."""
    part = replace(
        _speech_part(),
        house="senate",
        chamber="main",
        speech_part_type="speech",
    )

    metadata = build_source_metadata(part, _talker())

    assert metadata["house"] == "senate"
    assert metadata["chamber"] == "main"
    assert metadata["speech_part_type"] == "speech"


def test_merge_chunk_metadata_preserves_existing_fields() -> None:
    """Update document provenance without discarding unrelated chunk metadata."""
    existing = '{"custom":"keep","doc_metadata":{"legacy":"keep","debate_category":"old"}}'

    merged = merge_chunk_metadata(
        existing,
        {"debate_category": "Bills", "bill_ids": ["bill-1"]},
    )

    assert merged == (
        '{"custom": "keep", "doc_metadata": {"legacy": "keep", '
        '"debate_category": "Bills", "bill_ids": ["bill-1"]}}'
    )


def test_backfill_updates_matching_chunks_and_preserves_existing_data(tmp_path: Path) -> None:
    """Update all matching speech chunks without changing text or unrelated metadata."""
    database_path = tmp_path / "chunks.db"
    connection = sqlite3.connect(database_path)
    connection.execute(
        """
        CREATE TABLE document_chunks (
            chunk_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            source_document_id TEXT NOT NULL,
            chunk_text TEXT NOT NULL,
            metadata TEXT
        )
        """,
    )
    connection.executemany(
        """
        INSERT INTO document_chunks (
            chunk_id, source_id, source_document_id, chunk_text, metadata
        ) VALUES (?, ?, ?, ?, ?)
        """,
        [
            ("chunk-1", "hansard", "talker-1_part-1", "speech passage", '{"custom":"keep"}'),
            ("chunk-2", "hansard", "talker-1_part-1", "another passage", None),
            ("unrelated", "other", "talker-1_part-1", "other source", None),
        ],
    )
    connection.commit()
    connection.close()

    updated_documents, updated_chunks, unmatched_parts = _backfill_database(
        database_path,
        [
            replace(
                _speech_part(),
                house="senate",
                chamber="main",
                speech_part_type="speech",
            ),
        ],
        {"talker-1": _talker()},
        dry_run=False,
    )

    assert (updated_documents, updated_chunks, unmatched_parts) == (1, 2, 0)
    connection = sqlite3.connect(database_path)
    rows = connection.execute(
        """
        SELECT chunk_id, chunk_text, published_at, speaker_name, talker_id,
               house, chamber, bill_ids, debate_category, debate_titles,
               speech_part_type, metadata
        FROM document_chunks
        WHERE source_id = 'hansard'
        ORDER BY chunk_id
        """,
    ).fetchall()
    connection.close()

    assert rows[0][:11] == (
        "chunk-1",
        "speech passage",
        "2024-06-01T00:00:00+00:00",
        "Member A",
        "talker-1",
        "senate",
        "main",
        '["bill-1"]',
        "Bills",
        '["Second reading"]',
        "speech",
    )
    assert json.loads(rows[0][11]) == {
        "custom": "keep",
        "doc_metadata": build_source_metadata(_speech_part(), _talker()),
    }
    assert json.loads(rows[1][11])["doc_metadata"]["bill_ids"] == ["bill-1"]
