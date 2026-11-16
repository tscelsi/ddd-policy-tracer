"""Verify typed chunk metadata persistence and legacy schema migration."""

from pathlib import Path

from ddd_policy_tracer.enrichment.chunking.adapters import SQLiteDocumentChunkRepository
from ddd_policy_tracer.enrichment.chunking.models import DocumentChunk
from ddd_policy_tracer.utils.db_manager import DbManager


async def test_chunk_metadata_migrates_and_round_trips(tmp_path: Path) -> None:
    """Add typed fields to an existing chunk schema and preserve their values."""
    manager = DbManager(str(tmp_path / "chunks.db"))
    async with manager.with_transaction() as db:
        await db.execute(
            """
            CREATE TABLE document_chunks (
                chunk_id TEXT PRIMARY KEY NOT NULL,
                source_id TEXT NOT NULL,
                source_document_id TEXT NOT NULL,
                document_checksum TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                topics TEXT,
                metadata TEXT
            )
            """,
        )
        await db.execute(
            """
            INSERT INTO document_chunks (
                chunk_id, source_id, source_document_id, document_checksum,
                chunk_index, start_char, end_char, chunk_text
            ) VALUES ('legacy', 'hansard', 'old-speech', 'checksum', 0, 0, 5, 'legacy')
            """,
        )

    repository = SQLiteDocumentChunkRepository(manager)
    await repository.init()
    legacy_chunk = await repository.get_chunk(chunk_id="legacy")
    assert legacy_chunk is not None
    assert legacy_chunk.published_at is None

    chunk = DocumentChunk(
        chunk_id="new",
        source_id="hansard",
        source_document_id="speech-1",
        document_checksum="checksum",
        chunk_index=0,
        start_char=0,
        end_char=5,
        chunk_text="speech",
        published_at="2024-06-01",
        speaker_name="Member A",
        talker_id="talker-1",
        party="Example Party",
        electorate="Example Electorate",
        house="senate",
        chamber="main",
        bill_ids=["bill-1"],
        debate_category="Bills",
        debate_titles=["Second reading"],
        speech_part_type="speech",
    )
    await repository.save(chunk)

    saved = await repository.get_chunk(chunk_id="new")
    listed = await repository.list_chunks(source_id="hansard")
    without_embedding = await repository.list_chunks_without_embedding(source_id="hansard")

    assert saved == chunk
    assert chunk in listed
    assert chunk in without_embedding
    connection = await manager.get_db()
    await connection.close()
