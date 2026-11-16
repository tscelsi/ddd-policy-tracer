"""Tests for persisted Hansard metadata used by parliamentary retrieval filters."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

from ddd_policy_tracer.ingestion.adapters import FilesystemSourceDocumentRepository
from ddd_policy_tracer.ingestion.hansard.part_repository import (
    ChamberType,
    HouseType,
    PartType,
    SpeechPart,
    SpeechPartType,
)
from ddd_policy_tracer.ingestion.hansard.service_layer import convert_speeches_to_documents
from ddd_policy_tracer.ingestion.hansard.talker_repository import Talker


class StubSpeechPartRepository:
    """Provide one parliamentary speech part for metadata conversion."""

    async def list_all_speech_parts(self) -> list[SpeechPart]:
        """Return the representative Hansard speech part."""
        return [
            SpeechPart(
                id="2025-02-12_hor_main_1_0_0_1_0",
                date=datetime(2025, 2, 12, 1, 0, tzinfo=timezone(timedelta(hours=11))),
                bill_ids=None,
                house=HouseType.HOR,
                chamber=ChamberType.HOR_MAIN,
                type=PartType.SPEECH,
                debate_category="bill",
                debate_seq=1,
                subdebate_1_title="A Bill",
                subdebate_1_info=None,
                subdebate_1_seq=None,
                subdebate_2_title=None,
                subdebate_2_info=None,
                subdebate_2_seq=None,
                speech_id="speech-1",
                speech_seq=1,
                part_seq=1,
                talker_id="haines-id",
                speech_content="A parliamentary speech.",
                speech_part_type=SpeechPartType.SPEECH,
            ),
        ]


class StubTalkerRepository:
    """Provide one catalog speaker for document conversion."""

    async def list_talkers(self) -> list[Talker]:
        """Return the representative Hansard speaker."""
        return [
            Talker(
                id="haines-id",
                name="Haines, Helen MP",
                first_name="Helen",
                last_name="Haines",
                electorate="Indi",
                party="Independent",
            ),
        ]


async def test_hansard_conversion_persists_utc_date_and_talker_identity(
    tmp_path: Path,
) -> None:
    """Persist UTC publication instants and stable talker IDs in chunkable metadata."""
    repository = FilesystemSourceDocumentRepository(tmp_path / "source-documents.jsonl")

    await convert_speeches_to_documents(
        StubSpeechPartRepository(),
        StubTalkerRepository(),
        repository,
    )

    document = repository.list_versions(source_id="hansard")[0]
    assert document.published_at == datetime(2025, 2, 11, 14, tzinfo=UTC).isoformat()
    assert document.metadata["talker_id"] == "haines-id"
    assert document.metadata["published_at"] == document.published_at
