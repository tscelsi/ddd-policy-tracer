"""Verify Hansard metadata survives conversion and chunking."""

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

from ddd_policy_tracer.enrichment.chunking.chunking import MarkdownChunker
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
from ddd_policy_tracer.utils.time_helpers import utc_isoformat


class _SpeechPartRepository:
    """Provide one Hansard speech part for conversion tests."""

    async def list_all_speech_parts(self) -> list[SpeechPart]:
        """Return the representative speech part."""
        return [
            SpeechPart(
                id="part-1",
                date=datetime(2024, 6, 1, tzinfo=UTC),
                bill_ids=["bill-1", "bill-2"],
                house=HouseType.SENATE,
                chamber=ChamberType.HOR_MAIN,
                type=PartType.SPEECH,
                debate_category="Bills",
                debate_seq=1,
                subdebate_1_title="Second reading",
                subdebate_1_info=None,
                subdebate_1_seq=1,
                subdebate_2_title="Amendments",
                subdebate_2_info=None,
                subdebate_2_seq=1,
                speech_id="speech-1",
                speech_seq=1,
                part_seq=1,
                talker_id="talker-1",
                speech_content="A parliamentary speech.",
                speech_part_type=SpeechPartType.SPEECH,
            ),
        ]


class _TalkerRepository:
    """Provide one Hansard speaker for conversion tests."""

    async def list_talkers(self) -> list[Talker]:
        """Return the representative speaker."""
        return [
            Talker(
                id="talker-1",
                name="Member A",
                first_name="Member",
                last_name="A",
                electorate="Example Electorate",
                party="Example Party",
            ),
        ]


async def test_hansard_fields_survive_ingestion_and_chunking(tmp_path: Path) -> None:
    """Carry speech-part and speaker details into first-class chunk metadata."""
    repository = FilesystemSourceDocumentRepository(tmp_path / "source_documents.jsonl")

    await convert_speeches_to_documents(
        speech_part_repo=_SpeechPartRepository(),
        talker_repo=_TalkerRepository(),
        doc_repo=repository,
    )
    document = repository.list_versions(source_id="hansard")[0]
    chunk = MarkdownChunker().chunk_document(doc=document)[0]

    assert document.metadata is not None
    assert document.metadata["talker_id"] == "talker-1"
    assert document.metadata["house"] == "senate"
    assert document.metadata["chamber"] == "main"
    assert document.metadata["bill_ids"] == ["bill-1", "bill-2"]
    assert document.metadata["debate_category"] == "Bills"
    assert document.metadata["debate_titles"] == ["Second reading", "Amendments"]
    assert document.metadata["speech_part_type"] == "speech"
    assert chunk.published_at == "2024-06-01T00:00:00+00:00"
    assert chunk.speaker_name == "Member A"
    assert chunk.talker_id == "talker-1"
    assert chunk.party == "Example Party"
    assert chunk.electorate == "Example Electorate"
    assert chunk.bill_ids == ["bill-1", "bill-2"]
    assert chunk.debate_titles == ["Second reading", "Amendments"]


def test_utc_isoformat_normalizes_aware_and_naive_mongo_dates() -> None:
    """Keep local-offset and Mongo-naive timestamps comparable to UTC range bounds."""
    local_timestamp = datetime(2024, 6, 1, 10, tzinfo=timezone(timedelta(hours=10)))
    mongo_timestamp = datetime(2024, 6, 1, 0)

    assert utc_isoformat(local_timestamp) == "2024-06-01T00:00:00+00:00"
    assert utc_isoformat(mongo_timestamp) == "2024-06-01T00:00:00+00:00"
