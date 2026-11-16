"""Service layer for converting Hansard speech parts and talker info into source documents."""

import logging

from ddd_policy_tracer.ingestion.adapters import FilesystemSourceDocumentRepository
from ddd_policy_tracer.ingestion.domain import SourceDocument, compute_checksum
from ddd_policy_tracer.ingestion.hansard.part_repository import AbstractPartRepository
from ddd_policy_tracer.ingestion.hansard.talker_repository import AbstractTalkerRepository
from ddd_policy_tracer.utils.time_helpers import utc_isoformat, utc_now_isoformat

logger = logging.getLogger(__name__)


async def convert_speeches_to_documents(
    speech_part_repo: AbstractPartRepository,
    talker_repo: AbstractTalkerRepository,
    doc_repo: FilesystemSourceDocumentRepository,
) -> None:
    """Convert speech parts and associated talker info into source documents and persist them."""
    speech_parts = await speech_part_repo.list_all_speech_parts()
    talkers = await talker_repo.list_talkers()
    talker_map = {talker.id: talker for talker in talkers if talker.id is not None}
    processed_documents = 0
    ingested_documents = 0
    failed_documents = 0
    now = utc_now_isoformat()
    for part in speech_parts:
        talker = talker_map.get(part.talker_id)
        if talker is None:
            logger.warning(f"Skipping speech part {part.id} with missing talker {part.talker_id}")
            continue
        document = SourceDocument(
            source_id="hansard",
            source_document_id=f"{talker.id}_{part.id}",
            source_url=None,
            published_at=utc_isoformat(part.date),
            retrieved_at=now,
            checksum=compute_checksum(part.speech_content.encode("utf-8")),
            normalized_text=part.speech_content,
            raw_content_ref=None,
            content_type="text/plain",
            created_at=now,
            updated_at=now,
            metadata={
                "talker": talker.model_dump(),
                "talker_id": part.talker_id,
                "house": part.house.value,
                "chamber": part.chamber.value,
                "bill_ids": part.bill_ids,
                "debate_category": part.debate_category,
                "debate_titles": [
                    title for title in (part.subdebate_1_title, part.subdebate_2_title) if title
                ],
                "speech_part_type": part.speech_part_type.value,
            },
        )
        processed_documents += 1
        try:
            doc_repo.add_version(document)
            ingested_documents += 1
        except Exception as e:
            logger.error(f"Failed to save document for speech part {part.id}: {e}")
            failed_documents += 1
    logger.info(
        f"""Finished converting speeches to documents:
        processed={processed_documents},
        ingested={ingested_documents},
        failed={failed_documents}""",
    )
