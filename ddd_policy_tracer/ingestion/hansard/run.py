"""Main execution function for converting Hansard speech parts into source documents."""

from typing import Any

from pymongo import AsyncMongoClient

from ddd_policy_tracer.ingestion.adapters import FilesystemSourceDocumentRepository
from ddd_policy_tracer.ingestion.hansard.part_repository import MongoPartRepository
from ddd_policy_tracer.ingestion.hansard.service_layer import convert_speeches_to_documents
from ddd_policy_tracer.ingestion.hansard.talker_repository import MongoTalkerRepository
from ddd_policy_tracer.paths import DATA_DIR


async def main() -> None:
    """Main execution function to convert Hansard speech parts into source documents."""
    client = AsyncMongoClient[Any]("mongodb://localhost:27017")
    speech_part_repo = MongoPartRepository(client, "tmp")
    talker_repo = MongoTalkerRepository(client, "tmp")
    doc_repo = FilesystemSourceDocumentRepository(DATA_DIR / "documents_hansard.jsonl")
    await convert_speeches_to_documents(
        speech_part_repo=speech_part_repo,
        talker_repo=talker_repo,
        doc_repo=doc_repo,
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
