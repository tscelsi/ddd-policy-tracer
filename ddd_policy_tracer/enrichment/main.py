"""Enrichment pipeline entry point."""

import logging

from ddd_policy_tracer.enrichment.chunking.adapters import SQLiteDocumentChunkRepository
from ddd_policy_tracer.enrichment.embedding import EmbeddingService
from ddd_policy_tracer.enrichment.service_layer import (
    run_full_pipeline_on_chunks,
)
from ddd_policy_tracer.utils.db_manager import DbManager

logging.basicConfig(level=logging.INFO)


async def main() -> None:
    """Run the enrichment pipeline."""
    chunk_repo = SQLiteDocumentChunkRepository(
        db_manager=DbManager.get_instance("data/new2.db"),
    )
    await chunk_repo.init()
    chunks = await chunk_repo.list_chunks(source_id="committees")
    await run_full_pipeline_on_chunks(
        chunks=chunks,
        chunk_repo=chunk_repo,
        embedding_service=EmbeddingService(),
        batch_size=104,
    )
    db = await DbManager.get_instance("data/new2.db").get_db()
    await db.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
