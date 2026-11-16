"""Script to visualise chunk texts for a given source_id."""

import logging
import textwrap

from ddd_policy_tracer.utils.db_manager import DbManager

TEXT_WIDTH = 80


async def visualise_chunk_sample(source_id: str | None = None) -> None:
    """Print a sample of chunk texts for visual inspection."""
    from ddd_policy_tracer.enrichment.chunking.adapters import SQLiteDocumentChunkRepository

    repository = SQLiteDocumentChunkRepository(db_manager=DbManager.get_instance())
    await repository.init()
    chunks = await repository.list_chunks(source_id=source_id)
    for chunk in chunks:
        logging.info("\n".join(textwrap.wrap(chunk.chunk_text, TEXT_WIDTH, break_long_words=False)))
        logging.info("=" * TEXT_WIDTH)
        logging.info("=" * TEXT_WIDTH)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualise chunk texts for a given source_id.")
    parser.add_argument(
        "--source_id",
        type=str,
        help="Filter chunks by source_id (optional)",
        default=None,
    )
    args = parser.parse_args()

    import asyncio

    asyncio.run(visualise_chunk_sample(source_id=args.source_id))
