"""Main application entry point for the Policy Tracer API, built with FastAPI."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ddd_policy_tracer.__DEPRECATED_analysis.entities._catalog.router import router as catalog_router
from ddd_policy_tracer.enrichment.embedding import EmbeddingService
from ddd_policy_tracer.utils.db_manager import get_db_async

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Async context manager for application lifespan events (startup and shutdown)."""
    embed_service = EmbeddingService.get_instance()
    yield {"embed_service": embed_service}
    # Place any shutdown/cleanup logic here
    conn = await get_db_async()
    await conn.close()


app = FastAPI(
    title="Policy Tracer API",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)


app.include_router(catalog_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=5000)
