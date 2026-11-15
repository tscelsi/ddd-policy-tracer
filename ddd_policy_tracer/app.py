"""Main application entry point for the Policy Tracer API, built with FastAPI."""

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

from ddd_policy_tracer.__DEPRECATED_analysis.entities._catalog.router import (
    router as catalog_router,
)
from ddd_policy_tracer.answering.completion_service import RagCompletionService
from ddd_policy_tracer.answering.ollama_generator import (
    OllamaAnswerGenerator,
    OllamaAnswerGeneratorConfig,
)
from ddd_policy_tracer.answering.service_layer import AnswerSynthesizer
from ddd_policy_tracer.api.completion import router as completion_router
from ddd_policy_tracer.enrichment.embedding import EmbeddingService
from ddd_policy_tracer.paths import DATA_DIR
from ddd_policy_tracer.retrieval.sqlite_retrieval import SQLiteHybridRetriever
from ddd_policy_tracer.utils.db_manager import get_db_async

logging.basicConfig(level=logging.INFO)

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def load_app_environment(env_file: Path = ENV_FILE) -> None:
    """Load local app settings without overriding explicitly configured environment values."""
    load_dotenv(dotenv_path=env_file, override=False)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[dict[str, object], None]:
    """Load shared retrieval and generation resources for the application lifetime."""
    load_app_environment()
    embed_service = EmbeddingService.get_instance()
    database_path = Path(os.environ.get("RAG_DATABASE_PATH", str(DATA_DIR / "new3.db")))
    retriever = SQLiteHybridRetriever(database_path)
    completion_service = RagCompletionService(
        retriever=retriever,
        embedding_service=embed_service,
        synthesizer=AnswerSynthesizer(
            OllamaAnswerGenerator(
                OllamaAnswerGeneratorConfig(
                    model=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
                    host=os.environ.get("OLLAMA_HOST"),
                ),
            ),
        ),
    )
    try:
        yield {
            "embed_service": embed_service,
            "rag_completion_service": completion_service,
        }
    finally:
        retriever.close()
        conn = await get_db_async()
        await conn.close()


app = FastAPI(
    title="Policy Tracer API",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)


app.include_router(catalog_router)
app.include_router(completion_router, prefix="/api")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=5000)
