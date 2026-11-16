"""Main application entry point for the Policy Tracer API, built with FastAPI."""

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv
from fastapi import FastAPI
from pymongo import MongoClient

from ddd_policy_tracer.__DEPRECATED_analysis.entities._catalog.router import (
    router as catalog_router,
)
from ddd_policy_tracer.answering.completion_service import RagCompletionService
from ddd_policy_tracer.answering.metadata_filter_extraction import MetadataFilterExtractor
from ddd_policy_tracer.answering.ollama_generator import (
    OllamaAnswerGenerator,
    OllamaAnswerGeneratorConfig,
)
from ddd_policy_tracer.answering.openai_generator import (
    OpenAIAnswerGenerator,
    OpenAIAnswerGeneratorConfig,
)
from ddd_policy_tracer.answering.service_layer import AnswerGenerator, AnswerSynthesizer
from ddd_policy_tracer.answering.speaker_resolution import TalkerNameResolver
from ddd_policy_tracer.api.completion import router as completion_router
from ddd_policy_tracer.enrichment.embedding import EmbeddingService
from ddd_policy_tracer.ingestion.hansard.talker_repository import MongoTalkerCatalog
from ddd_policy_tracer.paths import DATA_DIR
from ddd_policy_tracer.retrieval.sqlite_retrieval import SQLiteHybridRetriever
from ddd_policy_tracer.utils.db_manager import get_db_async

logging.basicConfig(level=logging.INFO)

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class ConfiguredAnswerProvider(AnswerGenerator, MetadataFilterExtractor, Protocol):
    """Expose answer synthesis and structured filter extraction from one provider."""


def load_app_environment(env_file: Path = ENV_FILE) -> None:
    """Load local app settings without overriding explicitly configured environment values."""
    load_dotenv(dotenv_path=env_file, override=False)


def create_answer_generator() -> ConfiguredAnswerProvider:
    """Build the configured provider for both answer synthesis and filter extraction."""
    provider = os.environ.get("RAG_ANSWER_GENERATOR", "ollama").strip().lower()
    if provider == "ollama":
        return OllamaAnswerGenerator(
            OllamaAnswerGeneratorConfig(
                model=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
                host=os.environ.get("OLLAMA_HOST"),
            ),
        )
    if provider == "openai":
        return OpenAIAnswerGenerator(
            OpenAIAnswerGeneratorConfig(
                model=os.environ.get("OPENAI_MODEL", "gpt-6-luna"),
            ),
        )
    raise ValueError("RAG_ANSWER_GENERATOR must be either 'ollama' or 'openai'")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[dict[str, object], None]:
    """Load shared retrieval and generation resources for the application lifetime."""
    load_app_environment()
    embed_service = EmbeddingService.get_instance()
    database_path = Path(os.environ.get("RAG_DATABASE_PATH", str(DATA_DIR / "new3.db")))
    retriever = SQLiteHybridRetriever(database_path)
    mongo_client: MongoClient[dict[str, Any]] = MongoClient(
        os.environ.get("MONGO_URI", "mongodb://localhost:27017"),
        serverSelectionTimeoutMS=5000,
    )
    answer_provider = create_answer_generator()
    completion_service = RagCompletionService(
        retriever=retriever,
        embedding_service=embed_service,
        synthesizer=AnswerSynthesizer(answer_provider),
        filter_extractor=answer_provider,
        speaker_resolver=TalkerNameResolver(MongoTalkerCatalog(mongo_client)),
    )
    try:
        yield {
            "embed_service": embed_service,
            "rag_completion_service": completion_service,
        }
    finally:
        retriever.close()
        mongo_client.close()
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
