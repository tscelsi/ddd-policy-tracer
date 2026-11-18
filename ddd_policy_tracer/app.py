"""Main application entry point for the Policy Tracer API, built with FastAPI."""

import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from pymongo import AsyncMongoClient

from ddd_policy_tracer.answering.completion_service import RagCompletionService
from ddd_policy_tracer.answering.metadata_filter_extractor import (
    MetadataFilterExtractorConfig,
    OllamaMetadataFilterExtractor,
    OpenAIMetadataFilterExtractor,
)
from ddd_policy_tracer.answering.ollama_generator import (
    OllamaAnswerGenerator,
    OllamaAnswerGeneratorConfig,
)
from ddd_policy_tracer.answering.openai_generator import (
    OpenAIAnswerGenerator,
    OpenAIAnswerGeneratorConfig,
)
from ddd_policy_tracer.answering.query_variants import (
    OllamaQueryVariantDrafter,
    OpenAIQueryVariantDrafter,
    QueryVariantDrafter,
)
from ddd_policy_tracer.answering.service_layer import AnswerGenerator, AnswerSynthesizer
from ddd_policy_tracer.api.completion import router as completion_router
from ddd_policy_tracer.enrichment.chunking.adapters import SQLiteDocumentChunkRepository
from ddd_policy_tracer.enrichment.embedding import EmbeddingService
from ddd_policy_tracer.ingestion.hansard.talker_repository import MongoTalkerRepository
from ddd_policy_tracer.paths import DATA_DIR
from ddd_policy_tracer.retrieval.speaker_resolution import MongoTalkerNameResolver
from ddd_policy_tracer.retrieval.sqlite_retrieval import SQLiteHybridRetriever
from ddd_policy_tracer.utils.db_manager import DbManager, get_db_async

logging.basicConfig(level=logging.INFO)

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def load_app_environment(env_file: Path = ENV_FILE) -> None:
    """Load local app settings without overriding explicitly configured environment values."""
    load_dotenv(dotenv_path=env_file, override=False)


def create_answer_generator() -> AnswerGenerator:
    """Build the configured local or hosted answer-generation adapter."""
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


def create_query_variant_drafter() -> QueryVariantDrafter:
    """Build the query-review drafter with the application's configured provider and model."""
    provider = os.environ.get("RAG_ANSWER_GENERATOR", "ollama").strip().lower()
    if provider == "ollama":
        return OllamaQueryVariantDrafter(
            model=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
            host=os.environ.get("OLLAMA_HOST"),
        )
    if provider == "openai":
        return OpenAIQueryVariantDrafter(
            model=os.environ.get("OPENAI_MODEL", "gpt-6-luna"),
        )
    raise ValueError("RAG_ANSWER_GENERATOR must be either 'ollama' or 'openai'")


def create_metadata_filter_extractor() -> (
    OllamaMetadataFilterExtractor | OpenAIMetadataFilterExtractor
):
    """Build the metadata extractor with the configured answer-generation provider."""
    provider = os.environ.get("RAG_ANSWER_GENERATOR", "ollama").strip().lower()
    if provider == "ollama":
        config = MetadataFilterExtractorConfig(
            model=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
            host=os.environ.get("OLLAMA_HOST"),
        )
        return OllamaMetadataFilterExtractor(config)
    if provider == "openai":
        config = MetadataFilterExtractorConfig(
            model=os.environ.get("OPENAI_MODEL", "gpt-6-luna"),
            temperature=1.0,
        )
        return OpenAIMetadataFilterExtractor(config)
    raise ValueError("RAG_ANSWER_GENERATOR must be either 'ollama' or 'openai'")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[dict[str, object], None]:
    """Load shared retrieval and generation resources for the application lifetime."""
    load_app_environment()
    embed_service = EmbeddingService.get_instance()
    database_path = Path(os.environ.get("RAG_DATABASE_PATH", str(DATA_DIR / "new4.db")))
    database_manager = DbManager(db_path=str(database_path))
    database_connection = None
    retriever = None
    mongo_client = None
    app_started = False
    try:
        if await asyncio.to_thread(database_path.is_file):
            database_connection = await database_manager.get_db()
            await SQLiteDocumentChunkRepository(database_manager).init()
        retriever = SQLiteHybridRetriever(database_path)
        mongo_client = AsyncMongoClient(
            os.environ.get("MONGODB_URI", "mongodb://localhost:27017"),
        )
        talker_repository = MongoTalkerRepository(
            mongo_client,
            os.environ.get("HANSARD_MONGO_DATABASE", "hansard"),
        )
        completion_service = RagCompletionService(
            retriever=retriever,
            embedding_service=embed_service,
            synthesizer=AnswerSynthesizer(create_answer_generator()),
            metadata_filter_extractor=create_metadata_filter_extractor(),
            speaker_name_resolver=MongoTalkerNameResolver(talker_repository),
        )
        app_started = True
        yield {
            "embed_service": embed_service,
            "rag_completion_service": completion_service,
        }
    finally:
        if retriever is not None:
            retriever.close()
        if mongo_client is not None:
            await mongo_client.close()
        if database_connection is not None:
            await database_connection.close()
        if app_started:
            conn = await get_db_async()
            await conn.close()


app = FastAPI(
    title="Policy Tracer API",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)


app.include_router(completion_router, prefix="/api")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=5000)
