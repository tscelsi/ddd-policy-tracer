import asyncio

from graphrag_sdk import ConnectionConfig, GraphRAG

from ddd_policy_tracer.graph_rag.embedder import LocalEmbedder, LocalLLM
from ddd_policy_tracer.paths import DATA_DIR
import logging

logging.basicConfig(level=logging.INFO)

async def main():
    async with GraphRAG(
        connection=ConnectionConfig(
            host="localhost",
            graph_name="my_graph",
        ),  # graph_name = per-tenant isolation
        llm=LocalLLM(model_name="hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF:latest"),
        embedder=LocalEmbedder(),
        embedding_dimension=1024,
    ) as rag:
        # result = await rag.ingest([str(f) for f in (DATA_DIR / "committee_artifacts").iterdir()])
        # print(f"Nodes: {result.nodes_created}, Edges: {result.relationships_created}")

        # Finalize: deduplicate entities, backfill embeddings, create indexes
        await rag.finalize()

        # Full RAG: retrieve + generate
        # answer = await rag.completion("Where does Alice work?")
        # print(answer.answer)


if __name__ == "__main__":
    asyncio.run(main())
