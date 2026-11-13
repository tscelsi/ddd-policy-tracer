from typing import Any

from graphrag_sdk import Embedder, LLMInterface
from graphrag_sdk.core.models import LLMResponse
from sentence_transformers import SentenceTransformer


class LocalEmbedder(Embedder):
    def __init__(self):
        self._model_name = "Qwen/Qwen3-Embedding-0.6B"
        self._model = SentenceTransformer(
            self._model_name,
            # model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
            # tokenizer_kwargs={"padding_side": "left"},
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Embed a single text string into a float vector."""
        return self._model.encode(text, prompt_name="query").tolist()

    def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Batch embed multiple texts. Default: sequential fallback."""
        return self._model.encode(texts, prompt_name="document").tolist()


class LocalLLM(LLMInterface):
    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "ollama is required for the Local provider. Install with: uv add ollama",
            )
        response = ollama.chat(
            **{
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        content = response.message.content
        return LLMResponse(content=content)
