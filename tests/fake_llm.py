"""Offline stand-in for the Groq LLM, so the tests need no API key or internet."""
from typing import Any, Callable

from llama_index.core.llms import CompletionResponse, CompletionResponseGen, CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback


class FakeLLM(CustomLLM):
    """Answers with `responder(prompt)` and records every prompt it receives."""

    responder: Callable[[str], str]
    prompts: list = []

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name="fake", context_window=32000, num_output=1024)

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        self.prompts.append(prompt)
        return CompletionResponse(text=self.responder(prompt))

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        yield self.complete(prompt)


# --- offline embedding -------------------------------------------------------
import hashlib  # noqa: E402
import re  # noqa: E402
import unicodedata  # noqa: E402

from llama_index.core.base.embeddings.base import BaseEmbedding  # noqa: E402

DIM = 512


def _tokens(texto: str):
    texto = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()
    return re.findall(r"[a-z]{3,}", texto)


def _vector(texto: str) -> list[float]:
    v = [0.0] * DIM
    for tok in _tokens(texto):
        v[int(hashlib.md5(tok[:6].encode()).hexdigest(), 16) % DIM] += 1.0
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


class HashEmbedding(BaseEmbedding):
    """Bag-of-words hashing: crude, but retrieval by keywords works without downloading a model."""

    def _get_query_embedding(self, query: str) -> list[float]:
        return _vector(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return _vector(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return _vector(text)
