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
