"""The Groq client is configured from environment variables only."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import llm_config  # noqa: E402


def test_model_and_key_from_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    llm = llm_config.get_llm()
    assert llm.model == llm_config.DEFAULT_MODEL == "openai/gpt-oss-20b"
    monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
    assert llm_config.get_llm().model == "llama-3.1-8b-instant"


def test_colab_secret_name_is_accepted(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API", "colab-key")
    assert llm_config.groq_api_key() == "colab-key"


def test_missing_key_has_clear_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        llm_config.groq_api_key()
