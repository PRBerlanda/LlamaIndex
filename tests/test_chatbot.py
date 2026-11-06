"""Offline tests: pytest tests/  (keyword embedding + FakeLLM, no download or API key)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import serenatto_chatbot as sc  # noqa: E402
from fake_llm import FakeLLM, HashEmbedding  # noqa: E402

GRAOS = ["Bourbon", "Catuaí", "Geisha", "Yirgacheffe"]


@pytest.fixture(scope="module")
def index():
    return sc.construir_indice(HashEmbedding(), persist_dir=None, collection="teste")


def test_chunks_are_clean_text():
    chunks = sc.carregar_chunks()
    assert len(chunks) == 10
    texto = " ".join(c.get_content() for c in chunks)
    assert "\n" not in texto and "  " not in texto
    assert all(g in texto for g in GRAOS)


def test_retrieval_finds_the_right_coffee(index):
    nos = sc.trechos_relevantes(index, "Quero saber mais detalhes sobre o Geisha", top_k=1)
    assert "Geisha" in nos[0].node.get_content()


def test_chat_engine_sends_context_and_keeps_history(index):
    def responder(prompt: str) -> str:
        return "Temos o Geisha!" if "Geisha" in prompt else "Não sei."

    llm = FakeLLM(responder=responder, prompts=[])
    engine = sc.criar_chat_engine(index, llm)
    assert engine.chat("Vocês vendem o café Geisha?").response == "Temos o Geisha!"
    assert any(sc.SYSTEM_PROMPT[:40] in p for p in llm.prompts)  # system prompt reached the LLM
    assert len(engine.chat_history) == 2
    engine.reset()
    assert engine.chat_history == []


def test_gradio_handlers(index):
    import app

    app._estado["engine"] = sc.criar_chat_engine(index, FakeLLM(responder=lambda p: "Olá!", prompts=[]))
    texto, historico = app.converse_com_bot("Oi", None)
    assert texto == "" and historico == [
        {"role": "user", "content": "Oi"},
        {"role": "assistant", "content": "Olá!"},
    ]
    assert app.resetar_chat() == []
