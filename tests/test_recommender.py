"""Offline tests: pytest tests/

The embedding model is replaced by a keyword-hashing embedding and the LLM by FakeLLM,
so the tests need no download, API key or internet.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import book_recommender as br  # noqa: E402
from fake_llm import FakeLLM, HashEmbedding  # noqa: E402

DF = pd.read_csv(br.CATALOGO)


@pytest.fixture(scope="module")
def index():
    return br.construir_indice(HashEmbedding(), persist_dir=None, collection="teste")


def fake_responder(prompt: str) -> str:
    """Recommend the catalogue titles quoted in the retrieved context, wrapped in ```json."""
    if "GERE APENAS um objeto JSON" in prompt:
        contexto = prompt.split("---------------------")[1]
        titulos = [t for t in DF["nome"] if t in contexto][:3]
        livros = [{"titulo": t, "categoria": "Romances", "justificativa": "Está no contexto"} for t in titulos]
        return "```json\n" + json.dumps({"data": {"livros": livros}}, ensure_ascii=False) + "\n```"
    return "Olá! Recomendo: " + ", ".join(re.findall(r'"titulo": "([^"]+)"', prompt))


def test_catalogue():
    assert DF.shape == (175, 4)
    assert {"nome", "categoria", "sinopse", "preco"} <= set(DF.columns)
    assert len(list(br.ARTIGOS.glob("*.txt"))) == 41


def test_index_contains_catalogue_and_articles(index):
    fontes = {n.metadata.get("file_name") for n in index.docstore.docs.values()} if index.docstore.docs else set()
    nodes = index.as_retriever(similarity_top_k=200).retrieve("livros")
    fontes |= {n.node.metadata.get("file_name") for n in nodes}
    assert "livros_mais_vendidos.csv" in fontes
    assert any(f and f.startswith("artigo_") for f in fontes)


def test_romance_query_retrieves_romance_article(index):
    # keyword embedding used only in tests: the real E5 model also understands "Quais livros de romance?"
    nodes = index.as_retriever(similarity_top_k=3).retrieve("romances de amor")
    assert any("artigo_Romances" in n.node.metadata.get("file_name", "") for n in nodes)


def test_full_pipeline_only_recommends_catalogue_books(index):
    llm = FakeLLM(responder=fake_responder, prompts=[])
    engine = br.RAGStringQueryEngine(retriever=index.as_retriever(similarity_top_k=3), llm=llm)
    resposta = br.pipeline_completo("romances de amor", engine, llm, DF)
    assert resposta.startswith("Olá! Recomendo:")
    recomendados = re.findall(r'"titulo": "([^"]+)"', llm.prompts[-1])
    assert recomendados and set(recomendados) <= set(DF["nome"])


def test_extract_json_with_fences_and_text():
    assert br.extrair_json('Claro!\n```json\n{"data": {"livros": []}}\n```') == {"data": {"livros": []}}
    assert br.extrair_json('Aqui: {"data": {"livros": []}} fim') == {"data": {"livros": []}}
    with pytest.raises(ValueError):
        br.extrair_json("sem json")


def test_validation_checks_every_title():
    # The course version only checked the LAST title of the list.
    falso_primeiro = {"data": {"livros": [{"titulo": "Não existe"}, {"titulo": "TUDO É RIO"}]}}
    with pytest.raises(br.LivroNaoEncontrado):
        br.validando(falso_primeiro, DF)


def test_validation_handles_parentheses_and_case():
    # str.contains() treated "(EDIÇÃO ...)" as a regex group; exact, case-insensitive match does not.
    ok = {"data": {"livros": [{"titulo": "é assim que começa (edição de colecionador)"}]}}
    assert br.validando(ok, DF)


def test_persisted_index_is_reused(tmp_path):
    # The course notebook called load_index_from_storage(), which needs a persisted docstore
    # that Chroma does not create; the index is now reopened from the vector store.
    primeiro = br.construir_indice(HashEmbedding(), persist_dir=tmp_path, collection="persistido")
    total = len(primeiro.as_retriever(similarity_top_k=500).retrieve("livros"))
    segundo = br.construir_indice(HashEmbedding(), persist_dir=tmp_path, collection="persistido")
    assert len(segundo.as_retriever(similarity_top_k=500).retrieve("livros")) == total


def test_app_builds():
    import app

    assert app.app is not None
    assert app.recomendar("") == ""
