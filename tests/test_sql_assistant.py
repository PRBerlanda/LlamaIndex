"""Offline tests: pytest tests/

FakeLLM returns the SQL a good model would write for each course question; the result
is compared with an answer key computed directly with sqlite3.
"""
import re
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

import sql_assistant as sa  # noqa: E402
from fake_llm import FakeLLM, HashEmbedding  # noqa: E402

CON = sqlite3.connect(sa.DB_PATH)

CASOS = {
    "Quais estados mais frequentes na tabela Clientes?": (
        "Clientes",
        "```sql\nSELECT Estado, COUNT(*) AS Frequencia FROM Clientes GROUP BY Estado "
        "ORDER BY Frequencia DESC LIMIT 5\n```\nResultadoSQL: ...",
        CON.execute("SELECT Estado, COUNT(*) c FROM Clientes GROUP BY Estado ORDER BY c DESC LIMIT 5").fetchall(),
    ),
    "Quais as fornecedoras localizadas na cidade de São Paulo?": (
        "Fornecedores",
        "SELECT Nome FROM Fornecedores WHERE Cidade = 'sao paulo';",
        CON.execute("SELECT Nome FROM Fornecedores WHERE Cidade = 'sao paulo'").fetchall(),
    ),
    "Quantos funcionários com o cargo de vendedores temos cadastrados?": (
        "Funcionarios",
        "SELECT COUNT(*) FROM Funcionarios WHERE Cargo = 'vendas'",
        CON.execute("SELECT COUNT(*) FROM Funcionarios WHERE Cargo = 'vendas'").fetchall(),
    ),
}


def responder_fake(prompt: str) -> str:
    if "ConsultaSQL:" in prompt and "Pergunta:" in prompt:
        for pergunta, (_, sql, _) in CASOS.items():
            if f"Pergunta: {pergunta}" in prompt:
                return sql
    return "Resposta: " + prompt.split("Resultado SQL:", 1)[-1].strip()


@pytest.fixture(scope="module")
def assistente():
    llm = FakeLLM(responder=responder_fake, prompts=[])
    sql_database, retriever = sa.montar_assistente(llm, HashEmbedding())
    return llm, sql_database, retriever


def test_database_tables():
    _, metadata = sa.conectar()
    assert set(metadata.tables) == {"Clientes", "Fornecedores", "Funcionarios"}
    assert CON.execute("SELECT COUNT(*) FROM Clientes").fetchone()[0] == 100


def test_cached_descriptions_cover_every_table():
    import json

    descricoes = json.loads(sa.DESCRICOES_PATH.read_text(encoding="utf-8"))
    assert set(descricoes) == {"Clientes", "Fornecedores", "Funcionarios"}


@pytest.mark.parametrize("pergunta", list(CASOS))
def test_question_to_sql_to_answer(assistente, pergunta):
    llm, sql_database, retriever = assistente
    tabela, _, esperado = CASOS[pergunta]
    r = sa.responder(pergunta, llm, sql_database, retriever)
    assert not r.sql.lower().startswith("sql") and not r.sql.endswith(";")
    assert eval(r.resultado) == esperado  # SQLDatabase returns the rows as a Python literal
    assert r.resposta.startswith("Resposta:")


def test_answer_keys():
    # values match the course notebook: 3 sellers and 2 suppliers in Sao Paulo
    assert CASOS["Quantos funcionários com o cargo de vendedores temos cadastrados?"][2] == [(3,)]
    assert len(CASOS["Quais as fornecedoras localizadas na cidade de São Paulo?"][2]) == 2


def test_schema_prompt_contains_table_and_description(assistente):
    llm, sql_database, retriever = assistente
    sa.responder("Quantos funcionários com o cargo de vendedores temos cadastrados?", llm, sql_database, retriever)
    prompt_1 = [p for p in llm.prompts if "ConsultaSQL:" in p][-1]
    assert re.search(r"Table 'Funcionarios' has columns", prompt_1)
    assert "A descrição da tabela é:" in prompt_1


@pytest.mark.parametrize("sql", [
    "DROP TABLE Clientes",
    "DELETE FROM Clientes",
    "SELECT 1; DROP TABLE Clientes",
    "UPDATE Clientes SET Nome = 'x'",
])
def test_write_queries_are_blocked(sql):
    with pytest.raises(sa.SQLNaoPermitido):
        sa.validar_sql(sql)


def test_blocked_query_does_not_touch_db(assistente):
    _, sql_database, retriever = assistente
    llm = FakeLLM(responder=lambda p: "DELETE FROM Clientes" if "ConsultaSQL:" in p else "ok", prompts=[])
    r = sa.responder("apague tudo", llm, sql_database, retriever)
    assert r.resultado.startswith("Erro ao executar a consulta")
    assert CON.execute("SELECT COUNT(*) FROM Clientes").fetchone()[0] == 100


def test_gradio_handler(assistente):
    import app

    llm, sql_database, retriever = assistente
    app._estado.update(llm=llm, sql_database=sql_database, retriever=retriever)
    texto, historico = app.atualizar_historico(
        "Quantos funcionários com o cargo de vendedores temos cadastrados?", None)
    assert texto == "" and historico[0]["role"] == "user" and historico[1]["role"] == "assistant"
    assert "3" in historico[1]["content"]
