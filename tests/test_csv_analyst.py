"""Offline tests: pytest tests/

The LLM is replaced by FakeLLM, which returns the pandas expression a good model
would write. Each result is checked against an answer key computed directly with pandas.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from csv_analyst import UnsafeCodeError, consultar, executar_expressao, gerar_pdf, limpar_codigo  # noqa: E402
from fake_llm import FakeLLM  # noqa: E402

DF = pd.read_csv(ROOT / "data" / "vendas.csv")

CASOS = {
    "Qual é a forma de pagamento mais utilizada pelos clientes?":
        ("df['forma_pagamento'].value_counts().idxmax()", DF["forma_pagamento"].mode()[0]),
    "Qual é a avaliação média de cada filial?":
        ("df.groupby('filial')['avaliacao'].mean().round(2)", DF.groupby("filial")["avaliacao"].mean().round(2)),
    "Qual é a média gasta por cada tipo de cliente?":
        ("```python\ndf.groupby('tipo_cliente')['total'].mean()\n```", DF.groupby("tipo_cliente")["total"].mean()),
}


def responder(prompt: str) -> str:
    if "Expressão:" in prompt:  # prompt 1 -> pandas code
        for pergunta, (codigo, _) in CASOS.items():
            if pergunta in prompt:
                return codigo
        return "df.shape[0]"
    return "Resposta sintetizada."  # prompt 2 -> final text


def test_dataset_shape():
    assert DF.shape == (1000, 14)
    assert not DF.duplicated().any()


@pytest.mark.parametrize("pergunta", list(CASOS))
def test_pipeline_matches_answer_key(pergunta):
    llm = FakeLLM(responder=responder, prompts=[])
    resultado = consultar(DF, pergunta, llm)
    esperado = CASOS[pergunta][1]
    if isinstance(esperado, pd.Series):
        pd.testing.assert_series_equal(resultado.saida, esperado)
    else:
        assert resultado.saida == esperado
    assert resultado.resposta == "Resposta sintetizada."
    # prompt 1 carries the column types; prompt 2 carries the expression and its output
    assert "`forma_pagamento`: object" in llm.prompts[0]
    assert resultado.expressao in llm.prompts[1]


def test_payment_answer_key():
    # Same numbers shown in the course notebook: Carteira Digital 345, Dinheiro 344
    assert DF["forma_pagamento"].value_counts().to_dict()["Carteira Digital"] == 345


def test_clean_code_fences():
    assert limpar_codigo("```python\ndf.head()\n```") == "df.head()"
    assert limpar_codigo("Expressão: df.shape") == "df.shape"


def test_multiline_code():
    assert executar_expressao("x = df['total'].sum()\nround(x, 2)", DF) == round(DF["total"].sum(), 2)


@pytest.mark.parametrize("codigo", [
    "__import__('os').system('echo hi')",
    "import os\nos.listdir('.')",
    "open('/etc/passwd').read()",
    "df.__class__.__bases__",
])
def test_unsafe_code_is_blocked(codigo):
    with pytest.raises(UnsafeCodeError):
        executar_expressao(codigo, DF)


def test_execution_error_goes_to_answer_not_crash():
    llm = FakeLLM(responder=lambda p: "df['coluna_inexistente'].sum()" if "Expressão:" in p else "ok", prompts=[])
    resultado = consultar(DF, "pergunta qualquer", llm)
    assert str(resultado.saida).startswith("Erro ao executar a expressão")


def test_pdf_handles_non_latin_text(tmp_path):
    caminho = gerar_pdf([("Qual a média?", "A média é 7,03 ✅ — ótimo → ok")], tmp_path)
    assert caminho.exists() and caminho.read_bytes().startswith(b"%PDF")
    assert gerar_pdf([], tmp_path) is None


def test_app_builds():
    import app

    assert app.add_historico("p", "r", None) == [("p", "r")]
    assert app.add_historico("", "", [("a", "b")]) == [("a", "b")]
    assert app.app is not None
