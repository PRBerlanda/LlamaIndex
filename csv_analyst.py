"""Ask questions in Portuguese about a CSV file; an LLM writes the pandas code.

Pipeline (same steps as the course's QueryPipeline, written as plain functions
because QueryPipeline was removed from llama-index-core 0.13+):

    question -> prompt 1 (columns + df.head()) -> LLM -> pandas expression
             -> safe evaluation on the DataFrame -> result
             -> prompt 2 (question + expression + result) -> LLM -> final answer
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from llama_index.core import PromptTemplate

INSTRUCTION_STR = (
    "1. Converta a consulta para código Python executável usando Pandas.\n"
    "2. A linha final do código deve ser uma expressão Python que possa ser chamada com a função `eval()`.\n"
    "3. O código deve representar uma solução para a consulta.\n"
    "4. IMPRIMA APENAS A EXPRESSÃO.\n"
    "5. Não coloque a expressão entre aspas.\n"
)

PANDAS_PROMPT = PromptTemplate(
    "Você está trabalhando com um dataframe do pandas em Python chamado `df`.\n"
    "{colunas_detalhes}\n\n"
    "Este é o resultado de `print(df.head())`:\n"
    "{df_str}\n\n"
    "Siga estas instruções:\n"
    "{instruction_str}\n"
    "Consulta: {query_str}\n\n"
    "Expressão:"
)

SYNTHESIS_PROMPT = PromptTemplate(
    "Dada uma pergunta de entrada, atue como analista de dados e elabore uma resposta "
    "a partir dos resultados da consulta.\n"
    "Responda em português do Brasil, de forma natural, sem introduções como 'A resposta é:'.\n"
    "Consulta: {query_str}\n\n"
    "Instruções do Pandas:\n{pandas_instructions}\n\n"
    "Saída do Pandas: {pandas_output}\n\n"
    "Resposta: \n\n"
    "Ao final, exiba o código usado para gerar a resposta, no formato: "
    "O código utilizado foi `{pandas_instructions}`"
)

# Names the generated code may not touch (the LLM output is untrusted code).
_FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "input", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "__import__", "breakpoint", "help",
    "os", "sys", "subprocess", "shutil", "pathlib", "builtins",
}
_SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in ("abs", "all", "any", "bool", "dict", "float", "int", "len", "list",
                 "max", "min", "range", "round", "set", "sorted", "str", "sum", "tuple", "zip")
}


class UnsafeCodeError(ValueError):
    """Raised when the generated code tries something outside pandas analysis."""


@dataclass
class Resultado:
    pergunta: str
    expressao: str
    saida: object
    resposta: str


def descricao_colunas(df: pd.DataFrame) -> str:
    descricao = "\n".join(f"`{col}`: {df[col].dtype}" for col in df.columns)
    return "Aqui estão os detalhes das colunas do dataframe:\n" + descricao


def limpar_codigo(texto: str) -> str:
    """Remove markdown fences / 'Expressão:' labels the LLM sometimes adds."""
    bloco = re.search(r"```(?:python)?\s*(.*?)```", texto, flags=re.S)
    codigo = bloco.group(1) if bloco else texto
    codigo = re.sub(r"^\s*(Expressão|Expression)\s*:\s*", "", codigo.strip(), flags=re.I)
    return codigo.strip().strip("`").strip()


def _validar(arvore: ast.AST) -> None:
    for no in ast.walk(arvore):
        if isinstance(no, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Delete)):
            raise UnsafeCodeError("imports/global statements are not allowed")
        if isinstance(no, ast.Attribute) and no.attr.startswith("_"):
            raise UnsafeCodeError(f"private attribute not allowed: {no.attr}")
        if isinstance(no, ast.Name) and (no.id in _FORBIDDEN_NAMES or no.id.startswith("__")):
            raise UnsafeCodeError(f"name not allowed: {no.id}")


def executar_expressao(codigo: str, df: pd.DataFrame):
    """Run the generated code: every line but the last is exec'd, the last is eval'd.

    This is a guard-rail against obvious misuse, not a full sandbox: only run it
    on your own machine and with data you trust.
    """
    codigo = limpar_codigo(codigo)
    arvore = ast.parse(codigo)
    _validar(arvore)
    if not arvore.body:
        raise ValueError("empty expression")
    contexto = {"__builtins__": _SAFE_BUILTINS, "df": df, "pd": pd, "np": np}
    *corpo, ultima = arvore.body
    if corpo:
        exec(compile(ast.Module(body=corpo, type_ignores=[]), "<llm>", "exec"), contexto)
    if isinstance(ultima, ast.Expr):
        return eval(compile(ast.Expression(ultima.value), "<llm>", "eval"), contexto)
    exec(compile(ast.Module(body=[ultima], type_ignores=[]), "<llm>", "exec"), contexto)
    return None


def consultar(df: pd.DataFrame, pergunta: str, llm) -> Resultado:
    """Full pipeline: question -> pandas expression -> result -> answer."""
    prompt_1 = PANDAS_PROMPT.format(
        colunas_detalhes=descricao_colunas(df),
        df_str=df.head(5),
        instruction_str=INSTRUCTION_STR,
        query_str=pergunta,
    )
    expressao = limpar_codigo(llm.complete(prompt_1).text)
    try:
        saida = executar_expressao(expressao, df)
    except Exception as erro:  # report the error to the LLM instead of crashing the app
        saida = f"Erro ao executar a expressão: {erro}"
    prompt_2 = SYNTHESIS_PROMPT.format(
        query_str=pergunta, pandas_instructions=expressao, pandas_output=saida
    )
    resposta = llm.complete(prompt_2).text.strip()
    return Resultado(pergunta, expressao, saida, resposta)


def _latin1(texto: str) -> str:
    """The core PDF fonts only cover Latin-1; replace anything else (emojis, arrows...)."""
    return (texto.replace("—", "-").replace("–", "-").replace("’", "'")
            .replace("“", '"').replace("”", '"')
            .encode("latin-1", "replace").decode("latin-1"))


def gerar_pdf(historico: list[tuple[str, str]], pasta: str | Path = ".") -> Path | None:
    """Save the question/answer history as a PDF and return its path."""
    from fpdf import FPDF

    if not historico:
        return None
    caminho = Path(pasta) / f"relatorio_perguntas_respostas_{datetime.now():%Y%m%d%H%M%S}.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    for pergunta, resposta in historico:
        pdf.set_font("Helvetica", "B", 14)
        pdf.multi_cell(0, 8, text=_latin1(pergunta), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 12)
        pdf.multi_cell(0, 8, text=_latin1(resposta), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(6)
    pdf.output(str(caminho))
    return caminho
