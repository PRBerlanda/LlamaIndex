"""Text-to-SQL assistant for the Zoop e-commerce database (data/ecommerce.db).

Steps (the course's QueryPipeline, written as plain functions because QueryPipeline
was removed from llama-index-core 0.13+):

    question -> table retriever (embeddings of table descriptions) -> table schema + description
             -> prompt 1 -> LLM -> SQL query -> run on SQLite (read-only) -> result
             -> prompt 2 -> LLM -> final answer
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from llama_index.core import PromptTemplate, SQLDatabase, VectorStoreIndex
from llama_index.core.objects import ObjectIndex, SQLTableNodeMapping, SQLTableSchema
from sqlalchemy import MetaData, create_engine

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "ecommerce.db"
DESCRICOES_PATH = ROOT / "data" / "descricoes_tabelas.json"
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")

PROMPT_DESCRICAO = (
    "Analise a amostra da tabela '{nome_tabela}' abaixo e forneça uma curta e breve descrição "
    "do conteúdo dessa tabela.\n"
    "Informe até o máximo de 5 valores únicos de cada coluna.\n\n"
    "Amostra da Tabela:\n{df_amostra}\n\nDescrição:\n"
)

TEXTO2SQL = PromptTemplate(
    "Dada uma pergunta em linguagem natural, crie uma consulta {dialect} sintaticamente correta "
    "para executar e, em seguida, verifique os resultados da consulta e retorne a resposta. "
    "Você pode ordenar os resultados por uma coluna relevante para retornar os exemplos mais "
    "informativos no banco de dados.\n\n"
    "Nunca consulte todas as colunas de uma tabela específica. Pergunte apenas por algumas "
    "colunas relevantes, de acordo com a pergunta.\n\n"
    "Preste atenção para usar apenas os nomes de colunas que você pode ver na descrição do "
    "esquema. Tenha cuidado para não consultar colunas que não existem. Preste atenção em qual "
    "coluna está em qual tabela. Além disso, qualifique os nomes das colunas com o nome da "
    "tabela quando necessário. Os valores de texto estão em minúsculas e sem acentos.\n\n"
    "Use o seguinte formato, cada um em uma linha:\n\n"
    "Pergunta: Pergunta aqui\n"
    "ConsultaSQL: Consulta SQL para executar\n"
    "ResultadoSQL: Resultado da ConsultaSQL\n"
    "Resposta: Resposta final aqui\n\n"
    "Use apenas as tabelas listadas abaixo.\n\n"
    "{schema}\n\n"
    "Pergunta: {pergunta_user}\n"
    "ConsultaSQL:"
)

PROMPT_2 = PromptTemplate(
    'Você é o "Assistente de consulta de banco de dados da Zoop".\n'
    "Dada a seguinte pergunta, a consulta SQL correspondente e o resultado SQL, responda à "
    "pergunta de modo agradável e objetivamente.\n"
    'Evite iniciar conversas com cumprimentos e apresentações, como "Olá".\n\n'
    "Pergunta: {pergunta_user}\n"
    "Consulta SQL: {consulta}\n"
    "Resultado SQL: {resultado}\n"
    "Resposta:"
)


class SQLNaoPermitido(ValueError):
    """The generated SQL is not a single read-only SELECT."""


@dataclass
class Resposta:
    pergunta: str
    tabelas: list[str]
    sql: str
    resultado: str
    resposta: str


def conectar(db_path: Path = DB_PATH):
    engine = create_engine(f"sqlite:///{db_path}")
    metadata = MetaData()
    metadata.reflect(engine)
    return engine, metadata


def embedding_padrao():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(model_name=EMBED_MODEL)


def descrever_tabelas(engine, metadata, llm, cache: Path | None = DESCRICOES_PATH) -> dict[str, str]:
    """Ask the LLM for a short description of every table (cached in a JSON file)."""
    if cache and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    descricoes = {}
    for nome in metadata.tables:
        amostra = pd.read_sql_table(nome, engine).head(5).to_string()
        descricoes[nome] = llm.complete(PROMPT_DESCRICAO.format(nome_tabela=nome, df_amostra=amostra)).text
    if cache:
        cache.write_text(json.dumps(descricoes, ensure_ascii=False, indent=2), encoding="utf-8")
    return descricoes


def criar_retriever_tabelas(sql_database, descricoes: dict[str, str], embed_model, top_k: int = 1):
    """Vector index over the table descriptions: picks the table(s) a question is about."""
    schemas = [SQLTableSchema(table_name=nome, context_str=desc) for nome, desc in descricoes.items()]
    obj_index = ObjectIndex.from_objects(
        schemas, SQLTableNodeMapping(sql_database), VectorStoreIndex, embed_model=embed_model
    )
    return obj_index.as_retriever(similarity_top_k=top_k)


def descricao_tabela(sql_database, schemas: list[SQLTableSchema]) -> str:
    partes = []
    for schema in schemas:
        info = sql_database.get_single_table_info(schema.table_name)
        if schema.context_str:
            info += " A descrição da tabela é: " + schema.context_str
        partes.append(info)
    return "\n\n".join(partes)


def extrair_sql(texto: str) -> str:
    """Take the SQL out of the LLM answer.

    The course version used .strip('```'), which leaves the language tag when the
    model answers with ```sql ... ``` (the query then starts with "sql\\nSELECT").
    """
    texto = texto.split("ConsultaSQL:", 1)[-1]
    texto = re.split(r"\n\s*(ResultadoSQL|Resposta)\s*:", texto, maxsplit=1)[0]
    bloco = re.search(r"```(?:sql)?\s*(.*?)```", texto, flags=re.S | re.I)
    if bloco:
        texto = bloco.group(1)
    return texto.strip().strip("`").strip().rstrip(";").strip()


def validar_sql(sql: str) -> str:
    """Only one SELECT/WITH statement: the LLM must never change the database."""
    if ";" in sql or not re.match(r"^\s*(SELECT|WITH)\b", sql, flags=re.I):
        raise SQLNaoPermitido(f"Apenas uma consulta SELECT é permitida: {sql[:120]!r}")
    if re.search(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|PRAGMA)\b", sql, flags=re.I):
        raise SQLNaoPermitido(f"Comando não permitido: {sql[:120]!r}")
    return sql


def responder(pergunta: str, llm, sql_database, retriever) -> Resposta:
    schemas = retriever.retrieve(pergunta)
    prompt_1 = TEXTO2SQL.format(
        dialect=sql_database.engine.dialect.name,
        schema=descricao_tabela(sql_database, schemas),
        pergunta_user=pergunta,
    )
    sql = extrair_sql(llm.complete(prompt_1).text)
    try:
        resultado, _ = sql_database.run_sql(validar_sql(sql))
    except Exception as erro:  # send the error to the second prompt instead of crashing
        resultado = f"Erro ao executar a consulta: {erro}"
    final = llm.complete(PROMPT_2.format(pergunta_user=pergunta, consulta=sql, resultado=resultado)).text
    return Resposta(pergunta, [s.table_name for s in schemas], sql, str(resultado), final.strip())


def montar_assistente(llm, embed_model, db_path: Path = DB_PATH, cache: Path | None = DESCRICOES_PATH):
    """Everything the app needs: (sql_database, table retriever)."""
    engine, metadata = conectar(db_path)
    sql_database = SQLDatabase(engine)
    descricoes = descrever_tabelas(engine, metadata, llm, cache)
    return sql_database, criar_retriever_tabelas(sql_database, descricoes, embed_model)
