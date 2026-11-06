"""Book recommender for the "Buscante" bookstore: RAG + answers forced into JSON.

1. The best-seller catalogue (data/livros_mais_vendidos.csv) and the category
   articles (data/artigos/*.txt) are embedded and stored in a Chroma vector database.
2. For each question, the most similar chunks are retrieved and sent to the LLM
   with a prompt that requires a JSON answer.
3. The JSON is parsed and every recommended title is checked against the catalogue,
   so the bot never recommends a book the store does not sell.
4. A second LLM call turns the validated JSON into a friendly answer.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd
from llama_index.core import PromptTemplate, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CATALOGO = DATA_DIR / "livros_mais_vendidos.csv"
ARTIGOS = DATA_DIR / "artigos"
EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-large")

QA_PROMPT = PromptTemplate(
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "Responda em português.\n"
    "GERE APENAS um objeto JSON válido, e NADA MAIS. Sem texto adicional, explicações ou conversa.\n"
    "O formato JSON DEVE ser exatamente este:\n"
    "{\n"
    '  "data": {\n'
    '    "livros": [\n'
    '      {"titulo": "Titulo do livro", "categoria": "Categoria do livro", '
    '"justificativa": "Porque recomenda"}\n'
    "    ]\n"
    "  }\n"
    "}\n"
    "Use o título exatamente como aparece no contexto. "
    "Só recomende livros dessa base de dados para a query: {query_str}\n"
)

FORMAT_PROMPT = (
    "Formate a resposta para o usuário.\n"
    "Sua persona é um atendente da livraria Buscante que vai recomendar livros usando a consulta "
    "do usuário como referência.\n"
    "Recomende somente os livros que estão em response.\n"
    "consulta: {consulta}\n"
    "response: {response}\n"
)


class LivroNaoEncontrado(ValueError):
    """The LLM recommended a title that is not in the catalogue."""


# --- index -------------------------------------------------------------------

def embedding_padrao():
    """E5 models expect the 'query: ' / 'passage: ' prefixes."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(
        model_name=EMBED_MODEL, query_instruction="query: ", text_instruction="passage: "
    )


def carregar_documentos():
    """Catalogue rows (CSV) + category articles (TXT)."""
    catalogo = SimpleDirectoryReader(input_files=[str(CATALOGO)]).load_data()
    artigos = SimpleDirectoryReader(input_dir=str(ARTIGOS), required_exts=[".txt"]).load_data()
    return catalogo + artigos


def construir_indice(embed_model, persist_dir: str | Path | None = ROOT / "chroma_db",
                     collection: str = "livros"):
    """Build the Chroma index once; on later runs reuse the stored vectors."""
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore

    client = chromadb.PersistentClient(path=str(persist_dir)) if persist_dir else chromadb.EphemeralClient()
    chroma_collection = client.get_or_create_collection(collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    if chroma_collection.count() > 0:  # already embedded: do not pay the embedding time again
        return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(
        carregar_documentos(), storage_context=storage_context, embed_model=embed_model
    )


class RAGStringQueryEngine(CustomQueryEngine):
    """Retrieve context, fill QA_PROMPT and return the raw LLM text."""

    retriever: BaseRetriever
    llm: object
    qa_prompt: PromptTemplate = QA_PROMPT

    def custom_query(self, query_str: str) -> str:
        nodes = self.retriever.retrieve(query_str)
        context_str = "\n\n".join(n.node.get_content() for n in nodes)
        return self.llm.complete(self.qa_prompt.format(context_str=context_str, query_str=query_str)).text


# --- JSON handling -------------------------------------------------------------

def extrair_json(texto: str) -> dict:
    """Parse the LLM answer, tolerating ```json fences or text around the object."""
    bloco = re.search(r"```(?:json)?\s*(.*?)```", texto, flags=re.S)
    if bloco:
        texto = bloco.group(1)
    inicio, fim = texto.find("{"), texto.rfind("}")
    if inicio == -1 or fim == -1:
        raise ValueError(f"A resposta não contém JSON: {texto[:200]!r}")
    return json.loads(texto[inicio:fim + 1])


def normalizar(titulo: str) -> str:
    titulo = unicodedata.normalize("NFC", str(titulo)).upper().strip()
    return re.sub(r"\s+", " ", titulo)


def validando(resposta_json: dict, df: pd.DataFrame) -> list[dict]:
    """Check that EVERY recommended title exists in the catalogue (exact match, case-insensitive)."""
    titulos = set(df["nome"].map(normalizar))
    livros = resposta_json["data"]["livros"]
    for livro in livros:
        titulo = livro.get("titulo", "")
        if normalizar(titulo) not in titulos:
            raise LivroNaoEncontrado(f"Livro {titulo!r} não encontrado no catálogo")
    return livros


def formatando_respostas(response: str, consulta: str, llm) -> str:
    return llm.complete(FORMAT_PROMPT.format(consulta=consulta, response=response)).text


def pipeline_completo(consulta: str, query_engine, llm, df: pd.DataFrame) -> str:
    resposta = str(query_engine.query(consulta))
    resposta_json = extrair_json(resposta)
    validando(resposta_json, df)
    return formatando_respostas(json.dumps(resposta_json, ensure_ascii=False), consulta, llm)
