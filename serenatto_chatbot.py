"""RAG chatbot for Serenatto, an online store of specialty coffee.

PDF (FAQ + product sheet) -> chunks (SentenceSplitter) -> embeddings (multilingual E5)
-> Chroma vector store -> chat engine in "context" mode with a summarised memory.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter

ROOT = Path(__file__).resolve().parent
PDF = ROOT / "data" / "serenatto_cafes_especiais.pdf"
EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-large")

SYSTEM_PROMPT = (
    "Você é especialista em cafés especiais da Serenatto, uma loja online que vende grãos de "
    "cafés torrados. Sua função é tirar dúvidas de forma simpática e natural sobre os grãos "
    "disponíveis, usando apenas as informações do contexto. Se não souber, diga que não sabe."
)


def embedding_padrao():
    """E5 models expect the 'query: ' / 'passage: ' prefixes."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(
        model_name=EMBED_MODEL, query_instruction="query: ", text_instruction="passage: "
    )


def limpar_texto(texto: str) -> str:
    """pypdf extracts this PDF with a line break between every word ("Loja\\n \\nonline");
    collapse the whitespace so chunks hold real sentences and embeddings are not diluted."""
    return re.sub(r"\s+", " ", texto).strip()


def carregar_chunks(pdf: Path = PDF, chunk_size: int = 1200):
    """Read the PDF (one Document per page), clean the text and split it into chunks."""
    documentos = SimpleDirectoryReader(input_files=[str(pdf)]).load_data()
    for doc in documentos:
        doc.set_content(limpar_texto(doc.get_content()))
    return SentenceSplitter(chunk_size=chunk_size).get_nodes_from_documents(documentos)


def construir_indice(embed_model, persist_dir: str | Path | None = ROOT / "chroma_db",
                     collection: str = "documentos"):
    """Embed the chunks once; later runs reopen the stored Chroma collection."""
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore

    client = chromadb.PersistentClient(path=str(persist_dir)) if persist_dir else chromadb.EphemeralClient()
    chroma_collection = client.get_or_create_collection(collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    if chroma_collection.count() > 0:
        return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex(carregar_chunks(), storage_context=storage_context, embed_model=embed_model)


def criar_chat_engine(index, llm, token_limit: int = 256, top_k: int = 2):
    """Chat engine that retrieves context for every message and keeps a summarised history."""
    memory = ChatSummaryMemoryBuffer.from_defaults(llm=llm, token_limit=token_limit)
    return index.as_chat_engine(
        chat_mode="context",
        llm=llm,
        memory=memory,
        similarity_top_k=top_k,
        system_prompt=SYSTEM_PROMPT,
    )


def trechos_relevantes(index, pergunta: str, top_k: int = 2):
    """Chunks the retriever selects for a question (useful to inspect the RAG step)."""
    return index.as_retriever(similarity_top_k=top_k).retrieve(pergunta)
